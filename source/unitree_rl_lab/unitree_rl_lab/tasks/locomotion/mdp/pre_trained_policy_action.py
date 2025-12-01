from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



LOW_LEVEL_OBS_ORDER = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "position_commands",
    "fr_target_xy_rel",
    "leg_position",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "ft_stack",           # ← 本当に学習時に入っていたかは要確認
    # ... 実際に rsl_rl_cfg に書いてあったもの全部

    # "base_lin_vel", "base_ang_vel", "projected_gravity",
    # "position_commands", "joint_pos_rel", "joint_vel_rel", "last_action",

    # "base_lin_vel",
    # "base_ang_vel",
    # "projected_gravity",
    # "position_commands",
    # "joint_pos_rel",
    # "joint_vel_rel",
    # "last_action", 
    # "fr_target_xy_rel",
    # "leg_position",
    # # "camera_image",
    # # "front_depth",
    # # "front_normals",
    # # "height_scanner",
    # "ft_stack"

]


class FootstepPolicyAction(ActionTerm):
    r"""High-level footstep action term using a pre-trained low-level policy.

    High-level policy outputs 4-feet foothold targets.
    Low-level policy takes those as commands + proprio/force and outputs joint targets.
    """

    cfg: FootstepPolicyActionCfg

    def __init__(self, cfg: FootstepPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # --- load low-level policy (already trained) ---
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # --- high-level "raw" actions = foothold commands ---
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # --- low-level action term (e.g. JointPositionAction) ---
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(
            cfg.low_level_actions, env
        )
        self.low_level_actions = torch.zeros(
            self.num_envs,
            self._low_level_action_term.action_dim,
            device=self.device,
        )

       
        # 低レベル観測CFGをメンバに保持しておく
        self._low_level_obs_cfg = cfg.low_level_observations

        self.step_cmd_term = env.command_manager.get_term("step_fr_to_block")

        if hasattr(self._low_level_action_term, "_joint_ids"):
            self._joint_ids = self._low_level_action_term._joint_ids
        else:
            # 念のため fallback
            self._joint_ids = slice(None)


        # 前回の low-level action を観測に入れたいとき用
        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # ---- ここが重要：低レベル観測の "コマンド" 部分を差し替える ----
        # あなたの low-level ObsGroup で「足先コマンド」に相当する term 名を使う
        # ここでは仮に "foot_commands" と呼ぶことにします。
        # cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        # cfg.low_level_observations.actions.params = dict()

        # cfg.low_level_observations.position_commands.func = (
        #     lambda dummy_env: self._raw_actions
        # )
        # cfg.low_level_observations.position_commands.params = dict()

        # 1) 低レベルが使う "last_action" 観測を内部バッファで上書き
        # cfg.low_level_observations.last_action.func = lambda dummy_env: last_action()
        # cfg.low_level_observations.last_action.params = dict()

        # # 2) 低レベルが使う "position_commands" 観測を「上位の raw_actions」で上書き
        # #    （上位が出した 12 次元の足先コマンドを、そのまま低レベルの観測に渡すイメージ）
        # cfg.low_level_observations.position_commands.func = lambda dummy_env: self._raw_actions
        # cfg.low_level_observations.params = dict()

        low_obs = cfg.low_level_observations  # LOW_LEVEL_ENV_CFG.observations.policy

        # 1) low-level の "last_action" を内部バッファから取るように書き換え
        low_obs.last_action.func = lambda dummy_env: last_action()
        low_obs.last_action.params = dict()   # ★ これが重要

        # 2) low-level の "position_commands" を「上位 raw_actions」で上書き
        low_obs.position_commands.func = lambda dummy_env: self._raw_actions
        low_obs.position_commands.params = dict()   # ★ これも重要

        # それ以外（proprio, force など）は cfg.low_level_observations に定義済みのまま

        self._low_level_obs_manager = ObservationManager(
            {"ll_policy": cfg.low_level_observations}, env
        )

        self._counter = 0

    # -------- Properties --------

    @property
    def action_dim(self) -> int:
        # 4 feet × (x,y,z) in base frame
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # 上位RLのアクションは [-1,1] だったりするので、
        # ここで "有効な足先エリア" にスケーリングしてもよい
        return self.raw_actions

    # -------- Operations from RL side --------

    def process_actions(self, actions: torch.Tensor):
        # もし [-1,1] を物理座標に変換したいならここでやる：
        #   e.g. self._raw_actions[:] = self._unnormalize(actions)
        self._raw_actions[:] = actions

    # def apply_actions(self):
    #     # low-level policy の decimation に合わせる
    #     if self._counter % self.cfg.low_level_decimation == 0:

    #         # ① 上位の raw_actions (self._raw_actions) を
    #         #    step_fr_to_block の command バッファに書き込む
    #         self.step_cmd_term._command[:] = self._raw_actions

    #         # low-level policy が訓練時と同じ観測を再構成
    #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
    #         # low-level policy 推論 → joint targets など
    #         self.low_level_actions[:] = self.policy(low_level_obs)
    #         self._low_level_action_term.process_actions(self.low_level_actions)
    #         self._counter = 0

    #     # 最終的な low-level actions をロボットに適用
    #     self._low_level_action_term.apply_actions()
    #     self._counter += 1

    
    # def apply_actions(self):
    #     if self._counter % self.cfg.low_level_decimation == 0:
    #         # 1) 低レベル観測（dict）を取得
    #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

    #         # 2) dict -> Tensor にフラット化
    #         if isinstance(low_level_obs, dict):
    #             obs_tensors = []
    #             # term_cfgs は定義順が保持されている前提
    #             for name, term_cfg in self._low_level_obs_cfg.term_cfgs.items():
    #                 # low_level_obs[name]: [B, ....]
    #                 t = low_level_obs[name]
    #                 t = t.view(t.shape[0], -1)  # 各 term を [B, dim_i] に
    #                 obs_tensors.append(t)
    #             low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]
    #         else:
    #             # もし将来 io_descriptor を入れて Tensor で返ってくるようになったとき用
    #             low_level_obs_tensor = low_level_obs

    #         # 3) 低レベルポリシーを実行
    #         with torch.no_grad():
    #             self.low_level_actions[:] = self.policy(low_level_obs_tensor)

    #         # 4) 関節アクションへ反映
    #         self._low_level_action_term.process_actions(self.low_level_actions)
    #         self._counter = 0

    #     self._low_level_action_term.apply_actions()
    #     self._counter += 1

    

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:

             # ① 上位の raw_actions (self._raw_actions) を
            #   step_fr_to_block の command バッファに書き込む
            self.step_cmd_term._command[:] = self._raw_actions
            # 1) 低レベル観測を取得
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # 2) dict の場合はフラット化して Tensor にする
            if isinstance(low_level_obs, dict):
                B = next(iter(low_level_obs.values())).shape[0]

                # print("=== low_level_obs terms and dims ===")
                # total = 0
                # for k, v in low_level_obs.items():
                #     dim_k = int(v[0].numel())
                #     print(f"{k}: {dim_k}")
                #     total += dim_k
                # print("total dim (flat):", total)
                # ここで total が今 225 になっている

                # 必要なら self.policy 側の期待次元も見ておく
                # 例: norm の mean の長さを使う（構造による）
                try:
                    expected = int(self.policy.normalizer.mean.shape[0])
                    print("policy expects obs_dim =", expected)
                except Exception:
                    pass
                obs_tensors = []
                # ★ ここポイント：cfg.term_cfgs ではなく「実際の dict のキー順」を使う
                # for name, t in low_level_obs.items():
                #     # t: [B, ...] を [B, dim_i] に畳む
                #     t = t.view(t.shape[0], -1)
                #     obs_tensors.append(t)
                # low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

                for key in LOW_LEVEL_OBS_ORDER:
                    t = low_level_obs[key]          # [B, ...]
                    t = t.view(B, -1)               # [B, dim_i]
                    obs_tensors.append(t)
                low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

            
            else:
                # もし将来 io_descriptor を使って最初から Tensor で返ってくるようになった場合
                low_level_obs_tensor = low_level_obs

            
            # ★ ここで policy が期待する obs_dim に合わせてパディング　TODO
            try:
                expected_dim = int(self.policy.normalizer.mean.shape[0])
            except Exception:
                # 正しく取れない場合はエラーから分かっている 256 に決め打ちでもよい
                expected_dim = 256

            cur_dim = low_level_obs_tensor.shape[1]
            if cur_dim < expected_dim:
                pad = expected_dim - cur_dim
                pad_zeros = torch.zeros(
                    (low_level_obs_tensor.shape[0], pad),
                    device=low_level_obs_tensor.device,
                    dtype=low_level_obs_tensor.dtype,
                )
                low_level_obs_tensor = torch.cat([low_level_obs_tensor, pad_zeros], dim=-1)
            elif cur_dim > expected_dim:
                # 念のため大きすぎた場合は切り詰め（多分発生しない）
                low_level_obs_tensor = low_level_obs_tensor[:, :expected_dim]



            # 3) 低レベルポリシーを走らせる
            with torch.no_grad():
                self.low_level_actions[:] = self.policy(low_level_obs_tensor)

            # 4) 関節アクションへ反映
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1



@configclass
class FootstepPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FootstepPolicyAction
    asset_name: str = MISSING              # e.g. "robot"
    policy_path: str = MISSING             # low-level .pt
    low_level_decimation: int = 4          # low-level を何ステップに1回回すか
    low_level_actions: ActionTermCfg = MISSING      # 既存の PD アクション
    low_level_observations: ObservationGroupCfg = MISSING  # low-level 訓練時と同じ
    debug_vis: bool = False

    scale: float = 1.0


