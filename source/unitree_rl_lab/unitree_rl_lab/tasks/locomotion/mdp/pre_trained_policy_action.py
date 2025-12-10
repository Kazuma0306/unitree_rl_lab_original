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
    # "base_lin_vel",
    # "base_ang_vel",
    # "projected_gravity",
    # "position_commands",
    # "fr_target_xy_rel",
    # "leg_position",
    # "joint_pos_rel",
    # "joint_vel_rel",
    # "last_action",
    # "ft_stack",           # ← 本当に学習時に入っていたかは要確認
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

    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "position_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action", 
    "fr_target_xy_rel",
    "leg_position",
    "ft_stack"

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


        self.nominal_footholds_b = torch.tensor(
            [
                [ 0.25, 0.18, -0.22],  # FL
                [ 0.25,  -0.18, -0.22],  # FR
                [-0.25, 0.18, -0.22],  # RL
                [-0.25,  -0.18, -0.22],  # RR
            ],
            device=env.device,
            dtype=torch.float32,
        )

        self.delta_x_max = 0.2  # [m] 前後には ±15cm までしか動かさない
        self.delta_y_max = 0.2 # [m] 左右には ±10cm まで
        self.delta_z_max = 0.02  # [m] 高さ方向 ±5cm








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
        # self._raw_actions[:] = actions


        B = self.num_envs

        # [-1,1] にクリップ（保険）
        a = torch.clamp(actions, -1.0, 1.0).view(B, 4, 3)  # [B,4,3]

        # 軸ごとにスケーリング → 残差 [m]
        dx = a[..., 0] * self.delta_x_max  # [B,4]
        dy = a[..., 1] * self.delta_y_max
        dz = a[..., 2] * self.delta_z_max

        delta = torch.stack([dx, dy, dz], dim=-1)  # [B,4,3]

        # 名目足位置 + 残差
        foot_targets_b = self.nominal_footholds_b.unsqueeze(0) + delta  # [B,4,3]

        self._raw_actions[:] = foot_targets_b.view(B, -1)


        # 上位の PPO を無視して、自前で simple なコマンドを入れる
        # B = self._raw_actions.shape[0]
        # foot_targets = torch.zeros(B, 4, 3, device=self.device)

        # # FL だけ前に出す、他は名目スタンス
        # foot_targets[:, 0, :] = torch.tensor([0.35,  0.15, -0.30], device=self.device)
        # foot_targets[:, 1, :] = torch.tensor([0.25, -0.15, -0.30], device=self.device)
        # foot_targets[:, 2, :] = torch.tensor([-0.15,  0.15, -0.30], device=self.device)
        # foot_targets[:, 3, :] = torch.tensor([-0.15, -0.15, -0.30], device=self.device)

        # self._raw_actions[:] = foot_targets.view(B, -1)



    

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:

             # ① 上位の raw_actions (self._raw_actions) を
            #   step_fr_to_block の command バッファに書き込む
            # self.step_cmd_term._command[:] = self._raw_actions
            self.step_cmd_term.set_foot_targets_base(self._raw_actions)

            # 1) 低レベル観測を取得
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # 2) dict の場合はフラット化して Tensor にする
            if isinstance(low_level_obs, dict):
                B = next(iter(low_level_obs.values())).shape[0]


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
            # try:
            #     expected_dim = int(self.policy.normalizer.mean.shape[0])
            # except Exception:
            #     # 正しく取れない場合はエラーから分かっている 256 に決め打ちでもよい
            #     expected_dim = 256

            # cur_dim = low_level_obs_tensor.shape[1]
            # if cur_dim < expected_dim:
            #     pad = expected_dim - cur_dim
            #     pad_zeros = torch.zeros(
            #         (low_level_obs_tensor.shape[0], pad),
            #         device=low_level_obs_tensor.device,
            #         dtype=low_level_obs_tensor.dtype,
            #     )
            #     low_level_obs_tensor = torch.cat([low_level_obs_tensor, pad_zeros], dim=-1)
            # elif cur_dim > expected_dim:
            #     # 念のため大きすぎた場合は切り詰め（多分発生しない）
            #     low_level_obs_tensor = low_level_obs_tensor[:, :expected_dim]



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


