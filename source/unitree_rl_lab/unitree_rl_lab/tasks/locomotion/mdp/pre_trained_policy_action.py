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





# class FootstepPolicyAction(ActionTerm):
#     r"""High-level footstep action term using a pre-trained low-level policy.

#     High-level policy outputs 4-feet foothold targets.
#     Low-level policy takes those as commands + proprio/force and outputs joint targets.
#     """

#     cfg: FootstepPolicyActionCfg

#     def __init__(self, cfg: FootstepPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
#         super().__init__(cfg, env)

#         self.robot: Articulation = env.scene[cfg.asset_name]

#         # --- load low-level policy (already trained) ---
#         if not check_file_path(cfg.policy_path):
#             raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
#         file_bytes = read_file(cfg.policy_path)
#         self.policy = torch.jit.load(file_bytes).to(env.device).eval()

#         # --- high-level "raw" actions = foothold commands ---
#         self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

#         # --- low-level action term (e.g. JointPositionAction) ---
#         self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(
#             cfg.low_level_actions, env
#         )
#         self.low_level_actions = torch.zeros(
#             self.num_envs,
#             self._low_level_action_term.action_dim,
#             device=self.device,
#         )

       
#         # 低レベル観測CFGをメンバに保持しておく
#         self._low_level_obs_cfg = cfg.low_level_observations

#         self.step_cmd_term = env.command_manager.get_term("step_fr_to_block")

#         if hasattr(self._low_level_action_term, "_joint_ids"):
#             self._joint_ids = self._low_level_action_term._joint_ids
#         else:
#             # 念のため fallback
#             self._joint_ids = slice(None)


#         # 前回の low-level action を観測に入れたいとき用
#         def last_action():
#             if hasattr(env, "episode_length_buf"):
#                 self.low_level_actions[env.episode_length_buf == 0, :] = 0
#             return self.low_level_actions


#         low_obs = cfg.low_level_observations  # LOW_LEVEL_ENV_CFG.observations.policy

#         # 1) low-level の "last_action" を内部バッファから取るように書き換え
#         low_obs.last_action.func = lambda dummy_env: last_action()
#         low_obs.last_action.params = dict()   # ★ これが重要

#         # 2) low-level の "position_commands" を「上位 raw_actions」で上書き
#         low_obs.position_commands.func = lambda dummy_env: self._raw_actions
#         low_obs.position_commands.params = dict()   # ★ これも重要

#         # それ以外（proprio, force など）は cfg.low_level_observations に定義済みのまま

#         self._low_level_obs_manager = ObservationManager(
#             {"ll_policy": cfg.low_level_observations}, env
#         )

#         self._counter = 0


#         self.nominal_footholds_b = torch.tensor(
#             [
#                 [ 0.25, 0.15, -0.35],  # FL
#                 [ 0.25,  -0.15, -0.35],  # FR
#                 [-0.25, 0.15, -0.35],  # RL
#                 [-0.25,  -0.15, -0.35],  # RR
#             ],
#             device=env.device,
#             dtype=torch.float32,
#         )

#         self.delta_x_max = 0.2  # [m] 前後には ±15cm までしか動かさない
#         self.delta_y_max = 0.12 # [m] 左右には ±10cm まで
#         self.delta_z_max = 0.01  # [m] 高さ方向 ±5cm








#     # -------- Properties --------

#     @property
#     def action_dim(self) -> int:
#         # 4 feet × (x,y,z) in base frame
#         return 12

#     @property
#     def raw_actions(self) -> torch.Tensor:
#         return self._raw_actions

#     @property
#     def processed_actions(self) -> torch.Tensor:
#         # 上位RLのアクションは [-1,1] だったりするので、
#         # ここで "有効な足先エリア" にスケーリングしてもよい
#         return self.raw_actions

#     # -------- Operations from RL side --------

#     def process_actions(self, actions: torch.Tensor):
#         # もし [-1,1] を物理座標に変換したいならここでやる：
#         #   e.g. self._raw_actions[:] = self._unnormalize(actions)
#         # self._raw_actions[:] = actions


#         B = self.num_envs

#         # [-1,1] にクリップ（保険）
#         a = torch.clamp(actions, -1.0, 1.0).view(B, 4, 3)  # [B,4,3]

#         # 軸ごとにスケーリング → 残差 [m]
#         dx = a[..., 0] * self.delta_x_max  # [B,4]
#         dy = a[..., 1] * self.delta_y_max
#         dz = a[..., 2] * self.delta_z_max

#         delta = torch.stack([dx, dy, dz], dim=-1)  # [B,4,3]

#         # 名目足位置 + 残差
#         foot_targets_b = self.nominal_footholds_b.unsqueeze(0) + delta  # [B,4,3]

#         self._raw_actions[:] = foot_targets_b.view(B, -1)


#         # 上位の PPO を無視して、自前で simple なコマンドを入れる
#         # B = self._raw_actions.shape[0]
#         # foot_targets = torch.zeros(B, 4, 3, device=self.device)

#         # # FL だけ前に出す、他は名目スタンス
#         # foot_targets[:, 0, :] = torch.tensor([0.35,  0.15, -0.30], device=self.device)
#         # foot_targets[:, 1, :] = torch.tensor([0.25, -0.15, -0.30], device=self.device)
#         # foot_targets[:, 2, :] = torch.tensor([-0.15,  0.15, -0.30], device=self.device)
#         # foot_targets[:, 3, :] = torch.tensor([-0.15, -0.15, -0.30], device=self.device)

#         # self._raw_actions[:] = foot_targets.view(B, -1)



    

#     def apply_actions(self):
#         if self._counter % self.cfg.low_level_decimation == 0:

#              # ① 上位の raw_actions (self._raw_actions) を
#             #   step_fr_to_block の command バッファに書き込む
#             # self.step_cmd_term._command[:] = self._raw_actions
#             self.step_cmd_term.set_foot_targets_base(self._raw_actions)

#             # 1) 低レベル観測を取得
#             low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

#             # 2) dict の場合はフラット化して Tensor にする
#             if isinstance(low_level_obs, dict):
#                 B = next(iter(low_level_obs.values())).shape[0]


#                 # 必要なら self.policy 側の期待次元も見ておく
#                 # 例: norm の mean の長さを使う（構造による）
#                 try:
#                     expected = int(self.policy.normalizer.mean.shape[0])
#                     print("policy expects obs_dim =", expected)
#                 except Exception:
#                     pass
#                 obs_tensors = []
#                 # ★ ここポイント：cfg.term_cfgs ではなく「実際の dict のキー順」を使う
#                 # for name, t in low_level_obs.items():
#                 #     # t: [B, ...] を [B, dim_i] に畳む
#                 #     t = t.view(t.shape[0], -1)
#                 #     obs_tensors.append(t)
#                 # low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

#                 for key in LOW_LEVEL_OBS_ORDER:
#                     t = low_level_obs[key]          # [B, ...]
#                     t = t.view(B, -1)               # [B, dim_i]
#                     obs_tensors.append(t)
#                 low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

            
#             else:
#                 # もし将来 io_descriptor を使って最初から Tensor で返ってくるようになった場合
#                 low_level_obs_tensor = low_level_obs



#             # 3) 低レベルポリシーを走らせる
#             with torch.no_grad():
#                 self.low_level_actions[:] = self.policy(low_level_obs_tensor)

#             # 4) 関節アクションへ反映
#             self._low_level_action_term.process_actions(self.low_level_actions)
#             self._counter = 0

#         self._low_level_action_term.apply_actions()
#         self._counter += 1













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
                [ 0.25, 0.15, -0.35],  # FL
                [ 0.25,  -0.15, -0.35],  # FR
                [-0.25, 0.15, -0.35],  # RL
                [-0.25,  -0.15, -0.35],  # RR
            ],
            device=env.device,
            dtype=torch.float32,
        )

        self.delta_x_max = 0.12  # [m] 前後には ±15cm までしか動かさない
        self.delta_y_max = 0.12 # [m] 左右には ±10cm まで
        self.delta_z_max = 0.01  # [m] 高さ方向 ±5cm


        # __init__ 内
        # self._cf_sensor = env.scene.sensors["contact_forces"]

        # # 例：足リンク名（あなたのURDF/IsaacLab設定に合わせる）
        # self._foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # # sensor が body_names を持っている想定（無ければ print して確認）
        # sensor_body_names = list(self._cf_sensor.cfg.body_names) if hasattr(self._cf_sensor, "cfg") else None
        # # if sensor_body_names is None:
        # #     # ここは環境依存なので、最悪固定indexで対応
        # #     # ただし可能なら sensor_body_names を取れるようにするのがおすすめ
        # #     self._foot_ids = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)
        # # else:
        # #     self._foot_ids = torch.tensor([sensor_body_names.index(n) for n in self._foot_body_names],
        # #                                 device=self.device, dtype=torch.long)

        # # liftoff 用状態
        # self._contact_prev = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # self._since_update = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # # 閾値（まずは固定Nで開始 → 後で mg スケールにしてもOK）
        # self.F_ON  = 15.0  # 接触ON判定 [N]（Go2ならこの辺からスタートでOKなことが多い）
        # self.F_OFF = 8.0   # 接触OFF判定 [N]

        # # チャタリング対策
        # self.MIN_INTERVAL = 2    # 50Hzなら0.04s
        # self.MAX_HOLD     = 20   # 50Hzなら0.4s（イベント来ない時の保険）



        self._pending_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # 初期値：名目足位置（もしくはゼロでもOKだが、名目の方が安定）
        self._latched_actions = self.nominal_footholds_b.unsqueeze(0).repeat(self.num_envs, 1, 1).view(self.num_envs, -1)

        # ★低位観測へ渡すコマンドは latched
        low_obs.position_commands.func = lambda dummy_env: self._latched_actions
        low_obs.position_commands.params = dict()

        # contact_forces センサ参照
        self._cf_sensor = env.scene.sensors["contact_forces"]

        # # 足index（例：センサが body_names を持ってる想定。無ければ固定index）
        # sensor_body_names = list(self._cf_sensor.cfg.body_names)
        # foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        # self._foot_ids = torch.tensor([sensor_body_names.index(n) for n in foot_names], device=self.device)


        # センサが保持している “body_names” を使う（cfgじゃない）
        # ContactSensor.body_names はセンサが捕捉している剛体の名前リスト（順序付き）
        # さらに find_bodies で index を引ける
        foot_ids, foot_names = self._cf_sensor.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        )
        self._foot_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)  # shape [4]

        # 接触判定状態
        self._contact_prev = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._since_update = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # 閾値＆ガード（要調整）
        self.F_ON, self.F_OFF = 15.0, 8.0
        self.MIN_INTERVAL = 2
        self.MAX_HOLD = 60


        self._dbg_count = 0
        self._dbg_every = 50   # 50Hzなら約10秒おき（適宜100〜1000で調整）











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

    # def process_actions(self, actions: torch.Tensor):
    #     # もし [-1,1] を物理座標に変換したいならここでやる：
    #     #   e.g. self._raw_actions[:] = self._unnormalize(actions)
    #     # self._raw_actions[:] = actions


    #     B = self.num_envs

    #     # [-1,1] にクリップ（保険）
    #     a = torch.clamp(actions, -1.0, 1.0).view(B, 4, 3)  # [B,4,3]

    #     # 軸ごとにスケーリング → 残差 [m]
    #     dx = a[..., 0] * self.delta_x_max  # [B,4]
    #     dy = a[..., 1] * self.delta_y_max
    #     dz = a[..., 2] * self.delta_z_max

    #     delta = torch.stack([dx, dy, dz], dim=-1)  # [B,4,3]

    #     # 名目足位置 + 残差
    #     foot_targets_b = self.nominal_footholds_b.unsqueeze(0) + delta  # [B,4,3]

    #     self._raw_actions[:] = foot_targets_b.view(B, -1)


    #     # 上位の PPO を無視して、自前で simple なコマンドを入れる
    #     # B = self._raw_actions.shape[0]
    #     # foot_targets = torch.zeros(B, 4, 3, device=self.device)

    #     # # FL だけ前に出す、他は名目スタンス
    #     # foot_targets[:, 0, :] = torch.tensor([0.35,  0.15, -0.30], device=self.device)
    #     # foot_targets[:, 1, :] = torch.tensor([0.25, -0.15, -0.30], device=self.device)
    #     # foot_targets[:, 2, :] = torch.tensor([-0.15,  0.15, -0.30], device=self.device)
    #     # foot_targets[:, 3, :] = torch.tensor([-0.15, -0.15, -0.30], device=self.device)

    #     # self._raw_actions[:] = foot_targets.view(B, -1)


    def process_actions(self, actions: torch.Tensor):
        B = self.num_envs
        a = torch.clamp(actions, -1.0, 1.0).view(B, 4, 3)
        dx = a[..., 0] * self.delta_x_max
        dy = a[..., 1] * self.delta_y_max
        dz = a[..., 2] * self.delta_z_max
        foot_targets_b = self.nominal_footholds_b.unsqueeze(0) + torch.stack([dx, dy, dz], dim=-1)
        self._pending_actions[:] = foot_targets_b.view(B, -1)



    

    def _estimate_contact(self) -> torch.Tensor:
        forces_w = self._cf_sensor.data.net_forces_w  # [B, n_bodies, 3] ここは実フィールド名に合わせる
        f_foot = forces_w[:, self._foot_ids, :]       # [B,4,3]
        fmag = torch.linalg.vector_norm(f_foot, dim=-1)  # [B,4]
        contact_now = torch.where(self._contact_prev, fmag > self.F_OFF, fmag > self.F_ON)
        return contact_now




    # def apply_actions(self):
    #     if self._counter % self.cfg.low_level_decimation == 0:

    #          # ① 上位の raw_actions (self._raw_actions) を
    #         #   step_fr_to_block の command バッファに書き込む
    #         # self.step_cmd_term._command[:] = self._raw_actions
    #         self.step_cmd_term.set_foot_targets_base(self._raw_actions)

    #         # 1) 低レベル観測を取得
    #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

    #         # 2) dict の場合はフラット化して Tensor にする
    #         if isinstance(low_level_obs, dict):
    #             B = next(iter(low_level_obs.values())).shape[0]


    #             # 必要なら self.policy 側の期待次元も見ておく
    #             # 例: norm の mean の長さを使う（構造による）
    #             try:
    #                 expected = int(self.policy.normalizer.mean.shape[0])
    #                 print("policy expects obs_dim =", expected)
    #             except Exception:
    #                 pass
    #             obs_tensors = []
    #             # ★ ここポイント：cfg.term_cfgs ではなく「実際の dict のキー順」を使う
    #             # for name, t in low_level_obs.items():
    #             #     # t: [B, ...] を [B, dim_i] に畳む
    #             #     t = t.view(t.shape[0], -1)
    #             #     obs_tensors.append(t)
    #             # low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

    #             for key in LOW_LEVEL_OBS_ORDER:
    #                 t = low_level_obs[key]          # [B, ...]
    #                 t = t.view(B, -1)               # [B, dim_i]
    #                 obs_tensors.append(t)
    #             low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)  # [B, obs_dim]

            
    #         else:
    #             # もし将来 io_descriptor を使って最初から Tensor で返ってくるようになった場合
    #             low_level_obs_tensor = low_level_obs



    #         # 3) 低レベルポリシーを走らせる
    #         with torch.no_grad():
    #             self.low_level_actions[:] = self.policy(low_level_obs_tensor)

    #         # 4) 関節アクションへ反映
    #         self._low_level_action_term.process_actions(self.low_level_actions)
    #         self._counter = 0

    #     self._low_level_action_term.apply_actions()
    #     self._counter += 1




    def apply_actions(self):



        contact_now = self._estimate_contact()
        liftoff = self._contact_prev & (~contact_now)   # [B,4]
        self._contact_prev = contact_now

        # どの脚のliftoffで更新するか（まずは any でOK）
        update_mask = liftoff.any(dim=-1)               # [B]
        # 例：FRだけなら update_mask = liftoff[:, 1]

        # ガード
        self._since_update += 1
        allow = self._since_update >= self.MIN_INTERVAL
        force = self._since_update >= self.MAX_HOLD
        update_mask = (update_mask & allow) | force

        # ★ latched 更新
        if update_mask.any():
            self._latched_actions[update_mask] = self._pending_actions[update_mask]
            self._since_update[update_mask] = 0

        self.step_cmd_term.set_foot_targets_base(self._latched_actions)


        # ---- 低位 decimation 部分は “latched” を使う以外そのまま ----
        if self._counter % self.cfg.low_level_decimation == 0:
            # self.step_cmd_term.set_foot_targets_base(self._latched_actions)  # ★ここが latched

            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            if isinstance(low_level_obs, dict):
                B = next(iter(low_level_obs.values())).shape[0]
                obs_tensors = []
                for key in LOW_LEVEL_OBS_ORDER:
                    t = low_level_obs[key].reshape(B, -1)
                    obs_tensors.append(t)
                low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)
            else:
                low_level_obs_tensor = low_level_obs

            with torch.no_grad():
                self.low_level_actions[:] = self.policy(low_level_obs_tensor)

            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1



        self._dbg_count += 1
        if (self._dbg_count % self._dbg_every) == 0:
            # スカラー
            upd_mean = update_mask.float().mean().item()
            since_max = int(self._since_update.max().item())

            # 脚ごと [4]
            c_mean = contact_now.float().mean(dim=0)   # [4]
            l_mean = liftoff.float().mean(dim=0)       # [4]
            force_mean = force.float().mean().item()
            since_mean = self._since_update.float().mean().item()

            # tensor を Python list にして見やすくする
            print(
                f"[HL dbg] upd_mean={upd_mean:.3f} since_max={since_max} "
                f"contact_mean={c_mean.detach().cpu().tolist()} "
                f"liftoff_mean={l_mean.detach().cpu().tolist()}"
                f"force_mean={force_mean:.3f} since_mean={since_mean:.1f}"
            )



    





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