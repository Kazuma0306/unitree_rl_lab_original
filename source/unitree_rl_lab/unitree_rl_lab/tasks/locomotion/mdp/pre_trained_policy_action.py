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


import math
import torch
import torch.nn.functional as F
from isaaclab.sensors import patterns
from isaaclab.utils.math import matrix_from_quat  # (w,x,y,z) 前提


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



        # --- projection settings ---
        self._proj_enable = True
        self._proj_alpha = 1.0          # まずは常に投影（あとで0へ下げる）
        self._proj_margin_m = 0.03      # 石のエッジから何m内側を安全にするか（足裏半径+余裕）

        # --- height scanner sensor ---
        # あなたのscene側の名前に合わせる（多くの例で "height_scanner"） :contentReference[oaicite:5]{index=5}
        self._height_scanner = env.scene.sensors["height_scanner"]

        # --- get grid pattern (exact ordering matches ray order) ---
        pat_cfg = self._height_scanner.cfg.pattern_cfg
        ray_starts_l, _ = patterns.grid_pattern(pat_cfg, str(env.device))  # :contentReference[oaicite:6]{index=6}
        self._grid_xy = ray_starts_l[:, 0:2].to(env.device)               # [R,2]
        self._grid_R = self._grid_xy.shape[0]

        # infer grid dims robustly
        xs = torch.unique(self._grid_xy[:, 0])
        ys = torch.unique(self._grid_xy[:, 1])
        self._nx = xs.numel()
        self._ny = ys.numel()

        self._grid_ordering = getattr(pat_cfg, "ordering", "xy")  # :contentReference[oaicite:7]{index=7}
        self._grid_res = float(pat_cfg.resolution)
        self._margin_px = int(math.ceil(self._proj_margin_m / self._grid_res))



        self._proj_alpha = 1.0
        self._proj_alpha_min = 0.0          # 0.1 にしてもOK
        self._proj_warmup_steps = 0         # 最初は固定で1.0にしたいなら >0
        self._proj_anneal_steps = 200000000  # 例：総env-step（後で合わせる）
        self._global_step = 0









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

    

    def _quat_wxyz_to_yaw(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B,4) in (w,x,y,z) -> yaw (B,)"""
        w, x, y, z = q.unbind(-1)
        # yaw around z
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))



    @torch.no_grad()
    def _project_footholds(self, env_ids: torch.Tensor, foot_targets_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        env_ids: [K]
        foot_targets_b: [K,4,3]  (base frame targets)
        returns:
        projected_targets_b: [K,4,3]
        proj_dist_xy: [K,4]  (yaw-frameでどれくらい動かしたか)
        """
        K = env_ids.numel()
        device = self.device

        # --- base pose ---
        root = self.robot.data.root_state_w[env_ids]      # [K,13]
        base_pos_w = root[:, 0:3]
        base_quat_wxyz = root[:, 3:7]                     # IsaacLabは (w,x,y,z) が基本 :contentReference[oaicite:8]{index=8}

        # full rotation (world<-base)
        R_wb = matrix_from_quat(base_quat_wxyz)           # [K,3,3]
        yaw = self._quat_wxyz_to_yaw(base_quat_wxyz)      # [K]
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        # --- convert footholds: base -> world ---
        rel_b = foot_targets_b                             # [K,4,3]
        rel_w = torch.matmul(R_wb.unsqueeze(1), rel_b.unsqueeze(-1)).squeeze(-1)  # [K,4,3]

        # world->yaw-frame XY (yaw only)
        xw, yw, zw = rel_w[..., 0], rel_w[..., 1], rel_w[..., 2]
        x_yaw =  cy[:, None] * xw + sy[:, None] * yw
        y_yaw = -sy[:, None] * xw + cy[:, None] * yw
        des_xy = torch.stack([x_yaw, y_yaw], dim=-1)      # [K,4,2]

        # --- get ray hits z ---
        # ray_hits_w shape is (N_sensors, N_rays, 3) :contentReference[oaicite:9]{index=9}
        hits_w = self._height_scanner.data.ray_hits_w[env_ids]   # [K,R,3]
        hits_z = hits_w[..., 2]
        hits_z = torch.nan_to_num(hits_z, nan=-1.0e6, neginf=-1.0e6, posinf=-1.0e6)

        # stone vs hole (auto-threshold per env)
        zmin = hits_z.amin(dim=1, keepdim=True)
        zmax = hits_z.amax(dim=1, keepdim=True)
        span = (zmax - zmin).clamp(min=1.0e-6)
        # stepping stonesは holes_depth が深い（負）ので、min/max間で分離しやすい :contentReference[oaicite:10]{index=10}
        thresh = zmin + 0.5 * span
        stone = hits_z > thresh  # [K,R]

        # flatなどでspanが小さいときは全部OK扱い
        stone = torch.where((span.squeeze(1) < 0.2)[:, None], torch.ones_like(stone, dtype=torch.bool), stone)

        # --- reshape to 2D for dilation/erosion (depends on ordering) :contentReference[oaicite:11]{index=11}
        if self._grid_ordering == "xy":
            # idx = ix * ny + iy なので (nx,ny) -> (ny,nx) にして pool2d の(H,W)に合わせる
            stone_2d = stone.view(K, self._nx, self._ny).permute(0, 2, 1)  # [K,ny,nx]
        else:  # "yx"
            stone_2d = stone.view(K, self._ny, self._nx)                   # [K,ny,nx]

        # --- build safe region: "not near holes" by dilating holes ---
        hole_2d = (~stone_2d).float().unsqueeze(1)                         # [K,1,ny,nx]
        r = self._margin_px
        if r > 0:
            hole_dil = F.max_pool2d(hole_2d, kernel_size=2*r+1, stride=1, padding=r)
            safe_2d = (hole_dil < 0.5).squeeze(1)                          # [K,ny,nx]
        else:
            safe_2d = stone_2d

        safe = safe_2d.reshape(K, -1)                                      # [K,R]
        # ordering=="xy" のときは safe_2d を permute したので、flatten順が元のray順と一致するよう戻す
        if self._grid_ordering == "xy":
            safe = safe_2d.permute(0, 2, 1).reshape(K, -1)                 # [K,R]

        

        # --- (追加) reachability mask: grid candidates must be within nominal ± delta ---
        # nominal footholds in yaw-frame (same transform as des_xy)
        nom_b = self.nominal_footholds_b.unsqueeze(0).expand(K, -1, -1)  # [K,4,3]
        nom_w = torch.matmul(R_wb.unsqueeze(1), nom_b.unsqueeze(-1)).squeeze(-1)  # [K,4,3]
        nom_xw, nom_yw = nom_w[..., 0], nom_w[..., 1]
        nom_x_yaw =  cy[:, None] * nom_xw + sy[:, None] * nom_yw
        nom_y_yaw = -sy[:, None] * nom_xw + cy[:, None] * nom_yw
        nom_xy = torch.stack([nom_x_yaw, nom_y_yaw], dim=-1)  # [K,4,2]

        # reach mask over grid points: [K,4,R]
        gx = self._grid_xy[:, 0].view(1, 1, self._grid_R)  # [1,1,R]
        gy = self._grid_xy[:, 1].view(1, 1, self._grid_R)  # [1,1,R]

        reach = (gx - nom_xy[..., 0].unsqueeze(-1)).abs() <= self.delta_x_max
        reach = reach & ((gy - nom_xy[..., 1].unsqueeze(-1)).abs() <= self.delta_y_max)

        # combine with safe: safe is [K,R] -> [K,1,R]
        safe_reach = safe.unsqueeze(1) & reach  # [K,4,R]




        # fallback: safeが空なら投影しない
        has_safe = safe.any(dim=1)                                         # [K]

        # --- nearest safe grid point for each foot ---
        # dist^2: [K,4,R]
        # d2 = (des_xy.unsqueeze(2) - self._grid_xy.view(1, 1, self._grid_R, 2)).pow(2).sum(dim=-1)
        # d2 = d2.masked_fill(~safe.unsqueeze(1), 1.0e9)
        # idx = d2.argmin(dim=-1)                                            # [K,4]
        # proj_xy = self._grid_xy[idx]                                       # [K,4,2]

        # # no-safe envs: keep original
        # proj_xy = torch.where(has_safe[:, None, None], proj_xy, des_xy)
        # proj_dist = torch.linalg.vector_norm(proj_xy - des_xy, dim=-1)     # [K,4]


        # --- nearest candidate: prefer safe&reach, fallback to safe only, else keep original ---
        d2 = (des_xy.unsqueeze(2) - self._grid_xy.view(1, 1, self._grid_R, 2)).pow(2).sum(dim=-1)  # [K,4,R]

        # 1) safe & reachable
        d2_sr = d2.masked_fill(~safe_reach, 1.0e9)
        idx_sr = d2_sr.argmin(dim=-1)                 # [K,4]
        has_sr = safe_reach.any(dim=-1)               # [K,4]

        # 2) fallback: safe only（reach内に安全点がない場合）
        d2_s = d2.masked_fill(~safe.unsqueeze(1), 1.0e9)
        idx_s = d2_s.argmin(dim=-1)                   # [K,4]
        has_s = safe.any(dim=-1).unsqueeze(1)         # [K,1]

        idx = torch.where(has_sr, idx_sr, idx_s)      # [K,4]

        proj_xy = self._grid_xy[idx]                  # [K,4,2]

        # 3) fallback: safe自体が無ければ元の点
        proj_xy = torch.where(has_s.unsqueeze(-1), proj_xy, des_xy)
        proj_dist = torch.linalg.vector_norm(proj_xy - des_xy, dim=-1)  # [K,4]


        # --- yaw-frame -> world (XYだけ置き換え) ---
        xw_new =  cy[:, None] * proj_xy[..., 0] - sy[:, None] * proj_xy[..., 1]
        yw_new =  sy[:, None] * proj_xy[..., 0] + cy[:, None] * proj_xy[..., 1]
        rel_w_new = torch.stack([xw_new, yw_new, zw], dim=-1)              # [K,4,3] (zは元のまま)

        # --- world -> base ---
        R_bw = R_wb.transpose(1, 2)
        rel_b_new = torch.matmul(R_bw.unsqueeze(1), rel_w_new.unsqueeze(-1)).squeeze(-1)  # [K,4,3]

        return rel_b_new, proj_dist






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
        # if update_mask.any():
        #     self._latched_actions[update_mask] = self._pending_actions[update_mask]
        #     self._since_update[update_mask] = 0



        if update_mask.any():
            env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)

            # pending: [K,4,3]
            pending = self._pending_actions[env_ids].view(-1, 4, 3)

            if self._proj_enable:
                proj_pending, proj_dist = self._project_footholds(env_ids, pending)

                # 確率alphaで投影（まずはalpha=1.0でOK）
                if self._proj_alpha >= 1.0:
                    pending_exec = proj_pending
                elif self._proj_alpha <= 0.0:
                    pending_exec = pending
                else:
                    use = (torch.rand(env_ids.numel(), device=self.device) < self._proj_alpha)
                    pending_exec = torch.where(use[:, None, None], proj_pending, pending)

                # 書き戻し
                self._pending_actions[env_ids] = pending_exec.reshape(-1, 12)

                # （任意）デバッグ：投影量
                if (self._dbg_count % self._dbg_every) == 0:
                    print(f"[proj] mean_xy_shift={proj_dist.mean().item():.3f} m, alpha={self._proj_alpha:.2f}")

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


        # 1 env step 進んだらカウント（num_envs並列でも「env step」を基準にする）
        self._global_step += self.num_envs

        t = max(0, self._global_step - self._proj_warmup_steps)
        frac = min(1.0, t / float(self._proj_anneal_steps))
        self._proj_alpha = (1.0 - frac) * 1.0 + frac * self._proj_alpha_min




    





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