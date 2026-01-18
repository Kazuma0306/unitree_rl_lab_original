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


from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply

from isaaclab.utils.math import yaw_quat, quat_rotate_inverse
from .observations import *

import torch
from isaaclab.managers import SceneEntityCfg


def quat_wxyz_to_yaw(q_wxyz: torch.Tensor) -> torch.Tensor:
    """q_wxyz: [B,4] (w,x,y,z) -> yaw [B] (rad), Z-up想定"""
    w, x, y, z = q_wxyz.unbind(dim=-1)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """yaw: [B] -> q_wxyz: [B,4]"""
    half = 0.5 * yaw
    cy = torch.cos(half)
    sy = torch.sin(half)
    zeros = torch.zeros_like(cy)
    return torch.stack([cy, zeros, zeros, sy], dim=-1)  # (w,0,0,z)

# def yaw_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
#     """roll/pitch除去してyawだけ残したquat"""
#     yaw = quat_wxyz_to_yaw(q_wxyz)
#     return yaw_to_quat_wxyz(yaw)





def quat_conj(q):
    # q: (...,4) wxyz
    out = q.clone()
    out[..., 1:] *= -1
    return out

def quat_apply(q, v):
    # q: (...,4) wxyz, v: (...,3)
    w, x, y, z = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    # t = 2 * cross(q_vec, v)
    tx = 2*(y*vz - z*vy)
    ty = 2*(z*vx - x*vz)
    tz = 2*(x*vy - y*vx)
    # v' = v + w*t + cross(q_vec, t)
    vpx = vx + w*tx + (y*tz - z*ty)
    vpy = vy + w*ty + (z*tx - x*tz)
    vpz = vz + w*tz + (x*ty - y*tx)
    return torch.stack([vpx, vpy, vpz], dim=-1)

def yaw_quat(q):
    # yaw-only quaternion from full quaternion (wxyz)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)

@torch.no_grad()
def get_foot_pos_yawbase(raw, foot_body_ids):
    robot = raw.scene["robot"]
    root = robot.data.root_state_w            # (B,13)
    base_pos = root[:, 0:3]                   # (B,3)
    base_q   = root[:, 3:7]                   # (B,4) wxyz

    foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :]   # (B,4,3)
    rel_w = foot_pos_w - base_pos[:, None, :]                 # ★引き算

    qy = yaw_quat(base_q)                                      # (B,4)
    rel_b = quat_apply(quat_conj(qy)[:, None, :], rel_w)       # ★yaw逆回転


    return rel_b












# from scripts.rsl_rl.student_train import StudentNetInBEV_FiLM2D_OS8

from unitree_rl_lab.models.student_train import StudentNetInBEV_FiLM2D_OS8


# ---- 新BEV入力のメタ ----
X_MIN_IN = 0.4
X_MAX_IN = 1.2
Y_WIDTH_IN = 0.8
HB_IN = 128
WB_IN = 64

Hb_out, Wb_out = 61, 31  # 旧のまま（H5から取ってOK）

# img meta
IMG_H = 64; IMG_W = 64
FOCAL_LEN = 10.5
H_APERTURE = 20.955
V_APERTURE = H_APERTURE * (IMG_H / IMG_W)
fx = IMG_W * FOCAL_LEN / H_APERTURE
fy = IMG_H * FOCAL_LEN / V_APERTURE
cx = (IMG_W - 1) * 0.5
cy = (IMG_H - 1) * 0.5

t_bc = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32)
# q_bc_wxyz = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)
q_bc_wxyz = torch.tensor([0.2418, -0.6645,  0.6645, -0.2418], dtype=torch.float32)



def quat_to_R_wxyz(q):
    w,x,y,z = q.unbind(-1)
    ww,xx,yy,zz = w*w,x*x,y*y,z*z
    wx,wy,wz = w*x,w*y,w*z
    xy,xz,yz = x*y,x*z,y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

# 例: Base高さが30cmなら、Baseから見て地面は z = -0.30
Z_GROUND_BASE = -0.30

def build_bev_grid_in(device):
    # BEV grid points in base frame (Hb,Wb,3)
    xs = torch.linspace(X_MAX_IN, X_MIN_IN, HB_IN, device=device)
    ys = torch.linspace(Y_WIDTH_IN/2, -Y_WIDTH_IN/2, WB_IN, device=device)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    # zg = torch.zeros_like(xg)
    zg = torch.full_like(xg, Z_GROUND_BASE)
    pts_b = torch.stack([xg, yg, zg], dim=-1)  # (Hb,Wb,3)

    R_bc = quat_to_R_wxyz(q_bc_wxyz.to(device))
    t = t_bc.to(device)

    # base->cam : P_cam = R_bc^T (P_base - t_bc)
    pts_c = torch.einsum("...j,ij->...i", (pts_b - t), R_bc.t())
    X = pts_c[...,0]; Y = pts_c[...,1]; Z = pts_c[...,2]
    valid = Z > 0.1

    u = fx * (X / Z.clamp(min=1e-6)) + cx
    v = fy * (Y / Z.clamp(min=1e-6)) + cy

    grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
    grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0
    grid_x = grid_x.masked_fill(~valid, -2.0)
    grid_y = grid_y.masked_fill(~valid, -2.0)

    return torch.stack([grid_x, grid_y], dim=-1)  # (Hb,Wb,2)




def rgb_to_bev_in(rgb, bev_grid_in):
    """
    rgb: (B,H,W,3) or (B,3,H,W)
    bev_grid_in: (Hb_in, Wb_in, 2)  in [-1,1] for grid_sample
    return: (B,3,Hb_in,Wb_in)
    """
    if rgb.ndim != 4:
        raise RuntimeError(f"rgb must be 4D, got shape={tuple(rgb.shape)}")

    # --- HWC or CHW を判定 ---
    if rgb.shape[-1] == 3:          # (B,H,W,3)
        img = rgb.permute(0, 3, 1, 2).contiguous()
    elif rgb.shape[1] == 3:         # (B,3,H,W)
        img = rgb.contiguous()
    else:
        raise RuntimeError(f"rgb must be (B,H,W,3) or (B,3,H,W), got shape={tuple(rgb.shape)}")

    # --- float化＆正規化 ---
    if img.dtype != torch.float32:
        img = img.float()
    # uint8(0..255) のときだけ割る（floatでも 0..255 の場合があるので max で判定）
    if img.max() > 1.5:
        img = img / 255.0

    B = img.shape[0]
    grid = bev_grid_in.unsqueeze(0).expand(B, -1, -1, -1)  # (B,Hb,Wb,2)
    bev = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return bev





def index_to_xy_round(idx_b4, Hb, Wb, L, W, res):
    ix = idx_b4 // Wb
    iy = idx_b4 %  Wb
    x = ix.float() * res - L/2
    y = iy.float() * res - W/2
    return torch.stack([x, y], dim=-1)  # (B,4,2) or (Bu,2)








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


LEG2I = {"FL":0, "FR":1, "RL":2, "RR":3}



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


        self._latched_cmd = torch.zeros(self.num_envs, 12, device=self.device)
        self._hl_counter = 0
        self.hl_decimation = 10  # 例: 上位を 1/10 更新（50Hzなら5Hz）




        self.Teacher_mode =False # Using Teacher command for walking





        # 2) low-level の "position_commands" を「上位 raw_actions」で上書き

    

        # low_obs.position_commands.func = lambda dummy_env: self._raw_actions
        # low_obs.position_commands.params = dict()   # ★ これも重要

        # Teacherで歩かせたいなら、position_commandsはTeacherのコマンドを読む TODO

        if self.Teacher_mode == True:
            low_obs.position_commands.func = lambda dummy_env: env.command_manager.get_command("step_fr_to_block")
            low_obs.position_commands.params = dict()

        else: 

            low_obs.position_commands.func = lambda dummy_env: self._raw_actions
            low_obs.position_commands.params = dict()


        # low_obs.position_commands.func = lambda dummy_env: self._latched_cmd #TODO
        # low_obs.position_commands.params = dict()



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

        self.delta_x_max = 0.15  # [m] 前後には ±15cm までしか動かさない
        self.delta_y_max = 0.1 # [m] 左右には ±10cm まで
        self.delta_z_max = 0.01  # [m] 高さ方向 ±5cm



        self.hold_current = False
        self._hold_actions = None

        self.use_teacher_cmd = True  # ★Teacherで歩かせる評価モード

        self._pending_actions = torch.zeros(self.num_envs, self.action_dim, device=env.device)
        self._latched_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)




        # for state schedule
        self.T_hold_s   = 0.2   
        self.leg_order = ["FL_foot","FR_foot","RL_foot","RR_foot"]
        self.contact_sensor = env.scene.sensors["contact_forces"]
        names = self.contact_sensor.body_names
        self.sensor_cols = [names.index(n) for n in self.leg_order]
        self.contact_on  = 10.0
        self.contact_off = 6.0
        self.near_on  = 0.16
        self.near_off = 0.20

        self.dropout_s = 0.2
        self.cooldown_s = 0.2  
        self.hold_t = torch.zeros(self.num_envs, device=self.device)
        self.bad_t  = torch.zeros(self.num_envs, device=self.device)

        self.armed_prev   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_state = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.near_state    = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        self._swing_order = torch.tensor([LEG2I["FR"], LEG2I["RL"], LEG2I["FL"], LEG2I["RR"]],
                                 device=env.device, dtype=torch.long)
        

        self._swing_pos = torch.zeros(self.num_envs, device=env.device, dtype=torch.long)  # (B,) まず0から

        self._swing_leg_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._issue_last    = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


        feet_cfg = SceneEntityCfg(
            "robot",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )
        feet_cfg.resolve(self._env.unwrapped.scene)
        self._foot_body_ids = feet_cfg.body_ids  



        # ---- debug (online eval / visualization) ----
        self.debug_issue = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)   # (B,)
        self.debug_issue_leg = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # (B,)


        self._swing_leg_next = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


        self._initialized = False


        self.debug_teacher_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.debug_latched_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.debug_root13 = torch.zeros(self.num_envs, 13, device=self.device)
        self.debug_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


        self.debug_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_pending_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.debug_pending_latched_b = torch.zeros(self.num_envs,4,3, device=self.device)










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

        # if self.use_teacher_cmd:
        #     # Teacherが作っている command（B,12）をそのまま使う
        #     cmd = self._env.command_manager.get_command("step_fr_to_block2")  # (B,12)
        #     self._raw_actions[:] = cmd
        #     return
        
        foot_targets_b = actions.view(B, 4, 3)

        # 任意：安全のためクリップ（範囲はあなたの下位が学習した範囲に合わせる）
        # foot_targets_b[..., 0] = foot_targets_b[..., 0].clamp(-0.6, 0.6)
        # foot_targets_b[..., 1] = foot_targets_b[..., 1].clamp(-0.4, 0.4)
        # foot_targets_b[..., 2] = foot_targets_b[..., 2].clamp(-0.8, -0.15)

        self._raw_actions[:] = foot_targets_b.view(B, -1)


    

    def reset(self, env_ids=None):
        print("[ActionTerm.reset] called", env_ids)
        if env_ids is None:
            env_ids = slice(None)

        foot_pos_b = self._get_foot_pos_b()  # (B,4,3)
        latched_b = self._latched_actions.view(self.num_envs,4,3)
        pending_b = self._pending_actions.view(self.num_envs,4,3)

        latched_b[env_ids] = foot_pos_b[env_ids]
        pending_b[env_ids] = foot_pos_b[env_ids]

        self.hold_t[env_ids] = 0.0
        self.bad_t[env_ids]  = 0.0
        self.armed_prev[env_ids] = False
        self.contact_state[env_ids] = False
        self.near_state[env_ids] = False
        self._swing_pos[env_ids] = 0


    
    @property
    def swing_leg_now(self):
        return self._swing_leg_now  # (B,) long
    

    @property
    def swing_leg_next(self):
        return self._swing_leg_next


    @property
    def issue_last(self):
        return self._issue_last     # (B,) bool
    


    def _get_foot_pos_b(self) -> torch.Tensor:
        """Return foot positions in base frame (full base, not yaw-only).  (B,4,3)"""
        root = self.robot.data.root_state_w  # (B,13+)  pos(3), quat(wxyz)(4), ...
        base_pos_w = root[:, 0:3]
        base_quat_wxyz = root[:, 3:7]

        foot_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids, :]  # (B,4,3)

        rel_w = foot_pos_w - base_pos_w[:, None, :]  # (B,4,3)

        # quat inverse (wxyz)
        q = base_quat_wxyz
        q_inv = q.clone()
        q_inv[:, 1:] *= -1.0

        # IsaacLab の quat_apply を使ってる前提（あなたの既存utilでOK）
        foot_pos_b = quat_apply(q_inv[:, None, :], rel_w)  # (B,4,3)
        return foot_pos_b




    

    def apply_actions(self):

        B = self.num_envs

      


        if self._counter % self.cfg.low_level_decimation == 0:

             # ① 上位の raw_actions (self._raw_actions) を
            #   step_fr_to_block の command バッファに書き込む
            # self.step_cmd_term._command[:] = self._raw_actions



            # # 今の足位置をそのままコマンドにする（=完全に静止しやすい）
            # foot_b43 = ee_pos_base_obs(self._env)   # (B,4,3) すでにObsTermで使ってるやつ
            # self._raw_actions[:] = foot_b43.view(self.num_envs, -1)

            feet_cfg = SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # ★順序固定
            )
            feet_cfg.resolve(self._env.unwrapped.scene)     # ★これで body_ids が入る
            foot_body_ids = feet_cfg.body_ids  

            B = self.num_envs


            # if self._hold_actions is None:
            #     # ★yaw-only base の実足位置を一回だけラッチ
            #     foot_b43 = get_foot_pos_yawbase(self._env.unwrapped, foot_body_ids)  # (B,4,3)
            #     self._hold_actions = foot_b43.view(B, -1).clone()
            # self._raw_actions[:] = self._hold_actions





            # self.step_cmd_term.set_foot_targets_base(self._raw_actions)

            # if self.Teacher_mode == False: # TODO
            #     self.step_cmd_term.set_foot_targets_base(self._raw_actions)
                # self.step_cmd_term.set_foot_targets_base(self._latched_actions)


            # else:
            #     self.step_cmd_term.set_foot_targets_base("step_fr_to_block")
                


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



            # 3) 低レベルポリシーを走らせる
            with torch.no_grad():
                self.low_level_actions[:] = self.policy(low_level_obs_tensor)

            # 4) 関節アクションへ反映
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1





    # def apply_actions(self):
    #     if self._counter % self.cfg.low_level_decimation == 0:

    #          # ① 上位の raw_actions (self._raw_actions) を
    #         #   step_fr_to_block の command バッファに書き込む
    #         # self.step_cmd_term._command[:] = self._raw_actions



    #         # # 今の足位置をそのままコマンドにする（=完全に静止しやすい）
    #         # foot_b43 = ee_pos_base_obs(self._env)   # (B,4,3) すでにObsTermで使ってるやつ
    #         # self._raw_actions[:] = foot_b43.view(self.num_envs, -1)

    #         feet_cfg = SceneEntityCfg(
    #             "robot",
    #             body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # ★順序固定
    #         )
    #         feet_cfg.resolve(self._env.unwrapped.scene)     # ★これで body_ids が入る
    #         foot_body_ids = feet_cfg.body_ids  

    #         B = self.num_envs
    #         dev = self.device





    #         # --- phase change 検出 ---
    #         curr_phase = self.cmd_term.phase                 # [B]
    #         changed = curr_phase != self.prev_phase          # [B]
    #         env_ids = changed.nonzero(as_tuple=False).squeeze(-1)

    #         if env_ids.numel() > 0:
    #             # (A) RGB取得（例：カメラセンサから）
    #             # rgb: (B,64,64,3) uint8 か float
    #             rgb = self._get_front_rgb()                  # <- あなたの実装に合わせる
    #             rgb_u = rgb[env_ids].to(dev)

    #             # (B) 学習時と同じIPMで BEV生成
    #             bev = rgb_to_bev_in(rgb_u, self.bev_grid_in) # (Bu,3,128,64)

    #             # (C) proprio入力
    #             foot_pos_b = self._get_foot_pos_b()[env_ids] # (Bu,4,3)
    #             root = self._get_root_vec()[env_ids]         # (Bu,13) など

    #             with torch.inference_mode():
    #                 foot_logits, _ = self.student_model(bev, foot_pos_b=foot_pos_b, root=root)
    #                 # foot_logits: (Bu,4,K)

    #             # (D) どの脚を更新するか
    #             # 1) phaseが脚順に対応してるならこれが簡単
    #             leg_ids = (curr_phase[env_ids] % 4).long()   # (Bu,)
    #             # もし脚順が違うならここでマップする

    #             # (E) logits -> idx -> (x,y,z)
    #             Bu, _, K = foot_logits.shape
    #             logits_leg = foot_logits[torch.arange(Bu, device=dev), leg_ids, :]  # (Bu,K)
    #             idx = torch.argmax(logits_leg, dim=-1)                              # (Bu,)

    #             xy = index_to_xy_round(idx, Hb=self.Hb_out, Wb=self.Wb_out,
    #                                 L=self.L, W=self.W, res=self.res)            # (Bu,2)
    #             z  = torch.full((Bu,1), self.z_const, device=dev)
    #             tgt_b = torch.cat([xy, z], dim=-1)                                  # (Bu,3)

    #             # (F) holdの肝：その脚の pending だけ更新、他は触らない

    #             raw_b = self._raw_actions.view(B, 4, 3)  # ★これが保持されるターゲット
    #             raw_b[env_ids, leg_ids, :] = tgt_b       # ★該当脚だけ更新

             

    #         # phase記録更新（次ステップで再発火しない）
    #         if changed.any():
    #             self.prev_phase[changed] = curr_phase[changed]







    #         # if self._hold_actions is None:
    #         #     # ★yaw-only base の実足位置を一回だけラッチ
    #         #     foot_b43 = get_foot_pos_yawbase(self._env.unwrapped, foot_body_ids)  # (B,4,3)
    #         #     self._hold_actions = foot_b43.view(B, -1).clone()
    #         # self._raw_actions[:] = self._hold_actions





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

#         # ---- ここで nominal / pending / latched を先に作る ----
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

#         self.delta_x_max = 0.15  # [m] 前後には ±15cm までしか動かさない
#         self.delta_y_max = 0.10 # [m] 左右には ±10cm まで
#         self.delta_z_max = 0.01  # [m] 高さ方向 ±5cm


#         self._pending_actions = torch.zeros(self.num_envs, 12, device=self.device)
#         self._latched_actions = self.nominal_footholds_b.unsqueeze(0).repeat(self.num_envs, 1, 1).view(self.num_envs, -1)

#         # ★LLへ渡す position_commands は必ず latched(12)
#         low_obs.position_commands.func = lambda dummy_env: self._latched_actions
#         low_obs.position_commands.params = dict()

#         # ★ここで ObservationManager を作る（この順番が重要）
#         self._low_level_obs_manager = ObservationManager({"ll_policy": low_obs}, env)


#         self._counter = 0


        


       



#         # self._pending_actions = torch.zeros(self.num_envs, 12, device=self.device)

#         # # 初期値：名目足位置（もしくはゼロでもOKだが、名目の方が安定）
#         # self._latched_actions = self.nominal_footholds_b.unsqueeze(0).repeat(self.num_envs, 1, 1).view(self.num_envs, -1)

#         # # ★低位観測へ渡すコマンドは latched
#         # low_obs.position_commands.func = lambda dummy_env: self._latched_actions
#         # low_obs.position_commands.params = dict()

#         # contact_forces センサ参照
#         self._cf_sensor = env.scene.sensors["contact_forces"]

#         # # 足index（例：センサが body_names を持ってる想定。無ければ固定index）
#         # sensor_body_names = list(self._cf_sensor.cfg.body_names)
#         # foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
#         # self._foot_ids = torch.tensor([sensor_body_names.index(n) for n in foot_names], device=self.device)


#         # センサが保持している “body_names” を使う（cfgじゃない）
#         # ContactSensor.body_names はセンサが捕捉している剛体の名前リスト（順序付き）
#         # さらに find_bodies で index を引ける
#         foot_ids, foot_names = self._cf_sensor.find_bodies(
#             ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
#             preserve_order=True,
#         )
#         self._foot_ids = torch.tensor(foot_ids, device=self.device, dtype=torch.long)  # shape [4]

#         # 接触判定状態
#         self._contact_prev = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
#         self._since_update = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

#         # 閾値＆ガード（要調整）
#         self.F_ON, self.F_OFF = 20.0, 8.0
#         self.MIN_INTERVAL = 2
#         self.MAX_HOLD = 60


#         self._dbg_count = 0
#         self._dbg_every = 50   # 50Hzなら約10秒おき（適宜100〜1000で調整）



#         # --- projection settings ---
#         self._proj_enable = False
#         self._proj_margin_m = 0.04      # 石のエッジから何m内側を安全にするか（足裏半径+余裕）

#         # --- height scanner sensor ---
#         # あなたのscene側の名前に合わせる（多くの例で "height_scanner"） :contentReference[oaicite:5]{index=5}
#         self._height_scanner = env.scene.sensors["height_scanner"]

#         # --- get grid pattern (exact ordering matches ray order) ---
#         pat_cfg = self._height_scanner.cfg.pattern_cfg
#         ray_starts_l, _ = patterns.grid_pattern(pat_cfg, str(env.device))  # :contentReference[oaicite:6]{index=6}
#         self._grid_xy = ray_starts_l[:, 0:2].to(env.device)               # [R,2]
#         self._grid_R = self._grid_xy.shape[0]

#         # infer grid dims robustly
#         xs = torch.unique(self._grid_xy[:, 0])
#         ys = torch.unique(self._grid_xy[:, 1])
#         self._nx = xs.numel()
#         self._ny = ys.numel()

#         self._grid_ordering = getattr(pat_cfg, "ordering", "xy")  # :contentReference[oaicite:7]{index=7}
#         self._grid_res = float(pat_cfg.resolution)
#         self._margin_px = int(math.ceil(self._proj_margin_m / self._grid_res))



#         self._proj_alpha = 1.0
#         self._proj_alpha_min = 0.1          # 0.1 にしてもOK
#         self._proj_warmup_steps = 0         # 最初は固定で1.0にしたいなら >0
#         self._proj_anneal_steps = 600000000  # 例：総env-step（後で合わせる）
#         self._global_step = 0



#         # --- FSM for swing scheduling ---
#         # self._swing_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0..3
#         self._in_swing  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
#         # self._swing_t   = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

#         self._swing_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)   # 0..3
#         self._liftoff_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

#         # swing_t は bool index 加算より安全にするため dtype は long 推奨
#         self._swing_t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


#         # どの順番で脚を動かすか（例：FR->FL->RR->RL）
#         # foot index: [FL, FR, RL, RR] = [0,1,2,3] 前提　FR　RL　FL　RR
#         self._swing_order = torch.tensor([1, 2, 0, 3], dtype=torch.long, device=self.device)
#         self._swing_order_pos = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # orderの何番目か

#         # ガード（適宜）
#         self.MIN_SWING_STEPS = 8
#         self.MAX_SWING_STEPS = 30

#         # self.MIN_SWING_STEPS = 4
#         # self.MAX_SWING_STEPS = 60




#         # 例：初期値（飛び石ならこれが無難）
#         # self.MIN_SWING_SEC = 0.1   # 120ms（ノイズで即終了を防ぐ）
#         # self.MAX_SWING_SEC = 0.25   # 500ms（詰まり防止）
#         dt = float(self._env.physics_dt)  # sim dt

#         self.MIN_SWING_SEC = self.MIN_SWING_STEPS * dt   # 旧4step相当
#         self.MAX_SWING_SEC = self.MAX_SWING_STEPS * dt   # 旧30step相当

#         # __init__ 等で
#         self._swing_time = torch.zeros(self.num_envs, device=self.device)  # [B] sec




#         # __init__ のどこかで（robot は self.robot）
#         foot_body_ids, foot_names = self.robot.find_bodies(
#             ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
#             preserve_order=True,
#         )
#         self._foot_body_ids = torch.tensor(foot_body_ids, device=self.device, dtype=torch.long)
#         # ここで foot_names が想定順(FL,FR,RL,RR)か print で確認推奨

#         self.swing_clearance = 0.03  # 6cm まずは 3〜8cmで調整


#         # height scanner grid metadata（projectionで既に持ってるならそれを使う）
      

#         self._grid_res = 0.02         # resolution
#         self._grid_L   = 1.2          # size[0] (x方向の長さ)
#         self._grid_W   = 1.2          # size[1] (y方向の幅)
#         self._grid_ordering = "xy"    # GridPatternCfg.ordering と一致させる

#         self._nx = int(round(self._grid_L / self._grid_res)) + 1
#         self._ny = int(round(self._grid_W / self._grid_res)) + 1
#         self._grid_R = self._nx * self._ny

#         # 高さベースの接触判定しきい値（ヒステリシス）
#         self.CLEAR_ON  = 0.001  # [m] これ未満なら「接地」
#         self.CLEAR_OFF = 0.010  # [m] これ超えたら「離地」

#         self.foot_sole_offset = 0.03  # まず 3cm（要調整）

#         self.STOP_RADIUS  = 0.1














#     # -------- Properties --------

#     # @property
#     # def action_dim(self) -> int:
#     #     # 4 feet × (x,y,z) in base frame
#     #     return 12

    
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




#     def process_actions(self, actions: torch.Tensor):
#         B = self.num_envs
#         a = torch.clamp(actions, -1.0, 1.0).view(B, 4, 3)
#         dx = a[..., 0] * self.delta_x_max
#         dy = a[..., 1] * self.delta_y_max
#         dz = a[..., 2] * self.delta_z_max
#         foot_targets_b = self.nominal_footholds_b.unsqueeze(0) + torch.stack([dx, dy, dz], dim=-1)
#         self._pending_actions[:] = foot_targets_b.view(B, -1)


   



    

#     # def _estimate_contact(self) -> torch.Tensor:
#     #     forces_w = self._cf_sensor.data.net_forces_w  # [B, n_bodies, 3] ここは実フィールド名に合わせる
#     #     f_foot = forces_w[:, self._foot_ids, :]       # [B,4,3]
#     #     fmag = torch.linalg.vector_norm(f_foot, dim=-1)  # [B,4]
#     #     contact_now = torch.where(self._contact_prev, fmag > self.F_OFF, fmag > self.F_ON)
#     #     return contact_now


    

#     # def _estimate_contact(self) -> torch.Tensor:
#     #     forces_w = self._cf_sensor.data.net_forces_w
#     #     f_foot = forces_w[:, self._foot_ids, :]     # ← sensor側のfoot idsにするのが重要
#     #     fz = torch.clamp(torch.nan_to_num(f_foot[..., 2], 0.0), min=0.0)  # [B,4]

#     #     # 低パス（任意だけど効く）
#     #     if not hasattr(self, "_fz_ema"):
#     #         self._fz_ema = fz.clone()
#     #     alpha = getattr(self, "contact_ema_alpha", 0.2)   # 0.1〜0.3
#     #     self._fz_ema = alpha * fz + (1 - alpha) * self._fz_ema
#     #     f = self._fz_ema

#     #     # ヒステリシス（元の思想は維持）
#     #     raw = torch.where(self._contact_prev, f > self.F_OFF, f > self.F_ON)  # [B,4]

#     #     # デバウンス（連続カウント）
#     #     if not hasattr(self, "_contact_cnt"):
#     #         self._contact_cnt = torch.zeros_like(raw, dtype=torch.int16)

#     #     onN  = int(getattr(self, "contact_on_N",  2))
#     #     offN = int(getattr(self, "contact_off_N", 3))

#     #     cnt = self._contact_cnt
#     #     cnt = torch.where(raw, torch.clamp(cnt + 1, 0, onN), torch.clamp(cnt - 1, -offN, 0))
#     #     self._contact_cnt = cnt

#     #     contact_now = cnt > 0
#     #     return contact_now


    
#     def _estimate_contact(self) -> torch.Tensor:
#         forces_w = self._cf_sensor.data.net_forces_w
#         f_foot = forces_w[:, self._foot_ids, :]
#         fz = torch.clamp(torch.nan_to_num(f_foot[..., 2], 0.0), min=0.0)  # [B,4]

#         # EMA
#         if not hasattr(self, "_fz_ema"):
#             self._fz_ema = fz.clone()
#         alpha = float(getattr(self, "contact_ema_alpha", 0.2))
#         self._fz_ema = alpha * fz + (1 - alpha) * self._fz_ema
#         f = self._fz_ema

#         # ヒステリシス用の raw_on/off
#         raw_on  = f > self.F_ON
#         raw_off = f < self.F_OFF

#         # デバウンス用カウンタ
#         if not hasattr(self, "_on_cnt"):
#             self._on_cnt = torch.zeros_like(raw_on, dtype=torch.int16)
#             self._off_cnt = torch.zeros_like(raw_on, dtype=torch.int16)

#         onN  = int(getattr(self, "contact_on_N",  2))   # 連続ON回数
#         offN = int(getattr(self, "contact_off_N", 3))   # 連続OFF回数

#         # ON判定カウンタ：未接触のときだけ数える（接触中はリセット）
#         self._on_cnt = torch.where(
#             (~self._contact_prev) & raw_on,
#             torch.clamp(self._on_cnt + 1, 0, onN),
#             torch.zeros_like(self._on_cnt),
#         )

#         # OFF判定カウンタ：接触中のときだけ数える（未接触はリセット）
#         self._off_cnt = torch.where(
#             (self._contact_prev) & raw_off,
#             torch.clamp(self._off_cnt + 1, 0, offN),
#             torch.zeros_like(self._off_cnt),
#         )

#         contact_now = self._contact_prev.clone()
#         contact_now = torch.where((~self._contact_prev) & (self._on_cnt >= onN),  True,  contact_now)
#         contact_now = torch.where(( self._contact_prev) & (self._off_cnt >= offN), False, contact_now)

#         return contact_now






    
#     # def _estimate_contact(self) -> torch.Tensor:
#     #     # current_air_time: [B, n_bodies]  (track_Contactair_time=True のとき)
#     #     air_t = self._cf_sensor.data.current_air_time[:, self._foot_ids]  # [B,4]
#     #     # 接触中は air_time が 0 に張り付く（空中だと増える）
#     #     contact_now = air_t <= 1.0e-6
#     #     return contact_now

    

#     # @torch.no_grad()
#     # def _ground_z_under_feet_from_scanner(self, foot_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     #     """
#     #     foot_pos_w: [B,4,3] world
#     #     returns:
#     #     ground_z: [B,4]  (footの真下っぽい地面z)
#     #     valid:    [B,4]  (scanner範囲内 & hit有効)
#     #     """
#     #     B = foot_pos_w.shape[0]
#     #     device = foot_pos_w.device
#     #     idxB = torch.arange(B, device=device)

#     #     root = self.robot.data.root_state_w   # [B,13]
#     #     base_pos_w = root[:, 0:3]
#     #     base_quat  = root[:, 3:7]  # (wxyz)

#     #     yaw = self._quat_wxyz_to_yaw(base_quat)
#     #     cy, sy = torch.cos(yaw), torch.sin(yaw)

#     #     rel_w = foot_pos_w - base_pos_w[:, None, :]   # [B,4,3]
#     #     xw, yw = rel_w[..., 0], rel_w[..., 1]

#     #     # world -> yaw frame（あなたのprojectionと同じ式）
#     #     x =  cy[:, None] * xw + sy[:, None] * yw
#     #     y = -sy[:, None] * xw + cy[:, None] * yw

#     #     # 範囲チェック（scannerのsize内か）
#     #     x_min, x_max = -0.5 * self._grid_L, 0.5 * self._grid_L
#     #     y_min, y_max = -0.5 * self._grid_W, 0.5 * self._grid_W
#     #     in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

#     #     # 量子化して index 化
#     #     ix = torch.round((x - x_min) / self._grid_res).long().clamp(0, self._nx - 1)
#     #     iy = torch.round((y - y_min) / self._grid_res).long().clamp(0, self._ny - 1)

#     #     if self._grid_ordering == "xy":
#     #         ridx = ix * self._ny + iy
#     #     else:  # "yx"
#     #         ridx = iy * self._nx + ix

#     #     hits_w = self._height_scanner.data.ray_hits_w   # [B,R,3]
#     #     hits_z = hits_w[..., 2]
#     #     hits_z = torch.nan_to_num(hits_z, nan=-1.0e6, neginf=-1.0e6, posinf=-1.0e6)

#     #     ground_z = hits_z[idxB[:, None], ridx]   # [B,4]
#     #     valid = in_range & (ground_z > -1.0e5)   # nan_to_num対策

#     #     return ground_z, valid



#     # @torch.no_grad()
#     # def _estimate_contact_from_height(self) -> tuple[torch.Tensor, torch.Tensor]:
#     #     """
#     #     returns:
#     #     contact_now: [B,4] bool
#     #     clearance:   [B,4] float  (foot_z - ground_z)
#     #     """
#     #     # foot world pos（あなたは _get_foot_pos_b() を持ってるのでそれをworldへ）
#     #     foot_b = self._get_foot_pos_b()  # [B,4,3] base
#     #     root = self.robot.data.root_state_w
#     #     base_pos_w = root[:, 0:3]
#     #     base_quat  = root[:, 3:7]
#     #     from isaaclab.utils.math import quat_apply
#     #     foot_w = quat_apply(base_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4), foot_b.view(-1,3)).view(foot_b.shape) + base_pos_w[:,None,:]

#     #     ground_z, valid = self._ground_z_under_feet_from_scanner(foot_w)
#     #     # clearance = foot_w[..., 2] - ground_z  # [B,4]


#     #     foot_z_eff = foot_w[..., 2] - self.foot_sole_offset
#     #     clearance = foot_z_eff - ground_z

#     #     prev = self._contact_prev
#     #     contact_now = torch.where(prev, clearance < self.CLEAR_OFF, clearance < self.CLEAR_ON)
#     #     contact_now = contact_now & valid      # 範囲外は “非接触” 扱い（保守的）

#     #     return contact_now, clearance




    

#     def _quat_wxyz_to_yaw(self, q: torch.Tensor) -> torch.Tensor:
#         """q: (B,4) in (w,x,y,z) -> yaw (B,)"""
#         w, x, y, z = q.unbind(-1)
#         # yaw around z
#         return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    

#     def _get_foot_pos_b(self) -> torch.Tensor:
#         """return: foot_pos_b [B,4,3]  (base frame)"""
#         root = self.robot.data.root_state_w        # [B,13]
#         base_pos_w = root[:, 0:3]                  # [B,3]
#         base_quat_wxyz = root[:, 3:7]              # [B,4] (w,x,y,z)

#         R_wb = matrix_from_quat(base_quat_wxyz)    # [B,3,3]
#         R_bw = R_wb.transpose(1, 2)                # [B,3,3]

#         foot_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids, :]  # [B,4,3]
#         rel_w = foot_pos_w - base_pos_w[:, None, :]                         # [B,4,3]
#         foot_pos_b = torch.matmul(R_bw.unsqueeze(1), rel_w.unsqueeze(-1)).squeeze(-1)  # [B,4,3]
#         return foot_pos_b



#     # def _get_foot_pos_b(self) -> torch.Tensor:
#     #     root = self.robot.data.root_state_w
#     #     base_pos_w = root[:, 0:3]
#     #     base_quat_wxyz = root[:, 3:7]  # (w,x,y,z)

#     #     foot_pos_w = self.robot.data.body_pos_w[:, self._foot_body_ids, :]  # [B,4,3]
#     #     rel_w = foot_pos_w - base_pos_w[:, None, :]                         # [B,4,3]

#     #     qy = yaw_quat(base_quat_wxyz)                                       # [B,4] yaw-only
#     #     qy_rep = qy[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
#     #     rel_w_f = rel_w.reshape(-1, 3)

#     #     foot_b_f = quat_apply_inverse(qy_rep, rel_w_f)                      # [B*4,3]
#     #     return foot_b_f.view(rel_w.shape)




  


#     # @torch.no_grad()
#     # def _project_footholds(
#     #     self,
#     #     env_ids: torch.Tensor,
#     #     foot_targets_b: torch.Tensor,
#     #     foot_ids: torch.Tensor | None = None,   # [K,Nf] indices in {0,1,2,3}
#     # ) -> tuple[torch.Tensor, torch.Tensor]:
#     #     """
#     #     env_ids: [K]
#     #     foot_targets_b: [K,Nf,3]  (base frame targets; Nf can be 1 or 4)
#     #     foot_ids(optional): [K,Nf]
#     #     returns:
#     #     projected_targets_b: [K,Nf,3]
#     #     proj_dist_xy: [K,Nf]
#     #     """

#     #     assert self._grid_xy.shape[0] == self._height_scanner.data.ray_hits_w.shape[1]
#     #     assert self._grid_xy.shape[0] == self._nx * self._ny

#     #     K = env_ids.numel()
#     #     device = self.device

#     #     assert foot_targets_b.ndim == 3 and foot_targets_b.shape[0] == K and foot_targets_b.shape[2] == 3
#     #     Nf = foot_targets_b.shape[1]

#     #     # ----------------------------
#     #     # base pose (yaw for yaw-frame conversion)
#     #     # ----------------------------
#     #     root = self.robot.data.root_state_w[env_ids]       # [K,13]
#     #     base_quat_wxyz = root[:, 3:7]
#     #     R_wb = matrix_from_quat(base_quat_wxyz)            # [K,3,3]  (full rot)
#     #     yaw = self._quat_wxyz_to_yaw(base_quat_wxyz)       # [K]
#     #     cy, sy = torch.cos(yaw), torch.sin(yaw)

#     #     # ----------------------------
#     #     # desired XY in yaw-frame
#     #     #   base vec -> world (full) -> yaw-only frame
#     #     # ----------------------------
#     #     rel_b = foot_targets_b                                              # [K,Nf,3]
#     #     rel_w = torch.matmul(R_wb.unsqueeze(1), rel_b.unsqueeze(-1)).squeeze(-1)  # [K,Nf,3]
#     #     xw, yw = rel_w[..., 0], rel_w[..., 1]
#     #     x_yaw =  cy[:, None] * xw + sy[:, None] * yw
#     #     y_yaw = -sy[:, None] * xw + cy[:, None] * yw
#     #     des_xy = torch.stack([x_yaw, y_yaw], dim=-1)                         # [K,Nf,2]

#     #     # ----------------------------
#     #     # ray hits -> stone mask
#     #     # ----------------------------
#     #     hits_w = self._height_scanner.data.ray_hits_w[env_ids]               # [K,R,3]
#     #     hits_z = hits_w[..., 2]
#     #     hits_z = torch.nan_to_num(hits_z, nan=-1.0e6, neginf=-1.0e6, posinf=-1.0e6)
#     #     valid = hits_z > -1e5                                                # [K,R]

#     #     zmax = hits_z.masked_fill(~valid, -1.0e9).amax(dim=1, keepdim=True)  # [K,1]

#     #     # ★ 石判定の厚み：太すぎると縁がstone扱いになりやすい
#     #     top_tol = getattr(self, "_top_tol", 0.03)  # 0.02〜0.03推奨（解像度2cmなら特に）
#     #     stone = valid & (hits_z >= (zmax - top_tol))                         # [K,R]

#     #     # reshape to 2D
#     #     if self._grid_ordering == "xy":
#     #         stone_2d = stone.view(K, self._nx, self._ny).permute(0, 2, 1)    # [K,ny,nx]
#     #     else:
#     #         stone_2d = stone.view(K, self._ny, self._nx)                      # [K,ny,nx]

#     #     # ----------------------------
#     #     # safe region: dilate holes by margin_px
#     #     # ----------------------------
#     #     hole_2d = (~stone_2d).float().unsqueeze(1)                            # [K,1,ny,nx]
#     #     r = int(getattr(self, "_margin_px", 0))
#     #     if r > 0:
#     #         hole_dil = F.max_pool2d(hole_2d, kernel_size=2*r+1, stride=1, padding=r)
#     #         safe_2d = (hole_dil < 0.5).squeeze(1)                             # [K,ny,nx]
#     #     else:
#     #         safe_2d = stone_2d

#     #     if self._grid_ordering == "xy":
#     #         safe = safe_2d.permute(0, 2, 1).reshape(K, -1)                    # [K,R]
#     #     else:
#     #         safe = safe_2d.reshape(K, -1)                                     # [K,R]

#     #     # ----------------------------
#     #     # foot_ids (required for reach mask)
#     #     # ----------------------------
#     #     if foot_ids is None:
#     #         if Nf == 4:
#     #             foot_ids = torch.arange(4, device=device, dtype=torch.long).view(1, 4).expand(K, 4)
#     #         elif Nf == 1:
#     #             # 1脚投影は「そのenvで動かす脚」想定（FSM設計に合わせる）
#     #             leg = self._swing_order[self._swing_order_pos[env_ids]]       # [K]
#     #             foot_ids = leg.view(K, 1)
#     #         else:
#     #             raise ValueError(f"foot_ids is required when Nf={Nf} (not 1 or 4).")
#     #     assert foot_ids.shape == (K, Nf)

#     #     # nominal footholds -> yaw-frame XY
#     #     nom_b = self.nominal_footholds_b[foot_ids]                            # [K,Nf,3]
#     #     nom_w = torch.matmul(R_wb.unsqueeze(1), nom_b.unsqueeze(-1)).squeeze(-1)
#     #     nom_xw, nom_yw = nom_w[..., 0], nom_w[..., 1]
#     #     nom_x_yaw =  cy[:, None] * nom_xw + sy[:, None] * nom_yw
#     #     nom_y_yaw = -sy[:, None] * nom_xw + cy[:, None] * nom_yw
#     #     nom_xy = torch.stack([nom_x_yaw, nom_y_yaw], dim=-1)                  # [K,Nf,2]

#     #     # grid points
#     #     R = self._grid_R
#     #     gx = self._grid_xy[:, 0].view(1, 1, R)                                # [1,1,R]
#     #     gy = self._grid_xy[:, 1].view(1, 1, R)

#     #     reach = (gx - nom_xy[..., 0].unsqueeze(-1)).abs() <= self.delta_x_max
#     #     reach = reach & ((gy - nom_xy[..., 1].unsqueeze(-1)).abs() <= self.delta_y_max)  # [K,Nf,R]

#     #     safe_reach  = safe.unsqueeze(1) & reach                               # [K,Nf,R]
#     #     stone_reach = stone.unsqueeze(1) & reach                              # [K,Nf,R]
#     #     safe_only   = safe.unsqueeze(1)                                       # [K,1,R]
#     #     stone_only  = stone.unsqueeze(1)                                      # [K,1,R]

#     #     # distances to all grid cells
#     #     d2 = (des_xy.unsqueeze(2) - self._grid_xy.view(1, 1, R, 2)).pow(2).sum(dim=-1)  # [K,Nf,R]

#     #     BIG = 1.0e9
#     #     # tier 1: safe & reachable
#     #     d2_sr  = d2.masked_fill(~safe_reach, BIG)
#     #     idx_sr = d2_sr.argmin(dim=-1)
#     #     has_sr = safe_reach.any(dim=-1)

#     #     # tier 2: safe only
#     #     d2_s  = d2.masked_fill(~safe_only, BIG)
#     #     idx_s = d2_s.argmin(dim=-1)
#     #     has_s = safe.any(dim=-1).unsqueeze(1).expand(K, Nf)

#     #     # tier 3: stone & reachable（safeが空/厳しすぎるときの保険。穴は絶対選ばない）
#     #     d2_tr  = d2.masked_fill(~stone_reach, BIG)
#     #     idx_tr = d2_tr.argmin(dim=-1)
#     #     has_tr = stone_reach.any(dim=-1)

#     #     # tier 4: stone only（最後の“穴回避”）
#     #     d2_t  = d2.masked_fill(~stone_only, BIG)
#     #     idx_t = d2_t.argmin(dim=-1)
#     #     has_t = stone.any(dim=-1).unsqueeze(1).expand(K, Nf)

#     #     # select index by priority: sr -> s -> tr -> t -> (no projection)
#     #     idx_sel = torch.where(has_sr, idx_sr, idx_s)
#     #     idx_sel = torch.where((~has_sr) & (~has_s) & has_tr, idx_tr, idx_sel)
#     #     idx_sel = torch.where((~has_sr) & (~has_s) & (~has_tr) & has_t, idx_t, idx_sel)


#     #     stone_sel = stone.unsqueeze(1).gather(2, idx_sel.unsqueeze(-1)).squeeze(-1)  # [K,Nf]
#     #     safe_sel  = safe.unsqueeze(1).gather(2, idx_sel.unsqueeze(-1)).squeeze(-1)   # [K,Nf]
#     #     z_sel     = hits_z[:, None, :].expand(K, Nf, hits_z.shape[1]).gather(2, idx_sel.unsqueeze(-1)).squeeze(-1)

#     #     # print("stone_sel_mean", float(stone_sel.float().mean()),
#     #     #     "safe_sel_mean",  float(safe_sel.float().mean()),
#     #     #     "z_sel_min/mean", float(z_sel.min()), float(z_sel.mean()))


#     #     # if no stone at all, keep original target (no projection)
#     #     has_any = has_sr | has_s | has_tr | has_t   # [K,Nf]
#     #     proj_xy = self._grid_xy[idx_sel]            # [K,Nf,2]
#     #     proj_xy = torch.where(has_any.unsqueeze(-1), proj_xy, des_xy)

#     #     proj_dist = torch.linalg.vector_norm(proj_xy - des_xy, dim=-1)        # [K,Nf]

#     #     # --- yaw-frame -> world (XYだけ) ---
#     #     xw_new =  cy[:, None] * proj_xy[..., 0] - sy[:, None] * proj_xy[..., 1]
#     #     yw_new =  sy[:, None] * proj_xy[..., 0] + cy[:, None] * proj_xy[..., 1]
#     #     rel_w_new_xy = torch.stack([xw_new, yw_new], dim=-1)                  # [K,Nf,2]

#     #     # world -> base（XYだけ戻す）
#     #     # ここでは “元の目標のZ(base)” を保持する（投影でZを壊さない）
#     #     R_bw = R_wb.transpose(1, 2)                                           # [K,3,3]
#     #     rel_w_new = torch.zeros_like(rel_w)
#     #     rel_w_new[..., 0:2] = rel_w_new_xy
#     #     rel_b_new = torch.matmul(R_bw.unsqueeze(1), rel_w_new.unsqueeze(-1)).squeeze(-1)  # [K,Nf,3]
#     #     rel_b_new[..., 2] = foot_targets_b[..., 2]                            # ★Zは元のまま

#     #     return rel_b_new, proj_dist


    

#     @torch.no_grad()
#     def _project_footholds(self, env_ids, foot_targets_b, foot_ids=None):
#         K = env_ids.numel()
#         device = self.device
#         Nf = foot_targets_b.shape[1]

#         # --- base pose ---
#         root = self.robot.data.root_state_w[env_ids]
#         base_quat_wxyz = root[:, 3:7]
#         R_wb = matrix_from_quat(base_quat_wxyz)
#         yaw = self._quat_wxyz_to_yaw(base_quat_wxyz)
#         cy, sy = torch.cos(yaw), torch.sin(yaw)

#         # --- desired XY in yaw-frame ---
#         rel_b = foot_targets_b
#         rel_w = torch.matmul(R_wb.unsqueeze(1), rel_b.unsqueeze(-1)).squeeze(-1)
#         xw, yw = rel_w[..., 0], rel_w[..., 1]
#         x_yaw =  cy[:, None] * xw + sy[:, None] * yw
#         y_yaw = -sy[:, None] * xw + cy[:, None] * yw
#         des_xy = torch.stack([x_yaw, y_yaw], dim=-1)  # [K,Nf,2]

#         # --- hits -> stone mask ---
#         hits_w = self._height_scanner.data.ray_hits_w[env_ids]  # [K,R,3]
#         hits_z = torch.nan_to_num(hits_w[..., 2], nan=-1e6, neginf=-1e6, posinf=-1e6)
#         valid = hits_z > -1e5

#         zmax = hits_z.masked_fill(~valid, -1e9).amax(dim=1, keepdim=True)
#         top_tol = getattr(self, "_top_tol", 0.03)
#         stone = valid & (hits_z >= (zmax - top_tol))  # [K,R]

#         # reshape to 2D
#         if self._grid_ordering == "xy":
#             stone_2d = stone.view(K, self._nx, self._ny).permute(0, 2, 1)  # [K,ny,nx]
#         else:
#             stone_2d = stone.view(K, self._ny, self._nx)                    # [K,ny,nx]

#         # --- safe: hole dilate by margin_px (★ここが縁回避の本体) ---
#         # 重要: margin_pxは 1以上推奨。0だと縁OKになって溝に見える。
#         r = int(getattr(self, "_margin_px", 2))  # ←まず2を推奨（解像度2cmなら4cmバッファ）
#         hole_2d = (~stone_2d).float().unsqueeze(1)
#         hole_dil = F.max_pool2d(hole_2d, kernel_size=2*r+1, stride=1, padding=r)
#         safe_2d = (hole_dil < 0.5).squeeze(1)  # [K,ny,nx]

#         if self._grid_ordering == "xy":
#             safe = safe_2d.permute(0, 2, 1).reshape(K, -1)
#         else:
#             safe = safe_2d.reshape(K, -1)

#         # --- foot_ids ---
#         if foot_ids is None:
#             if Nf == 4:
#                 foot_ids = torch.arange(4, device=device).view(1,4).expand(K,4)
#             elif Nf == 1:
#                 leg = self._swing_order[self._swing_order_pos[env_ids]]
#                 foot_ids = leg.view(K,1)
#             else:
#                 raise ValueError("foot_ids required")
#         # [K,Nf]

#         # --- reach mask ---
#         nom_b = self.nominal_footholds_b[foot_ids]  # [K,Nf,3]
#         nom_w = torch.matmul(R_wb.unsqueeze(1), nom_b.unsqueeze(-1)).squeeze(-1)
#         nom_xw, nom_yw = nom_w[...,0], nom_w[...,1]
#         nom_x_yaw =  cy[:, None]*nom_xw + sy[:, None]*nom_yw
#         nom_y_yaw = -sy[:, None]*nom_xw + cy[:, None]*nom_yw
#         nom_xy = torch.stack([nom_x_yaw, nom_y_yaw], dim=-1)  # [K,Nf,2]

#         R = self._grid_R
#         gx = self._grid_xy[:,0].view(1,1,R)
#         gy = self._grid_xy[:,1].view(1,1,R)
#         reach = (gx - nom_xy[...,0].unsqueeze(-1)).abs() <= self.delta_x_max
#         reach = reach & ((gy - nom_xy[...,1].unsqueeze(-1)).abs() <= self.delta_y_max)  # [K,Nf,R]

#         safe_reach = safe.unsqueeze(1) & reach  # [K,Nf,R]

#         # --- nearest safe&reach ---
#         d2 = (des_xy.unsqueeze(2) - self._grid_xy.view(1,1,R,2)).pow(2).sum(dim=-1)
#         BIG = 1e9
#         d2_sr = d2.masked_fill(~safe_reach, BIG)
#         idx_sr = d2_sr.argmin(dim=-1)              # [K,Nf]
#         has_sr = safe_reach.any(dim=-1)            # [K,Nf]

#         # ★ フォールバック：safe&reach が無いなら「今の足位置」にする（動かさない）
#         # 現在足(base) -> world(full) -> yaw で cur_xy を作る
#         feet_b_all = self._get_foot_pos_b()[env_ids]  # [K,4,3]
#         feet_b = feet_b_all.gather(
#             1, foot_ids[..., None].expand(K, Nf, 3)
#         )  # [K,Nf,3]
#         feet_w = torch.matmul(R_wb.unsqueeze(1), feet_b.unsqueeze(-1)).squeeze(-1)
#         fxw, fyw = feet_w[...,0], feet_w[...,1]
#         cur_x_yaw =  cy[:, None]*fxw + sy[:, None]*fyw
#         cur_y_yaw = -sy[:, None]*fxw + cy[:, None]*fyw
#         cur_xy = torch.stack([cur_x_yaw, cur_y_yaw], dim=-1)  # [K,Nf,2]

#         proj_xy = self._grid_xy[idx_sr]  # [K,Nf,2]
#         proj_xy = torch.where(has_sr.unsqueeze(-1), proj_xy, cur_xy)  # ★ここ

#         proj_dist = torch.linalg.vector_norm(proj_xy - des_xy, dim=-1)

#         # yaw -> world XY
#         xw_new =  cy[:, None]*proj_xy[...,0] - sy[:, None]*proj_xy[...,1]
#         yw_new =  sy[:, None]*proj_xy[...,0] + cy[:, None]*proj_xy[...,1]
#         rel_w_new = torch.zeros_like(rel_w)
#         rel_w_new[...,0] = xw_new
#         rel_w_new[...,1] = yw_new

#         # world -> base
#         R_bw = R_wb.transpose(1,2)
#         rel_b_new = torch.matmul(R_bw.unsqueeze(1), rel_w_new.unsqueeze(-1)).squeeze(-1)

#         # Zは元のターゲットを維持（必要ならここで足現在ZにしてもOK）
#         rel_b_new[...,2] = foot_targets_b[...,2]

#         return rel_b_new, proj_dist




   





#     # def apply_actions(self):



#     #     contact_now = self._estimate_contact()
#     #     liftoff = self._contact_prev & (~contact_now)   # [B,4]
#     #     self._contact_prev = contact_now

#     #     # どの脚のliftoffで更新するか（まずは any でOK）
#     #     update_mask = liftoff.any(dim=-1)               # [B]
#     #     # 例：FRだけなら update_mask = liftoff[:, 1]

#     #     # ガード
#     #     self._since_update += 1
#     #     allow = self._since_update >= self.MIN_INTERVAL
#     #     force = self._since_update >= self.MAX_HOLD
#     #     update_mask = (update_mask & allow) | force

#     #     # ★ latched 更新
#     #     # if update_mask.any():
#     #     #     self._latched_actions[update_mask] = self._pending_actions[update_mask]
#     #     #     self._since_update[update_mask] = 0



#     #     if update_mask.any():
#     #         env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)

#     #         # pending: [K,4,3]
#     #         pending = self._pending_actions[env_ids].view(-1, 4, 3)

#     #         if self._proj_enable:
#     #             proj_pending, proj_dist = self._project_footholds(env_ids, pending)

#     #             # 確率alphaで投影（まずはalpha=1.0でOK）
#     #             if self._proj_alpha >= 1.0:
#     #                 pending_exec = proj_pending
#     #             elif self._proj_alpha <= 0.0:
#     #                 pending_exec = pending
#     #             else:
#     #                 use = (torch.rand(env_ids.numel(), device=self.device) < self._proj_alpha)
#     #                 pending_exec = torch.where(use[:, None, None], proj_pending, pending)

#     #             # 書き戻し
#     #             self._pending_actions[env_ids] = pending_exec.reshape(-1, 12)

#     #             # （任意）デバッグ：投影量
#     #             if (self._dbg_count % self._dbg_every) == 0:
#     #                 print(f"[proj] mean_xy_shift={proj_dist.mean().item():.3f} m, alpha={self._proj_alpha:.2f}")

#     #         self._latched_actions[update_mask] = self._pending_actions[update_mask]
#     #         self._since_update[update_mask] = 0

#     #     self.step_cmd_term.set_foot_targets_base(self._latched_actions)


#     #     # ---- 低位 decimation 部分は “latched” を使う以外そのまま ----
#     #     if self._counter % self.cfg.low_level_decimation == 0:
#     #         # self.step_cmd_term.set_foot_targets_base(self._latched_actions)  # ★ここが latched

#     #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

#     #         if isinstance(low_level_obs, dict):
#     #             B = next(iter(low_level_obs.values())).shape[0]
#     #             obs_tensors = []
#     #             for key in LOW_LEVEL_OBS_ORDER:
#     #                 t = low_level_obs[key].reshape(B, -1)
#     #                 obs_tensors.append(t)
#     #             low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)
#     #         else:
#     #             low_level_obs_tensor = low_level_obs

#     #         with torch.no_grad():
#     #             self.low_level_actions[:] = self.policy(low_level_obs_tensor)

#     #         self._low_level_action_term.process_actions(self.low_level_actions)
#     #         self._counter = 0

#     #     self._low_level_action_term.apply_actions()
#     #     self._counter += 1



#     #     self._dbg_count += 1
#     #     if (self._dbg_count % self._dbg_every) == 0:
#     #         # スカラー
#     #         upd_mean = update_mask.float().mean().item()
#     #         since_max = int(self._since_update.max().item())

#     #         # 脚ごと [4]
#     #         c_mean = contact_now.float().mean(dim=0)   # [4]
#     #         l_mean = liftoff.float().mean(dim=0)       # [4]
#     #         force_mean = force.float().mean().item()
#     #         since_mean = self._since_update.float().mean().item()

#     #         # tensor を Python list にして見やすくする
#     #         print(
#     #             f"[HL dbg] upd_mean={upd_mean:.3f} since_max={since_max} "
#     #             f"contact_mean={c_mean.detach().cpu().tolist()} "
#     #             f"liftoff_mean={l_mean.detach().cpu().tolist()}"
#     #             f"force_mean={force_mean:.3f} since_mean={since_mean:.1f}"
#     #         )


#     #     # 1 env step 進んだらカウント（num_envs並列でも「env step」を基準にする）
#     #     self._global_step += self.num_envs

#     #     t = max(0, self._global_step - self._proj_warmup_steps)
#     #     frac = min(1.0, t / float(self._proj_anneal_steps))
#     #     self._proj_alpha = (1.0 - frac) * 1.0 + frac * self._proj_alpha_min

    


#     # def apply_actions(self):


#     #     B = self.num_envs
#     #     device = self.device
#     #     idx = torch.arange(B, device=device)

#     #     # ---- contact + events（順序はこのままでOK）----
#     #     contact_now = self._estimate_contact()                      # [B,4]

#     #     # contact_now, clearance = self._estimate_contact_from_height()
#     #     prev = self._contact_prev
#     #     liftoff   = prev & (~contact_now)
#     #     touchdown = (~prev) & contact_now
#     #     self._contact_prev = contact_now

#     #     pending_b = self._pending_actions.view(B, 4, 3)
#     #     latched_b = self._latched_actions.view(B, 4, 3)


#     #     foot_pos_b = self._get_foot_pos_b()   # [B,4,3]

#     #     # “いまスイング対象の脚” を env ごとに決める
#     #     # in_swing 中は固定脚、それ以外は次に動かす脚
#     #     swing_leg_now = torch.where(
#     #         self._in_swing,
#     #         self._swing_leg,
#     #         self._swing_order[self._swing_order_pos],
#     #     )  # [B] long

#     #     mask_stance = contact_now.clone()  # [B,4]  接触してる脚だけホールド対象
#     #     mask_stance[idx, swing_leg_now] = False     # スイング脚だけは上書きしない


#     #             # stance (接地) 脚だけ XY を実足に合わせる（横ズレ抑制）

#     #     # Z は実足より少し“下”を目標にして反力を作る（数mm〜2cm程度）
#     #     stance_z_bias = 0.01  # まず1cm

#     #     ms = mask_stance  # [B,4]

#     #     # td_stance = touchdown  # [B,4]（スイング脚除外したければマスクする）
#     #     # latched_b[..., 0] = torch.where(td_stance, foot_pos_b[..., 0], latched_b[..., 0])
#     #     # latched_b[..., 1] = torch.where(td_stance, foot_pos_b[..., 1], latched_b[..., 1])
#     #     # latched_b[..., 2] = torch.where(td_stance, foot_pos_b[..., 2], latched_b[..., 2])

#     #     latched_b[..., 0] = torch.where(ms, foot_pos_b[..., 0], latched_b[..., 0])
#     #     latched_b[..., 1] = torch.where(ms, foot_pos_b[..., 1], latched_b[..., 1])

#     #     # # XYだけホールド
#     #     # latched_b[..., 0] = torch.where(ms, foot_pos_b[..., 0], latched_b[..., 0])
#     #     # latched_b[..., 1] = torch.where(ms, foot_pos_b[..., 1], latched_b[..., 1])

#     #     # # Zは押し込み（支持用）
#     #     # latched_b[..., 2] = torch.where(ms, foot_pos_b[..., 2] - stance_z_bias, latched_b[..., 2])



#     #     # ---- timer（安全な増やし方）----
#     #     self._swing_t += self._in_swing.to(self._swing_t.dtype)      # [B]

#     #     # =========================================
#     #     # 1) done判定（“スイング開始した脚”で見る）
#     #     # =========================================
#     #     if self._in_swing.any():
#     #         leg = self._swing_leg                                     # [B] ★固定脚
#     #         lf_leg = liftoff[idx, leg]                                # [B]
#     #         td_leg = touchdown[idx, leg]                              # [B]

#     #         # liftoff を一度でも見たか（擦って接触が切れない対策にもなる）
#     #         self._liftoff_seen |= (self._in_swing & lf_leg)

#     #         done_td = self._in_swing & self._liftoff_seen & td_leg & (self._swing_t >= self.MIN_SWING_STEPS)
#     #         done_to = self._in_swing & (self._swing_t >= self.MAX_SWING_STEPS)
#     #         done = done_td | done_to

#     #         if done.any():
#     #             self._in_swing[done] = False
#     #             self._swing_t[done] = 0
#     #             self._liftoff_seen[done] = False
#     #             self._swing_order_pos[done] = (self._swing_order_pos[done] + 1) % 4

#     #     # “今スイング対象として観測に出す脚”は、in_swingなら固定脚、そうでなければ次脚
#     #     leg_for_obs = torch.where(self._in_swing, self._swing_leg, self._swing_order[self._swing_order_pos])
#     #     onehot = torch.zeros(B, 4, device=device)
#     #     onehot.scatter_(1, leg_for_obs.view(-1, 1), 1.0)
#     #     self.step_cmd_term.set_swing_leg_onehot(onehot)

#     #     # =========================================
#     #     # 2) start（in_swingでないenvだけ：その脚だけlatched更新）
#     #     # =========================================
#     #     start = ~self._in_swing
#     #     if start.any():
#     #         env_ids = start.nonzero(as_tuple=False).squeeze(-1)
#     #         leg_s = self._swing_order[self._swing_order_pos[env_ids]]          # [K]

#     #         tgt = pending_b[env_ids, leg_s, :]                                 # [K,3]

#     #         if getattr(self, "_proj_enable", False):
#     #             tgt_in = tgt.view(-1, 1, 3)
#     #             tgt_proj, _ = self._project_footholds(env_ids, tgt_in)         # [K,1,3]
#     #             tgt = tgt_proj[:, 0, :]


#     #         # # cur = latched_b[env_ids, leg_s, :2]
#     #         # cur = foot_pos_b[env_ids, leg_s, :2]   # ★実足位置（base）

#     #         # dxy = torch.linalg.vector_norm(tgt[:, :2] - cur, dim=-1)
#     #         # min_step = 0.06

#     #         # # --- Zを持ち上げる（base座標で“上”＝zを増やす＝負なら0に近づける）---
#     #         # # nom_z = self.nominal_footholds_b[leg_s, 2]          # [K]
#     #         # # tgt[:, 2] = torch.maximum(tgt[:, 2], nom_z + self.swing_clearance)
#     #         # cur_z = foot_pos_b[env_ids, leg_s, 2]
#     #         # tgt[:, 2] = torch.maximum(tgt[:, 2], cur_z + self.swing_clearance)

#     #         # scale = (min_step / (dxy + 1e-6)).clamp(min=1.0)
#     #         # tgt[:, :2] = cur + (tgt[:, :2] - cur) * scale.unsqueeze(-1)

#     #         latched_b[env_ids, leg_s, :] = tgt

#     #         self._in_swing[env_ids] = True
#     #         self._swing_t[env_ids] = 0
#     #         self._liftoff_seen[env_ids] = False
#     #         self._swing_leg[env_ids] = leg_s                                   # ★固定




#     # # 書き戻し
#     #     self._latched_actions[:] = latched_b.view(B, -1)
#     #     self.step_cmd_term.set_foot_targets_base(self._latched_actions)



    

#     #     # ---- 以下、低位 decimation 部分はあなたの現状のままでOK ----
#     #     if self._counter % self.cfg.low_level_decimation == 0:
#     #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

#     #         if isinstance(low_level_obs, dict):
#     #             B2 = next(iter(low_level_obs.values())).shape[0]
#     #             obs_tensors = []
#     #             for key in LOW_LEVEL_OBS_ORDER:
#     #                 obs_tensors.append(low_level_obs[key].reshape(B2, -1))
#     #             low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)
#     #         else:
#     #             low_level_obs_tensor = low_level_obs

#     #         with torch.no_grad():
#     #             self.low_level_actions[:] = self.policy(low_level_obs_tensor)

#     #         self._low_level_action_term.process_actions(self.low_level_actions)
#     #         self._counter = 0

#     #     self._low_level_action_term.apply_actions()
#     #     self._counter += 1



#     #     B = self.num_envs
#     #     idx = torch.arange(B, device=self.device)

#     #     # “今スイング中の脚” をちゃんと保存している前提（無ければ一旦 leg_sched を使う）
#     #     leg = getattr(self, "_swing_leg", None)
#     #     if leg is None:
#     #         leg = self._swing_order[self._swing_order_pos]   # 暫定

#     #     c_leg = contact_now[idx, leg]
#     #     # td_leg = ((~self._contact_prev[idx, leg]) & contact_now[idx, leg])  # prev更新前に計算してる前提
#     #     td_leg = ((~prev[idx, leg]) & contact_now[idx, leg])  # ←これでOK


      
#     #     if (self._dbg_count % 200) == 0:
#     #         # 以前のコードで再計算していた部分を削除し、冒頭の変数を利用する
            
#     #         # どの脚を見るか（FSMロジックと同じ leg を使うのが正しい）
#     #         # self._swing_leg は FSM内で更新されている可能性があるため、
#     #         # "そのフレームで着地判定に使われた脚" を見るのが理想ですが、
#     #         # 簡易的には現在の _swing_leg でOKです。
#     #         leg_dbg = self._swing_leg if hasattr(self, "_swing_leg") else self._swing_order[self._swing_order_pos]
            
#     #         # contact_now は [B, 4] なので、該当脚の状態を抽出
#     #         c_leg_dbg = contact_now[idx, leg_dbg]
            
#     #         # touchdown も [B, 4] として冒頭で計算済み
#     #         td_leg_dbg = touchdown[idx, leg_dbg] 

#     #         uniq, cnt = torch.unique(leg_dbg, return_counts=True)

#     #         lf_leg = liftoff[idx, 0]   # leg_for_debug は “固定した swing_leg” 推奨

#     #         print("[FSM dbg] leg_dist:", list(zip(uniq.tolist(), cnt.tolist())),
#     #             "in_swing_mean:", float(self._in_swing.float().mean().item()),
#     #             "c_leg_mean:", float(c_leg_dbg.float().mean().item()),
#     #             "td_leg_mean:", float(td_leg_dbg.float().mean().item()),  # これで正しく出るはず
#     #             "swing_t_max:", int(self._swing_t.max().item()))
            
#     #         print("lf_leg_mean:", float(lf_leg.float().mean().item()))

#     #         leg_dbg = self._swing_leg                         # [B]
#     #         c = contact_now[idx, leg_dbg]                     # [B]

#     #         # base座標での足z（負が下なら、浮くと 0 に近づく）
#     #         z_b = foot_pos_b[idx, leg_dbg, 2]                 # [B]
#     #         z_cmd = latched_b[idx, leg_dbg, 2]                # [B]

#     #         print("z_cmd_mean", float(z_cmd.mean().item()),
#     #             "z_b_mean", float(z_b.mean().item()),
#     #             "z_err_mean", float((z_cmd - z_b).abs().mean().item()),
#     #             "c_leg_mean", float(c.float().mean().item()))

#     #         def _stat(x: torch.Tensor):
#     #             n = x.numel()
#     #             if n == 0:
#     #                 return 0, float("nan"), float("nan"), float("nan")
#     #             return n, float(x.mean().item()), float(x.min().item()), float(x.max().item())

#     #         contact_f = self._estimate_contact()          # 既存の力ベース（デバッグ用）

#     #         print(self._proj_alpha)


#     #         tmax = int(self._swing_t.max().item())
#     #         tmean = float(self._swing_t.float().mean().item())
#     #         timeout_frac = float((self._swing_t >= (self.MAX_SWING_STEPS - 1)).float().mean().item())
#     #         print("tmax", tmax, "tmean", tmean, "timeout_frac", timeout_frac)





#     #         # c = clearance[contact_f]
#     #         # a = clearance[~contact_f]

#     #         # nC, mC, mnC, mxC = _stat(c)
#     #         # nA, mA, mnA, mxA = _stat(a)

#     #         # print(f"clear(contact) n={nC} mean/min/max: {mC:.4f} {mnC:.4f} {mxC:.4f}")
#     #         # print(f"clear(air)     n={nA} mean/min/max: {mA:.4f} {mnA:.4f} {mxA:.4f}")


#     #     self._dbg_count += 1


#     #     # 1 env step 進んだらカウント（num_envs並列でも「env step」を基準にする）
#     #     self._global_step += self.num_envs

#     #     t = max(0, self._global_step - self._proj_warmup_steps)
#     #     frac = min(1.0, t / float(self._proj_anneal_steps))
#     #     self._proj_alpha = (1.0 - frac) * 1.0 + frac * self._proj_alpha_min




    
#     # def reset(self, env_ids=None):
#     #     # ActionManager.reset() から env_ids が渡ってくる想定
#     #     if env_ids is None:
#     #         env_ids = slice(None)

#     #     # --- FSM / timers ---
#     #     self._in_swing[env_ids] = False
#     #     self._liftoff_seen[env_ids] = False
#     #     if hasattr(self, "_swing_t"):
#     #         self._swing_t[env_ids] = 0
#     #     if hasattr(self, "_swing_time"):
#     #         self._swing_time[env_ids] = 0.0

#     #     # 接触ヒステリシスも初期化（初手で誤判定しにくくする）
#     #     self._contact_prev[env_ids] = False

#     #     # スケジュール位置（好み：0に戻す/戻さない。まずは戻すのが無難）
#     #     self._swing_order_pos[env_ids] = 0

#     #     # --- stance anchor (world-fixed) ---
#     #     if hasattr(self, "_stance_anchor_valid"):
#     #         self._stance_anchor_valid[env_ids] = False
#     #     # anchor_w自体は残っててもOK（valid=falseなら参照しない）
#     #     # ただし気持ち悪ければゼロクリアしても良い
#     #     if hasattr(self, "_stance_anchor_w"):
#     #         self._stance_anchor_w[env_ids] = 0.0


    
#     def reset(self, env_ids=None):
#         if env_ids is None:
#             env_ids = torch.arange(self.num_envs, device=self.device)
#         elif isinstance(env_ids, slice):
#             env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
#         else:
#             env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

#         # --- FSM / timers ---
#         self._in_swing[env_ids] = False
#         self._liftoff_seen[env_ids] = False
#         if hasattr(self, "_swing_time"):
#             self._swing_time[env_ids] = 0.0
#         self._swing_order_pos[env_ids] = 0

#         # ★ reset直後は少し待つ（落ち着かせる）
#         if hasattr(self, "_cooldown_steps"):
#             settle = int(getattr(self, "reset_settle_steps", 8))  # 例: 8〜20
#             self._cooldown_steps[env_ids] = settle

#         # --- 現在状態を取る ---
#         robot = self.robot
#         base_pos_w  = robot.data.root_pos_w[env_ids]
#         base_quat_w = robot.data.root_quat_w[env_ids]
#         foot_pos_w  = robot.data.body_pos_w[env_ids][:, self._foot_body_ids, :]   # [K,4,3]

#         contact_now = self._estimate_contact()[env_ids]  # [K,4]

#         # ★ ヒステリシス prev は False ではなく「今の接触」に合わせる（誤イベント減る）
#         self._contact_prev[env_ids] = contact_now

#         # --- event debounce state を reset ---
#         if hasattr(self, "_contact_evt_state"):
#             self._contact_evt_state[env_ids] = contact_now

#             onN  = int(getattr(self, "event_on_N",  2))
#             offN = int(getattr(self, "event_off_N", 2))

#             # 接地なら「on成立済み」, 非接地なら「off成立済み」にする
#             self._evt_on_cnt[env_ids]  = torch.where(contact_now,
#                                                     torch.full_like(contact_now, onN, dtype=torch.int16),
#                                                     torch.zeros_like(contact_now, dtype=torch.int16))
#             self._evt_off_cnt[env_ids] = torch.where(~contact_now,
#                                                     torch.full_like(contact_now, offN, dtype=torch.int16),
#                                                     torch.zeros_like(contact_now, dtype=torch.int16))

#         # --- world固定ターゲットを「実足」で初期化 ---
#         if hasattr(self, "_tgt_w"):
#             self._tgt_w[env_ids] = foot_pos_w
#             self._tgt_w_valid[env_ids] = True

#         # --- low-level へ渡すホールド目標も「今のターゲット」から埋める ---
#         if hasattr(self, "_cmd_b_hold") and hasattr(self, "_tgt_w"):
#             rel_w = self._tgt_w[env_ids] - base_pos_w[:, None, :]      # [K,4,3]
#             rel_w_f = rel_w.reshape(-1, 3)
#             q_rep = base_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
#             cmd_b_f = quat_apply_inverse(q_rep, rel_w_f)
#             cmd_b = cmd_b_f.view(-1, 4, 3)
#             self._cmd_b_hold[env_ids] = cmd_b



    


#     # def apply_actions(self):

#     #     B = self.num_envs
#     #     device = self.device
#     #     idx = torch.arange(B, device=device)


#     #     # B, device, idx の直後あたり
#     #     if not hasattr(self, "_cooldown_steps"):
#     #         self._cooldown_steps = torch.zeros(B, device=device, dtype=torch.int16)


#     #     # # ---- contact + events ----
#     #     # contact_now = self._estimate_contact()  # [B,4]
#     #     # prev = self._contact_prev
#     #     # liftoff   = prev & (~contact_now)
#     #     # touchdown = (~prev) & contact_now
#     #     # self._contact_prev = contact_now  # ※デバッグで prev を使うので prev は保存しておく


#     #     # ---- contact ----
#     #     contact_now = self._estimate_contact()  # [B,4]  ← これは今まで通り使う


#     #     # ★デバッグ用 prev（1フレ前のcontact）
#     #     if not hasattr(self, "_contact_prev_dbg"):
#     #         self._contact_prev_dbg = contact_now.clone()
#     #     prev = self._contact_prev_dbg
#     #     self._contact_prev_dbg = contact_now.clone()

#     #     self._contact_prev = contact_now        # ← _estimate_contact のヒステリシス用 prev はこれで更新（今まで通り）

#     #     # ---- events: debounce only ----
#     #     if not hasattr(self, "_contact_evt_state"):
#     #         # イベント確定用の状態（初期は現在値）
#     #         self._contact_evt_state = contact_now.clone()
#     #         self._evt_on_cnt  = torch.zeros_like(contact_now, dtype=torch.int16)
#     #         self._evt_off_cnt = torch.zeros_like(contact_now, dtype=torch.int16)

#     #     onN  = int(getattr(self, "event_on_N",  2))  # touchdown 確定に必要な連続ON
#     #     offN = int(getattr(self, "event_off_N", 2))  # liftoff  確定に必要な連続OFF

#     #     raw = contact_now

#     #     # raw が True 連続なら on_cnt を増やす、False ならリセット
#     #     self._evt_on_cnt = torch.where(
#     #         raw, torch.clamp(self._evt_on_cnt + 1, 0, onN), torch.zeros_like(self._evt_on_cnt)
#     #     )

#     #     # raw が False 連続なら off_cnt を増やす、True ならリセット
#     #     self._evt_off_cnt = torch.where(
#     #         ~raw, torch.clamp(self._evt_off_cnt + 1, 0, offN), torch.zeros_like(self._evt_off_cnt)
#     #     )

#     #     # まだ「未接触」だと確定している脚が、onN 連続 True になったら touchdown
#     #     touchdown = (~self._contact_evt_state) & (self._evt_on_cnt >= onN)

#     #     # まだ「接触中」だと確定している脚が、offN 連続 False になったら liftoff
#     #     liftoff = (self._contact_evt_state) & (self._evt_off_cnt >= offN)

#     #     # 確定状態を更新（イベントが起きた脚だけ）
#     #     if touchdown.any():
#     #         self._contact_evt_state[touchdown] = True
#     #     if liftoff.any():
#     #         self._contact_evt_state[liftoff] = False



#     #     pending_b = self._pending_actions.view(B, 4, 3)
#     #     latched_b = self._latched_actions.view(B, 4, 3)

#     #     foot_pos_b = self._get_foot_pos_b()   # [B,4,3]

#     #     # “いまスイング対象の脚” を env ごとに決める
#     #     swing_leg_now = torch.where(
#     #         self._in_swing,
#     #         self._swing_leg,
#     #         self._swing_order[self._swing_order_pos],
#     #     )  # [B] long

#     #     # stance脚マスク（接地していて、かつスイング脚ではない）
#     #     mask_stance = contact_now.clone()
#     #     mask_stance[idx, swing_leg_now] = False
#     #     ms = mask_stance

#     #     # # # ========== ここから置き換え ==========

#     #     # # # foot_pos_w を取る（world） 
#     #     # # # self.robot / self._foot_ids がある前提（あなたの他コードと同じ） 
#     #     # robot = self.robot 
#     #     # foot_pos_w = robot.data.body_pos_w[:, self._foot_ids, :] # [B,4,3] 
#     #     # base_pos_w = robot.data.root_pos_w # [B,3] 
#     #     # base_quat_w = robot.data.root_quat_w # [B,4] wxyz


#     #     # # 初回確保
#     #     # if not hasattr(self, "_stance_anchor_w"):
#     #     #     self._stance_anchor_w = foot_pos_w.clone()              # [B,4,3]
#     #     #     self._stance_anchor_valid = torch.zeros_like(contact_now, dtype=torch.bool)  # [B,4]

#     #     # # 1) liftoff で無効化
#     #     # if liftoff.any():
#     #     #     self._stance_anchor_valid[liftoff] = False

#     #     # # 2) touchdown でアンカー更新（ここは従来通り）
#     #     # if touchdown.any():
#     #     #     self._stance_anchor_w[touchdown] = foot_pos_w[touchdown]
#     #     #     self._stance_anchor_valid[touchdown] = True

#     #     # # 3) 「接地してるのにアンカー無い」脚を必ず埋める（reset直後/チャタリング対策）
#     #     # need_anchor = contact_now & (~self._stance_anchor_valid)
#     #     # if need_anchor.any():
#     #     #     self._stance_anchor_w[need_anchor] = foot_pos_w[need_anchor]
#     #     #     self._stance_anchor_valid[need_anchor] = True

#     #     # # 4) worldアンカー -> yaw-base へ変換（ブロードキャストでOK）
#     #     # qy = yaw_quat(base_quat_w)                                  # (w,x,y,z) :contentReference[oaicite:5]{index=5}
#     #     # rel_w = self._stance_anchor_w - base_pos_w[:, None, :]      # [B,4,3]
#     #     # # rel_b = quat_rotate_inverse(qy[:, None, :], rel_w)          # [B,4,3] :contentReference[oaicite:6]{index=6}

#     #     # # ★ B と B*4 を揃える
#     #     # qy_rep  = qy[:, None, :].expand(-1, 4, -1).reshape(-1, 4)  # [B*4,4]
#     #     # rel_w_f = rel_w.reshape(-1, 3)                             # [B*4,3]

#     #     # rel_b_f = quat_apply_inverse(qy_rep, rel_w_f)              # [B*4,3]
#     #     # rel_b   = rel_b_f.view(B, 4, 3)                            # [B,4,3]


#     #     # # # 5) “強すぎ”防止：現在足位置からの補正量をクランプして入れる（超おすすめ）
#     #     # # max_xy = getattr(self, "stance_hold_xy_max", 0.0)  # 3cm から（2~5cmで調整）
#     #     # # err_xy = rel_b[..., 0:2] - foot_pos_b[..., 0:2]
#     #     # # err_xy = torch.clamp(err_xy, min=-max_xy, max=max_xy)

#     #     # # # 6) 支持脚(ms) だけ反映（＋アンカー有効の脚だけ）
#     #     # # use = ms & self._stance_anchor_valid
#     #     # # latched_b[..., 0] = torch.where(use, foot_pos_b[..., 0] + err_xy[..., 0], latched_b[..., 0])
#     #     # # latched_b[..., 1] = torch.where(use, foot_pos_b[..., 1] + err_xy[..., 1], latched_b[..., 1])


#     #     # anchor_xy = rel_b[..., 0:2]                    # [B,4,2]

#     #     # # 2) latched をアンカーに「速度制限付きで」寄せる（1stepで動かしすぎない）
#     #     # dt = float(getattr(self._env, "physics_dt", 0.005))
#     #     # vmax = getattr(self, "stance_hold_vmax", 0.3)   # m/s (0.2〜0.6で調整)
#     #     # step_max = vmax * dt                           # 例: 0.3*0.005=1.5mm

#     #     # delta = anchor_xy - latched_b[..., 0:2]
#     #     # delta = torch.clamp(delta, -step_max, step_max)

#     #     # use = ms & self._stance_anchor_valid
#     #     # latched_b[..., 0] = torch.where(use, latched_b[..., 0] + delta[..., 0], latched_b[..., 0])
#     #     # latched_b[..., 1] = torch.where(use, latched_b[..., 1] + delta[..., 1], latched_b[..., 1])






#     #     # 支持脚：タッチダウンした瞬間だけアンカー更新（base座標）
#     #     # td_stance = touchdown.clone()
#     #     # td_stance[idx, swing_leg_now] = False  # スイング脚除外

#     #     # latched_b[..., 0] = torch.where(td_stance, foot_pos_b[..., 0], latched_b[..., 0])
#     #     # latched_b[..., 1] = torch.where(td_stance, foot_pos_b[..., 1], latched_b[..., 1])


#     #     # # stance脚
#     #     # # latched_b[..., 0] = torch.where(ms, foot_pos_b[..., 0], latched_b[..., 0])  # xは毎step合わせる

#     #     # # touchdownした脚（＝着地直後の脚）だけ y を固定したいなら swing除外しない
#     #     # td = touchdown  # ← swing_leg_now を消す
#     #     # latched_b[..., 1] = torch.where(td, foot_pos_b[..., 1], latched_b[..., 1])  # yだけ固定


#     #     # if not hasattr(self, "_td_plan_b"):
#     #     #     self._td_plan_b = torch.zeros_like(latched_b)  # [B,4,3]
#     #     #     self._td_act_b  = torch.zeros_like(latched_b)  # [B,4,3]
#     #     #     self._td_mask   = torch.zeros_like(contact_now, dtype=torch.bool)

#     #     # if touchdown.any():
#     #     #     self._td_plan_b[touchdown] = latched_b[touchdown]   # planned at touchdown
#     #     #     self._td_act_b[touchdown]  = foot_pos_b[touchdown]  # actual at touchdown
#     #     #     self._td_mask[touchdown]   = True




#     #     # # stance脚は毎ステップXYを実足に固定（横ズレ抑制）
#     #     # latched_b[..., 0] = torch.where(ms, foot_pos_b[..., 0], latched_b[..., 0])
#     #     # latched_b[..., 1] = torch.where(ms, foot_pos_b[..., 1], latched_b[..., 1])
#     #     # Z押し込みを使うならここ（まずはOFF推奨）
#     #     # stance_z_bias = 0.01
#     #     # latched_b[..., 2] = torch.where(ms, foot_pos_b[..., 2] - stance_z_bias, latched_b[..., 2])

#     #     # ---- timer: 秒で管理（sim step ごとに physics_dt を加算）----
#     #     env = self._env
#     #     dt = float(getattr(env, "physics_dt", 0.0))
#     #     if dt <= 0.0:
#     #         # フォールバック（環境によっては step_dt / decimation から推定）
#     #         step_dt = float(getattr(env, "step_dt", 0.0))
#     #         decim = int(getattr(getattr(env, "cfg", None), "decimation", 1))
#     #         dt = step_dt / max(decim, 1)

#     #     # dt = float(getattr(env, "physics_dt", 0.0))
#     #     # if dt <= 0.0:
#     #     #     dt = float(getattr(env, "step_dt", 0.0))   # ★ /decimation しない


#     #     self._swing_time += self._in_swing.float() * dt  # [B] sec

#     #     # =========================================
#     #     # 1) done判定（“スイング開始した脚”で見る）
#     #     # =========================================
#     #     if self._in_swing.any():
#     #         # leg = self._swing_leg                  # [B] 固定脚
#     #         # lf_leg = liftoff[idx, leg]             # [B]
#     #         # td_leg = touchdown[idx, leg]           # [B]

#     #         # # liftoff を一度でも見たか
#     #         # self._liftoff_seen |= (self._in_swing & lf_leg)

#     #         # # ★ 秒で判定
#     #         # done_td = self._in_swing & self._liftoff_seen & td_leg & (self._swing_time >= self.MIN_SWING_SEC)
#     #         # done_to = self._in_swing & self._liftoff_seen &(self._swing_time >= self.MAX_SWING_SEC)
#     #         # done = done_td | done_to

#     #         # if done.any():
#     #         #     self._in_swing[done] = False
#     #         #     self._swing_time[done] = 0.0
#     #         #     self._liftoff_seen[done] = False
#     #         #     self._swing_order_pos[done] = (self._swing_order_pos[done] + 1) % 4

#     #         # 例: dt=0.005なら 0.08s = 16 step
#     #         self.MAX_LIFTOFF_SEC = getattr(self, "MAX_LIFTOFF_SEC", 0.08)

#     #         leg = self._swing_leg
#     #         lf_leg = liftoff[idx, leg]
#     #         td_leg = touchdown[idx, leg]

#     #         self._liftoff_seen |= (self._in_swing & lf_leg)

#     #         done_td = self._in_swing & self._liftoff_seen & td_leg & (self._swing_time >= self.MIN_SWING_SEC)

#     #         # ① 離れた後のタイムアウト（従来MAX）
#     #         to_air = self._in_swing & self._liftoff_seen & (self._swing_time >= self.MAX_SWING_SEC)

#     #         # ② 離れないままのタイムアウト（短い猶予で脱出）
#     #         to_stuck = self._in_swing & (~self._liftoff_seen) & (self._swing_time >= self.MAX_LIFTOFF_SEC)

#     #         done = done_td | to_air | to_stuck

#     #         if done.any():
#     #             # self._in_swing[done] = False
#     #             # self._swing_time[done] = 0.0
#     #             # self._liftoff_seen[done] = False
#     #             # self._swing_order_pos[done] = (self._swing_order_pos[done] + 1) % 4


#     #             self._in_swing[done] = False
#     #             self._swing_time[done] = 0.0
#     #             self._liftoff_seen[done] = False

#     #             # ★ order_pos は「成功 or 離陸後タイムアウト」だけ進める
#     #             adv = done_td | to_air
#     #             self._swing_order_pos[adv] = (self._swing_order_pos[adv] + 1) % 4

#     #                 # ★ done直後は少し待つ（見た目の同時っぽさ＆連続開始を抑える）
#     #             cd = int(getattr(self, "swing_cooldown_steps", 4))  # 2〜6あたりで調整
#     #             self._cooldown_steps[done] = cd


            
    


#     #     # 観測用 onehot
#     #     # leg_for_obs = torch.where(self._in_swing, self._swing_leg, self._swing_order[self._swing_order_pos])
#     #     # onehot = torch.zeros(B, 4, device=device)
#     #     # onehot.scatter_(1, leg_for_obs.view(-1, 1), 1.0)
#     #     # self.step_cmd_term.set_swing_leg_onehot(onehot)


#     #     onehot = torch.zeros(B, 4, device=device)
#     #     env_ids = self._in_swing.nonzero(as_tuple=False).squeeze(-1)
#     #     if env_ids.numel() > 0:
#     #         leg_ids = self._swing_leg[env_ids]
#     #         onehot[env_ids, leg_ids] = 1.0
#     #     self.step_cmd_term.set_swing_leg_onehot(onehot)


#     #     cmd = self._env.command_manager.get_command("pose_command")  # 例：UniformPose2dCommand
#     #     des_xy_b = cmd[:, :2]
#     #     dist = torch.linalg.vector_norm(des_xy_b, dim=1)             # [B]
#     #     near_goal = dist < self.STOP_RADIUS                          # 例 0.25

#     #     # 2) start 判定をこうする
#     #     # start = (~self._in_swing) & (~near_goal)

#     #     # =========================================
#     #     # 2) start（in_swingでないenvだけ：その脚だけlatched更新）
#     #     # =========================================

#     #     # start 判定の前
#     #     self._cooldown_steps = torch.clamp(self._cooldown_steps - 1, min=0)


#     #     # start = ~self._in_swing
#     #     start = (~self._in_swing) & (self._cooldown_steps == 0)

#     #     if start.any():
#     #         env_ids = start.nonzero(as_tuple=False).squeeze(-1)
#     #         leg_s = self._swing_order[self._swing_order_pos[env_ids]]  # [K]

#     #         tgt = pending_b[env_ids, leg_s, :]  # [K,3]

#     #         if getattr(self, "_proj_enable", False):
#     #             tgt_in = tgt.view(-1, 1, 3)
#     #             tgt_proj, _ = self._project_footholds(env_ids, tgt_in)  # [K,1,3]
#     #             tgt = tgt_proj[:, 0, :]

#     #         # ★ スイング脚はZクリアランスを入れる（引っかかり防止）
#     #         # cur_z = foot_pos_b[env_ids, leg_s, 2]
#     #         # tgt[:, 2] = torch.maximum(tgt[:, 2], cur_z + self.swing_clearance)


#     #         # ===== ここから追加：近すぎるならスイング開始しない =====
#     #         # min_step = getattr(self, "MIN_STEP_M", 0.02)  # 2cm（4cm溝なら 2〜3cm推奨）
#     #         # cur_xy = foot_pos_b[env_ids, leg_s, 0:2]      # [K,2]
#     #         # d = torch.linalg.vector_norm(tgt[:, 0:2] - cur_xy, dim=-1)  # [K]


#     #         # ok = d >= min_step
#     #         # skip = ~ok

#     #         # # 近すぎるenvは「その脚をスキップして次脚」へ（停止防止）
#     #         # if skip.any():
#     #         #     env_sk = env_ids[skip]
#     #         #     self._swing_order_pos[env_sk] = (self._swing_order_pos[env_sk] + 1) % 4
#     #         #     # in_swing は False のまま、latched も触らない

#     #         # # 以降の開始処理は ok の env だけに絞る
#     #         # if not ok.any():
#     #         #     # 全部近すぎて開始なし
#     #         #     pass
#     #         # else:
#     #         #     env_ids = env_ids[ok]
#     #         #     leg_s = leg_s[ok]
#     #         #     tgt = tgt[ok]



#     #         # ---- 近すぎ対策：移動中だけ押し出し ----
#     #         min_step = getattr(self, "MIN_STEP_M", 0.02)  # 2cm〜
#     #         cur_xy = foot_pos_b[env_ids, leg_s, 0:2]
#     #         delta = tgt[:, 0:2] - cur_xy
#     #         d = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

#     #         # 押し出し方向：ゴール方向（des_xy_b）を使う
#     #         dir_xy = des_xy_b[env_ids]
#     #         n = torch.linalg.vector_norm(dir_xy, dim=-1, keepdim=True)
#     #         dir_xy = torch.where(n > 1e-6, dir_xy / n,
#     #                             torch.tensor([1.0, 0.0], device=device).view(1,2))

#     #         # d < min_step のときだけ、min_step まで押し出す（大股にはしない）
#     #         tgt_xy = torch.where(d < min_step, cur_xy + dir_xy * min_step, tgt[:, 0:2])
#     #         tgt[:, 0:2] = tgt_xy
#     #         # ===== 追加ここまで =====


#     #         latched_b[env_ids, leg_s, :] = tgt





#     #         # ★ planned unsafe 用：更新イベントをラッチ
#     #         # env = self._env
#     #         # if not hasattr(env, "_planned_update_mask"):
#     #         #     env._planned_update_mask = torch.zeros(B, 4, device=device, dtype=torch.bool)

#     #         # # ここで zero_ しない（←重要）
#     #         # env._planned_update_mask[env_ids, leg_s] = True




#     #         self._in_swing[env_ids] = True
#     #         self._swing_time[env_ids] = 0.0
#     #         self._liftoff_seen[env_ids] = False
#     #         self._swing_leg[env_ids] = leg_s  # 固定

#     #     # ---- 書き戻し ----
#     #     self._latched_actions[:] = latched_b.view(B, -1)



#     #     # ---- start ブロックが終わった後（cmd_b作る直前）----

#     #     # swing_leg_now2 = torch.where(
#     #     #     self._in_swing,
#     #     #     self._swing_leg,
#     #     #     self._swing_order[self._swing_order_pos],
#     #     # )  # [B]

#     #     # ms2 = contact_now.clone()
#     #     # ms2[idx, swing_leg_now2] = False  # stanceのみ True

#     #     # cmd_b = latched_b.clone()
#     #     # cmd_b[..., 0] = torch.where(ms2, foot_pos_b[..., 0], cmd_b[..., 0])
#     #     # cmd_b[..., 1] = torch.where(ms2, foot_pos_b[..., 1], cmd_b[..., 1])

#     #     # self.step_cmd_term.set_foot_targets_base(cmd_b.view(B, -1))

#     #     self._latched_actions[:] = latched_b.view(B, -1)
#     #     self.step_cmd_term.set_foot_targets_base(self._latched_actions)  # = latched をそのまま





#     #     # self.step_cmd_term.set_foot_targets_base(self._latched_actions)

#     #     # ---- 低位 decimation 部分（そのまま）----
#     #     if self._counter % self.cfg.low_level_decimation == 0:
#     #         low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

#     #         if isinstance(low_level_obs, dict):
#     #             B2 = next(iter(low_level_obs.values())).shape[0]
#     #             obs_tensors = []
#     #             for key in LOW_LEVEL_OBS_ORDER:
#     #                 obs_tensors.append(low_level_obs[key].reshape(B2, -1))
#     #             low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)
#     #         else:
#     #             low_level_obs_tensor = low_level_obs

#     #         with torch.no_grad():
#     #             self.low_level_actions[:] = self.policy(low_level_obs_tensor)

#     #         self._low_level_action_term.process_actions(self.low_level_actions)
#     #         self._counter = 0

#     #     self._low_level_action_term.apply_actions()
#     #     self._counter += 1

#     #     # ---- debug ----
#     #     leg = getattr(self, "_swing_leg", None)
#     #     if leg is None:
#     #         leg = self._swing_order[self._swing_order_pos]

#     #     c_leg = contact_now[idx, leg]
#     #     td_leg_dbg = ((~prev[idx, leg]) & contact_now[idx, leg])  # prevを使う（OK）

#     #     if (self._dbg_count % 200) == 0:
#     #         leg_dbg = self._swing_leg if hasattr(self, "_swing_leg") else self._swing_order[self._swing_order_pos]
#     #         c_leg_dbg = contact_now[idx, leg_dbg]
#     #         td_leg_dbg2 = touchdown[idx, leg_dbg]
#     #         uniq, cnt = torch.unique(leg_dbg, return_counts=True)

#     #         # liftoffはスイング脚で見るのが筋
#     #         lf_leg_dbg = liftoff[idx, leg_dbg]

#     #         print("[FSM dbg] leg_dist:", list(zip(uniq.tolist(), cnt.tolist())),
#     #             "in_swing_mean:", float(self._in_swing.float().mean().item()),
#     #             "c_leg_mean:", float(c_leg_dbg.float().mean().item()),
#     #             "td_leg_mean:", float(td_leg_dbg2.float().mean().item()),
#     #             "swing_time_max:", float(self._swing_time.max().item()))

#     #         print("lf_leg_mean:", float(lf_leg_dbg.float().mean().item()))

#     #         z_b = foot_pos_b[idx, leg_dbg, 2]
#     #         z_cmd = latched_b[idx, leg_dbg, 2]

#     #         print("z_cmd_mean", float(z_cmd.mean().item()),
#     #             "z_b_mean", float(z_b.mean().item()),
#     #             "z_err_mean", float((z_cmd - z_b).abs().mean().item()),
#     #             "c_leg_mean", float(c_leg_dbg.float().mean().item()))

#     #         tmax = float(self._swing_time.max().item())
#     #         tmean = float(self._swing_time.mean().item())
#     #         timeout_frac = float((self._swing_time >= (self.MAX_SWING_SEC - dt)).float().mean().item())
#     #         print("tmax_sec", tmax, "tmean_sec", tmean, "timeout_frac", timeout_frac)

#     #         print(self._proj_alpha)

#     #     self._dbg_count += 1

#     #     # 1 env step 進んだらカウント（※apply_actionsがsim stepならここは速すぎるので注意）
#     #     # env step基準にしたいなら、env側の env-step 境界で増やすのが安全
#     #     self._global_step += self.num_envs

#     #     t = max(0, self._global_step - self._proj_warmup_steps)
#     #     frac = min(1.0, t / float(self._proj_anneal_steps))
#     #     self._proj_alpha = (1.0 - frac) * 1.0 + frac * self._proj_alpha_min





#     def apply_actions(self):

#         B = self.num_envs
#         device = self.device
#         idx = torch.arange(B, device=device)

#         if not hasattr(self, "_cmd_b_hold"):
#             self._cmd_b_hold = torch.zeros((B, 4, 3), device=device)



#         # B, device, idx の直後あたり
#         if not hasattr(self, "_cooldown_steps"):
#             self._cooldown_steps = torch.zeros(B, device=device, dtype=torch.int16)


#         # ---- contact ----
#         contact_now = self._estimate_contact()  # [B,4]  ← これは今まで通り使う


#         # ★デバッグ用 prev（1フレ前のcontact）
#         if not hasattr(self, "_contact_prev_dbg"):
#             self._contact_prev_dbg = contact_now.clone()
#         prev = self._contact_prev_dbg
#         self._contact_prev_dbg = contact_now.clone()

#         self._contact_prev = contact_now        # ← _estimate_contact のヒステリシス用 prev はこれで更新（今まで通り）

#         # ---- events: debounce only ----
#         if not hasattr(self, "_contact_evt_state"):
#             # イベント確定用の状態（初期は現在値）
#             self._contact_evt_state = contact_now.clone()
#             self._evt_on_cnt  = torch.zeros_like(contact_now, dtype=torch.int16)
#             self._evt_off_cnt = torch.zeros_like(contact_now, dtype=torch.int16)

#         onN  = int(getattr(self, "event_on_N",  2))  # touchdown 確定に必要な連続ON
#         offN = int(getattr(self, "event_off_N", 2))  # liftoff  確定に必要な連続OFF

#         raw = contact_now

#         # raw が True 連続なら on_cnt を増やす、False ならリセット
#         self._evt_on_cnt = torch.where(
#             raw, torch.clamp(self._evt_on_cnt + 1, 0, onN), torch.zeros_like(self._evt_on_cnt)
#         )

#         # raw が False 連続なら off_cnt を増やす、True ならリセット
#         self._evt_off_cnt = torch.where(
#             ~raw, torch.clamp(self._evt_off_cnt + 1, 0, offN), torch.zeros_like(self._evt_off_cnt)
#         )

#         # まだ「未接触」だと確定している脚が、onN 連続 True になったら touchdown
#         touchdown = (~self._contact_evt_state) & (self._evt_on_cnt >= onN)

#         # まだ「接触中」だと確定している脚が、offN 連続 False になったら liftoff
#         liftoff = (self._contact_evt_state) & (self._evt_off_cnt >= offN)


#         # --- base pose & yaw quat ---
#         robot = self.robot
#         base_pos_w  = robot.data.root_pos_w          # [B,3]
#         base_quat_w = robot.data.root_quat_w         # [B,4] wxyz
#         qy = yaw_quat(base_quat_w)                   # [B,4] wxyz yaw-only

#         foot_pos_w = robot.data.body_pos_w[:, self._foot_body_ids, :]  # [B,4,3]

#         # --- init target_w buffer ---
#         if not hasattr(self, "_tgt_w"):
#             self._tgt_w = foot_pos_w.clone()  # 初期は実足位置
#             self._tgt_w_valid = torch.ones_like(contact_now, dtype=torch.bool)

#         # --- touchdown: stanceアンカーを実足で更新（world固定）---
#         if touchdown.any():
#             self._tgt_w[touchdown] = foot_pos_w[touchdown]
#             self._tgt_w_valid[touchdown] = True

#         # 確定状態を更新（イベントが起きた脚だけ）
#         if touchdown.any():
#             self._contact_evt_state[touchdown] = True
#         if liftoff.any():
#             self._contact_evt_state[liftoff] = False



#         pending_b = self._pending_actions.view(B, 4, 3)
#         latched_b = self._latched_actions.view(B, 4, 3)

#         foot_pos_b = self._get_foot_pos_b()   # [B,4,3]

#         # “いまスイング対象の脚” を env ごとに決める
#         swing_leg_now = torch.where(
#             self._in_swing,
#             self._swing_leg,
#             self._swing_order[self._swing_order_pos],
#         )  # [B] long

#         # stance脚マスク（接地していて、かつスイング脚ではない）
#         # mask_stance = contact_now.clone()
#         mask_stance = self._contact_evt_state.clone()

#         mask_stance[idx, swing_leg_now] = False
#         ms = mask_stance

     
#         # ---- timer: 秒で管理（sim step ごとに physics_dt を加算）----
#         env = self._env
#         dt = float(getattr(env, "physics_dt", 0.0))
#         if dt <= 0.0:
#             # フォールバック（環境によっては step_dt / decimation から推定）
#             step_dt = float(getattr(env, "step_dt", 0.0))
#             decim = int(getattr(getattr(env, "cfg", None), "decimation", 1))
#             dt = step_dt / max(decim, 1)


#         self._swing_time += self._in_swing.float() * dt  # [B] sec

#         # =========================================
#         # 1) done判定（“スイング開始した脚”で見る）
#         # =========================================
#         if self._in_swing.any():
      
#             self.MAX_LIFTOFF_SEC = getattr(self, "MAX_LIFTOFF_SEC", 0.08)

#             leg = self._swing_leg
#             lf_leg = liftoff[idx, leg]
#             td_leg = touchdown[idx, leg]

#             self._liftoff_seen |= (self._in_swing & lf_leg)

#             done_td = self._in_swing & self._liftoff_seen & td_leg & (self._swing_time >= self.MIN_SWING_SEC)

#             # ① 離れた後のタイムアウト（従来MAX）
#             to_air = self._in_swing & self._liftoff_seen & (self._swing_time >= self.MAX_SWING_SEC)

#             # ② 離れないままのタイムアウト（短い猶予で脱出）
#             to_stuck = self._in_swing & (~self._liftoff_seen) & (self._swing_time >= self.MAX_LIFTOFF_SEC)

#             done = done_td | to_air | to_stuck

#             if done.any():
#                 self._in_swing[done] = False
#                 self._swing_time[done] = 0.0
#                 self._liftoff_seen[done] = False
#                 self._swing_order_pos[done] = (self._swing_order_pos[done] + 1) % 4


#                     # ★ done直後は少し待つ（見た目の同時っぽさ＆連続開始を抑える）
#                 cd = int(getattr(self, "swing_cooldown_steps", 4))  # 2〜6あたりで調整
#                 self._cooldown_steps[done] = cd


#         # 観測用 onehot

#         onehot = torch.zeros(B, 4, device=device)
#         env_ids = self._in_swing.nonzero(as_tuple=False).squeeze(-1)
#         if env_ids.numel() > 0:
#             leg_ids = self._swing_leg[env_ids]
#             onehot[env_ids, leg_ids] = 1.0
#         self.step_cmd_term.set_swing_leg_onehot(onehot)


#         cmd = self._env.command_manager.get_command("pose_command")  # 例：UniformPose2dCommand
#         des_xy_b = cmd[:, :2]
#         dist = torch.linalg.vector_norm(des_xy_b, dim=1)             # [B]
#         near_goal = dist < self.STOP_RADIUS                          # 例 0.25


#         # =========================================
#         # 2) start（in_swingでないenvだけ：その脚だけlatched更新）
#         # =========================================

#         # start 判定の前
#         self._cooldown_steps = torch.clamp(self._cooldown_steps - 1, min=0)


#         # start = ~self._in_swing
#         start = (~self._in_swing) & (self._cooldown_steps == 0)

#         if start.any():
#             env_ids = start.nonzero(as_tuple=False).squeeze(-1)
#             leg_s = self._swing_order[self._swing_order_pos[env_ids]]  # [K]

#             tgt = pending_b[env_ids, leg_s, :]  # [K,3]

#             if getattr(self, "_proj_enable", False):
#                 tgt_in = tgt.view(-1, 1, 3)
#                 tgt_proj, _ = self._project_footholds(env_ids, tgt_in)  # [K,1,3]
#                 tgt = tgt_proj[:, 0, :]

      
#             # ---- 近すぎ対策：移動中だけ押し出し ----
#             min_step = getattr(self, "MIN_STEP_M", 0.02)  # 2cm〜
#             cur_xy = foot_pos_b[env_ids, leg_s, 0:2]
#             delta = tgt[:, 0:2] - cur_xy
#             d = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

#             # 押し出し方向：ゴール方向（des_xy_b）を使う
#             dir_xy = des_xy_b[env_ids]
#             n = torch.linalg.vector_norm(dir_xy, dim=-1, keepdim=True)
#             dir_xy = torch.where(n > 1e-6, dir_xy / n,
#                                 torch.tensor([1.0, 0.0], device=device).view(1,2))

#             # d < min_step のときだけ、min_step まで押し出す（大股にはしない）
#             tgt_xy = torch.where(d < min_step, cur_xy + dir_xy * min_step, tgt[:, 0:2])
#             tgt[:, 0:2] = tgt_xy


#             # ===== 追加：swing start で tgt を world にラッチ =====
#             # qy_sel   = qy[env_ids]          # [K,4]
#             base_sel = base_pos_w[env_ids]  # [K,3]

#             # tgt_w = base_sel + quat_apply(qy_sel, tgt)   # tgt は base座標 [K,3]
#             # self._tgt_w[env_ids, leg_s, :] = tgt_w

#             qy_sel = yaw_quat(base_quat_w[env_ids])          # yaw-only
#             tgt_w  = base_sel + quat_apply(qy_sel, tgt)      # tgt は yaw-base で解釈
#             self._tgt_w[env_ids, leg_s, :] = tgt_w


#             # start時
#             # q_sel = base_quat_w[env_ids]                 # フルquat
#             # tgt_w = base_sel + quat_apply(q_sel, tgt)    # tgtはbase座標
#             # self._tgt_w[env_ids, leg_s, :] = tgt_w


#             latched_b[env_ids, leg_s, :] = tgt


#             self._in_swing[env_ids] = True
#             self._swing_time[env_ids] = 0.0
#             self._liftoff_seen[env_ids] = False
#             self._swing_leg[env_ids] = leg_s  # 固定

#         # ---- 書き戻し ----


#         self._latched_actions[:] = latched_b.view(B, -1)


#         # --- every step: world固定ターゲットを base(yaw) に戻して低位へ渡す ---
#         # cmd_b = R_bw(yaw) * (tgt_w - base_pos_w)
#         rel_w = self._tgt_w - base_pos_w[:, None, :]           # [B,4,3]
#         # qy_rep  = qy[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
#         rel_w_f = rel_w.reshape(-1, 3)
#         # cmd_b_f = quat_apply_inverse(qy_rep, rel_w_f)          # [B*4,3]
#         # cmd_b   = cmd_b_f.view(B, 4, 3)

#         # 毎step world->base
#         q_rep = base_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
#         cmd_b_f = quat_apply_inverse(q_rep, rel_w_f)
#         cmd_b = cmd_b_f.view(B,4,3)


#         # low-level が更新されるタイミングだけターゲットを更新
#         if (self._counter % self.cfg.low_level_decimation) == 0:
#             self._cmd_b_hold[:] = cmd_b

#         # 毎stepは “ホールドした値” を渡す
#         self.step_cmd_term.set_foot_targets_base(self._cmd_b_hold.view(B, -1))

#         # self.step_cmd_term.set_foot_targets_base(self._latched_actions)  # = latched をそのまま
#         # self.step_cmd_term.set_foot_targets_base(cmd_b.view(B, -1))




#         # ---- 低位 decimation 部分（そのまま）----
#         if self._counter % self.cfg.low_level_decimation == 0:
#             low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

#             if isinstance(low_level_obs, dict):
#                 B2 = next(iter(low_level_obs.values())).shape[0]
#                 obs_tensors = []
#                 for key in LOW_LEVEL_OBS_ORDER:
#                     obs_tensors.append(low_level_obs[key].reshape(B2, -1))
#                 low_level_obs_tensor = torch.cat(obs_tensors, dim=-1)
#             else:
#                 low_level_obs_tensor = low_level_obs

#             with torch.no_grad():
#                 self.low_level_actions[:] = self.policy(low_level_obs_tensor)

#             self._low_level_action_term.process_actions(self.low_level_actions)
#             self._counter = 0

#         self._low_level_action_term.apply_actions()
#         self._counter += 1

#         # ---- debug ----
#         leg = getattr(self, "_swing_leg", None)
#         if leg is None:
#             leg = self._swing_order[self._swing_order_pos]

#         c_leg = contact_now[idx, leg]
#         td_leg_dbg = ((~prev[idx, leg]) & contact_now[idx, leg])  # prevを使う（OK）

#         if (self._dbg_count % 200) == 0:
#             leg_dbg = self._swing_leg if hasattr(self, "_swing_leg") else self._swing_order[self._swing_order_pos]
#             c_leg_dbg = contact_now[idx, leg_dbg]
#             td_leg_dbg2 = touchdown[idx, leg_dbg]
#             uniq, cnt = torch.unique(leg_dbg, return_counts=True)

#             # liftoffはスイング脚で見るのが筋
#             lf_leg_dbg = liftoff[idx, leg_dbg]

#             print("[FSM dbg] leg_dist:", list(zip(uniq.tolist(), cnt.tolist())),
#                 "in_swing_mean:", float(self._in_swing.float().mean().item()),
#                 "c_leg_mean:", float(c_leg_dbg.float().mean().item()),
#                 "td_leg_mean:", float(td_leg_dbg2.float().mean().item()),
#                 "swing_time_max:", float(self._swing_time.max().item()))

#             print("lf_leg_mean:", float(lf_leg_dbg.float().mean().item()))

#             z_b = foot_pos_b[idx, leg_dbg, 2]
#             z_cmd = latched_b[idx, leg_dbg, 2]

#             print("z_cmd_mean", float(z_cmd.mean().item()),
#                 "z_b_mean", float(z_b.mean().item()),
#                 "z_err_mean", float((z_cmd - z_b).abs().mean().item()),
#                 "c_leg_mean", float(c_leg_dbg.float().mean().item()))

#             tmax = float(self._swing_time.max().item())
#             tmean = float(self._swing_time.mean().item())
#             timeout_frac = float((self._swing_time >= (self.MAX_SWING_SEC - dt)).float().mean().item())
#             print("tmax_sec", tmax, "tmean_sec", tmean, "timeout_frac", timeout_frac)

#             print(self._proj_alpha)

#         self._dbg_count += 1

#         # 1 env step 進んだらカウント（※apply_actionsがsim stepならここは速すぎるので注意）
#         # env step基準にしたいなら、env側の env-step 境界で増やすのが安全
#         self._global_step += self.num_envs

#         t = max(0, self._global_step - self._proj_warmup_steps)
#         frac = min(1.0, t / float(self._proj_anneal_steps))
#         self._proj_alpha = (1.0 - frac) * 1.0 + frac * self._proj_alpha_min

        


        


    






    





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