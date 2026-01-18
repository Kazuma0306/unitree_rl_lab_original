
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.tasks.locomotion.robots.go2.locotransformer import VisionMLPActorCritic, LocoTransformerActorCritic, LocoTransformerFinetune
from unitree_rl_lab.tasks.locomotion.robots.go2.locotransformer_force import LocoTransformerHFP, MlpHFP, VisionHighLevelAC, VisionHighRNN




# play_student_upper.py
import torch
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv

import torch.nn.functional as Fun

# ---- あなたの学習時と同じ関数をここに置く（そのままコピペ推奨）----
# build_bev_grid_in(device)
# rgb_to_bev_in(rgb, bev_grid_in)
# index_to_xy_round(idx, Hb, Wb, L, W, res)

def yaw_quat(q_wxyz):
    # q=(w,x,y,z) -> yaw-only quaternion (w,x,y,z)
    w,x,y,z = q_wxyz.unbind(-1)
    # yaw from quaternion
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)

def quat_apply(q, v):
    # q: (...,4) wxyz, v:(...,3)
    w,x,y,z = q.unbind(-1)
    vx,vy,vz = v.unbind(-1)
    # t = 2 * cross(q_vec, v)
    tx = 2*(y*vz - z*vy)
    ty = 2*(z*vx - x*vz)
    tz = 2*(x*vy - y*vx)
    # v' = v + w*t + cross(q_vec, t)
    vpx = vx + w*tx + (y*tz - z*ty)
    vpy = vy + w*ty + (z*tx - x*tz)
    vpz = vz + w*tz + (x*ty - y*tx)
    return torch.stack([vpx, vpy, vpz], dim=-1)

def get_front_rgb(env, camera_name="camera"):
    cam = env.scene.sensors[camera_name]

    rgb0   = cam.data.output["rgb"].clone()
    x = rgb0

    # d = cam.data
    # # よくある候補を順に試す（環境で名前が違うので安全策）
    # if hasattr(d, "output") and isinstance(d.output, dict):
    #     for k in ["rgb"]:
    #         if k in d.output:
    #             x = d.output[k]
    #             break
    #     else:
    #         raise RuntimeError(f"camera.data.output keys = {list(d.output.keys())}")
    # elif hasattr(d, "rgb"):
    #     x = d.rgb
    # else:
    #     raise RuntimeError("Cannot find RGB in camera sensor data.")
    return x  # shape: (B,H,W,3) or (B,3,H,W)

def get_foot_pos_yawbase(env, foot_body_ids):
    robot = env.scene.articulations["robot"]
    root = robot.data.root_state_w
    base_p = root[:, 0:3]
    base_q = root[:, 3:7]  # wxyz
    qy = yaw_quat(base_q)

    feet_w = robot.data.body_pos_w[:, foot_body_ids, :]  # (B,4,3)
    rel_w = feet_w - base_p[:, None, :]
    # yaw-only inverse rotation: apply conj(qy)
    qy_conj = qy.clone()
    qy_conj[:, 1:] *= -1
    feet_b = quat_apply(qy_conj[:, None, :], rel_w)
    return feet_b








from unitree_rl_lab.models.student_train import StudentNetInBEV_FiLM2D_OS8, SimpleConcatFootNet


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

# def build_bev_grid_in(device):
#     # BEV grid points in base frame (Hb,Wb,3)
#     # xs = torch.linspace(X_MAX_IN, X_MIN_IN, HB_IN, device=device)
#     # ys = torch.linspace(Y_WIDTH_IN/2, -Y_WIDTH_IN/2, WB_IN, device=device)
#     # xg, yg = torch.meshgrid(xs, ys, indexing="ij")

#     xs = torch.linspace(X_MIN_IN, X_MAX_IN, HB_IN, device=device)                 # 昇順
#     ys = torch.linspace(-Y_WIDTH_IN/2, Y_WIDTH_IN/2, WB_IN, device=device)        # 昇順
#     xg, yg = torch.meshgrid(xs, ys, indexing="ij")

#     # zg = torch.zeros_like(xg)
#     zg = torch.full_like(xg, Z_GROUND_BASE)
#     pts_b = torch.stack([xg, yg, zg], dim=-1)  # (Hb,Wb,3)

#     R_bc = quat_to_R_wxyz(q_bc_wxyz.to(device))
#     t = t_bc.to(device)

#     # base->cam : P_cam = R_bc^T (P_base - t_bc)
#     pts_c = torch.einsum("...j,ij->...i", (pts_b - t), R_bc.t())
#     X = pts_c[...,0]; Y = pts_c[...,1]; Z = pts_c[...,2]
#     valid = Z > 0.1

#     u = fx * (X / Z.clamp(min=1e-6)) + cx
#     v = fy * (Y / Z.clamp(min=1e-6)) + cy

#     grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
#     grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0
#     grid_x = grid_x.masked_fill(~valid, -2.0)
#     grid_y = grid_y.masked_fill(~valid, -2.0)

#     return torch.stack([grid_x, grid_y], dim=-1)  # (Hb,Wb,2)



# def build_bev_grid_in(device):
#     """
#     BEV定義（yaw-base / base平面）:
#       - x: 前方 [X_MIN_IN .. X_MAX_IN]（行方向、上が奥・下が手前にしたいなら昇順OK）
#       - y: 左右 [-Y_WIDTH_IN/2 .. +Y_WIDTH_IN/2]（列方向、左=-y, 右=+y ならこのまま）
#     画像投影:
#       - u = fx * (X/Z) + cx
#       - v = fy * (-Y/Z) + cy   ★ここが重要（上下反転の根本修正）
#     """
#     # (1) BEV grid in base frame
#     xs = torch.linspace(X_MIN_IN, X_MAX_IN, HB_IN, device=device)          # x: near->far (昇順)
#     ys = torch.linspace(-Y_WIDTH_IN/2, Y_WIDTH_IN/2, WB_IN, device=device) # y: left(-)->right(+) (昇順)
#     xg, yg = torch.meshgrid(xs, ys, indexing="ij")

#     zg = torch.full_like(xg, Z_GROUND_BASE)  # base座標の仮想地面高さ
#     pts_b = torch.stack([xg, yg, zg], dim=-1)  # (Hb,Wb,3)

#     # (2) base -> cam
#     R_bc = quat_to_R_wxyz(q_bc_wxyz.to(device))  # base->cam の回転（あなたの定義のまま）
#     t = t_bc.to(device)

#     pts_c = torch.einsum("...j,ij->...i", (pts_b - t), R_bc.t())  # P_cam = R^T (P_base - t)
#     X = pts_c[..., 0]
#     Y = pts_c[..., 1]
#     Z = pts_c[..., 2]

#     # (3) project (★ vは -Y/Z を使う)
#     Zc = Z.clamp(min=1e-6)
#     u = fx * (X / Zc) + cx
#     v = fy * (-Y / Zc) + cy  # ★上下反転修正

#     # (4) valid mask: カメラ前方 + 画像内
#     in_front = Z > 0.1
#     in_img = (u >= 0.0) & (u <= (IMG_W - 1.0)) & (v >= 0.0) & (v <= (IMG_H - 1.0))
#     valid = in_front & in_img

#     # (5) normalize to [-1,1] for grid_sample
#     grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
#     grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0

#     # invalid -> outside so grid_sample returns 0
#     grid_x = grid_x.masked_fill(~valid, -2.0)
#     grid_y = grid_y.masked_fill(~valid, -2.0)

#     return torch.stack([grid_x, grid_y], dim=-1)  # (Hb,Wb,2)



def quat_conj(q):
    out = q.clone()
    out[..., 1:] *= -1
    return out

def quat_mul(q1, q2):
    # (w,x,y,z)
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)








# def build_bev_grid_in(
#     root_state_w,          # (B,13)  base world pose
#     t_bc, q_bc,            # base->cam offset in base frame (wxyz)
#     fx, fy, cx, cy,        # intrinsics
#     IMG_H, IMG_W,          # image size
#     X_MIN, X_MAX, Y_WIDTH, # BEV領域（yaw-base）
#     HB, WB,                # BEV解像度
#     world_ground_z=0.0,    # 地面の world z
#     v_flip=False,          # まだ上下逆なら True にする（後述）
# ):
#     device = root_state_w.device
#     B = root_state_w.shape[0]

#     base_pos = root_state_w[:, 0:3]      # (B,3)
#     q_full   = root_state_w[:, 3:7]      # (B,4) wxyz
#     q_yaw    = yaw_quat(q_full)          # (B,4)

#     # 1) yaw-base上の格子点（zはダミーで0）
#     xs = torch.linspace(X_MIN, X_MAX, HB, device=device)
#     ys = torch.linspace(-Y_WIDTH/2, Y_WIDTH/2, WB, device=device)
#     xg, yg = torch.meshgrid(xs, ys, indexing="ij")
#     pts_y = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)   # (HB,WB,3)

#     # 2) yawだけで world XY に回す（roll/pitch 無視）
#     pts_w = quat_apply(q_yaw[:, None, None, :], pts_y[None, ...]) + base_pos[:, None, None, :]
#     pts_w[..., 2] = world_ground_z  # ★ world地面に固定

#     # 3) camera world pose（base pose + offset）
#     cam_pos_w = base_pos + quat_apply(q_full, t_bc.to(device))     # (B,3)
#     cam_q_w   = quat_mul(q_full, q_bc.to(device))                  # (B,4)

#     # 4) world -> camera
#     pts_c = quat_apply(quat_conj(cam_q_w)[:, None, None, :], pts_w - cam_pos_w[:, None, None, :])
#     Xc, Yc, Zc = pts_c.unbind(-1)
#     valid = Zc > 1e-3

#     # 5) project
#     u = fx * (Xc / Zc.clamp(min=1e-6)) + cx
#     if not v_flip:
#         v = fy * (Yc / Zc.clamp(min=1e-6)) + cy
#     else:
#         v = cy - fy * (Yc / Zc.clamp(min=1e-6))  # ★上下が逆ならこっち

#     grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
#     grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0

#     grid_x = grid_x.masked_fill(~valid, -2.0)
#     grid_y = grid_y.masked_fill(~valid, -2.0)

#     grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,HB,WB,2)
#     return grid, valid




def build_bev_grid_in(device, flip_v=True):
    # row0 = far (X_MAX), col0 = left (+Y)  ←あなたの元の定義
    xs = torch.linspace(X_MAX_IN, X_MIN_IN, HB_IN, device=device)        # 奥→手前
    ys = torch.linspace(Y_WIDTH_IN/2, -Y_WIDTH_IN/2, WB_IN, device=device)  # 左→右
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")

    zg = torch.full_like(xg, Z_GROUND_BASE)
    pts_b = torch.stack([xg, yg, zg], dim=-1)  # (Hb,Wb,3)

    R_bc = quat_to_R_wxyz(q_bc_wxyz.to(device))
    t = t_bc.to(device)

    # base->cam : P_cam = R_bc^T (P_base - t_bc)
    pts_c = torch.einsum("...j,ij->...i", (pts_b - t), R_bc.t())
    X = pts_c[..., 0]; Y = pts_c[..., 1]; Z = pts_c[..., 2]
    valid = Z > 0.1

    u = fx * (X / Z.clamp(min=1e-6)) + cx
    v = fy * (Y / Z.clamp(min=1e-6)) + cy

    if flip_v:
        # ★ここが本命：画像の上下座標系を合わせる
        v = (IMG_H - 1.0) - v

    grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
    grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0

    grid_x = grid_x.masked_fill(~valid, -2.0)
    grid_y = grid_y.masked_fill(~valid, -2.0)

    return torch.stack([grid_x, grid_y], dim=-1)  #




# def rgb_to_bev_in(rgb, bev_grid_in):
#     """
#     rgb: (B,H,W,3) or (B,3,H,W)
#     bev_grid_in: (Hb_in, Wb_in, 2)  in [-1,1] for grid_sample
#     return: (B,3,Hb_in,Wb_in)
#     """
#     if rgb.ndim != 4:
#         raise RuntimeError(f"rgb must be 4D, got shape={tuple(rgb.shape)}")

#     # --- HWC or CHW を判定 ---
#     if rgb.shape[-1] == 3:          # (B,H,W,3)
#         img = rgb.permute(0, 3, 1, 2).contiguous()
#     elif rgb.shape[1] == 3:         # (B,3,H,W)
#         img = rgb.contiguous()
#     else:
#         raise RuntimeError(f"rgb must be (B,H,W,3) or (B,3,H,W), got shape={tuple(rgb.shape)}")

#     # --- float化＆正規化 ---
#     if img.dtype != torch.float32:
#         img = img.float()
#     # uint8(0..255) のときだけ割る（floatでも 0..255 の場合があるので max で判定）
#     if img.max() > 1.5:
#         img = img / 255.0

#     B = img.shape[0]
#     grid = bev_grid_in.unsqueeze(0).expand(B, -1, -1, -1)  # (B,Hb,Wb,2)
#     bev = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


#     return bev


def rgb_to_bev_in(rgb, bev_grid):
    """
    rgb: (B,H,W,3) or (B,3,H,W)
    bev_grid: (HB,WB,2)  または (B,HB,WB,2)
    return: (B,3,HB,WB)
    """
    if rgb.ndim != 4:
        raise RuntimeError(f"rgb must be 4D, got shape={tuple(rgb.shape)}")

    # HWC -> CHW
    if rgb.shape[-1] == 3:
        img = rgb.permute(0, 3, 1, 2).contiguous()
    elif rgb.shape[1] == 3:
        img = rgb.contiguous()
    else:
        raise RuntimeError(f"rgb must be (B,H,W,3) or (B,3,H,W), got shape={tuple(rgb.shape)}")

    if img.dtype != torch.float32:
        img = img.float()


    if img.max() > 1.5:
        img = img / 255.0

    B = img.shape[0]

    # grid shape handling
    if bev_grid.ndim == 3:
        # (HB,WB,2) -> (B,HB,WB,2)
        grid = bev_grid.unsqueeze(0).expand(B, -1, -1, -1)
    elif bev_grid.ndim == 4:
        # (B,HB,WB,2) already
        assert bev_grid.shape[0] == B, f"grid batch {bev_grid.shape[0]} != img batch {B}"
        grid = bev_grid
    else:
        raise RuntimeError(f"bev_grid must be 3D or 4D, got {bev_grid.ndim}D shape={tuple(bev_grid.shape)}")

    bev = Fun.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return bev



# def (idx_b4, Hb, Wb, L, W, res):
#     ix = idx_b4 // Wb
#     iy = idx_b4 %  Wb
#     x = ix.float() * res - L/2
#     y = iy.float() * res - W/2
#     return torchindex_to_xy_round.stack([x, y], dim=-1)  # (B,4,2) or (Bu,2)



def index_to_xy(idx_b4, x_coords, y_coords, Wb):
    ix = idx_b4 // Wb
    iy = idx_b4 % Wb
    x = x_coords[ix]
    y = y_coords[iy]
    return torch.stack([x, y], dim=-1)  # (B,4,2)

def xy_to_index(xy_b42, Hb, Wb, L, W, res):
    # xy_b42: (B,4,2) in meters, base frame
    x = xy_b42[..., 0]
    y = xy_b42[..., 1]
    ix = torch.round((x + L/2) / res).long().clamp(0, Hb - 1)
    iy = torch.round((y + W/2) / res).long().clamp(0, Wb - 1)
    idx = ix * Wb + iy  # (B,4)
    return idx



def quat_conjugate(q):
    # q: (...,4) wxyz
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def base_cmd_to_yawbase(cmd_b, root_state_w, yaw_quat_fn, quat_apply_fn):
    """
    cmd_b: (B,4,3) base(フル姿勢) 座標の相対点
    戻り: (B,4,3) yaw-base 座標の相対点
    """
    base_pos = root_state_w[:, 0:3]     # (B,3)
    q_full   = root_state_w[:, 3:7]     # (B,4) wxyz
    q_yaw    = yaw_quat_fn(q_full)      # (B,4) wxyz (yaw-only)

    # base -> world
    p_w = base_pos[:, None, :] + quat_apply_fn(q_full[:, None, :], cmd_b)  # (B,4,3)

    # world -> yaw-base
    p_y = quat_apply_fn(quat_conjugate(q_yaw)[:, None, :], p_w - base_pos[:, None, :])  # (B,4,3)
    return p_y

# def yawbase_to_base(p_y, root_state_w, yaw_quat_fn, quat_apply_fn):
#     """
#     p_y: (B,4,3) yaw-base 相対点
#     戻り: (B,4,3) base(フル姿勢) 相対点
#     """
#     base_pos = root_state_w[:, 0:3]
#     q_full   = root_state_w[:, 3:7]
#     q_yaw    = yaw_quat_fn(q_full)

#     # yaw-base -> world
#     p_w = base_pos[:, None, :] + quat_apply_fn(q_yaw[:, None, :], p_y)

#     # world -> base
#     p_b = quat_apply_fn(quat_conjugate(q_full)[:, None, :], p_w - base_pos[:, None, :])
#     return p_b







def root13_hist_to_feat(root_hist: torch.Tensor, keep_z: bool = True) -> torch.Tensor:
    """
    root_hist: (B,T,13)  [pos(3), quat(wxyz)(4), lin_vel_w(3), ang_vel_w(3)]
    return: (B,T,D)  D = 9 (keep_z=True) or 8 (keep_z=False)
      [pos_z?, sin(yaw), cos(yaw), lin_vel_yaw(3), ang_vel_yaw(3)]
    """
    assert root_hist.ndim == 3 and root_hist.shape[-1] >= 13, root_hist.shape

    pos = root_hist[..., 0:3]
    quat = root_hist[..., 3:7]        # wxyz
    v_w = root_hist[..., 7:10]
    w_w = root_hist[..., 10:13]

    w, x, y, z = quat.unbind(dim=-1)

    # yaw (wxyz) : atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (B,T)

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # world -> yaw-base (rotate by -yaw around z)
    vx, vy, vz = v_w.unbind(dim=-1)
    wx, wy, wz = w_w.unbind(dim=-1)

    vx_y = c * vx + s * vy
    vy_y = -s * vx + c * vy
    wx_y = c * wx + s * wy
    wy_y = -s * wx + c * wy

    v_yaw = torch.stack([vx_y, vy_y, vz], dim=-1)  # (B,T,3)
    w_yaw = torch.stack([wx_y, wy_y, wz], dim=-1)  # (B,T,3)

    yaw_sin = torch.sin(yaw)[..., None]
    yaw_cos = torch.cos(yaw)[..., None]

    if keep_z:
        pos_z = pos[..., 2:3]
        feat = torch.cat([pos_z, yaw_sin, yaw_cos, v_yaw, w_yaw], dim=-1)  # (B,T,9)
    else:
        feat = torch.cat([yaw_sin, yaw_cos, v_yaw, w_yaw], dim=-1)         # (B,T,8)

    return feat







def get_contact_4(env, sensor_name: str, sensor_cols: list[int], thresh: float, device):
    # (B, num_sensor_bodies, 3)
    F = env.scene.sensors[sensor_name].data.net_forces_w
    f_foot = F[:, sensor_cols, :]                 # (B,4,3)
    contact = (f_foot[..., 2].abs() > thresh)     # (B,4) bool
    return contact.to(device=device, dtype=torch.uint8)




def build_proprio_flat(root_seq, foot_seq, contact_seq=None):
    """
    root_seq:   (Bu,T,13)
    foot_seq:   (Bu,T,4,3)
    contact_seq:(Bu,T,4) uint8/float or None
    return: proprio_flat (Bu, T*P)
    """
    Bu, T, _ = root_seq.shape

    root_feat = root13_hist_to_feat(root_seq, keep_z=True)   # (Bu,T,9) ←あなたの定義
    foot_flat = foot_seq.reshape(Bu, T, -1)                  # (Bu,T,12)

    if contact_seq is not None:
        cont = contact_seq.float()                           # (Bu,T,4)
        proprio_hist = torch.cat([foot_flat, root_feat, cont], dim=-1)  # (Bu,T,25)
    else:
        proprio_hist = torch.cat([foot_flat, root_feat], dim=-1)        # (Bu,T,21)

    return proprio_hist.reshape(Bu, -1)                       # (Bu,T*P)



# def build_proprio_flat2(root_seq, foot_seq, cont_seq, fsm_seq):
#     # root_seq:(Bu,T,13) foot_seq:(Bu,T,4,3) cont_seq:(Bu,T,4) fsm_seq:(Bu,T,20)
#     Bu, T, _ = root_seq.shape
#     foot_flat = foot_seq.reshape(Bu, T, 12)
#     root_feat = root13_hist_to_feat(root_seq, keep_z=True)  # (Bu,T,9)
#     cont_f = (cont_seq > 0).to(torch.float32)

#     per_t = torch.cat([foot_flat, root_feat, cont_f, fsm_seq.to(torch.float32)], dim=-1)  # (Bu,T,45)
#     return per_t.reshape(Bu, -1)  # (Bu, 2160)


def build_proprio_flat2(root_seq, foot_seq, cont_seq, fsm_seq):
    """
    学習側の前処理に合わせた proprio flatten（45次元/step）
    root_seq:(Bu,T,13)
    foot_seq:(Bu,T,4,3)
    cont_seq:(Bu,T,4)   0/1 or u8
    fsm_seq :(Bu,T,20)  [cmd_age, hold_t, bad_t, armed, near4, contact4, stableC4, stableN4]
    returns: (Bu, T*45)
    """
    Bu, T, _ = root_seq.shape

    foot_flat = foot_seq.reshape(Bu, T, 12)
    root_feat = root13_hist_to_feat(root_seq, keep_z=True)  # (Bu,T,9)
    cont_f = (cont_seq > 0).to(torch.float32)               # (Bu,T,4)

    # ---- fsm normalize (match training) ----
    f = fsm_seq.to(torch.float32)
    if f.shape[-1] < 20:
        raise RuntimeError(f"fsm_seq dim must be >=20, got {f.shape}")

    f = f[..., :20]

    # 0〜2秒にクリップして 0〜1
    f_time  = torch.clamp(f[..., 0:3],   0.0, 2.0) / 2.0     # cmd_age, hold, bad
    f_armed = f[..., 3:4]                                   # 0/1
    f_bin   = torch.clamp(f[..., 4:12],  0.0, 1.0)           # near4, contact4
    f_stab  = torch.clamp(f[..., 12:20], 0.0, 2.0) / 2.0     # stableC4, stableN4

    fsm_feat = torch.cat([f_time, f_armed, f_bin, f_stab], dim=-1)  # (Bu,T,20)

    per_t = torch.cat([foot_flat, root_feat, cont_f, fsm_feat], dim=-1)  # (Bu,T,45)
    return per_t.reshape(Bu, -1)



def build_proprio_flat33(
    root_hist: torch.Tensor,   # (B,T,13)
    foot_hist: torch.Tensor,   # (B,T,4,3)
    fsm_hist: torch.Tensor | None,  # (B,T,20) or None
    *,
    keep_z: bool = True,
) -> torch.Tensor:
    """
    returns: proprio_flat (B, T*41)
      per-step: [foot(12), root_feat(9), fsm_feat(20)] = 41
    学習側の前処理（time/bin/stableのclamp&scale）と一致させる。
    """
    assert root_hist.ndim == 3 and foot_hist.ndim == 4
    B, T, _ = root_hist.shape

    # foot (B,T,12)
    foot_hist_flat = foot_hist.reshape(B, T, -1)

    # root feat (B,T,9)
    root_feat = root13_hist_to_feat(root_hist, keep_z=keep_z)

    # fsm feat (B,T,20) 正規化
    if fsm_hist is not None:
        assert fsm_hist.ndim == 3, fsm_hist.shape
        f = fsm_hist.float()
        if f.shape[-1] < 20:
            raise RuntimeError(f"fsm_hist dim must be >=20, got {f.shape}")

        f = f[..., :20]  # 念のため

        # 学習コードと同じ：0〜2秒でクリップして0〜1へ
        f_time  = torch.clamp(f[..., 0:3],  0.0, 2.0) / 2.0   # cmd_age, hold, bad
        f_armed = f[..., 3:4]                                # 0/1
        f_bin   = torch.clamp(f[..., 4:12], 0.0, 1.0)        # near4, contact4
        f_stab  = torch.clamp(f[..., 12:20],0.0, 2.0) / 2.0  # stableC4, stableN4

        fsm_feat = torch.cat([f_time, f_armed, f_bin, f_stab], dim=-1)  # (B,T,20)
    else:
        # 学習側で fsm が無いケースがあるなら、その時と同様に「入れない」設計に合わせる必要あり。
        # ここでは 0 埋めで 20 次元に合わせる（学習も0埋めなら一致）。
        fsm_feat = torch.zeros((B, T, 20), device=root_hist.device, dtype=root_hist.dtype)

    proprio_hist = torch.cat([foot_hist_flat, root_feat, fsm_feat], dim=-1)  # (B,T,41)
    proprio_flat = proprio_hist.reshape(B, -1)                               # (B,T*41)
    return proprio_flat




# def ensure_bhwc_u8(rgb: torch.Tensor) -> torch.Tensor:
#     # BCHW -> BHWC
#     if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
#         rgb = rgb.permute(0, 2, 3, 1).contiguous()

#     # float(0..1 or 0..255) -> uint8(0..255)
#     if rgb.dtype != torch.uint8:
#         # 0..1 想定。もし0..255ならclampで潰れないのでOK
#         rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

#     return rgb.contiguous()


def ensure_bhwc_u8(rgb):
    if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.permute(0,2,3,1).contiguous()
    if rgb.dtype == torch.uint8:
        return rgb.contiguous()
    rgb = rgb.float()
    if rgb.max() > 1.5:   # 0..255系
        rgb = (rgb.clamp(0.0, 255.0)).to(torch.uint8)
    else:                 # 0..1系
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb.contiguous()







import itertools



import itertools
import torch

LEG_NAMES = ["FL", "FR", "RL", "RR"]

# 24通りの脚対応 permutation（pred_i -> teacher_perm[i]）
PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)  # (24,4) CPU

perm_votes = torch.zeros(len(PERMS), dtype=torch.long)
LEG_PERM_FIXED = None  # ここが決まったら固定して使う

def best_perm_from_dist(D_b44: torch.Tensor):
    """
    D_b44: (Bu,4,4)  where D[i,j]=||pred_leg_i - teacher_leg_j||  (cmでもmでもOK)
    return:
      best_perm_b4: (Bu,4) pred_i -> teacher_j
      best_cost_b : (Bu,)
      best_k_b    : (Bu,)
    """
    Bu = D_b44.shape[0]
    ar = torch.arange(4)
    costs = []
    for k in range(PERMS.shape[0]):
        p = PERMS[k]
        costs.append(D_b44[:, ar, p].mean(dim=-1))  # (Bu,)
    costs = torch.stack(costs, dim=1)              # (Bu,24)
    best_k = costs.argmin(dim=1)                   # (Bu,)
    best_cost = costs[torch.arange(Bu), best_k]    # (Bu,)
    best_perm = PERMS[best_k]                      # (Bu,4)
    return best_perm, best_cost, best_k

def gather_perm_x(x_b4d: torch.Tensor, perm_b4: torch.Tensor):
    """x_b4d: (Bu,4,D), perm_b4: (Bu,4) -> x aligned: (Bu,4,D)"""
    idx = perm_b4[:, :, None].expand(-1, -1, x_b4d.shape[-1])
    return x_b4d.gather(1, idx)

def gather_perm_mask(mask_b4: torch.Tensor, perm_b4: torch.Tensor):
    """mask_b4: (Bu,4) bool -> aligned"""
    return mask_b4.gather(1, perm_b4)









import matplotlib.pyplot as plt
import numpy as np
import torch

# plt.ion()  # インタラクティブON（毎step更新したいので）

# def _to_u8_hwc_rgb(x):
#     """
#     x: torch Tensor
#       - (3,H,W) float[0..1] or uint8
#       - (H,W,3) float/uint8
#     return: uint8 HWC RGB
#     """
#     x = x.detach().cpu()
#     if x.ndim == 3 and x.shape[0] == 3:   # CHW -> HWC
#         x = x.permute(1, 2, 0).contiguous()
#     if x.dtype != torch.uint8:
#         x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
#     return x.numpy()  # HWC RGB uint8

# # 画面を最初に作って使い回す（毎回plt.figure作ると重い）
# _fig = None
# _axes = None
# _im = None

# def show_four_images(rgb_pre_hwc, rgb_post_hwc, bev_pre_chw, bev_post_chw, title=""):
#     global _fig, _axes, _im
#     imgs = [
#         _to_u8_hwc_rgb(rgb_pre_hwc),   # HWC
#         _to_u8_hwc_rgb(rgb_post_hwc),
#         _to_u8_hwc_rgb(bev_pre_chw),   # CHW -> HWC
#         _to_u8_hwc_rgb(bev_post_chw),
#     ]
#     names = ["RGB_PRE", "RGB_POST", "BEV_PRE (model input)", "BEV_POST (model input)"]

#     if _fig is None:
#         _fig, _axes = plt.subplots(2, 2, figsize=(10, 6))
#         _axes = _axes.reshape(-1)
#         _im = []
#         for ax, img, nm in zip(_axes, imgs, names):
#             ax.set_title(nm)
#             ax.axis("off")
#             _im.append(ax.imshow(img))
#         _fig.suptitle(title)
#         plt.show(block=False)
#     else:
#         for k in range(4):
#             _im[k].set_data(imgs[k])
#         _fig.suptitle(title)

#     _fig.canvas.draw_idle()
#     plt.pause(0.001)  # ここが“表示更新”







plt.ion()
_LEG = ["FL","FR","RL","RR"]
_fig_xy = None
_ax_xy  = None

def show_cmd_vs_pred_xy(teach_xy, pred_xy, foot_xy=None, L=1.2, W=0.6, title=""):
    """
    teach_xy: (4,2) teacher/cmd (yaw-base) [m]
    pred_xy : (4,2) pred            [m]
    foot_xy : (4,2) optional current feet (yaw-base) [m]
    """
    global _fig_xy, _ax_xy

    def _np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return np.asarray(x)

    T = _np(teach_xy)
    P = _np(pred_xy)
    F = _np(foot_xy)

    if _fig_xy is None:
        _fig_xy = plt.figure(figsize=(6,6))
        _ax_xy  = _fig_xy.add_subplot(1,1,1)
        plt.show(block=False)

    ax = _ax_xy
    ax.cla()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-L/2, L/2])
    ax.set_ylim([-W/2, W/2])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m] (yaw-base)")
    ax.set_ylabel("y [m] (yaw-base)")
    ax.set_title(title)

    # teacher
    ax.scatter(T[:,0], T[:,1], marker="o", label="teacher(cmd)")
    for i in range(4):
        ax.text(T[i,0], T[i,1], f"T{_LEG[i]}", fontsize=9)

    # pred
    ax.scatter(P[:,0], P[:,1], marker="x", label="pred")
    for i in range(4):
        ax.text(P[i,0], P[i,1], f"P{_LEG[i]}", fontsize=9)

    # foot (optional)
    # if F is not None:
    #     ax.scatter(F[:,0], F[:,1], marker="+", label="foot_now")
    #     for i in range(4):
    #         ax.text(F[i,0], F[i,1], f"F{_LEG[i]}", fontsize=9)

    # vectors pred -> teacher（脚ごと）
    # for i in range(4):
    #     ax.plot([P[i,0], T[i,0]], [P[i,1], T[i,1]], linewidth=2)

    # error numbers
    err_cm = np.linalg.norm(P - T, axis=1) * 100.0
    ax.text(0.02, 0.98, f"err(cm) = {[round(e,1) for e in err_cm]}",
            transform=ax.transAxes, va="top", ha="left")

    ax.legend(loc="best")
    _fig_xy.canvas.draw_idle()
    plt.pause(0.001)





@torch.no_grad()
def get_foot_pos_basefull(raw, foot_body_ids):
    robot = raw.scene["robot"]
    root = robot.data.root_state_w
    base_p = root[:, 0:3]
    q_full = root[:, 3:7]  # wxyz
    feet_w = robot.data.body_pos_w[:, foot_body_ids, :]  # (B,4,3)
    rel_w = feet_w - base_p[:, None, :]
    # world -> base(full): apply conj(q_full)
    q_conj = quat_conjugate(q_full)
    feet_b = quat_apply(q_conj[:, None, :], rel_w)
    return feet_b  # (B,4,3)




def yawbase_xy_to_base_cmd(pred_xy_y: torch.Tensor,
                           root13_now: torch.Tensor,
                           z_ref_b: torch.Tensor | None = None,
                           default_z: float = -0.35) -> torch.Tensor:
    """
    pred_xy_y: (B,4,2) yaw-base [m]
    root13_now: (B,13) root_state_w  (quat wxyz を含む)
    z_ref_b: (B,4,3) があればそのzを使う（教師zなど）
    return: (B,4,3) base座標の目標
    """
    B = pred_xy_y.shape[0]

    # z を用意
    if z_ref_b is None:
        z = torch.full((B,4), float(default_z), device=pred_xy_y.device, dtype=pred_xy_y.dtype)
    else:
        z = z_ref_b[..., 2].to(pred_xy_y.device).to(pred_xy_y.dtype)

    v_y = torch.zeros((B,4,3), device=pred_xy_y.device, dtype=pred_xy_y.dtype)
    v_y[..., :2] = pred_xy_y
    v_y[...,  2] = z

    # yaw-base -> base : +yaw 回転
    qy = yaw_quat(root13_now[:, 3:7].to(pred_xy_y.device))  # (B,4) wxyz yaw-only

    v_flat = v_y.reshape(B*4, 3)
    q_flat = qy.repeat_interleave(4, dim=0)
    v_b_flat = quat_apply(q_flat, v_flat)                   # (B*4,3)

    return v_b_flat.reshape(B,4,3)









LEG2I = {"FL":0, "FR":1, "RL":2, "RR":3}




# def select_xy_from_logits_topk(
#     logits_u,                 # (Bu,K) or (Bu,1,K)
#     prev_xy_u,                # (Bu,2)  同じ座標系（たぶん yawbase）
#     index_to_xy_fn,           # idx -> (Bu,Kcand,2)
#     Wb_out: int,
#     x_coords, y_coords,
#     Kcand: int = 0,
#     lam: float = 10.0,        # 距離ペナルティ係数（logit / m）
#     gap_th: float = 0.5,      # top1-top2がこれ未満なら「僅差」とみなす
# ):
#     if logits_u.ndim == 3:
#         logits_u = logits_u.squeeze(1)  # (Bu,K)

#     topv, topi = logits_u.topk(Kcand, dim=-1)     # (Bu,Kcand)
#     cand_xy = index_to_xy_fn(topi, x_coords, y_coords, Wb_out).float()  # (Bu,Kcand,2)

#     # top1/top2 gap（自信の粗い指標）
#     gap = topv[:, 0] - topv[:, 1]                 # (Bu,)

#     # 距離（前回目標に近いほど良い）
#     dist = torch.linalg.vector_norm(cand_xy - prev_xy_u[:, None, :], dim=-1)  # (Bu,Kcand)

#     # 僅差のときだけ距離を効かせる（普段はargmaxのまま）
#     use_bias = (gap < gap_th)[:, None]             # (Bu,1)
#     score = torch.where(use_bias, topv - lam * dist, topv)

#     best = score.argmax(dim=-1)                    # (Bu,)
#     pred_xy = cand_xy[torch.arange(cand_xy.shape[0], device=cand_xy.device), best]  # (Bu,2)
#     return pred_xy, gap



def select_xy_from_logits_topk(
    foot_logits_u,            # (Bu, Kall)
    x_coords, 
    y_coords,
    prev_xy_u=None,           # (Bu,2)  yawbase
    K_TOPK=50,
    ent_th=4.7,
    p1_th=0.03,
    margin_th=0.005,
    lam_prev=300.0,           # ★距離(m)の二乗に掛けるのでこれくらいから
    jump_max=0.25,            # ★25cmより遠い候補は捨てる（迷い時の暴走防止）
    
):
    Bu = foot_logits_u.shape[0]
    prob = torch.softmax(foot_logits_u, dim=-1)                          # (Bu,Kall)

    # --- 不確実性指標 ---
    top2_p, top2_idx = prob.topk(2, dim=-1)
    p1 = top2_p[:,0]
    margin = top2_p[:,0] - top2_p[:,1]
    ent = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)

    unsure = (ent > ent_th) | (p1 < p1_th) | (margin < margin_th)

    # --- argmax（通常時） ---
    arg_idx = prob.argmax(dim=-1)                                        # (Bu,)
    arg_xy  = index_to_xy(arg_idx, x_coords, y_coords, Wb_out)           # (Bu,2)

    if (prev_xy_u is None) or (unsure.sum().item() == 0):
        return arg_xy, dict(p1=p1, margin=margin, ent=ent, unsure=unsure)

    # --- topK候補 ---
    topk_p, topk_idx = prob.topk(K_TOPK, dim=-1)                         # (Bu,K)
    topk_xy = index_to_xy(topk_idx, x_coords, y_coords, Wb_out)          # (Bu,K,2)

    # --- スコア：-log p + lam*||xy-prev||^2（迷い時だけ） ---
    base = -torch.log(topk_p.clamp_min(1e-9))                             # (Bu,K)
    d2 = ((topk_xy - prev_xy_u[:,None,:])**2).sum(dim=-1)                 # (Bu,K)

    # jump制限（遠すぎる候補は除外）
    far = (d2 > (jump_max**2))
    score = base + lam_prev * d2
    score = score.masked_fill(far, 1e9)

    best = score.argmin(dim=-1)                                           # (Bu,)
    sel_xy = topk_xy[torch.arange(Bu, device=foot_logits_u.device), best] # (Bu,2)

    # unsure のときだけ sel_xy、それ以外は argmax
    out_xy = torch.where(unsure[:,None], sel_xy, arg_xy)
    return out_xy, dict(p1=p1, margin=margin, ent=ent, unsure=unsure, best=best, topk_idx=topk_idx)





def dump_rank_info(foot_logits_u, teacher_idx_u, topk=20):
    # foot_logits_u: (Bu,K)
    Bu, K = foot_logits_u.shape
    # teacher logit
    tlog = foot_logits_u[torch.arange(Bu, device=foot_logits_u.device), teacher_idx_u]  # (Bu,)
    # rank: 0が最高
    teacher_rank = (foot_logits_u > tlog[:, None]).sum(dim=-1)  # (Bu,)
    # topk membership
    topi = foot_logits_u.topk(min(topk, K), dim=-1).indices
    in_topk = (topi == teacher_idx_u[:, None]).any(dim=-1)
    # gap
    topv = foot_logits_u.topk(2, dim=-1).values
    gap = topv[:, 0] - topv[:, 1]
    return teacher_rank, in_topk, gap




























def main():


    # """Play with RSL-RL agent."""
    # # parse configuration
    # env_cfg = parse_env_cfg(
    #     args_cli.task,
    #     device=args_cli.device,
    #     num_envs=args_cli.num_envs,
    #     use_fabric=not args_cli.disable_fabric,
    #     entry_point_key="play_env_cfg_entry_point",
    # )
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # if args_cli.use_pretrained_checkpoint:
    #     resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    # else:
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # log_dir = os.path.dirname(resume_path)


    # create isaac environment
    # env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)



    # test_cfg = agent_cfg.to_dict()
    # # # 2. 次に、その「辞書」の中身をクラスオブジェクトで上書きします
    # # #    (辞書なので、アクセスは[]を使います)
    # # test_cfg["policy"]["class_name"] = VisionMLPActorCritic
    # # test_cfg["policy"]["class_name"] = LocoTransformerActorCritic
    # # test_cfg["policy"]["class_name"] = LocoTransformerFinetune
    # # test_cfg["policy"]["class_name"] = LocoTransformerHFP
    # test_cfg["policy"]["class_name"] = MlpHFP
    # # test_cfg["policy"]["class_name"] = VisionHighLevelAC
    # # test_cfg["policy"]["class_name"] = VisionHighRNN



    # print(f"[INFO]: Loading model checkpoint from: {resume_path}")


    # """Play with RSL-RL agent."""
    # # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # if args_cli.use_pretrained_checkpoint:
    #     resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    # else:
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # log_dir = os.path.dirname(resume_path)

    # create isaac environment
    device=args_cli.device,

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    raw = env.unwrapped

    cam = raw.scene.sensors["camera"]
    ray = raw.scene.sensors["height_scanner"]


    # # wrap for video recording
    # if args_cli.video:
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos", "play"),
    #         "step_trigger": lambda step: step == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     print("[INFO] Recording videos during training.")
    #     print_dict(video_kwargs, nesting=4)
    #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # device = "cuda:0"
    # --- app launch ---
    # app_launcher = AppLauncher({})
    # simulation_app = app_launcher.app

    # --- env create（あなたの PLAY cfg を使う） ---
    # from unitree_rl_lab.tasks... import YourEnvCfg_PLAY
    # cfg = YourEnvCfg_PLAY()
    # cfg.scene.num_envs = 64
    # cfg.sim.device = device
    # env = ManagerBasedRLEnv(cfg=cfg)

    # ---- student load（TorchScript 推奨） ----
    # student = torch.jit.load("student_ts.pt", map_location=device).eval()

    # print("device =", device, type(device))


    # ppo_runner = OnPolicyRunner(env, test_cfg, log_dir=log_dir, device=agent_cfg.device)
    # ppo_runner.load(resume_path)

    # # ★これが「policy（callable）」になる
    # policy = ppo_runner.get_inference_policy(device=agent_cfg.device)

    device = device[0] if isinstance(device, tuple) else device



    ckpt_path = "/home/digital/isaac_ws/unitree_rl_lab/runs/student/best.pt"


    ckpt = torch.load(ckpt_path, map_location=device)


    # model = StudentNetInBEV(Hb_out, Wb_out, use_proprio=use_proprio).to(device)

    Hb_out, Wb_out = 61, 31  # 旧のまま（H5から取ってOK）

    # student_model = StudentNetInBEV_FiLM2D_OS8(Hb_out, Wb_out, use_proprio=True, pretrained_backbone=True).to(device)
    student_model = SimpleConcatFootNet(Hb_out, Wb_out, pretrained_backbone=True).to(device)

    # state_dict ロード
    missing, unexpected = student_model.load_state_dict(ckpt["model"], strict=True)
    print("missing:", missing)
    print("unexpected:", unexpected)
    # self.student_model.freeze_backbone(True)   # backbone凍結（BNもevalに）

    student_model.eval()
    for p in student_model.parameters():
        p.requires_grad_(False)

    # ---- 学習時メタ（ここはあなたの保存値/固定値に合わせる） ----
    res, L, W = 0.02, 1.2, 0.6


    # ---- output grid meta (教師/評価で使う) ----
    Hb_out = int(ckpt["Hb"])
    Wb_out = int(ckpt["Wb"])
    res    = float(ckpt["ray_res"])
    L      = float(ckpt["ray_L"])
    W      = float(ckpt["ray_W"])



    bev_grid_in = build_bev_grid_in(torch.device(device))  # (128,64,2)

   

    # g = bev_grid_in
    # print("[GRIDCHK]",
    #     g[0,0].detach().cpu().numpy(),
    #     g[0,-1].detach().cpu().numpy(),
    #     g[-1,0].detach().cpu().numpy(),
    #     g[-1,-1].detach().cpu().numpy(),
    #     float(g.mean()), float(g.std()))

    # grid_flipv = bev_grid_in.clone()
    # grid_flipv[..., 1] *= -1                     # grid_yだけ反転（投影の上下反転）

    
    

    # ---- 足 body ids（1回だけ） ----
    robot = raw.scene.articulations["robot"]
    leg_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_body_ids = [robot.body_names.index(n) for n in leg_order]

    # ---- hold buffer 初期化 ----
    B = raw.num_envs


    foot_b43 = get_foot_pos_yawbase(raw, foot_body_ids)         # (B,4,3)
    upper_action = foot_b43.reshape(B, -1).clone().to(device)   # (B,12)

    # ---- phase 監視（cmd_term の phase） ----
    cmd_name = "step_fr_to_block"
    cmd_term = raw.command_manager._terms[cmd_name]
    prev_phase = cmd_term.phase.clone().detach()



    # T_hist = 64
    T_hist = 48
    T = T_hist
    ptr = 0
    D_FSM = 8


    root_hist = torch.zeros((B, T, 13), device=device, dtype=torch.float32)
    foot_hist = torch.zeros((B, T, 4, 3), device=device, dtype=torch.float32)
    contact_hist = torch.zeros((B, T, 4), device=device, dtype=torch.uint8)
    fsm_hist     = torch.zeros((B, T, D_FSM), device=device, dtype=torch.float32)



    # def hist_write(root13_b13, foot_b43, contact_b4_u8):
    #     nonlocal ptr
    #     root_hist[:, ptr, :] = root13_b13
    #     foot_hist[:, ptr, :, :] = foot_b43
    #     contact_hist[:, ptr, :] = contact_b4_u8
    #     ptr = (ptr + 1) % T

    # def hist_gather(env_ids):
    #     t = torch.arange(T, device=device)
    #     idx = (ptr + t) % T
    #     r = root_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,13)
    #     f = foot_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,4,3)
    #     c = contact_hist.index_select(0, env_ids).index_select(1, idx)   # (Bu,T,4)
    #     return r, f, c
    

    # def hist_fill(done_ids, root13_now, foot43_now, contact_now_u8):
    #     # done_ids: (Nd,) long
    #     # root13_now: (B,13), foot43_now: (B,4,3), contact_now: (B,4)
    #     contact_now_u8 = contact_now_u8.to(torch.uint8)
    #     for t in range(T):
    #         root_hist[done_ids, t, :] = root13_now[done_ids]
    #         foot_hist[done_ids, t, :, :] = foot43_now[done_ids]
    #         contact_hist[done_ids, t, :] = contact_now_u8[done_ids]

    
    def hist_write(root13_b13, foot_b43, contact_b4_u8, fsm_bD):
        nonlocal ptr
        root_hist[:, ptr, :] = root13_b13
        foot_hist[:, ptr, :, :] = foot_b43
        contact_hist[:, ptr, :] = contact_b4_u8
        fsm_hist[:, ptr, :] = fsm_bD
        ptr = (ptr + 1) % T

    def hist_gather(env_ids):
        t = torch.arange(T, device=device)
        idx = (ptr + t) % T
        r = root_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,13)
        f = foot_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,4,3)
        c = contact_hist.index_select(0, env_ids).index_select(1, idx)   # (Bu,T,4)
        s = fsm_hist.index_select(0, env_ids).index_select(1, idx)       # (Bu,T,20)
        return r, f, c, s

    def hist_fill(done_ids, root13_now, foot43_now, contact_now_u8):
        contact_now_u8 = contact_now_u8.to(torch.uint8)
        fsm0 = torch.zeros((done_ids.numel(), D_FSM), device=device, dtype=torch.float32)
        for t in range(T):
            root_hist[done_ids, t, :] = root13_now[done_ids]
            foot_hist[done_ids, t, :, :] = foot43_now[done_ids]
            contact_hist[done_ids, t, :] = contact_now_u8[done_ids]
            fsm_hist[done_ids, t, :] = fsm0






    base_env = raw  # OrderEnforcing 等を剥がしたいならさらに whileで剥がす
    # cmd_term = base_env.command_manager.get_term("step_fr_to_block")  # Teacher
    # prev_phase = cmd_term.phase.clone()

    # base_env = env.unwrapped
    


    # ループ前に追加（収集と同じバッファ）
    cmd_term = raw.command_manager.get_term("step_fr_to_block")

    cmd_ver_prev = cmd_term.cmd_version.clone()
    cmd_pre_b = cmd_term.command.view(B,4,3).clone()

    cmd_prev_update = cmd_term.command.view(B, 4, 3).clone()   # 「前回更新時」のcmd（moved算出用）
    cmd_ver_prev_end = cmd_term.cmd_version.clone()  # 前イテレーションの「step後」



    x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
    y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)



    # 観測バッファ（ステップ前）
    rgb_prev = None
    foot_prev = None
    root_prev = None


    # ログ用
    err_hist = []

    def _unwrap_reset_out(reset_out):
        # gymnasium: (obs, info)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            return reset_out[0]
        return reset_out


    # obs, _ = raw.reset()

    obs = _unwrap_reset_out(raw.reset()) # vec env




    # FSM internal state
    DT = float(raw.step_dt)
    T_HOLD  = 0.2
    DROPOUT = 0.2
    F_ON, F_OFF = 10.0, 6.0
    NEAR_ON, NEAR_OFF = 0.16, 0.20

    hold_t = torch.zeros((B,), device=device)
    bad_t  = torch.zeros((B,), device=device)
    cmd_age = torch.zeros((B,), device=device)

    contact_state = torch.zeros((B,4), dtype=torch.bool, device=device)
    near_state    = torch.zeros((B,4), dtype=torch.bool, device=device)
    stable_t_contact = torch.zeros((B,4), device=device)
    stable_t_near    = torch.zeros((B,4), device=device)

    cmd_ver_prev2 = cmd_term.cmd_version.clone()   # cmd_age用



    



    contact_sensor = raw.scene.sensors["contact_forces"]
    sensor_body_names = contact_sensor.body_names
    sensor_cols = [sensor_body_names.index(leg) for leg in ["FL_foot","FR_foot","RL_foot","RR_foot"]]
    F = contact_sensor.data.net_forces_w[:, sensor_cols, :]   # (B,4,3)
    Fz = F[..., 2].abs()                                      # (B,4)
    contacts = (Fz > 8).to(torch.uint8)        # (B,4) 0/1


    # reset直後のpost状態で埋める（これが無いと最初のTステップが壊れる）
    root13_0 = robot.data.root_state_w[:, :13].to(device).float()
    foot43_0 = get_foot_pos_yawbase(raw, foot_body_ids).to(device).float()
    # hist_fill(torch.arange(B, device=device), root13_0, foot43_0)


    # contacts0 = torch.zeros((B,4), device=device, dtype=torch.uint8)
    hist_fill(torch.arange(B, device=device), root13_0, foot43_0, contacts)









    # ===== ループ外で1回だけ作る（毎step作ると無駄＆バグ源）=====
    x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
    y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)
    all_ids  = torch.arange(B, device=device)

    def logits_to_xy_argmax(foot_logits):
        # foot_logits: (Bu,4,K)
        pred_idx = foot_logits.argmax(dim=-1)                          # (Bu,4)
        pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)   # (Bu,4,2)
        return pred_xy

    def quantize_xy(xy):
        # xy: (Bu,4,2) continuous -> index -> xy (学習指標と一致)
        idx = xy_to_index(xy, Hb_out, Wb_out, L, W, res).long()        # (Bu,4)
        xyq = index_to_xy(idx, x_coords, y_coords, Wb_out)             # (Bu,4,2)
        return xyq









    def run_pred_sliced(rgb_u8, root_seq, foot_seq, cont_seq):
        """
        ここ重要:
        - 既に slice 済みを渡す（中で env_ids を触らない）
        - pre/post の比較が env ずれしない
        """
        Bu = root_seq.shape[0]
        if Bu == 0:
            return None

        bev = rgb_to_bev_in(rgb_u8.to(device), bev_grid_in)  # (Bu,3,128,64)


        # print(f"BEV Range: {bev.min().item()} ~ {bev.max().item()}")

        # build_proprio_flat はあなたの既存関数でOK
        proprio_flat = build_proprio_flat(
            root_seq.to(device),
            foot_seq.to(device),
            cont_seq.to(device) if cont_seq is not None else None
        )  # (Bu, T*P)

        assert proprio_flat.shape[1] == T * 25, proprio_flat.shape


        foot_logits, _ = student_model(bev, proprio=proprio_flat)  # (Bu,4,K)
        return logits_to_xy_argmax(foot_logits).detach().cpu()     # (Bu,4,2)
    


    def run_pred_sliced2(rgb_u8, root_seq, foot_seq, cont_seq,fsm_seq):
        """
        ここ重要:
        - 既に slice 済みを渡す（中で env_ids を触らない）
        - pre/post の比較が env ずれしない
        """
        Bu = root_seq.shape[0]
        if Bu == 0:
            return None

        bev = rgb_to_bev_in(rgb_u8.to(device), bev_grid_in)  # (Bu,3,128,64)


        # print(f"BEV Range: {bev.min().item()} ~ {bev.max().item()}")

        # build_proprio_flat はあなたの既存関数でOK
        proprio_flat = build_proprio_flat2(
            root_seq.to(device),
            foot_seq.to(device),
            cont_seq.to(device) if cont_seq is not None else None,
            fsm_seq.to(device)
        )  # (Bu, T*P)

        assert proprio_flat.shape[1] == T * 45, proprio_flat.shape


        foot_logits, moved = student_model(bev, proprio=proprio_flat)  # (Bu,4,K)
        return logits_to_xy_argmax(foot_logits).detach(), moved   # (Bu,4,2)
    


    # def run_pred_1leg(rgb_u8, root_seq, foot_seq, cont_seq, swing_onehot=None):
    #     """
    #     1脚版（swing条件付き）:
    #     - slice済み(Bu)を渡す（中でenv_ids触らない）
    #     - FSMは使わない（互換のため引数だけ残す）
    #     - swing_onehot(Bu,4) を proprio に concat
    #     - 出力は (Bu,2) のみ（= 選ばれた脚の着地点）
    #     """
    #     Bu, T, _ = root_seq.shape
    #     if Bu == 0:
    #         return None, None

    #     if swing_onehot is None:
    #         raise RuntimeError("run_pred_sliced2: swing_onehot is required for 1-leg model.")

    #     # -----------------------------
    #     # (1) BEV
    #     # -----------------------------
    #     bev = rgb_to_bev_in(rgb_u8.to(device), bev_grid_in)  # (Bu,3,128,64)

    #     # -----------------------------
    #     # (2) proprio_flat = [foot(12), root_feat(9), contact(4)] * T  + swing(4)
    #     # -----------------------------
    #     root_seq = root_seq.to(device)
    #     foot_seq = foot_seq.to(device)

    #     foot_hist_flat = foot_seq.reshape(Bu, T, 12)                     # (Bu,T,12)
    #     root_feat = root13_hist_to_feat(root_seq, keep_z=True)           # (Bu,T,9)

    #     if cont_seq is None:
    #         ch = torch.zeros(Bu, T, 4, device=device, dtype=torch.float32)
    #     else:
    #         ch = (cont_seq.to(device) > 0.5).to(torch.float32)           # (Bu,T,4)

    #     proprio_hist = torch.cat([foot_hist_flat, root_feat, ch], dim=-1) # (Bu,T,25)
    #     proprio_flat = proprio_hist.reshape(Bu, -1)                       # (Bu,T*25)

    #     swing = swing_onehot.to(device).float()                           # (Bu,4)
    #     swing = (swing > 0.5).float()
    #     bad = (swing.sum(dim=1) == 0)
    #     if bad.any():
    #         swing[bad, 0] = 1.0                                           # 全ゼロ保険

    #     proprio_flat = torch.cat([proprio_flat, swing], dim=-1)           # (Bu, T*25 + 4)

    #     # 念のためshapeチェック（T=48なら 1204）
    #     expected = T * 25 + 4
    #     assert proprio_flat.shape[1] == expected, (proprio_flat.shape, expected)

    #     # -----------------------------
    #     # (3) forward: logits (Bu,1,K) -> pred_xy (Bu,2)
    #     # -----------------------------
    #     out = student_model(bev, proprio=proprio_flat)

    #     foot_logits = out[0] if isinstance(out, (tuple, list)) else out   # (Bu,1,K) or (Bu,K)

      
    #     if foot_logits.ndim == 3:
    #         foot_logits = foot_logits.squeeze(1)                          # (Bu,K)

    #     pred_idx = foot_logits.argmax(dim=-1)                             # (Bu,)
    #     pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)      # (Bu,2)

    #     swing_leg = swing.argmax(dim=-1)                                  # (Bu,)
    #     return pred_xy.detach(), swing_leg.detach()





    def normalize_fsm8(fsm_seq):  # (Bu,T,8) 想定
        f = fsm_seq.to(torch.float32)

        # 学習と同じ
        cmd_age = torch.clamp(f[..., 0:1], 0.0, 2.0) / 2.0
        hold_t  = torch.clamp(f[..., 1:2], 0.0, 2.0) / 2.0
        bad_t   = torch.clamp(f[..., 2:3], 0.0, 2.0) / 2.0
        armed   = torch.clamp(f[..., 3:4], 0.0, 1.0)

        contact4 = torch.clamp(f[..., 4:8], 0.0, 1.0)

        return torch.cat([cmd_age, hold_t, bad_t, armed, contact4], dim=-1)  # (Bu,T,8)
    


    def run_pred_1leg(
        rgb_u8, root_seq, foot_seq, cont_seq, fsm_seq,
        swing_onehot=None,
        prev_xy=None,                 # ★追加: (Bu,2) yawbase。同脚の前回採用目標
        topk=10,                      # ★追加
        lam=10.0,                     # ★追加: logit/m
        gap_th=0.5,                   # ★追加: top1-top2の僅差閾値
        return_logits=False
    ):
        Bu, T, _ = root_seq.shape
        if Bu == 0:
            return None, None

        if swing_onehot is None:
            raise RuntimeError("run_pred_1leg: swing_onehot is required for 1-leg model.")

        bev = rgb_to_bev_in(rgb_u8.to(device), bev_grid_in)  # (Bu,3,128,64)

        root_seq = root_seq.to(device)
        foot_seq = foot_seq.to(device)

        foot_hist_flat = foot_seq.reshape(Bu, T, 12)                     # (Bu,T,12)
        root_feat = root13_hist_to_feat(root_seq, keep_z=True)           # (Bu,T,9)

        if cont_seq is None:
            ch = torch.zeros(Bu, T, 4, device=device, dtype=torch.float32)
        else:
            ch = (cont_seq.to(device) > 0.5).to(torch.float32)           # (Bu,T,4)


        
        fsm_seq = normalize_fsm8(fsm_seq)          # ←これを必ず入れる


        proprio_hist = torch.cat([foot_hist_flat, root_feat, fsm_seq], dim=-1) # (Bu,T,25)
        # proprio_flat = proprio_hist.reshape(Bu, -1)                       # (Bu,T*25)

        swing = swing_onehot.to(device).float()
        swing = (swing > 0.5).float()
        bad = (swing.sum(dim=1) == 0)
        if bad.any():
            swing[bad, 0] = 1.0


        swing_rep = swing[:, None, :].expand(Bu, T, 4)  # (Bu,T,4)


        proprio_flat = torch.cat([proprio_hist, swing_rep], dim=-1)           # (Bu,T*25+4)
        proprio_flat = proprio_flat.reshape(B, -1)           # (B,T*33) = (B,1584)

        expected = T * 33
        assert proprio_flat.shape[1] == expected, (proprio_flat.shape, expected)

        out = student_model(bev, proprio=proprio_flat)
        foot_logits = out[0] if isinstance(out, (tuple, list)) else out   # (Bu,1,K) or (Bu,K)

        if foot_logits.ndim == 3:
            foot_logits = foot_logits.squeeze(1)                          # (Bu,K)

        # -----------------------------
        # ★ここから差し替え: argmax -> TopK + 近傍バイアス
        # -----------------------------
        if (prev_xy is None) or (topk <= 1):
            pred_idx = foot_logits.argmax(dim=-1)                         # (Bu,)
            pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)  # (Bu,2)
            gap = None
        else:
            prev_xy = prev_xy.to(device).float()                          # (Bu,2)

            Kcand = min(topk, foot_logits.shape[-1])
            topv, topi = foot_logits.topk(Kcand, dim=-1)                  # (Bu,Kcand)

            # 候補をxyへ
            cand_xy = index_to_xy(topi, x_coords, y_coords, Wb_out).float()  # (Bu,Kcand,2)

            # 自信（top1-top2差）
            gap = topv[:, 0] - topv[:, 1]                                 # (Bu,)

            # 距離（前回目標へ）
            dist = torch.linalg.vector_norm(cand_xy - prev_xy[:, None, :], dim=-1)  # (Bu,Kcand)

            use_bias = (gap < gap_th)[:, None]                             # (Bu,1)
            score = torch.where(use_bias, topv - lam * dist, topv)         # (Bu,Kcand)

            best = score.argmax(dim=-1)                                    # (Bu,)
            pred_xy = cand_xy[torch.arange(Bu, device=device), best]       # (Bu,2)

        swing_leg = swing.argmax(dim=-1)                                  # (Bu,)
        # return pred_xy.detach(), swing_leg.detach()  # gapも返したければ返せます
    
        if return_logits:
            return pred_xy.detach(), swing_leg.detach(), foot_logits.detach()
        else:
            return pred_xy.detach(), swing_leg.detach()

    




    def run_pred_sliced33(rgb_u8, root_seq, foot_seq, cont_seq, fsm_seq):
        Bu = root_seq.shape[0]
        if Bu == 0:
            return None

        bev = rgb_to_bev_in(rgb_u8.to(device), bev_grid_in)  # (Bu,3,128,64)

        proprio_flat = build_proprio_flat33(
            root_seq.to(device),
            foot_seq.to(device),
            fsm_seq.to(device),
            keep_z=True,
           
        )
        assert proprio_flat.shape[1] == T * 41, proprio_flat.shape

        foot_logits, _ = student_model(bev, proprio=proprio_flat)  # (Bu,4,K)
        return logits_to_xy_argmax(foot_logits).detach()           # GPUのまま

    












    # def run_pred_sliced(rgb_u8, root_seq, foot_seq, cont_seq):
    #     Bu = root_seq.shape[0]
    #     if Bu == 0:
    #         return None

    #     # ★動的grid：このスライスの「現在root」
    #     root_now = root_seq[:, -1, :13].to(device).float()   # (Bu,13)

    #     # grid, valid = build_bev_grid_in(
    #     #     root_state_w=root_now,
    #     #     t_bc=t_bc.to(device),              # base->cam offset (wxyz)
    #     #     q_bc=q_bc_wxyz.to(device),         # base->cam rot (wxyz)
    #     #     fx=fx, fy=fy, cx=cx, cy=cy,
    #     #     IMG_H=IMG_H, IMG_W=IMG_W,
    #     #     X_MIN=X_MIN_IN, X_MAX=X_MAX_IN, Y_WIDTH=Y_WIDTH_IN,
    #     #     HB=HB_IN, WB=WB_IN,
    #     #     world_ground_z=0.0,
    #     #     v_flip=False,  # 上下逆なら True
    #     # )

    #     # （デバッグしたいなら）
    #     # print(f"[IPM] valid_ratio={valid.float().mean().item():.3f}")

    #     bev = rgb_to_bev_in(rgb_u8.to(device), grid)  # ★ここが変わる

    #     proprio_flat = build_proprio_flat(
    #         root_seq.to(device),
    #         foot_seq.to(device),
    #         cont_seq.to(device) if cont_seq is not None else None
    #     )

    #     foot_logits, _ = student_model(bev, proprio=proprio_flat)
    #     pred_idx = foot_logits.argmax(dim=-1)
    #     pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)
    #     return pred_xy.detach().cpu()

    


    pending_mask = torch.zeros(B, dtype=torch.bool, device=device)
    pending_teacher_xy_q = torch.zeros(B, 4, 2, dtype=torch.float32, device=device)  # yaw-base, quantized



    def xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
        # (...,4) xyzw -> wxyz
        return q_xyzw[..., [3,0,1,2]]

    def get_sensor_world_pose_wxyz(sensor):
        """
        IsaacLab sensor から world pose を取る（頑健版）
        returns:
        pos_w: (B,3)
        quat_wxyz: (B,4)
        """
        d = sensor.data

        # 1) data に pos_w / quat_w がある場合（センサによってはある）
        if hasattr(d, "pos_w") and hasattr(d, "quat_w"):
            pos_w = d.pos_w
            quat_wxyz = d.quat_w
            return pos_w, quat_wxyz

        # 2) data に pos_w / rot_w (回転行列) がある場合
        if hasattr(d, "pos_w") and hasattr(d, "rot_w"):
            pos_w = d.pos_w
            R = d.rot_w  # (B,3,3)
            # 回転行列->quat(wxyz)（簡易。精度必要なら専用関数に置換してOK）
            qw = torch.sqrt(torch.clamp(1.0 + R[...,0,0] + R[...,1,1] + R[...,2,2], min=0.0)) / 2.0
            qx = (R[...,2,1] - R[...,1,2]) / torch.clamp(4*qw, min=1e-8)
            qy = (R[...,0,2] - R[...,2,0]) / torch.clamp(4*qw, min=1e-8)
            qz = (R[...,1,0] - R[...,0,1]) / torch.clamp(4*qw, min=1e-8)
            quat_wxyz = torch.stack([qw,qx,qy,qz], dim=-1)
            return pos_w, quat_wxyz

        # 3) view から取る（Camera はだいたいこれが通る）
        view = getattr(sensor, "_view", None) or getattr(sensor, "_sensor_view", None) or getattr(sensor, "view", None)
        if view is None or not hasattr(view, "get_world_poses"):
            # デバッグ用：使える属性名を見たい時
            keys = [k for k in dir(d) if any(s in k.lower() for s in ["pos", "quat", "rot", "pose", "tf"])]
            raise AttributeError(f"Cannot get camera world pose. CameraData attrs ~ {keys}, view={type(view)}")

        pos_w, quat_xyzw = view.get_world_poses()  # 多くの場合 xyzw
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        return pos_w, quat_wxyz


    robot = raw.scene.articulations["robot"]
    root = robot.data.root_state_w
    base_p = root[:,0:3]
    base_q = root[:,3:7]  # wxyz

    cam = raw.scene.sensors["camera"]
   
    cam_p, cam_q = get_sensor_world_pose_wxyz(cam)  # cam_q は wxyz になる


    def quat_conj(q):
        out = q.clone()
        out[...,1:] *= -1
        return out

    def quat_mul(a,b):
        aw,ax,ay,az = a.unbind(-1)
        bw,bx,by,bz = b.unbind(-1)
        return torch.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ], dim=-1)

    # base->cam の相対姿勢（base座標でのcam）
    q_bc_actual = quat_mul(quat_conj(base_q), cam_q)                    # (B,4)
    t_bc_actual = quat_apply(quat_conj(base_q), (cam_p - base_p))       # (B,3)

    # print("[CAM t_bc actual]", t_bc_actual[0].detach().cpu().numpy())
    # print("[CAM q_bc actual]", q_bc_actual[0].detach().cpu().numpy())
    # print("[CAM t_bc assumed]", t_bc.detach().cpu().numpy())
    # print("[CAM q_bc assumed]", q_bc_wxyz.detach().cpu().numpy())


    MAX_DELAY = 8  # 0〜7step後まで見る
    pending = torch.zeros(B, dtype=torch.bool, device=device)
    pending_k = torch.zeros(B, dtype=torch.long, device=device)
    pending_teacher = torch.zeros(B, 4, 2, dtype=torch.float32, device=device)


    am = raw.action_manager

    # IsaacLab は _terms が dict/list の両方あり得るので頑健に取る
    terms = getattr(am, "_terms", None)
    if isinstance(terms, dict):
        # 名前候補を探す
        print("[ACTION] term keys =", list(terms.keys()))
        term = terms.get("pre_trained_policy_action", None) or next(iter(terms.values()))
    elif isinstance(terms, (list, tuple)):
        print("[ACTION] n_terms =", len(terms))
        term = terms[0]
    else:
        term = None

    print("[ACTION] picked term =", type(term))


    foot0 = get_foot_pos_yawbase(raw, foot_body_ids).clone()   # (B,4,3)


    swing_order = torch.tensor([LEG2I["FR"], LEG2I["RL"], LEG2I["FL"], LEG2I["RR"]],
                                 device = device, dtype=torch.long)
    swing_pos = torch.zeros(B,device = device, dtype=torch.long)  # (B,) まず0から

    # cmd_age = torch.zeros(B, device=device)
    # armed_prev = torch.zeros(B, device=device, dtype=torch.bool)  # (B,)


    student_cmd_b = cmd_term.command.view(B,4,3).clone()  # 初期だけteacherでOK
    student_cmd_b = student_cmd_b.to(device)





    # ====== B: student側のゲート付きスケジューラ状態 ======
    DT = float(raw.step_dt)  # or env.step_dt
    T_HOLD   = 0.20
    DROPOUT  = 0.20

    F_ON  = 10.0
    F_OFF = 6.0
    NEAR_ON  = 0.16
    NEAR_OFF = 0.20

    active_leg  = swing_order[swing_pos]                            # (B,)

    # studentが実際に低レベルに出すコマンド（保持される）
    student_cmd_b = cmd_term.command.view(B,4,3).clone()            # 初期はteacherでOK
    # できれば reset直後は実足位置で初期化（後述）

    # ゲート判定の状態
    hold_t = torch.zeros(B, device=device)
    bad_t  = torch.zeros(B, device=device)
    armed_prev = torch.zeros(B, dtype=torch.bool, device=device)

    contact_state = torch.zeros(B, 4, dtype=torch.bool, device=device)
    near_state    = torch.zeros(B, 4, dtype=torch.bool, device=device)

    # reset直後クールダウン（任意だが強く推奨）
    cooldown_t = torch.zeros(B, device=device)
    COOLDOWN_S = 0.20


    Teacher_mode =False








    # ループ外で1回だけ
    last_pred_xy = torch.zeros(B, 4, 2, device=device)
    last_pred_valid = torch.zeros(B, 4, dtype=torch.bool, device=device)
    last_pred_step = torch.full((B,4), -1, device=device, dtype=torch.long)
    global_step = 0

    prev_xy = None  # ★先に必ず定

    prev_xy_mem   = torch.zeros(B,4,2, device=device)
    prev_xy_valid = torch.zeros(B,4,   device=device, dtype=torch.bool)


    # # # ループ外で保持
    # if 'rgb_prev' not in locals():
    #     rgb_prev = rgb_post.clone()



    




    with torch.inference_mode():
        while simulation_app.is_running():

            armed_f = torch.zeros(B, device=device, dtype=torch.float32)   # (B,)


            # -----------------------------
            # (1) step前スナップショット（全env）
            # -----------------------------
            cmd_ver_pre = cmd_term.cmd_version.clone()                 # (B,)
            cmd_pre_b   = cmd_term.command.view(B,4,3).clone()         # (B,4,3)
            swing_leg_pre = cmd_term.swing_leg.clone()   # 参考（脚条件に使うなら）

            rgb_pre   = get_front_rgb(raw).clone()                     # (B,H,W,3) uint8
            root_pre  = robot.data.root_state_w[:, :13].clone()        # (B,13)

            # ★重要：このstepで teacher yaw-base を作る参照rootを固定する
            # 収集側が「cmd更新直後のroot」で変換してるなら、評価も同じタイミングのrootに寄せるべき。
            # raw.stepの中でcmdが更新される前提でも、まずは "preのroot" で統一してズレを消す。
            root_ref_pre = root_pre.clone()                            # (B,13)

            # 履歴（pre）
            root_seq_pre_all, foot_seq_pre_all, cont_seq_pre_all, fsm_seq_pre_all = hist_gather(all_ids)

            # -----------------------------
            # (2) schedulerが指定する脚（外側は読むだけ）
            # -----------------------------
            # swing_leg_pre = term.swing_leg_now.clone()                 # (B,)
            # swing_leg_pre = term.swing_leg_next.clone()              # これが“今回使う予定の脚”

            # swing_oh_pre  = Fun.one_hot(swing_leg_pre, 4).float()         # (B,4)

            if Teacher_mode == True:
                teacher_leg = swing_leg_pre.long()
                teacher_oh  = torch.nn.functional.one_hot(teacher_leg, 4).float()

                # -----------------------------
                # (3) pred（pre観測で推論）
                # -----------------------------
                pred_xy_sel_pre, _ = run_pred_1leg(
                    rgb_pre,
                    root_seq_pre_all,
                    foot_seq_pre_all,
                    cont_seq_pre_all,
                    fsm_seq_pre_all,
                    # swing_oh_pre,   # ★条件：このstepでスケジューラが指定した脚
                    teacher_oh,
                    prev_xy
                )  # (B,2)  yaw-base

            # -----------------------------
            # (4) 上位アクション作成：latched を土台に swing脚だけ差し替え
            # -----------------------------
            latched_b_pre = term._latched_actions.view(B, 4, 3).clone()  # base(full)

            # base->yawbase（参照は root_ref_pre で統一）
            # latched_y = base_cmd_to_yawbase(latched_b_pre, root_ref_pre, yaw_quat, quat_apply)  # (B,4,3) yawbase
            # cmd_xy_y  = latched_y[:, :, :2].clone()                                            # (B,4,2)

            # idxB = torch.arange(B, device=device)
            # cmd_xy_y[idxB, swing_leg_pre, :] = pred_xy_sel_pre                                 # swing脚だけ差し替え

            # # yawbase(xy) -> base(xyz)
            # pred_cmd_b = yawbase_xy_to_base_cmd(cmd_xy_y, root_ref_pre, z_ref_b=latched_b_pre) # (B,4,3)
            # upper_action = pred_cmd_b.reshape(B, 12).contiguous()


            # cmd_y = base_cmd_to_yawbase(cmd_pre_b, root_pre, yaw_quat, quat_apply)   # (B,4,3) yawbase
            # cmd_xy_y = cmd_y[:,:,:2].clone()
            # cmd_xy_y[torch.arange(B, device=device), teacher_leg, :] = pred_xy_sel_pre
            # student_cmd_b = yawbase_xy_to_base_cmd(cmd_xy_y, root_pre, z_ref_b=cmd_pre_b)  # (B,4,3)
            # upper_action = student_cmd_b.reshape(B,12)


            swing_leg_pre = cmd_term.swing_leg.clone()        # (B,)
            cmd_pre_b = cmd_term.command.view(B,4,3).clone()


            # いったん action は「前回の student_cmd_b」を出す
            upper_action = student_cmd_b.reshape(B,12)

            # -----------------------------
            # (5) step 実行
            # -----------------------------
            obs, rew, terminated, truncated, info = raw.step(upper_action)
            done = (terminated | truncated).to(torch.bool)


            # step後
            swing_leg_post = cmd_term.swing_leg.clone()       # (B,)
            cmd_post_b = cmd_term.command.view(B,4,3).clone()




            # 「このstepで更新された」を差分で定義
            # delta = torch.linalg.vector_norm((cmd_post_b[..., :2] - cmd_pre_b[..., :2]), dim=-1)  # (B,4)
            # post_edge = (delta.max(dim=-1).values > 1e-2) & (~done)  # (B,)
            # cmd_updated = (delta.max(dim=-1).values > 1e-2) & (~done)  # (B,)


            # eval_ids = post_edge.nonzero(as_tuple=False).squeeze(-1)

            # leg_diff = delta.argmax(dim=-1)  # (B,)  ※更新がない時は意味がないので注意








            post_edge = (cmd_term.cmd_version != cmd_ver_pre) & (~done)
            ids = post_edge.nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                # “実際に更新された脚” をコマンド差分で取る（最も信頼できる）
                delta = torch.linalg.vector_norm(
                    (cmd_post_b[ids][..., :2] - cmd_pre_b[ids][..., :2]),
                    dim=-1
                )  # (Bu,4)
                leg_diff = delta.argmax(dim=-1)  # (Bu,)

                leg_pre  = swing_leg_pre[ids].long()
                leg_post = swing_leg_post[ids].long()

                m_pre  = (leg_pre  == leg_diff).float().mean().item()
                m_post = (leg_post == leg_diff).float().mean().item()
                print(f"[LEG-TIMING] match(pre)={m_pre:.3f} match(post)={m_post:.3f}")

            
            # ver_post = cmd_term.cmd_version.clone()
            # edge = (ver_post != ver_pre) & (~done)


            if post_edge.any() and Teacher_mode == True:
                ids = post_edge.nonzero(as_tuple=False).squeeze(-1)
                # leg = teacher_leg[ids]
                leg = swing_leg_post[ids].long()


                # student_cmd_b を yawbase にして、その脚だけ差し替え
                # student_y = base_cmd_to_yawbase(student_cmd_b, root_pre, yaw_quat, quat_apply)  # rootは一貫
                # student_xy = student_y[:, :, :2].clone()

                # idxB = torch.arange(B, device=device)
                # student_xy[idxB, leg, :] = pred_xy_sel_pre[ids]  # 更新脚だけ

                # # zは保持（student_cmd_b）か nominal で固定
                # student_cmd_b = yawbase_xy_to_base_cmd(student_xy, root_pre, z_ref_b=student_cmd_b)


                student_y  = base_cmd_to_yawbase(student_cmd_b, root_pre, yaw_quat, quat_apply)  # (B,4,3)
                student_xy = student_y[:, :, :2].clone()

                student_xy[ids, leg, :] = pred_xy_sel_pre[ids]   # ←これで「(env,leg)が1対1」

                student_cmd_b = yawbase_xy_to_base_cmd(student_xy, root_pre, z_ref_b=student_cmd_b)


            # -----------------------------
            # (6) step後スナップショット（全env）
            # -----------------------------
            cmd_ver_post = cmd_term.cmd_version.clone()
            cmd_post_b   = cmd_term.command.view(B,4,3).clone()

            root_post = robot.data.root_state_w[:, :13].clone()
            rgb_post  = get_front_rgb(raw).clone()
            foot_post = get_foot_pos_yawbase(raw, foot_body_ids).clone()







            rgb_delta = (rgb_post.float() - rgb_pre.float()).abs().mean(dim=(1,2,3))  # (B,)
            stale = (rgb_delta < 0.05)  # 閾値は適当に(まずは0.05〜0.2で試す)

            if stale.any():
                s = stale.nonzero(as_tuple=False).squeeze(-1)
                print(f"[RGB-STALE] n={s.numel()} e0={int(s[0])} delta={rgb_delta[s[0]].item():.4f}")

            rgb_pre = rgb_post.clone()





            # feet in base(full) が取れるならそれを使う（teacherと同じ座標系）
            feet_b = get_foot_pos_basefull(raw, foot_body_ids)      # (B,4,3) 既にある前提
        
            # === (B) near/contact 用の target を決める（ここだけ分岐）===
            if Teacher_mode:
                tgt_b = cmd_term.command.view(B,4,3).clone()   # teacher目標
            else:
                tgt_b = student_cmd_b   

            dist = torch.linalg.vector_norm(feet_b[..., :2] - tgt_b[..., :2], dim=-1)  # (B,4)

            # contact hysteresis
            F  = contact_sensor.data.net_forces_w[:, sensor_cols, :]   # (B,4,3)
            Fz = F[..., 2].abs()
            contact_on  = (Fz > F_ON)
            contact_off = (Fz < F_OFF)
            contact_state = torch.where(contact_state, ~contact_off, contact_on)

            # near hysteresis
            near_on  = (dist < NEAR_ON)
            near_off = (dist > NEAR_OFF)
            near_state = torch.where(near_state, ~near_off, near_on)

            # good_all（teacherと同じ）
            good_all = (contact_state & near_state).all(dim=-1)     # (B,)

            # dropout/hold
            bad_t = torch.where(good_all, torch.zeros_like(bad_t), bad_t + DT)
            broke = (bad_t > DROPOUT)
            hold_t = torch.where(
                good_all,
                hold_t + DT,
                torch.where(broke, torch.zeros_like(hold_t), hold_t)
            )


            # reset後クールダウン中は進行しない（推奨）
            if (cooldown_t > 0).any():
                cooldown_t = torch.clamp(cooldown_t - DT, min=0.0)
                good_all = good_all & (cooldown_t <= 0)


            
            armed = (hold_t >= T_HOLD)
            issue = armed & (~armed_prev)
            armed_prev = armed



            armed_f = armed.to(torch.float32)
            contact_f = contact_state.to(torch.float32)  # (B,4)



            if issue.any():
                ids = issue.nonzero(as_tuple=False).squeeze(-1)


                # ① 今回更新する脚（進める前）
                leg_now = swing_order[swing_pos][ids]          # (Bu,)

                teacher_oh  = torch.nn.functional.one_hot(leg_now, 4).float()

                rgb_post = get_front_rgb(raw).clone()
                root_seq_post_all, foot_seq_post_all, cont_seq_post_all, fsm_seq_all = hist_gather(all_ids)

                prev_xy = prev_xy_mem[ids, leg_now]    # (Bu,2)


                pred_xy, _ = run_pred_1leg(
                    rgb_post[ids],
                    root_seq_post_all[ids],
                    foot_seq_post_all[ids],
                    cont_seq_post_all[ids],
                    fsm_seq_all[ids],
                    teacher_oh,
                    prev_xy
                )  # (Bu,2) yawbase

                # student_cmd_b の “その脚だけ” XYを更新（他脚は絶対に触らない）
                idxB = torch.arange(ids.numel(), device=device)
                leg  = leg_now.long()                      # (Bu,)

                # 現在のstudent_cmdをyawbaseにして上書き → baseへ戻す（rootは統一）
                base_u = student_cmd_b[ids]                         # (Bu,4,3)
                root_u = root_post[ids]                             # postで統一（ここは一貫していればOK）

                y_u = base_cmd_to_yawbase(base_u, root_u, yaw_quat, quat_apply)     # (Bu,4,3)
                xy_u = y_u[:, :, :2].clone()

                xy_u[idxB, leg, :] = pred_xy

                # zは保持（student_cmd_bのzを参照）にするのが安定
                new_b = yawbase_xy_to_base_cmd(xy_u, root_u, z_ref_b=base_u)        # (Bu,4,3)
                student_cmd_b[ids] = new_b

                # タイマリセット（teacherと同じ）
                hold_t[ids] = 0.0
                bad_t[ids]  = 0.0

                print("[B-ISSUE] advance, leg =", leg[:8].tolist())


                    # ③ 更新が終わってから次へ
                swing_pos[ids] = (swing_pos[ids] + 1) % 4
                active_leg = swing_order[swing_pos]            # 次回用


                Bu = ids.numel()
                idx = torch.arange(Bu, device=device)

                # 1) いま保持している student 目標（base full）
                base_u = student_cmd_b[ids]                # (Bu,4,3)

                # 2) その瞬間のroot（predを作るタイミングと揃えるのが重要）
                root_u = root_post[ids]                    # (Bu,13) 例：postでpred作るならpost

                # 3) base -> yawbase に変換
                y_u = base_cmd_to_yawbase(base_u, root_u, yaw_quat, quat_apply)   # (Bu,4,3)

                # 4) 更新脚だけ取り出す（これが prev_xy）
                prev_xy = y_u[idx, leg, :2].contiguous()   # (Bu,2) yawbase


                prev_xy_mem[ids, leg] = pred_xy
                prev_xy_valid[ids, leg] = True

              





            # --- （デバッグ）このstepで yaw がどれくらい変わったか
            # ここが大きいほど、root_post参照でyawbase化するとズレやすい
            # print("[YAW-DELTA deg]",
            #       (yaw_from_root(root_post) - yaw_from_root(root_ref_pre)).abs().mean().item() * 57.3)

            # -----------------------------
            # (7) 履歴push（post）
            # -----------------------------
            # contact_u8 = contact_state.to(torch.uint8)


            cmd_ver_now = cmd_term.cmd_version.clone()
            cmd_updated = (cmd_ver_now != cmd_ver_prev) & (~done)
            cmd_age = torch.where(cmd_updated, torch.zeros_like(cmd_age), cmd_age + DT)
            cmd_ver_prev = cmd_ver_now

            # fsm_post: (B,4)
            # fsm_post = torch.stack([cmd_age, hold_t, bad_t, armed, contact_state], dim=-1).float()

            

            fsm_post = torch.cat([
                cmd_age.unsqueeze(-1),
                hold_t.unsqueeze(-1),
                bad_t.unsqueeze(-1),
                armed_f.unsqueeze(-1),
                contact_f,
            ], dim=-1)  # (B,20)


            hist_write(root_post.float(), foot_post.float(), contact_state, fsm_post)

            root_seq_post_all, foot_seq_post_all, cont_seq_post_all, fsm_seq_post_all = hist_gather(all_ids)


            root_seq_dbg, foot_seq_dbg, cont_seq_dbg, fsm_seq_dbg = hist_gather(all_ids)



            # 最新tのはず（-1）が “今書いたfsm_post” と一致するか
            fsm_last = fsm_seq_dbg[:, -1, :]                     # (B, D_FSM)
            diff = (fsm_last - fsm_post).abs().max(dim=-1).values # (B,)

            if (diff > 1e-5).any():
                bad = (diff > 1e-5).nonzero(as_tuple=False).squeeze(-1)
                print("[HIST-MISMATCH] n=", bad.numel(), "max=", diff[bad].max().item())
                i = int(bad[0].item())
                print("fsm_post[i] =", fsm_post[i].detach().cpu().numpy())
                print("fsm_last[i] =", fsm_last[i].detach().cpu().numpy())

            # -----------------------------
            # (8) 評価対象 env（cmd_version が変わったenv）
            # -----------------------------
            # あなたの方針に合わせる：このstep中にcmd_versionが変わったら評価する
            post_edge = (cmd_ver_post != cmd_ver_pre) & (~done)
            eval_ids = post_edge.nonzero(as_tuple=False).squeeze(-1)



            if eval_ids.numel() > 0:
                Bu = eval_ids.numel()

                # --- Teacherの更新脚は post の cmd_term.swing_leg が正解（あなたのログより） ---
                teacher_leg = cmd_term.swing_leg[eval_ids].long()  # (Bu,)
                teacher_oh  = torch.nn.functional.one_hot(teacher_leg, 4).float()  # (Bu,4)

                # teacher_leg = leg_diff[eval_ids].long()
                # teacher_oh  = torch.nn.functional.one_hot(teacher_leg, 4).float()


                # --- post の hist/obs で teacher_leg 条件の pred を出す（収集に寄せるならこっちが安全） ---
                rgb_u_post      = rgb_post[eval_ids]
                root_seq_u_post = root_seq_post_all[eval_ids]
                foot_seq_u_post = foot_seq_post_all[eval_ids]
                cont_seq_u_post = cont_seq_post_all[eval_ids]
                fsm_seq_u_post = fsm_seq_post_all[eval_ids]


                prev_xy = prev_xy_mem[eval_ids, teacher_leg]    # (Bu,2)


                pred_xy_teacher, _, logits = run_pred_1leg(
                    rgb_u_post,
                    root_seq_u_post,
                    foot_seq_u_post,
                    cont_seq_u_post,
                    fsm_seq_u_post,
                    teacher_oh,
                    prev_xy,
                    return_logits=True
                )  # (Bu,2)

                # --- teacher点（yawbase + quantize）も post root で作る ---
                cmd_post_b_u = cmd_term.command.view(B,4,3)[eval_ids]
                root_post_u  = root_post[eval_ids]
                teach_y = base_cmd_to_yawbase(cmd_post_b_u, root_post_u, yaw_quat, quat_apply)[..., :2]  # (Bu,4,2)
                teach_q = quantize_xy(teach_y)  # (Bu,4,2)

                idx = torch.arange(Bu, device=device)
                teach_xy = teach_q[idx, teacher_leg, :]  # (Bu,2)

                err_cm = torch.linalg.vector_norm(pred_xy_teacher - teach_xy, dim=-1) * 100.0
                print(f"[EVAL] Bu={Bu} teacherleg_err={err_cm.mean().item():.2f}cm")

                # --- 可視化（1脚だけ出す） ---
                i0 = 0
                leg = int(teacher_leg[i0].item())

                pred4  = torch.full((4,2), float("nan"))
                teach4 = torch.full((4,2), float("nan"))
                pred4[leg]  = pred_xy_teacher[i0].detach().cpu()
                teach4[leg] = teach_q[i0].detach().cpu()[leg]

                foot4 = foot_seq_u_post[i0, -1, :, :2].detach().cpu()

                show_cmd_vs_pred_xy(
                    teach_xy=teach4,
                    pred_xy=pred4,
                    foot_xy=foot4,
                    L=L, W=W,
                    title=f"env={int(eval_ids[i0])} teacher_leg={leg} phase={int(cmd_term.phase[int(eval_ids[i0])])}",
                )   



                prob = logits.softmax(dim=-1)  # (Bu,K)

                teacher_idx = xy_to_index(teach_xy, Hb_out, Wb_out, L, W, res).long()  # (Bu,)
                p_teacher = prob[torch.arange(Bu, device=device), teacher_idx]

                topk_prob, topk_idx = prob.topk(k=5, dim=-1)
                topk_xy = index_to_xy(topk_idx.reshape(-1), x_coords, y_coords, Wb_out).view(Bu,5,2)

                margin = topk_prob[:,0] - topk_prob[:,1]
                entropy = -(prob * (prob.clamp_min(1e-9).log())).sum(dim=-1)

                print(f"[OUT] p_teacher={p_teacher.mean().item():.4f} margin={margin.mean().item():.4f} ent={entropy.mean().item():.2f}")
                print(" top5_xy[0] =", topk_xy[0].detach().cpu().numpy(), " top5_p[0] =", topk_prob[0].detach().cpu().numpy())




                WARN_CM = 7.0
                BAD_CM  = 10.0

                bad_mask = (err_cm > WARN_CM)
                if bad_mask.any():
                    ids_bad = eval_ids[bad_mask]
                    Bu_bad = ids_bad.numel()
                    j = torch.arange(Bu_bad, device=device)

                    # --- 1) そのstepで実際に更新された脚（信頼できる）---
                    cmd_pre_u  = cmd_pre_b[ids_bad]          # (Bu,4,3)
                    cmd_post_u = cmd_post_b[ids_bad]         # (Bu,4,3)
                    delta = torch.linalg.vector_norm((cmd_post_u[..., :2] - cmd_pre_u[..., :2]), dim=-1)  # (Bu,4)
                    leg_diff = delta.argmax(dim=-1)          # (Bu,)

                    teacher_leg_post = cmd_term.swing_leg[ids_bad].long()  # (Bu,)
                    # teacher_leg_post = leg_diff[ids_bad].long()

                    match_leg = (teacher_leg_post == leg_diff)

                    # --- 2) teacher点を「root_pre」と「root_post」で両方作って差を見る（原因切り分け）---
                    root_pre_u  = root_pre[ids_bad]
                    root_post_u = root_post[ids_bad]

                    teach_pre  = base_cmd_to_yawbase(cmd_post_u, root_pre_u,  yaw_quat, quat_apply)[..., :2]   # (Bu,4,2)
                    teach_post = base_cmd_to_yawbase(cmd_post_u, root_post_u, yaw_quat, quat_apply)[..., :2]   # (Bu,4,2)

                    # その脚だけ抜く（teacher_leg_post基準）
                    leg_u = teacher_leg_post
                    teach_pre_leg  = teach_pre[j, leg_u, :]
                    teach_post_leg = teach_post[j, leg_u, :]

                    root_ref_shift_cm = torch.linalg.vector_norm(teach_pre_leg - teach_post_leg, dim=-1) * 100.0  # (Bu,)

                    # quantize境界の影響（どれくらい補正されるか）
                    teach_post_q = quantize_xy(teach_post)                     # (Bu,4,2)
                    q_shift_cm = torch.linalg.vector_norm(teach_post_q[j,leg_u,:] - teach_post_leg, dim=-1) * 100.0

                    # --- 3) 入力破綻っぽい兆候 ---
                    rgb = rgb_post[ids_bad].float()
                    rgb_mean = rgb.mean(dim=(1,2,3))
                    rgb_std  = rgb.std(dim=(1,2,3))

                    # hist の最後2step差分（ゼロ/異常に大きいを検知）
                    rseq = root_seq_post_all[ids_bad]  # (Bu,T,13)
                    fseq = foot_seq_post_all[ids_bad]  # (Bu,T,4,3)
                    cseq = cont_seq_post_all[ids_bad]  # (Bu,T,4)

                    root_d = torch.linalg.vector_norm(rseq[:,-1,:3] - rseq[:,-2,:3], dim=-1)  # (Bu,)
                    foot_d = torch.linalg.vector_norm(fseq[:,-1,:,:2] - fseq[:,-2,:,:2], dim=(-1,-2))  # (Bu,) ざっくり
                    cont_sum = cseq.sum(dim=(1,2))  # (Bu,) 0 なら全部0が多い等

                    print("==== STUDENT-TEACH OUTLIER ====")
                    for k in range(min(Bu_bad, 8)):
                        eid = int(ids_bad[k].item())
                        print(
                            f"[eid={eid}] err={err_cm[bad_mask][k].item():.2f}cm "
                            f"teacher_leg={int(teacher_leg_post[k])} leg_diff={int(leg_diff[k])} match={bool(match_leg[k].item())} "
                            f"root_ref_shift={root_ref_shift_cm[k].item():.2f}cm q_shift={q_shift_cm[k].item():.2f}cm "
                            f"rgb_mean={rgb_mean[k].item():.1f} rgb_std={rgb_std[k].item():.1f} "
                            f"root_d={root_d[k].item()*100:.2f}cm foot_d={foot_d[k].item()*100:.2f}cm cont_sum={int(cont_sum[k].item())}"
                        )




                    


                    leg = teacher_leg.long()                  # (Bu,)
                    eid = eval_ids.long()                     # (Bu,)
                    pred = pred_xy_teacher                    # (Bu,2)  ※あなたのEVALで使ってるpred

                    prev = last_pred_xy[eid, leg]             # (Bu,2)
                    had  = last_pred_valid[eid, leg]          # (Bu,)
                    jump_cm = torch.linalg.vector_norm(pred - prev, dim=-1) * 100.0
                    jump_cm = torch.where(had, jump_cm, torch.full_like(jump_cm, -1.0))

                    leg_age = torch.where(
                        had,
                        (global_step - last_pred_step[eid, leg]).to(torch.float32),
                        torch.full_like(jump_cm, -1.0)
                    )



                    # bad subset
                    pred_bad = pred_xy_teacher[bad_mask]          # (Bu_bad,2)
                    prev_bad = prev_xy[bad_mask]                  # (Bu_bad,2)  ※EVALで使ったprev

                    # 1) cmd_pre と cmd_post それぞれの teacher 点（同じrootで yawbase化）
                    teach_cmdpre = base_cmd_to_yawbase(cmd_pre_u,  root_post_u, yaw_quat, quat_apply)[..., :2]   # (Bu,4,2)
                    teach_cmdpost = base_cmd_to_yawbase(cmd_post_u, root_post_u, yaw_quat, quat_apply)[..., :2]  # (Bu,4,2)

                    teach_cmdpre_q  = quantize_xy(teach_cmdpre)
                    teach_cmdpost_q = quantize_xy(teach_cmdpost)

                    leg_u = teacher_leg_post  # (Bu_bad,)

                    teach_pre_leg_xy  = teach_cmdpre_q[j,  leg_u, :]   # (Bu_bad,2)
                    teach_post_leg_xy = teach_cmdpost_q[j, leg_u, :]   # (Bu_bad,2)

                    # 2) “cmdジャンプ量”と、“predがpre/postどっちに近いか”
                    cmd_jump_cm   = torch.linalg.vector_norm(teach_post_leg_xy - teach_pre_leg_xy, dim=-1) * 100.0
                    err_pre_cm    = torch.linalg.vector_norm(pred_bad - teach_pre_leg_xy,  dim=-1) * 100.0
                    err_post_cm   = torch.linalg.vector_norm(pred_bad - teach_post_leg_xy, dim=-1) * 100.0
                    pred_prev_cm  = torch.linalg.vector_norm(pred_bad - prev_bad,           dim=-1) * 100.0

                    print("[EDGE-ALIGN] cmd_jump_cm=", cmd_jump_cm[:8].tolist())
                    print("[EDGE-ALIGN] err_pre_cm=", err_pre_cm[:8].tolist(),
                        " err_post_cm=", err_post_cm[:8].tolist(),
                        " pred_prev_cm=", pred_prev_cm[:8].tolist())







            


           

            # -----------------------------
            # (9) reset対応（あなたのままでOK）
            # -----------------------------
            if done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                obs = _unwrap_reset_out(raw.reset(env_ids=reset_ids))

                root_r = robot.data.root_state_w[:, :13].to(device).float()
                foot_r = get_foot_pos_yawbase(raw, foot_body_ids).to(device).float()
                cont_r = torch.zeros((B,4), device=device, dtype=torch.uint8)

                hist_fill(reset_ids.to(device), root_r, foot_r, cont_r)


                # foot_b = get_foot_pos_yawbase(raw, foot_body_ids).to(device).float()             # (B,4,3) 取れないなら自前で
                # student_cmd_b[reset_ids] = foot_b[reset_ids] 

                feet_b_full = get_foot_pos_basefull(raw, foot_body_ids)  # base(full)
                student_cmd_b[reset_ids] = feet_b_full[reset_ids]


                if Teacher_mode == False:
                    feet_b_full = get_foot_pos_basefull(raw, foot_body_ids)   # (B,4,3) base(full)
                    student_cmd_b[reset_ids] = feet_b_full[reset_ids]


                hold_t[reset_ids] = 0.0
                bad_t[reset_ids]  = 0.0
                armed_prev[reset_ids] = False
                contact_state[reset_ids] = False
                near_state[reset_ids]    = False
                swing_pos[reset_ids] = 0
                cooldown_t[reset_ids] = COOLDOWN_S



                last_pred_valid[reset_ids] = False
                last_pred_step[reset_ids] = -1


                prev_xy_mem[reset_ids]   = foot_r[reset_ids, :, :2]
                prev_xy_valid[reset_ids] = True


            cmd_ver_prev_end = cmd_term.cmd_version.clone()

            global_step += 1
        


            

        

# raw.close()






    # with torch.inference_mode():
    #     while simulation_app.is_running():
    #         curr_phase = cmd_term.phase
    #         changed = curr_phase != prev_phase
    #         env_ids = changed.nonzero(as_tuple=False).squeeze(-1)


    #         # if env_ids.numel() > 0:

    #         #     rgb = get_front_rgb(raw, camera_name="camera").to(device)
    #         #     rgb_u = rgb[env_ids]
    #         #     bev = rgb_to_bev_in(rgb_u, bev_grid_in)  # (Bu,3,128,64)
    #         #     foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)[env_ids]  # (Bu,4,3)
    #         #     root = get_root_vec(raw)[env_ids]                               # (Bu,13)




    #         #     foot_logits, _ = student_model(bev, foot_pos_b, root)  # (Bu,4,K)

    #         #     leg_ids = (curr_phase[env_ids] % 4).long()        # (Bu,)
    #         #     Bu, _, K = foot_logits.shape
    #         #     logits_leg = foot_logits[torch.arange(Bu, device=rgb_u.device), leg_ids, :]  # (Bu,K)
    #         #     idx = torch.argmax(logits_leg, dim=-1)                                        # (Bu,)
    #         #     xy = index_to_xy_round(idx, Hb_out, Wb_out, L, W, res)  # (Bu,2)
    #         #     z = torch.full((Bu,1), z_const, device=xy.device)
    #         #     tgt_b = torch.cat([xy, z], dim=-1)                      # (Bu,3)

    #         #     ua = upper_action.view(B, 4, 3)
    #         #     ua[env_ids, leg_ids, :] = tgt_b                         # ★ここが hold の肝
    #         #     prev_phase[env_ids] = curr_phase[env_ids]


    #         # safe = torch.tensor([[ 0.25, 0.15, -0.35,
    #         #         0.25,-0.15, -0.35,
    #         #         -0.20, 0.15, -0.35,
    #         #         -0.20,-0.15, -0.35]], device=raw.device).repeat(B,1)



    #         # # ---- step前の観測を保存（学習時と合わせたいのでここが重要）----
    #         # rgb_prev  = get_front_rgb(base_env)                       # (B,64,64,3)
    #         # foot_prev = get_foot_pos_yawbase(base_env, foot_body_ids) # (B,4,3)
    #         # root_prev = robot.data.root_state_w[:, :13]               # (B,13)

    #         cmd_now_b = cmd_term.command.view(B,4,3)         # teacher（base）
    #         upper_action = cmd_now_b.reshape(B,12).clone()   # ★これで teacher rollout
    #         # obs, rew, terminated, truncated, info = raw.step(upper_action)




    #         cmd_pre_b = cmd_term.command.view(B,4,3).clone()
    #         rgb_pre  = get_front_rgb(raw).clone()
    #         foot_pre = get_foot_pos_yawbase(raw, foot_body_ids).clone()
    #         root_pre = robot.data.root_state_w[:, :13].clone()


    #         root_seq_pre, foot_seq_pre, cont_seq_pre = hist_gather(env_ids)  # ※この行を step 前に移動するのが理想


    #         # ===== step前スナップショット =====
    #         cmd_ver_pre = cmd_term.cmd_version.clone()
    #         cmd_pre_b   = cmd_term.command.view(B,4,3).clone()

    #         rgb_pre   = get_front_rgb(raw).clone()
    #         root_pre  = robot.data.root_state_w[:, :13].clone()
    #         foot_pre  = get_foot_pos_yawbase(raw, foot_body_ids).clone()
    #         # （contactは必要なら pre でも取る）
    #         # cont_pre  = get_contact_4(...)

    #         # 「step開始時点ですでにcmdが更新済み」か？
    #         pre_edge = (cmd_ver_pre != cmd_ver_prev_end)





    #         # env.step には常に (B,12) を渡す（中身は保持される）
    #         obs, rew, terminated, truncated, info = raw.step(upper_action)




    #         # post状態を取得
    #         # rgb_post  = get_front_rgb(base_env)    
    #         rgb_post   = cam.data.output["rgb"].clone()
    #                # (B,64,64,3) or (B,3,H,W) ※あなたの関数に合わせる
    #         foot_post = get_foot_pos_yawbase(base_env, foot_body_ids) # (B,4,3)
    #         root_post = robot.data.root_state_w[:, :13]               # (B,13)

    #         # deviceへ
    #         foot_post = foot_post.to(device).float()
    #         root_post = root_post.to(device).float()

    #         # 履歴にpush
    #         # hist_write(root_post, foot_post)
    #         contacts = (Fz > 8).to(torch.uint8).to(device)

    #         hist_write(root_post, foot_post, contacts)

    #         root_seq_post, foot_seq_post, cont_seq_post = hist_gather(env_ids)




    #         done = terminated | truncated



    #         # ===== step後スナップショット =====
    #         cmd_ver_post = cmd_term.cmd_version.clone()
    #         cmd_post_b   = cmd_term.command.view(B,4,3).clone()

    #         rgb_post   = get_front_rgb(raw).clone()
    #         root_post  = robot.data.root_state_w[:, :13].clone()
    #         foot_post  = get_foot_pos_yawbase(raw, foot_body_ids).clone()
    #         cont_post  = get_contact_4(raw, "contact_forces", sensor_cols, thresh=8.0, device=raw.device)

    #         # 履歴は「post」を積む（収集と同じ）
    #         hist_write(root_post, foot_post, cont_post)

    #         # 「stepの中（末尾）でcmdが更新された」か？
    #         post_edge = (cmd_ver_post != cmd_ver_pre)

    #         # ===== ここから評価する env_ids を作る =====
    #         env_ids_pre  = (pre_edge  & (~done)).nonzero(as_tuple=False).squeeze(-1)
    #         env_ids_post = (post_edge & (~done)).nonzero(as_tuple=False).squeeze(-1)

    #         print(f"[EDGE] pre_edge={int(pre_edge.sum())} post_edge={int(post_edge.sum())} done={int(done.sum())}")








    #         # ---- (A) step後に cmd update を検出（収集と同じ）----
    #         cmd_ver_now = cmd_term.cmd_version
    #         cmd_updated = (cmd_ver_now != cmd_ver_prev) & (~done)   # (B,)
    #         env_ids = cmd_updated.nonzero(as_tuple=False).squeeze(-1)






    #         # step後に取る
    #         cmd_post_b = cmd_term.command.view(B,4,3).clone()
    #         rgb_post  = get_front_rgb(raw).clone()
    #         foot_post = get_foot_pos_yawbase(raw, foot_body_ids).clone()
    #         root_post = robot.data.root_state_w[:, :13].clone()

    #         # cmd_updated を post で判定（あなたの今の方式）
    #         cmd_ver_now = cmd_term.cmd_version
    #         cmd_updated = (cmd_ver_now != cmd_ver_prev) & (~done)
    #         env_ids = cmd_updated.nonzero(as_tuple=False).squeeze(-1)





    #         # # ---- (B) step後観測を取得（ここが重要：post-step）----
    #         # rgb_post  = get_front_rgb(raw, camera_name="camera")              # (B,64,64,3) uint8
    #         # foot_post = get_foot_pos_yawbase(raw, foot_body_ids)              # (B,4,3)
    #         # root_post = get_root_vec(raw)                                     # (B,13) wxyz想定

    #         # assert rgb_post.dtype == torch.uint8 and rgb_post.ndim == 4 and rgb_post.shape[-1] == 3
    #         # assert rgb_post.shape[1:3] == (64,64), rgb_post.shape  # 学習と同じなら


    #         # ---- (C) 更新が起きたenvだけ評価 ----
    #         if env_ids.numel() > 0:
    #             Bu = env_ids.numel()
                


    #             # 教師：学習と同じソース（cmd_term.command）
    #             # cmd_now = cmd_term.command.view(B, 4, 3)
    #             cmd_now_b = cmd_term.command.view(B,4,3)
    #             cmd_now = base_cmd_to_yawbase(cmd_now_b, root_post, yaw_quat, quat_apply)
    #             teacher_xy = cmd_now[env_ids, :, :2]  # (Bu,4,2) ※量子化しない（学習も連続xy→index化）



    #             # root = get_root_vec(raw)  # (B,13) pos + quat[wxyz]
    #             # base_p = root[:,0:3]
    #             # q_full = root[:,3:7]
    #             # q_yaw  = yaw_quat(q_full)

    #             # # debug goal world（あなたのtermが保存してる）
    #             # LEG_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    #             # goal_w_dbg = torch.stack([cmd_term._debug_goal_w_cache[leg] for leg in LEG_ORDER], dim=1)  # (B,4,3)

    #             # # teacher yaw-base をあなたの変換関数で作る（ここはあなたの実装に合わせる）
    #             # teacher_y = base_cmd_to_yawbase(cmd_now_b, root, yaw_quat, quat_apply)  # (B,4,3)

    #             # # yaw-base -> world に戻す（検証用）
    #             # goal_w_from_yaw = base_p[:,None,:] + quat_apply(q_yaw[:,None,:], teacher_y)

    #             # err_cm = (goal_w_from_yaw - goal_w_dbg).norm(dim=-1).mean().item() * 100
    #             # print("[CHECK] yaw-teacher back-to-world vs debug goal =", err_cm, "cm")





    #             # moved脚：収集と同じ定義に寄せる（前回更新時との差）
    #             moved = (torch.norm(cmd_now - cmd_prev_update, dim=-1) > 2e-2)  # (B,4)
    #             moved_u = moved[env_ids]                                        # (Bu,4)

    #             # student（post-step観測で推論）
    #             rgb_u  = rgb_post[env_ids].to(device)
    #             bev    = rgb_to_bev_in(rgb_u, bev_grid_in)                      # (Bu,3,128,64)
    #             foot_u = foot_post[env_ids].to(device)
    #             root_u = root_post[env_ids].to(device)


    #             # 履歴窓を集める
    #             # root_seq, foot_seq = hist_gather(env_ids.to(device))  # (Bu,T,13), (Bu,T,4,3)
    #             root_seq, foot_seq, contact_seq = hist_gather(env_ids.to(device))      # (Bu,T,13),(Bu,T,4,3),(Bu,T,4)


    #             # root_featへ変換
    #             root_feat = root13_hist_to_feat(root_seq, keep_z=True)  # (Bu,T,9)
    #             foot_flat = foot_seq.reshape(Bu, T, -1)                 # (Bu,T,12)

    #             # proprio_hist = torch.cat([foot_flat, root_feat], dim=-1) # (Bu,T,21)

    #             contact_f = contact_seq.float()                                         # (Bu,T,4)

    #             proprio_hist = torch.cat([foot_flat, root_feat, contact_f], dim=-1)     # (Bu,T,25)

    #             proprio_flat = proprio_hist.reshape(Bu, -1)              # (Bu,T*21)
                

    #             # モデル推論（重要：proprio=proprio_flat を渡す）
    #             foot_logits, _ = student_model(bev, proprio=proprio_flat)  # (Bu,4,K)

    #             # foot_logits, _ = student_model(bev, foot_pos_b=foot_u, root=root_u)  # (Bu,4,K)






    #             # pred -> xy（まずは argmax で学習指標と揃えるのが安全）
    #             pred_idx = foot_logits.argmax(dim=-1)  # (Bu,4)
    #             # pred_xy = index_to_xy(
    #             #     pred_idx.flatten(), Hb_out, Wb_out, L, W, res
    #             # ).view(Bu, 4, 2)  # (Bu,4,2)
    #             # pred_xy = index_to_xy(
    #             #     pred_idx.flatten(), Hb_out, Wb_out, L, W, res
    #             # ).view(Bu, 4, 2)  # (Bu,4,2)
    #             x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
    #             y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)
    #             pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb_out) 

    #             # err_leg_cm = (pred_xy - teacher_xy).norm(dim=-1) * 100.0  # (Bu,4)

    #             # # 更新脚だけ平均（movedでマスク）
    #             # # err_changed = err_leg_cm[moved_u]
    #             # # mean_changed = err_changed.mean().item() if err_changed.numel() > 0 else float("nan")
    #             # mean_all4 = err_leg_cm.mean().item()
    #             # # n_ch = moved_u.sum(dim=-1).float().mean().item()

    #             # # print(f"[EVAL] Bu={Bu} mean_changed={mean_changed:.2f}cm mean_all4={mean_all4:.2f}cm avg_moved_legs={n_ch:.2f}")
    #             # print(f"[EVAL] Bu={Bu}  mean_all4={mean_all4:.2f}cm")


    #             # ★teacher も量子化してから戻す（これが学習と一致）
    #             teacher_idx = xy_to_index(teacher_xy, Hb_out, Wb_out, L, W, res).long()   # (Bu,4)
    #             teacher_xy_q = index_to_xy(teacher_idx, x_coords, y_coords, Wb_out)       # (Bu,4,2)

    #             err_leg_cm = (pred_xy - teacher_xy_q).norm(dim=-1) * 100.0
    #             mean_all4 = err_leg_cm.mean().item()
    #             print(f"[EVAL] Bu={Bu} mean_all4={mean_all4:.2f}cm")






    #             def logits_to_xy_argmax(foot_logits):
    #                 # foot_logits: (Bu,4,K)
    #                 pred_idx = foot_logits.argmax(dim=-1)  # (Bu,4)
    #                 pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)  # (Bu,4,2)
    #                 return pred_xy







    #             teach_pre  = base_cmd_to_yawbase(cmd_pre_b,  root_pre,  yaw_quat, quat_apply)[env_ids, :, :2]
    #             teach_post = base_cmd_to_yawbase(cmd_post_b, root_post, yaw_quat, quat_apply)[env_ids, :, :2]

    #             def run_pred(rgb_u8, root_seq, foot_seq, cont_seq):

    #                 Bu = root_seq.shape[0]
    #                 if Bu == 0:
    #                     # 何も予測しない（呼び出し側でスキップされる想定）
    #                     return None
                    
    #                 rgb_u = rgb_u8[env_ids].to(device)                           # (Bu,64,64,3) uint8
    #                 bev   = rgb_to_bev_in(rgb_u, bev_grid_in)                    # (Bu,3,128,64)
    #                 root_seq = root_seq.to(device)
    #                 foot_seq = foot_seq.to(device)
    #                 cont_seq = cont_seq.to(device) if cont_seq is not None else None

    #                 proprio_flat = build_proprio_flat(root_seq, foot_seq, cont_seq)  # (Bu,T*P)
    #                 foot_logits, _ = student_model(bev, proprio=proprio_flat)        # ★ここが要望の形
    #                 return logits_to_xy_argmax(foot_logits).detach().cpu()           # (Bu,4,2)

    #             p_pre  = run_pred(rgb_pre,  root_seq_pre,  foot_seq_pre,  cont_seq_pre)
    #             p_post = run_pred(rgb_post, root_seq_post, foot_seq_post, cont_seq_post)


    #             if (p_pre is not None) or (p_post is not None):
                   

    #                 # 4通り誤差
    #                 e_pre_pre   = ((p_pre  - teach_pre.cpu()).norm(dim=-1).mean()*100).item()
    #                 e_pre_post  = ((p_pre  - teach_post.cpu()).norm(dim=-1).mean()*100).item()
    #                 e_post_pre  = ((p_post - teach_pre.cpu()).norm(dim=-1).mean()*100).item()
    #                 e_post_post = ((p_post - teach_post.cpu()).norm(dim=-1).mean()*100).item()

    #                 print(f"[TIME+HIST] e(preObs,preCmd)={e_pre_pre:.2f}  e(preObs,postCmd)={e_pre_post:.2f}  "
    #                     f"e(postObs,preCmd)={e_post_pre:.2f}  e(postObs,postCmd)={e_post_post:.2f} cm")


    #             # ix = teacher_idx // Wb_out
    #             # iy = teacher_idx %  Wb_out
    #             # on_edge = (ix == 0) | (ix == Hb_out-1) | (iy == 0) | (iy == Wb_out-1)
    #             # edge_rate = on_edge.float().mean().item()

    #             # if mean_all4 > 6.0:
    #             #     mx = teacher_xy[...,0].abs().max().item()
    #             #     my = teacher_xy[...,1].abs().max().item()
    #             #     print(f"  [DBG] edge_rate={edge_rate:.2f}  max|x|={mx:.3f}  max|y|={my:.3f}")


    #             # if mean_all4 > 6.0:
    #             #     # 4通りの circular shift を試す
    #             #     best = 1e9
    #             #     best_s = None
    #             #     for s in range(4):
    #             #         e = (pred_xy.roll(shifts=s, dims=1) - teacher_xy_q).norm(dim=-1).mean().item() * 100.0
    #             #         if e < best:
    #             #             best = e
    #             #             best_s = s
    #             #     print(f"  [DBG] best_shift={best_s} best_err={best:.2f}cm (raw={mean_all4:.2f}cm)")





    #             # # teacher_xy: (Bu,4,2) 連続 yaw-base になっている前提
    #             # teacher_idx = xy_to_index(teacher_xy, Hb_out, Wb_out, L, W, res).long()  # (Bu,4)

    #             # # A) 学習時に使ってた座標復元（linspace）
    #             # teacher_xy_A = index_to_xy(teacher_idx, x_coords, y_coords, Wb_out)      # (Bu,4,2)

    #             # # B) 横評価で使ってる座標復元（round版）
    #             # teacher_xy_B = index_to_xy_round(teacher_idx.flatten(), Hb_out, Wb_out, L, W, res).view(Bu,4,2)

    #             # print("max |A-B| cm =", ((teacher_xy_A - teacher_xy_B).abs().max() * 100).item())
    #             # print("mean quant err A cm =", ((teacher_xy_A - teacher_xy).norm(dim=-1).mean() * 100).item())
    #             # print("mean quant err B cm =", ((teacher_xy_B - teacher_xy).norm(dim=-1).mean() * 100).item())


    #             # バッファ更新（収集と同じ）
    #             cmd_ver_prev[env_ids] = cmd_ver_now[env_ids]
    #             cmd_prev_update[env_ids] = cmd_now[env_ids].detach()

    #         # ---- reset対応（reset env はトリガ誤検知しないよう追従）----
    #         if done.any():
    #             reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
    #             obs, _ = raw.reset(env_ids=reset_ids)

    #             # hold初期化（あなたの元コード通り）
    #             foot_b43 = get_foot_pos_yawbase(raw, foot_body_ids)
    #             upper_action[reset_ids] = foot_b43[reset_ids].reshape(-1, 12)

    #             # cmd_version追従
    #             cmd_ver_prev[reset_ids] = cmd_term.cmd_version[reset_ids]
    #             # cmd_prev_update[reset_ids] = cmd_term.command.view(B,4,3)[reset_ids].detach()
    #             cmd_b = cmd_term.command.view(B,4,3)
    #             cmd_y = base_cmd_to_yawbase(cmd_b, root_post, yaw_quat, quat_apply)  # root_postはpost-stepで取ったやつ
    #             cmd_prev_update[reset_ids] = cmd_y[reset_ids].detach()



    #             # reset後の状態で履歴を同値埋め
    #             root13_r = robot.data.root_state_w[:, :13].to(device).float()
    #             foot43_r = get_foot_pos_yawbase(raw, foot_body_ids).to(device).float()
    #             # hist_fill(reset_ids.to(device), root13_r, foot43_r)
    #             F = contact_sensor.data.net_forces_w[:, sensor_cols, :]
    #             Fz = F[..., 2].abs()
    #             contact_r = (Fz >8 ).to(torch.uint8)

    #             hist_fill(reset_ids.to(device), root13_r, foot43_r, contact_r)


          

    #         # obs, rew, terminated, truncated, info = raw.step(safe)

    #         # cmd = raw.command_manager.get_command("step_fr_to_block")
    #         # print("max|cmd-safe| =", (cmd - safe).abs().max().item())


    #         # cmd = raw.command_manager.get_command("step_fr_to_block")  # (B,12) のはず
    #         # print("max|cmd-safe| =", (cmd - safe).abs().max().item())
    #         # print("cmd[0] =", cmd[0].tolist())
    #         # print("safe[0]=", safe[0].tolist())

    #         # foot_b43 = get_foot_pos_yawbase(raw, foot_body_ids)           # (B,4,3)
    #         # approx = foot_b43.view(B,12) + safe
    #         # cmd = raw.command_manager.get_command("step_fr_to_block")
    #         # print("max|cmd-(foot+safe)| =", (cmd - approx).abs().max().item())


    #         # if done.any():
    #         #     reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
    #         #     obs, _ = raw.reset(env_ids=reset_ids)

    #         #     # reset env は hold を初期化（事故防止）
    #         #     foot_b43 = get_foot_pos_yawbase(raw, foot_body_ids)
    #         #     upper_action[reset_ids] = foot_b43[reset_ids].reshape(-1, 12)
    #         #     prev_phase[reset_ids] = cmd_term.phase[reset_ids]

    #     # simulation_app.close()



        raw.close()

if __name__ == "__main__":
    main()

    simulation_app.close()

