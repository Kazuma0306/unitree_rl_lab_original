import argparse, os, math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


import csv
import matplotlib.pyplot as plt



# -------------------------
# Utils: coords <-> grid
# -------------------------
def make_bev_coords(Hb, Wb, L, W, device):
    x = torch.linspace(-L/2, L/2, Hb, device=device)  # (Hb,)
    y = torch.linspace(-W/2, W/2, Wb, device=device)  # (Wb,)
    return x, y

def xy_to_index(xy_b42, Hb, Wb, L, W, res):
    # xy_b42: (B,4,2) in meters, base frame
    x = xy_b42[..., 0]
    y = xy_b42[..., 1]
    ix = torch.round((x + L/2) / res).long().clamp(0, Hb - 1)
    iy = torch.round((y + W/2) / res).long().clamp(0, Wb - 1)
    idx = ix * Wb + iy  # (B,4)
    return idx

def index_to_xy(idx_b4, x_coords, y_coords, Wb):
    ix = idx_b4 // Wb
    iy = idx_b4 % Wb
    x = x_coords[ix]
    y = y_coords[iy]
    return torch.stack([x, y], dim=-1)  # (B,4,2)



def yaw_from_quat_wxyz(q):
    # q: (...,4) wxyz
    w, x, y, z = q[...,0], q[...,1], q[...,2], q[...,3]
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def world_xy_to_yawbase_xy(foot_xy_w, root_state_w):
    # foot_xy_w: (B,4,2) world
    # root_state_w: (B,13)
    base_xy = root_state_w[:, None, 0:2]     # (B,1,2)
    q = root_state_w[:, 3:7]                 # (B,4) wxyz
    yaw = yaw_from_quat_wxyz(q)              # (B,)

    rel = foot_xy_w - base_xy                # (B,4,2)

    c = torch.cos(-yaw)[:, None]             # (B,1)
    s = torch.sin(-yaw)[:, None]

    x = rel[...,0]
    y = rel[...,1]
    xb = c*x - s*y
    yb = s*x + c*y
    return torch.stack([xb, yb], dim=-1)     # (B,4,2)




############################## BEV Transformation ##########################

import torch
import torch.nn.functional as F

def quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """q: (...,4) wxyz -> R: (...,3,3)"""
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

def yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """q: (...,4) wxyz -> yaw (rad)"""
    w, x, y, z = q.unbind(-1)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    return torch.atan2(siny, cosy)

def rotz(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw); s = torch.sin(yaw)
    R = torch.zeros(yaw.shape + (3,3), dtype=yaw.dtype, device=yaw.device)
    R[...,0,0]=c; R[...,0,1]=-s
    R[...,1,0]=s; R[...,1,1]=c
    R[...,2,2]=1
    return R

def intrinsics_from_pinhole(width:int, height:int, focal_length:float, horizontal_aperture:float,
                            cx=None, cy=None, device="cpu", dtype=torch.float32):
    """
    IsaacのPinholeCameraCfgのパラメータ相当の近似:
      fx = width  * focal_length / horizontal_aperture
      fy = height * focal_length / vertical_aperture
    vertical_aperture未指定なら横×(H/W) と仮定（正方形なら同じ）
    """
    vertical_aperture = horizontal_aperture * (height / width)
    fx = width  * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    if cx is None: cx = (width - 1) * 0.5
    if cy is None: cy = (height - 1) * 0.5
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    return K

@torch.no_grad()
def warp_rgb_to_bev_ipm(
    rgb_bchw: torch.Tensor,        # (B,3,Himg,Wimg) float [0,1]
    base_pos_w: torch.Tensor,      # (B,3)
    base_quat_wxyz: torch.Tensor,  # (B,4)
    K: torch.Tensor,              # (3,3) or (B,3,3)
    t_bc: torch.Tensor,           # (3,) base frame
    q_bc_wxyz: torch.Tensor,      # (4,) wxyz : camera pose relative to base
    Hb:int, Wb:int,
    L:float, W:float,
    z_plane: float = 0.0,
    use_full_base_for_cam: bool = True,   # True推奨: カメラ姿勢はroll/pitch含める
    bc_is_cam_to_base: bool = True,       # ここが合わないときは False を試す
) -> torch.Tensor:
    """
    BEVセル中心 (x_b,y_b,z=z_plane in world) -> 画像に投影 -> grid_sample
    - BEV格子の向き: base yaw のみで (x,y) を worldへ回す（raycaster attach_yaw_only 相当）
    - カメラ姿勢: base姿勢(フル) × (固定オフセット) で合成
    """
    B, _, Himg, Wimg = rgb_bchw.shape
    dev = rgb_bchw.device
    dtype = rgb_bchw.dtype

    # Kをバッチへ
    if K.ndim == 2:
        Kb = K[None].expand(B, -1, -1).to(dev, dtype)
    else:
        Kb = K.to(dev, dtype)

    # BEV grid（base-yaw frame）
    xs = torch.linspace(-L/2, L/2, Hb, device=dev, dtype=dtype)
    ys = torch.linspace(-W/2, W/2, Wb, device=dev, dtype=dtype)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")  # (Hb,Wb)
    pw = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)  # (Hb,Wb,3)

    # base yaw で worldへ回す
    yaw = yaw_from_quat_wxyz(base_quat_wxyz)        # (B,)
    R_yaw = rotz(yaw)                               # (B,3,3)

    p0 = base_pos_w.clone()
    p0[:, 2] = z_plane
    Xw = (R_yaw[:,None,None,:,:] @ pw[None,:,:,:,None]).squeeze(-1) + p0[:,None,None,:]  # (B,Hb,Wb,3)

    # base->world (フル) 回転
    if use_full_base_for_cam:
        R_wb = quat_to_rotmat_wxyz(base_quat_wxyz)  # (B,3,3)
    else:
        # yaw-onlyでカメラも回す（通常は非推奨）
        R_wb = R_yaw

    # base<->cam 固定変換
    R_bc = quat_to_rotmat_wxyz(q_bc_wxyz[None].to(dev, dtype)).expand(B, -1, -1)  # (B,3,3)
    t_bc_ = t_bc.to(dev, dtype)[None].expand(B, -1)                               # (B,3)

    # 「T_bc が cam->base」か「base->cam」かで切替
    if bc_is_cam_to_base:
        # T_wc = T_wb * T_bc  （cam->world）
        R_wc = R_wb @ R_bc
        t_wc = base_pos_w + (R_wb @ t_bc_[..., None]).squeeze(-1)
    else:
        # T_bc を base->cam と解釈：T_wc = T_wb * T_bc(base->cam) なので cam->worldは逆になる
        # ここでは cam->world を作りたいので、base->cam を反転して cam->base を得る
        R_cb = R_bc
        t_cb = t_bc_
        R_bc_inv = R_cb.transpose(-1, -2)
        t_bc_inv = -(R_bc_inv @ t_cb[..., None]).squeeze(-1)
        R_wc = R_wb @ R_bc_inv
        t_wc = base_pos_w + (R_wb @ t_bc_inv[..., None]).squeeze(-1)

    # world->cam
    R_cw = R_wc.transpose(-1, -2)
    t_cw = -(R_cw @ t_wc[..., None]).squeeze(-1)  # (B,3)

    # project (ROS optical: +Z forward を想定して X/Z, Y/Z)
    Xc = (R_cw[:,None,None,:,:] @ Xw[:,:,:,None]).squeeze(-1) + t_cw[:,None,None,:]  # (B,Hb,Wb,3)
    Z = Xc[..., 2].clamp(min=1e-6)

    fx, fy = Kb[:,0,0], Kb[:,1,1]
    cx, cy = Kb[:,0,2], Kb[:,1,2]
    u = fx[:,None,None] * (Xc[...,0] / Z) + cx[:,None,None]
    v = fy[:,None,None] * (Xc[...,1] / Z) + cy[:,None,None]

    # grid_sample の正規化
    grid_x = 2.0 * (u / (Wimg - 1.0)) - 1.0
    grid_y = 2.0 * (v / (Himg - 1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,Hb,Wb,2)

    bev = F.grid_sample(rgb_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return bev  # (B,3,Hb,Wb)













# -------------------------
# Dataset
# -------------------------
class TeacherH5Dataset(Dataset):
    def __init__(self, path: str, use_trav: bool = True):
        self.f = h5py.File(path, "r")
        self.rgb = self.f["rgb"]            # (N,H,W,3) uint8
        self.foot_xy = self.f["foot_xy"]    # (N,4,2) float32
        self.use_trav = use_trav
        self.trav = self.f["trav"] if use_trav and "trav" in self.f else None

        # Optional proprio (if you saved them)
        self.foot_pos_b = self.f["foot_pos_b"] if "foot_pos_b" in self.f else None
        self.root = self.f["root_state_w"] if "root_state_w" in self.f else None

        # BEV meta from attrs (recommended)
        self.res = float(self.f.attrs["ray_resolution"]) if "ray_resolution" in self.f.attrs else None
        if "ray_size_xy" in self.f.attrs:
            sz = np.array(self.f.attrs["ray_size_xy"], dtype=np.float32).reshape(-1)
            self.L = float(sz[0]); self.W = float(sz[1])
        else:
            self.L = None; self.W = None

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, i):
        rgb = self.rgb[i]  # uint8 HWC
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        foot_xy = torch.from_numpy(self.foot_xy[i]).float()           # (4,2)

        sample = {"rgb": rgb, "foot_xy": foot_xy}

        if self.trav is not None:
            trav = torch.from_numpy(self.trav[i]).float()            # (Hb,Wb) 0/1
            sample["trav"] = trav

        # optional: current foot positions & root state (can help)
        if self.foot_pos_b is not None:
            fp = torch.from_numpy(self.foot_pos_b[i]).float()        # (4,3)
            sample["foot_pos_b"] = fp
        if self.root is not None:
            root = torch.from_numpy(self.root[i]).float()            # (13,)
            sample["root_state_w"] = root

        return sample













# -------------------------
# Model (baseline)
# -------------------------
# class StudentNet(nn.Module):
#     """
#     RGB -> (optional proprio) -> foot logits (4 x Hb*Wb) + optional trav logits (Hb*Wb)
#     """
#     def __init__(self, Hb, Wb, use_trav=True, use_proprio=True):
#         super().__init__()
#         self.Hb, self.Wb = Hb, Wb
#         self.use_trav = use_trav
#         self.use_proprio = use_proprio

#         # lightweight backbone: ResNet18
#         from torchvision.models import resnet18
#         m = resnet18(weights=None)
#         self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (B,512,1,1)
#         feat_dim = 512

#         prop_dim = 0
#         if use_proprio:
#             # foot_pos_b (4x3=12) + root_state_w (13) => 25 dims
#             prop_dim = 25
#             self.prop_mlp = nn.Sequential(
#                 nn.Linear(prop_dim, 128), nn.ReLU(),
#                 nn.Linear(128, 128), nn.ReLU(),
#             )
#             fuse_dim = feat_dim + 128
#         else:
#             fuse_dim = feat_dim

#         self.fuse = nn.Sequential(
#             nn.Linear(fuse_dim, 512), nn.ReLU(),
#             nn.Linear(512, 512), nn.ReLU(),
#         )

#         self.foot_head = nn.Linear(512, 4 * Hb * Wb)
#         self.trav_head = nn.Linear(512, Hb * Wb) if use_trav else None

#     def forward(self, rgb, foot_pos_b=None, root=None):
#         x = self.backbone(rgb).flatten(1)  # (B,512)

#         if self.use_proprio:
#             assert foot_pos_b is not None and root is not None
#             prop = torch.cat([foot_pos_b.flatten(1), root], dim=1)  # (B,25)
#             p = self.prop_mlp(prop)
#             x = torch.cat([x, p], dim=1)

#         h = self.fuse(x)  # (B,512)
#         foot_logits = self.foot_head(h).view(-1, 4, self.Hb * self.Wb)  # (B,4,K)

#         trav_logits = None
#         if self.use_trav:
#             trav_logits = self.trav_head(h).view(-1, self.Hb, self.Wb)  # (B,Hb,Wb)

#         return foot_logits, trav_logits



import torch
import torch.nn as nn
import torch.nn.functional as F

# class StudentBEVNet(nn.Module):
#     """
#     BEV RGB -> foot logits map (B,4,Hb,Wb) + optional trav logits (B,Hb,Wb)
#     optional proprio: (foot_pos_b(4,3)+root(13)) -> embed -> broadcast concat
#     """
#     def __init__(self, Hb, Wb, use_trav=True, use_proprio=True, in_ch=3):
#         super().__init__()
#         self.Hb, self.Wb = Hb, Wb
#         self.use_trav = use_trav
#         self.use_proprio = use_proprio

#         # BEV backbone（軽量CNN）
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#         )

#         prop_ch = 0
#         if use_proprio:
#             prop_dim = 12 + 13  # foot_pos_b + root(13)
#             self.prop_mlp = nn.Sequential(
#                 nn.Linear(prop_dim, 128), nn.ReLU(),
#                 nn.Linear(128, 64), nn.ReLU(),
#             )
#             prop_ch = 64

#         self.fuse = nn.Sequential(
#             nn.Conv2d(64 + prop_ch, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
#         )

#         self.foot_head = nn.Conv2d(128, 4, 1)                 # (B,4,Hb,Wb)
#         self.trav_head = nn.Conv2d(128, 1, 1) if use_trav else None  # (B,1,Hb,Wb)

#     def forward(self, bev_rgb, foot_pos_b=None, root=None):
#         """
#         bev_rgb: (B,3,Hb,Wb)
#         """
#         x = self.stem(bev_rgb)  # (B,64,Hb,Wb)

#         if self.use_proprio:
#             assert foot_pos_b is not None and root is not None
#             prop = torch.cat([foot_pos_b.flatten(1), root], dim=1)  # (B,25)
#             p = self.prop_mlp(prop)                                 # (B,64)
#             p = p[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
#             x = torch.cat([x, p], dim=1)

#         x = self.fuse(x)

#         foot_logits_map = self.foot_head(x)  # (B,4,Hb,Wb)
#         foot_logits = foot_logits_map.flatten(2)  # (B,4,Hb*Wb)

#         trav_logits = None
#         if self.use_trav:
#             trav_logits = self.trav_head(x).squeeze(1)  # (B,Hb,Wb)

#         return foot_logits, trav_logits




class StudentNetInBEV(nn.Module):
    def __init__(self, Hb_out, Wb_out, use_trav=False, use_proprio=True):
        super().__init__()
        self.Hb_out, self.Wb_out = Hb_out, Wb_out
        self.use_trav = use_trav
        self.use_proprio = use_proprio

        from torchvision.models import resnet18
        m = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (B,512,1,1)
        feat_dim = 512

        if use_proprio:
            self.prop_mlp = nn.Sequential(
                nn.Linear(25, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            fuse_dim = feat_dim + 128
        else:
            fuse_dim = feat_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )

        K = Hb_out * Wb_out
        self.foot_head = nn.Linear(512, 4 * K)
        self.trav_head = nn.Linear(512, K) if use_trav else None

    def forward(self, bev, foot_pos_b=None, root=None):
        x = self.backbone(bev).flatten(1)  # (B,512)

        if self.use_proprio:
            assert foot_pos_b is not None and root is not None
            prop = torch.cat([foot_pos_b.flatten(1), root], dim=1)  # (B,25)
            p = self.prop_mlp(prop)
            x = torch.cat([x, p], dim=1)

        h = self.fuse(x)

        K = self.Hb_out * self.Wb_out
        foot_logits = self.foot_head(h).view(-1, 4, K)  # (B,4,K)

        trav_logits = None
        if self.use_trav:
            trav_logits = self.trav_head(h).view(-1, self.Hb_out, self.Wb_out)  # (B,Hb,Wb)

        return foot_logits, trav_logits








import h5py, numpy as np

path = "teacher_static.h5"
with h5py.File(path, "r") as f:
    print("attrs ray_size_xy:", f.attrs.get("ray_size_xy"))
    print("attrs ray_resolution:", f.attrs.get("ray_resolution"))
    for k in ["rgb","hits_z","trav","foot_xy","foot_pos_b","root_state_w"]:
        if k in f:
            print(k, f[k].shape, f[k].dtype)

    foot_xy = f["foot_xy"][:]
    print("foot_xy x min/max:", foot_xy[...,0].min(), foot_xy[...,0].max())
    print("foot_xy y min/max:", foot_xy[...,1].min(), foot_xy[...,1].max())



import h5py, numpy as np

path="teacher_static.h5"
with h5py.File(path,"r") as f:
    foot_xy = f["foot_xy"][:]          # (N,4,2)
    root = f["root_state_w"][:]        # (N,13)
    base_xy = root[:, None, 0:2]       # (N,1,2)

    rel = foot_xy - base_xy            # (N,4,2) もしfoot_xyがworldなら、ここがロボット周りの値になる

    print("rel x min/max:", rel[...,0].min(), rel[...,0].max())
    print("rel y min/max:", rel[...,1].min(), rel[...,1].max())

    # 参考：foot_pos_b（ベース座標の足）と近いか？
    fp = f["foot_pos_b"][:]            # (N,4,3)
    print("foot_pos_b xy min/max:",
          fp[...,0].min(), fp[...,0].max(),
          fp[...,1].min(), fp[...,1].max())











# ---- 新BEV入力のメタ ----
X_MIN_IN = 0.4
X_MAX_IN = 1.2
Y_WIDTH_IN = 0.8
HB_IN = 128
WB_IN = 64

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
q_bc_wxyz = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)

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

def build_bev_grid_in(device):
    # BEV grid points in base frame (Hb,Wb,3)
    xs = torch.linspace(X_MAX_IN, X_MIN_IN, HB_IN, device=device)
    ys = torch.linspace(Y_WIDTH_IN/2, -Y_WIDTH_IN/2, WB_IN, device=device)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    zg = torch.zeros_like(xg)
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



import torch
import torch.nn.functional as F

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




import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def _idx_to_ij(idx: int, Wb: int):
    return idx // Wb, idx % Wb

@torch.no_grad()
def show_train_viz_block(
    rgb_bhwc_u8: torch.Tensor,      # (B,64,64,3) uint8
    bev_bchw: torch.Tensor,         # (B,3,128,64) float
    foot_logits_b4k: torch.Tensor,  # (B,4,K)
    tgt_idx_b4: torch.Tensor,       # (B,4)
    Hb_out: int,
    Wb_out: int,
    leg: int = 0,
    sample_i: int = 0,
    title: str = "",
):
    # 取り出し
    rgb = rgb_bhwc_u8[sample_i].detach().cpu().numpy()  # HWC uint8

    rgb_t = rgb_bhwc_u8[sample_i].detach().cpu()

    # (H,W,3) or (3,H,W) の両対応
    if rgb_t.ndim == 3 and rgb_t.shape[-1] == 3:
        rgb = rgb_t.numpy()
    elif rgb_t.ndim == 3 and rgb_t.shape[0] == 3:
        rgb = rgb_t.permute(1,2,0).numpy()
    else:
        raise RuntimeError(f"Unexpected rgb shape for imshow: {tuple(rgb_t.shape)}")

    # 0..1 float にするなら（uint8でもimshowはOK）
    if rgb.dtype != np.uint8 and rgb.max() > 1.5:
        rgb = (rgb / 255.0).clip(0,1)

    bev = bev_bchw[sample_i].detach().cpu().permute(1,2,0).clamp(0,1).numpy()  # HWC float

    logits = foot_logits_b4k[sample_i, leg].detach().cpu()  # (K,)
    prob = F.softmax(logits, dim=-1).view(Hb_out, Wb_out).numpy()

    gt = int(tgt_idx_b4[sample_i, leg].detach().cpu().item())
    pd = int(torch.argmax(foot_logits_b4k[sample_i, leg]).detach().cpu().item())
    gi, gj = _idx_to_ij(gt, Wb_out)
    pi, pj = _idx_to_ij(pd, Wb_out)

    # 描画
    plt.figure(figsize=(12,4))

    ax1 = plt.subplot(1,3,1)
    ax1.set_title("RGB")
    ax1.imshow(rgb)
    ax1.axis("off")

    ax2 = plt.subplot(1,3,2)
    ax2.set_title("Input BEV (new) 128x64")
    ax2.imshow(bev)
    ax2.axis("off")

    ax3 = plt.subplot(1,3,3)
    ax3.set_title(f"Output prob (old) leg={leg}\nGT=red Pred=cyan")
    ax3.imshow(prob, origin="upper", aspect="auto")
    ax3.scatter([gj],[gi], c="r", s=40)
    ax3.scatter([pj],[pi], c="c", s=40)
    ax3.set_xticks([]); ax3.set_yticks([])

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()  # ★ここがブロック（閉じるまで止まる）









# -------------------------
# Train
# -------------------------
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/student")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--use_trav", action="store_true")
    ap.add_argument("--no_proprio", action="store_true")

    # fallback if attrs missing
    ap.add_argument("--ray_res", type=float, default=0.02)
    ap.add_argument("--ray_L", type=float, default=1.2)
    ap.add_argument("--ray_W", type=float, default=0.6)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TeacherH5Dataset(args.data, use_trav=args.use_trav)
    N = len(ds)


    hist = {
        "epoch": [],
        "train_loss": [],
        "train_err_cm": [],
        "val_loss": [],
        "val_err_cm": [],
    }

    csv_path = os.path.join(args.out, "history.csv")
    png_path = os.path.join(args.out, "curves.png")



    bev_grid_in = build_bev_grid_in(device)


    # infer Hb,Wb from trav or hits_z if present
    if "trav" in ds.f:
        Hb, Wb = ds.f["trav"].shape[1], ds.f["trav"].shape[2]
    elif "hits_z" in ds.f:
        Hb, Wb = ds.f["hits_z"].shape[1], ds.f["hits_z"].shape[2]
    else:
        raise RuntimeError("Need trav or hits_z in HDF5 to infer (Hb,Wb).")

    # BEV meta
    res = ds.res if ds.res is not None else args.ray_res
    L = ds.L if ds.L is not None else args.ray_L
    W = ds.W if ds.W is not None else args.ray_W

    n_val = int(round(N * args.val_frac))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)


    use_proprio = not args.no_proprio and (ds.foot_pos_b is not None) and (ds.root is not None)
    # model = StudentNet(Hb, Wb, use_trav=args.use_trav, use_proprio=use_proprio).to(device)
    Hb_out, Wb_out = 61, 31  # 旧のまま（H5から取ってOK）
    model = StudentNetInBEV(Hb_out, Wb_out, use_proprio=use_proprio).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # class imbalance for trav (optional)
    trav_pos_weight = None
    if args.use_trav and "trav" in ds.f:
        # quick estimate using a small subset (fast)
        idxs = np.random.choice(N, size=min(N, 2000), replace=False)
        pos = 0.0; neg = 0.0
        for i in idxs:
            t = ds.f["trav"][i]
            pos += float(t.sum())
            neg += float(t.size - t.sum())
        pw = neg / max(pos, 1.0)
        trav_pos_weight = torch.tensor([pw], device=device)

    x_coords, y_coords = make_bev_coords(Hb, Wb, L, W, device=device)




    # ---- main() の中で固定メタを用意（あなたのcfgを反映） ----
    # カメラ intrinsics（あなたの PinholeCameraCfg 由来）
    K = intrinsics_from_pinhole(
        width=64, height=64,
        focal_length=10.5,
        horizontal_aperture=20.955,
        device=device
    )

    # カメラ offset（あなたの OffsetCfg）
    t_bc = torch.tensor([0.370, 0.0, 0.15], device=device)
    q_bc = torch.tensor([0.5, -0.5, 0.5, -0.5], device=device)  # wxyz

    # BEV 定義（raycasterと合わせる）
    z_plane = 0.0  # world ground
    # L, W, Hb, Wb は既に推定済みのものを使う（args.ray_L / args.ray_W / Hb / Wb）

    use_proprio = not args.no_proprio and (ds.foot_pos_b is not None) and (ds.root is not None)
    # model = StudentBEVNet(Hb, Wb, use_trav=args.use_trav, use_proprio=use_proprio, in_ch=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


    VIS_EVERY = 200   # 何ステップごとに表示するか
    VIS_LEG   = 0     # 0:FL 1:FR 2:RL 3:RR など（あなたの並びに合わせて）
    VIS_I     = 0     # バッチ内の何番目を表示するか



    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        total_foot_m = 0.0
        count = 0

        for step, batch in enumerate(loader):

            # ---- load batch ----
            # rgb = batch["rgb"].to(device, non_blocking=True)  # (B,H,W,3) uint8 or float
            # if rgb.dtype != torch.float32:
            #     rgb = rgb.float() / 255.0
            # if rgb.ndim == 4 and rgb.shape[-1] == 3:
            #     rgb = rgb.permute(0,3,1,2).contiguous()  # -> (B,3,H,W)


            # # root は IPMに必須（pos+quat）
            # root = batch.get("root", None)
            # if root is None:
            #     raise RuntimeError("batch['root'] is required for IPM (base pose).")
            # root = root.to(device, non_blocking=True)


            rgb = batch["rgb"].to(device)  # (B,64,64,3) uint8

            foot_xy = batch["foot_xy"].to(device)  # (B,4,2)

            foot_pos_b = batch.get("foot_pos_b", None)
            root = batch.get("root_state_w", None)   # ★キー注意

            if use_proprio:
                foot_pos_b = foot_pos_b.to(device)
                root = root.to(device)

            base_pos_w = root[:, 0:3]
            base_quat_wxyz = root[:, 3:7]


            bev = rgb_to_bev_in(rgb, bev_grid_in)  # (B,3,128,64)

          

            # targets (教師): foot_xy はすでに yaw-base で保存してる想定
            foot_xy = batch["foot_xy"].to(device, non_blocking=True)  # (B,4,2)
            tgt_idx = xy_to_index(foot_xy, Hb_out, Wb_out, L, W, res).long()  # (B,4)

            foot_pos_b = batch.get("foot_pos_b", None)
            if use_proprio:
                if foot_pos_b is None:
                    raise RuntimeError("use_proprio=True but foot_pos_b missing in batch.")
                foot_pos_b = foot_pos_b.to(device, non_blocking=True)

            # ---- forward & loss ----
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    foot_logits, trav_logits = model(bev, foot_pos_b=foot_pos_b, root=root)

                    # 4脚 CE（平均にするのが基本）
                    foot_loss = 0.0
                    for leg in range(4):
                        foot_loss = foot_loss + F.cross_entropy(foot_logits[:, leg, :], tgt_idx[:, leg])
                    foot_loss = foot_loss / 4.0

                    loss = foot_loss

                    if args.use_trav and ("trav" in batch):
                        trav = batch["trav"].to(device, non_blocking=True).float()  # (B,Hb,Wb)
                        bce = F.binary_cross_entropy_with_logits(trav_logits, trav, pos_weight=trav_pos_weight)
                        loss = loss + 0.5 * bce


            
            

            # foot_xy が world なら yaw-base に変換
            # foot_xy = world_xy_to_yawbase_xy(foot_xy, root)

            # 教師 idx は旧BEV範囲のまま

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                # show_train_viz_block(
                #     rgb_bhwc_u8=batch["rgb"],   # まだGPUに載せてないならそのままOK
                #     bev_bchw=bev,               # (B,3,128,64)
                #     foot_logits_b4k=foot_logits,
                #     tgt_idx_b4=tgt_idx,
                #     Hb_out=Hb_out,
                #     Wb_out=Wb_out,
                #     leg=VIS_LEG,
                #     sample_i=0,
                #     title=f"ep={ep} step={step}",
                # )

            # ---- metric: pred xy error ----
            with torch.no_grad():
                pred_idx = foot_logits.argmax(dim=-1)                 # (B,4)
                # x_coords, y_coords は1回だけ作って外に持つの推奨（ここでは仮に毎回でも可）
                x_coords = torch.linspace(-L/2, L/2, Hb, device=device)
                y_coords = torch.linspace(-W/2, W/2, Wb, device=device)
                pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb)  # (B,4,2)
                err = torch.norm(pred_xy - foot_xy, dim=-1).mean()

            B = rgb.shape[0]
            total_loss += float(loss.item()) * B
            total_foot_m += float(err.item()) * B
            count += B

            # debugは間引く（毎バッチprintはやめる）
            if step == 0 and (not train):
                pass

        return total_loss / max(count,1), total_foot_m / max(count,1)











    # def run_epoch(loader, train: bool):
    #     model.train(train)
    #     total_loss = 0.0
    #     total_foot_m = 0.0
    #     count = 0

    #     for batch in loader:
    #         rgb = batch["rgb"].to(device)
    #         foot_xy = batch["foot_xy"].to(device)  # (B,4,2)
    #         foot_xy = world_xy_to_yawbase_xy(foot_xy, root)  # ★ここでBEV座標に戻す

    #         B = rgb.shape[0]

           
    #         # BEV座標（生成時と同じ定義）
    #         x_coords = torch.linspace(-L/2, L/2, Hb, device=foot_xy.device)
    #         y_coords = torch.linspace(-W/2, W/2, Wb, device=foot_xy.device)

    #         # --- 1) BEV範囲外率 ---
    #         x = foot_xy[..., 0]  # (B,4)
    #         y = foot_xy[..., 1]
    #         oor = (x < -L/2) | (x > L/2) | (y < -W/2) | (y > W/2)

    #         print("[DEBUG] foot_xy x min/max:", x.min().item(), x.max().item(),
    #             " y min/max:", y.min().item(), y.max().item())
    #         print("[DEBUG] out-of-range ratio:", oor.float().mean().item())

    #         # --- 2) 量子化誤差（xy→idx→xy） ---
    #         # xy_to_index / index_to_xy をあなたの関数に合わせる
    #         tgt_idx = xy_to_index(foot_xy, Hb, Wb, L, W, res)  # (B,4) long
    #         xy_back = index_to_xy(tgt_idx, x_coords, y_coords, Wb)  # (B,4,2)
    #         qerr = torch.norm(xy_back - foot_xy, dim=-1).mean()  # m

    #         print("[DEBUG] quantization err:", qerr.item() * 100.0, "cm")


    #         foot_pos_b = batch.get("foot_pos_b", None)
    #         root = batch.get("root", None)
    #         if use_proprio:
    #             foot_pos_b = foot_pos_b.to(device)
    #             root = root.to(device)

    #         # targets: grid index
    #         tgt_idx = xy_to_index(foot_xy, Hb, Wb, L, W, res)  # (B,4)

    #         with torch.cuda.amp.autocast(enabled=(device == "cuda")):
    #             foot_logits, trav_logits = model(rgb, foot_pos_b=foot_pos_b, root=root)

    #             # foot CE (sum over legs)
    #             foot_loss = 0.0
    #             for leg in range(4):
    #                 foot_loss = foot_loss + F.cross_entropy(foot_logits[:, leg, :], tgt_idx[:, leg])

    #             loss = foot_loss/4.0

    #             if args.use_trav and ("trav" in batch):
    #                 trav = batch["trav"].to(device)  # (B,Hb,Wb) 0/1
    #                 bce = F.binary_cross_entropy_with_logits(
    #                     trav_logits, trav,
    #                     pos_weight=trav_pos_weight
    #                 )
    #                 loss = loss + 0.5 * bce  # lambda=0.5 まずは固定でOK

    #         if train:
    #             opt.zero_grad(set_to_none=True)
    #             scaler.scale(loss).backward()
    #             scaler.step(opt)
    #             scaler.update()

    #         # metric: predicted xy error (meters)
    #         with torch.no_grad():
    #             pred_idx = foot_logits.argmax(dim=-1)  # (B,4)
    #             pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb)  # (B,4,2)
    #             err = torch.norm(pred_xy - foot_xy, dim=-1).mean()       # mean over legs & batch

    #         total_loss += float(loss.item()) * B
    #         total_foot_m += float(err.item()) * B
    #         count += B

    #     return total_loss / max(count,1), total_foot_m / max(count,1)

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_err = run_epoch(train_loader, train=True)
        va_loss, va_err = run_epoch(val_loader, train=False)

        # ---- record ----
        hist["epoch"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["train_err_cm"].append(tr_err * 100.0)
        hist["val_loss"].append(va_loss)
        hist["val_err_cm"].append(va_err * 100.0)

        # ---- write csv (overwrite each epoch) ----
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_err_cm", "val_loss", "val_err_cm"])
            for i in range(len(hist["epoch"])):
                w.writerow([
                    hist["epoch"][i],
                    hist["train_loss"][i],
                    hist["train_err_cm"][i],
                    hist["val_loss"][i],
                    hist["val_err_cm"][i],
                ])



        print(f"ep {ep:03d} | train loss {tr_loss:.4f} err {tr_err*100:.2f}cm | "
              f"val loss {va_loss:.4f} err {va_err*100:.2f}cm")

        # save best by val err
        if va_err < best_val:
            best_val = va_err
            ckpt = {
                "model": model.state_dict(),
                "Hb": Hb, "Wb": Wb,
                "ray_res": res, "ray_L": L, "ray_W": W,
                "use_trav": args.use_trav,
                "use_proprio": use_proprio,
            }
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

    print("done. best val err =", best_val)



    # ---- plot ----
    epochs = hist["epoch"]

    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train loss")
    plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(epochs, hist["train_err_cm"], label="train err (cm)")
    plt.plot(epochs, hist["val_err_cm"], label="val err (cm)")
    plt.xlabel("epoch")
    plt.ylabel("error [cm]")
    plt.legend()
    plt.grid(True)

    # まとめて保存（2枚保存したければ別ファイルにしてOK）
    # plt.savefig(png_path, dpi=150)
    # print("saved:", csv_path)
    # print("saved:", png_path)

    # GUIがあるなら表示したければ
    plt.show()



if __name__ == "__main__":
    main()
