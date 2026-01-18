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

# @torch.no_grad()
# def warp_rgb_to_bev_ipm(
#     rgb_bchw: torch.Tensor,        # (B,3,Himg,Wimg) float [0,1]
#     base_pos_w: torch.Tensor,      # (B,3)
#     base_quat_wxyz: torch.Tensor,  # (B,4)
#     K: torch.Tensor,              # (3,3) or (B,3,3)
#     t_bc: torch.Tensor,           # (3,) base frame
#     q_bc_wxyz: torch.Tensor,      # (4,) wxyz : camera pose relative to base
#     Hb:int, Wb:int,
#     L:float, W:float,
#     z_plane: float = 0.0,
#     use_full_base_for_cam: bool = True,   # True推奨: カメラ姿勢はroll/pitch含める
#     bc_is_cam_to_base: bool = True,       # ここが合わないときは False を試す
# ) -> torch.Tensor:
#     """
#     BEVセル中心 (x_b,y_b,z=z_plane in world) -> 画像に投影 -> grid_sample
#     - BEV格子の向き: base yaw のみで (x,y) を worldへ回す（raycaster attach_yaw_only 相当）
#     - カメラ姿勢: base姿勢(フル) × (固定オフセット) で合成
#     """
#     B, _, Himg, Wimg = rgb_bchw.shape
#     dev = rgb_bchw.device
#     dtype = rgb_bchw.dtype

#     # Kをバッチへ
#     if K.ndim == 2:
#         Kb = K[None].expand(B, -1, -1).to(dev, dtype)
#     else:
#         Kb = K.to(dev, dtype)

#     # BEV grid（base-yaw frame）
#     xs = torch.linspace(-L/2, L/2, Hb, device=dev, dtype=dtype)
#     ys = torch.linspace(-W/2, W/2, Wb, device=dev, dtype=dtype)
#     xg, yg = torch.meshgrid(xs, ys, indexing="ij")  # (Hb,Wb)
#     pw = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)  # (Hb,Wb,3)

#     # base yaw で worldへ回す
#     yaw = yaw_from_quat_wxyz(base_quat_wxyz)        # (B,)
#     R_yaw = rotz(yaw)                               # (B,3,3)

#     p0 = base_pos_w.clone()
#     p0[:, 2] = z_plane
#     Xw = (R_yaw[:,None,None,:,:] @ pw[None,:,:,:,None]).squeeze(-1) + p0[:,None,None,:]  # (B,Hb,Wb,3)

#     # base->world (フル) 回転
#     if use_full_base_for_cam:
#         R_wb = quat_to_rotmat_wxyz(base_quat_wxyz)  # (B,3,3)
#     else:
#         # yaw-onlyでカメラも回す（通常は非推奨）
#         R_wb = R_yaw

#     # base<->cam 固定変換
#     R_bc = quat_to_rotmat_wxyz(q_bc_wxyz[None].to(dev, dtype)).expand(B, -1, -1)  # (B,3,3)
#     t_bc_ = t_bc.to(dev, dtype)[None].expand(B, -1)                               # (B,3)

#     # 「T_bc が cam->base」か「base->cam」かで切替
#     if bc_is_cam_to_base:
#         # T_wc = T_wb * T_bc  （cam->world）
#         R_wc = R_wb @ R_bc
#         t_wc = base_pos_w + (R_wb @ t_bc_[..., None]).squeeze(-1)
#     else:
#         # T_bc を base->cam と解釈：T_wc = T_wb * T_bc(base->cam) なので cam->worldは逆になる
#         # ここでは cam->world を作りたいので、base->cam を反転して cam->base を得る
#         R_cb = R_bc
#         t_cb = t_bc_
#         R_bc_inv = R_cb.transpose(-1, -2)
#         t_bc_inv = -(R_bc_inv @ t_cb[..., None]).squeeze(-1)
#         R_wc = R_wb @ R_bc_inv
#         t_wc = base_pos_w + (R_wb @ t_bc_inv[..., None]).squeeze(-1)

#     # world->cam
#     R_cw = R_wc.transpose(-1, -2)
#     t_cw = -(R_cw @ t_wc[..., None]).squeeze(-1)  # (B,3)

#     # project (ROS optical: +Z forward を想定して X/Z, Y/Z)
#     Xc = (R_cw[:,None,None,:,:] @ Xw[:,:,:,None]).squeeze(-1) + t_cw[:,None,None,:]  # (B,Hb,Wb,3)
#     Z = Xc[..., 2].clamp(min=1e-6)

#     fx, fy = Kb[:,0,0], Kb[:,1,1]
#     cx, cy = Kb[:,0,2], Kb[:,1,2]
#     u = fx[:,None,None] * (Xc[...,0] / Z) + cx[:,None,None]
#     v = fy[:,None,None] * (Xc[...,1] / Z) + cy[:,None,None]

#     # grid_sample の正規化
#     grid_x = 2.0 * (u / (Wimg - 1.0)) - 1.0
#     grid_y = 2.0 * (v / (Himg - 1.0)) - 1.0
#     grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,Hb,Wb,2)

#     bev = F.grid_sample(rgb_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return bev  # (B,3,Hb,Wb)













# -------------------------
# Dataset
# -------------------------
# class TeacherH5Dataset(Dataset):
#     def __init__(self, path: str, use_trav: bool = True):
#         self.f = h5py.File(path, "r")
#         self.rgb = self.f["rgb"]            # (N,H,W,3) uint8
#         # self.foot_xy = self.f["foot_xy"]    # (N,4,2) float32
#         self.use_trav = use_trav
#         self.trav = self.f["trav"] if use_trav and "trav" in self.f else None

#         # Optional proprio (if you saved them)
#         self.foot_pos_b = self.f["foot_pos_b"] if "foot_pos_b" in self.f else None
#         self.root = self.f["root_state_w"] if "root_state_w" in self.f else None

#         self.moved = self.f["moved"] if ("moved" in self.f) else None

#         self.cmd_xy = self.f["cmd_xy"] if "cmd_xy" in self.f else None
#         self.foot_xy = self.f["foot_xy"] if "foot_xy" in self.f else None  # 旧互換


#         # BEV meta from attrs (recommended)
#         self.res = float(self.f.attrs["ray_resolution"]) if "ray_resolution" in self.f.attrs else None
#         if "ray_size_xy" in self.f.attrs:
#             sz = np.array(self.f.attrs["ray_size_xy"], dtype=np.float32).reshape(-1)
#             self.L = float(sz[0]); self.W = float(sz[1])
#         else:
#             self.L = None; self.W = None


#         # History proprio (if saved)
#         self.root_hist = self.f["root_hist"] if "root_hist" in self.f else None   # (N,T,13)
#         self.foot_hist = self.f["foot_hist"] if "foot_hist" in self.f else None  # (N,T,4,3)
#         self.T_hist = int(self.f.attrs["T_hist"]) if "T_hist" in self.f.attrs else None
#         self.contact_hist = self.f["contact_hist"] if "contact_hist" in self.f else None  # (N,T,4)



#     def __len__(self):
#         return self.rgb.shape[0]

#     def __getitem__(self, i):
#         rgb = self.rgb[i]  # uint8 HWC
#         rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)

#         # foot_xy = torch.from_numpy(self.foot_xy[i]).float()           # (4,2)

        

#         sample = {"rgb": rgb}

#         # __getitem__
#         if self.cmd_xy is not None:
#             tgt_xy = torch.from_numpy(self.cmd_xy[i]).float()
#         # else:
#         #     tgt_xy = torch.from_numpy(self.foot_xy[i]).float()
#             sample["tgt_xy"] = tgt_xy

#         if self.trav is not None:
#             trav = torch.from_numpy(self.trav[i]).float()            # (Hb,Wb) 0/1
#             sample["trav"] = trav

#         # optional: current foot positions & root state (can help)
#         if self.foot_pos_b is not None:
#             fp = torch.from_numpy(self.foot_pos_b[i]).float()        # (4,3)
#             sample["foot_pos_b"] = fp
#         if self.root is not None:
#             root = torch.from_numpy(self.root[i]).float()            # (13,)
#             sample["root_state_w"] = root


#          # history: (T,13), (T,4,3)
#         if self.root_hist is not None:
#             rh = torch.from_numpy(self.root_hist[i]).float()
#             sample["root_hist"] = rh
#         if self.foot_hist is not None:
#             fh = torch.from_numpy(self.foot_hist[i]).float()
#             sample["foot_hist"] = fh

#         if self.contact_hist is not None:
#             ch = torch.from_numpy(self.contact_hist[i]).to(torch.uint8)  # (T,4) uint8
#             sample["contact_hist"] = ch

        
#         if self.moved is not None:
#             mv = torch.from_numpy(self.moved[i])                     # (4,) uint8 0/1
#             sample["moved"] = mv



#         return sample





class TeacherH5Dataset(Dataset):
    def __init__(self, path: str, use_trav: bool = True):
        self.path = path
        self.f = h5py.File(path, "r")

        self.rgb = self.f["rgb"]  # (N,H,W,3) uint8
        self.use_trav = use_trav
        self.trav = self.f["trav"] if use_trav and "trav" in self.f else None

        self.foot_pos_b = self.f["foot_pos_b"] if "foot_pos_b" in self.f else None
        self.root = self.f["root_state_w"] if "root_state_w" in self.f else None

        self.moved = self.f["moved"] if ("moved" in self.f) else None
        self.cmd_xy = self.f["cmd_xy"] if "cmd_xy" in self.f else None
        self.foot_xy = self.f["foot_xy"] if "foot_xy" in self.f else None  # 旧互換

        # History
        self.root_hist = self.f["root_hist"] if "root_hist" in self.f else None   # (N,T,13)
        self.foot_hist = self.f["foot_hist"] if "foot_hist" in self.f else None  # (N,T,4,3)
        self.contact_hist = self.f["contact_hist"] if "contact_hist" in self.f else None  # (N,T,4)

        # ★追加：FSM
        self.fsm_hist = self.f["fsm_hist"] if "fsm_hist" in self.f else None      # (N,T,D)

        self.T_hist = int(self.f.attrs["T_hist"]) if "T_hist" in self.f.attrs else None
        self.D_fsm = int(self.f.attrs["D_fsm"]) if "D_fsm" in self.f.attrs else (self.fsm_hist.shape[-1] if self.fsm_hist is not None else 0)

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, i):
        sample = {}

        rgb = self.rgb[i]  # uint8 HWC
        sample["rgb"] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        # teacher target
        if self.cmd_xy is not None:
            sample["tgt_xy"] = torch.from_numpy(self.cmd_xy[i]).float()  # (4,2)
        # elif self.foot_xy is not None:
        #     sample["tgt_xy"] = torch.from_numpy(self.foot_xy[i]).float() # (4,2)

        if self.trav is not None:
            trav = torch.from_numpy(self.trav[i]).float()  # (Hb,Wb)
            sample["trav"] = trav

        if self.foot_pos_b is not None:
            sample["foot_pos_b"] = torch.from_numpy(self.foot_pos_b[i]).float()  # (4,3)
        if self.root is not None:
            sample["root_state_w"] = torch.from_numpy(self.root[i]).float()      # (13,)

        if self.root_hist is not None:
            sample["root_hist"] = torch.from_numpy(self.root_hist[i]).float()    # (T,13)
        if self.foot_hist is not None:
            sample["foot_hist"] = torch.from_numpy(self.foot_hist[i]).float()    # (T,4,3)

        # ★ contact は uint8 のままだとモデル側で面倒なので、ここで float にしとくのが楽
        if self.contact_hist is not None:
            ch = torch.from_numpy(self.contact_hist[i]).float()                  # (T,4) 0/1
            sample["contact_hist"] = ch

        # ★ fsm_hist
        if self.fsm_hist is not None:
            fh = torch.from_numpy(self.fsm_hist[i]).float()                      # (T,D_fsm)
            sample["fsm_hist"] = fh

        # ★ moved: BCE に使うので float(0/1) にする
        if self.moved is not None:
            mv = torch.from_numpy(self.moved[i]).float()                         # (4,)
            sample["moved"] = mv

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


from torchvision.models import resnet18, ResNet18_Weights


class StudentNetInBEV(nn.Module):
    def __init__(self, Hb_out, Wb_out, use_trav=False, use_proprio=True):
        super().__init__()
        self.Hb_out, self.Wb_out = Hb_out, Wb_out
        self.use_trav = use_trav
        self.use_proprio = use_proprio

        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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






import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


def _resnet18_os8(
    pretrained: bool = False,
    in_channels: int = 3,
):
    """古いtorchvisionでも動くように、dilation無しで stride だけ潰してOS8化した ResNet18 を作る。"""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)

    # --- 入力ch対応 (pretrainedなら重み移植) ---
    if in_channels != 3:
        old_w = m.conv1.weight.data  # (64,3,7,7)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    m.conv1.weight.copy_(old_w.mean(dim=1, keepdim=True))
                else:
                    rep = (in_channels + 2) // 3
                    w = old_w.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
                    w = w * (3.0 / float(in_channels))
                    m.conv1.weight.copy_(w)

    # --- OS8化：layer3 と layer4 の「最初のブロック」だけ stride=1 にする ---
    # ResNet18 BasicBlock は conv1 に stride が入っているのでそこを変更
    for layer in [m.layer3, m.layer4]:
        b0 = layer[0]
        b0.conv1.stride = (1, 1)
        # downsample がある場合、そこも stride を合わせる（形が合わなくなるのを防ぐ）
        if b0.downsample is not None:
            b0.downsample[0].stride = (1, 1)

    # 2D特徴が欲しいので avgpool/fc なしで layer4 まで返す
    backbone2d = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3, m.layer4,
    )
    return backbone2d



def add_coord_channels(feat: torch.Tensor) -> torch.Tensor:
    # feat: (B,C,H,W)
    B, C, H, W = feat.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype),
        torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,H,W)
    return torch.cat([feat, coords], dim=1)  # (B,C+2,H,W)


# class StudentNetInBEV_FiLM2D_OS8(nn.Module):
#     """
#     - OS8化（stride潰し）した ResNet18 で (B,512,H',W') を保持
#     - FiLMで2D特徴をproprio条件付け
#     - Conv headで 4脚スコアマップ → Hb_out×Wb_outへupsample
#     - foot_logits: (B,4,K)
#     """

#     def __init__(
#         self,
#         Hb_out: int,
#         Wb_out: int,
#         use_trav: bool = False,
#         use_proprio: bool = True,
#         bev_in_channels: int = 3,
#         proprio_dim: int = 25,
#         pretrained_backbone: bool = False,
#         film_hidden: int = 256,
#         decoder_channels: int = 128,  # 0ならdecoder無しで1x1直出し
#     ):
#         super().__init__()
#         self.Hb_out, self.Wb_out = Hb_out, Wb_out
#         self.use_trav = use_trav
#         self.use_proprio = use_proprio

#         self.backbone2d = _resnet18_os8(
#             pretrained=pretrained_backbone,
#             in_channels=bev_in_channels,
#         )

#         feat_dim = 514

#         if use_proprio:
#             self.film = nn.Sequential(
#                 nn.Linear(proprio_dim, film_hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(film_hidden, 2 * feat_dim),
#             )

#         if decoder_channels and decoder_channels > 0:
#             self.decoder = nn.Sequential(
#                 nn.Conv2d(feat_dim, decoder_channels, 3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
#                 nn.ReLU(inplace=True),
#             )
#             head_in = decoder_channels
#         else:
#             self.decoder = nn.Identity()
#             head_in = feat_dim

#         self.foot_head = nn.Conv2d(head_in, 4, kernel_size=1)
#         self.trav_head = nn.Conv2d(head_in, 1, kernel_size=1) if use_trav else None

#         self._backbone_frozen = False


#         # ---- global branch：GAP -> MLP/Linear -> (B,4,K) ----
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))   # (B,C,1,1)
#         g_in = feat_dim + (proprio_dim if use_proprio else 0)

#         # “前のモデルっぽく”するなら Linear一本でも良いが、軽くMLP挟むと安定しやすい
#         self.global_mlp = nn.Sequential(
#             nn.Linear(g_in, 512), nn.ReLU(inplace=True),
#             nn.Linear(512, 512), nn.ReLU(inplace=True),
#         )
#         K = Hb_out * Wb_out
#         self.foot_head_global = nn.Linear(512, 4 * K)  # (B,4*K)

#         # alpha：固定でも学習でもOK
#         self.alpha = 0.4


#     def freeze_backbone(self, frozen: bool = True, bn_eval: bool = True):
#         self._backbone_frozen = frozen
#         for p in self.backbone2d.parameters():
#             p.requires_grad = not frozen
#         if frozen and bn_eval:
#             self.backbone2d.eval()

#     def _apply_film(self, feat2d: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
#         gb = self.film(proprio)               # (B,1024)
#         gamma, beta = gb.chunk(2, dim=1)      # (B,512),(B,512)
#         gamma = torch.tanh(gamma)             # 安定化
#         return feat2d * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

#     def forward(self, bev: torch.Tensor, foot_pos_b=None, root=None, proprio=None):
#         if self._backbone_frozen:
#             self.backbone2d.eval()

#         feat = self.backbone2d(bev)  # (B,512,H',W')
#         feat = add_coord_channels(feat)

#         if self.use_proprio:
#             if proprio is None:
#                 assert foot_pos_b is not None and root is not None
#                 proprio = torch.cat([foot_pos_b.flatten(1), root], dim=1)  # (B,25)
#             feat = self._apply_film(feat, proprio)

#         f2 = self.decoder(feat)

#         foot_map = self.foot_head(f2)  # (B,4,H',W')
#         foot_map = F.interpolate(
#             foot_map, size=(self.Hb_out, self.Wb_out),
#             mode="bilinear", align_corners=False
#         )
#         foot_logits = foot_map.flatten(2)  # (B,4,K)



#         # ----- global logits -----
#         g = self.gap(feat).flatten(1)         # (B,512)
#         if self.use_proprio:
#             g = torch.cat([g, proprio], dim=1)  # (B,512+25)
#         h = self.global_mlp(g)                # (B,512)

#         K = self.Hb_out * self.Wb_out
#         global_logits = self.foot_head_global(h).view(-1, 4, K)  # (B,4,K)

#         # ----- combine -----
#         foot_logits = foot_logits + self.alpha * global_logits


#         trav_logits = None
#         if self.use_trav:
#             trav_map = self.trav_head(feat)  # (B,1,H',W')
#             trav_map = F.interpolate(
#                 trav_map, size=(self.Hb_out, self.Wb_out),
#                 mode="bilinear", align_corners=False
#             )
#             trav_logits = trav_map[:, 0]  # (B,Hb,Wb)

#         return foot_logits, trav_logits
    





import h5py, numpy as np

# path = "teacher_static.h5"
# with h5py.File(path, "r") as f:
#     print("attrs ray_size_xy:", f.attrs.get("ray_size_xy"))
#     print("attrs ray_resolution:", f.attrs.get("ray_resolution"))
#     for k in ["rgb","hits_z","trav","foot_xy","foot_pos_b","root_state_w"]:
#         if k in f:
#             print(k, f[k].shape, f[k].dtype)

#     foot_xy = f["foot_xy"][:]
#     print("foot_xy x min/max:", foot_xy[...,0].min(), foot_xy[...,0].max())
#     print("foot_xy y min/max:", foot_xy[...,1].min(), foot_xy[...,1].max())



# import h5py, numpy as np

# path="teacher_static.h5"
# with h5py.File(path,"r") as f:
#     foot_xy = f["foot_xy"][:]          # (N,4,2)
#     root = f["root_state_w"][:]        # (N,13)
#     base_xy = root[:, None, 0:2]       # (N,1,2)

#     rel = foot_xy - base_xy            # (N,4,2) もしfoot_xyがworldなら、ここがロボット周りの値になる

#     print("rel x min/max:", rel[...,0].min(), rel[...,0].max())
#     print("rel y min/max:", rel[...,1].min(), rel[...,1].max())

#     # 参考：foot_pos_b（ベース座標の足）と近いか？
#     fp = f["foot_pos_b"][:]            # (N,4,3)
#     print("foot_pos_b xy min/max:",
#           fp[...,0].min(), fp[...,0].max(),
#           fp[...,1].min(), fp[...,1].max())











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

def quat_apply(q, v):
    w,x,y,z = q.unbind(-1)
    vx,vy,vz = v.unbind(-1)
    tx = 2*(y*vz - z*vy)
    ty = 2*(z*vx - x*vz)
    tz = 2*(x*vy - y*vx)
    vpx = vx + w*tx + (y*tz - z*ty)
    vpy = vy + w*ty + (z*tx - x*tz)
    vpz = vz + w*tz + (x*ty - y*tx)
    return torch.stack([vpx, vpy, vpz], dim=-1)

def yaw_quat(q):
    w,x,y,z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)



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

import torch
import torch.nn.functional as F

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







def gaussian_targets_from_index(tgt_idx: torch.Tensor, H: int, W: int, sigma_pix: float = 1.5):
    """
    tgt_idx: (B,4) 0..H*W-1 のindex
    return : (B,4,K) 各脚のソフトターゲット分布（K=H*W）
    """
    device = tgt_idx.device
    B, L = tgt_idx.shape
    K = H * W

    idx = tgt_idx.reshape(B * L)  # (B*4,)

    y0 = (idx // W).float()  # (B*4,)
    x0 = (idx %  W).float()

    yy = torch.arange(H, device=device).float()[None, :, None]  # (1,H,1)
    xx = torch.arange(W, device=device).float()[None, None, :]  # (1,1,W)

    # (B*4,H,W)
    g = torch.exp(-((xx - x0[:, None, None])**2 + (yy - y0[:, None, None])**2) / (2.0 * sigma_pix**2))

    # 正規化して確率分布に
    g = g / (g.sum(dim=(1, 2), keepdim=True) + 1e-8)

    g = g.view(B, L, H, W).flatten(2)  # (B,4,K)
    return g







from unitree_rl_lab.models.student_train import StudentNetInBEV_FiLM2D_OS8, SimpleConcatFootNet






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


# -------------------------
# Train
# -------------------------
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/student")
    ap.add_argument("--epochs", type=int, default=100)
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



    # g = bev_grid_in
    # print("[GRIDCHK]",
    #     g[0,0].detach().cpu().numpy(),
    #     g[0,-1].detach().cpu().numpy(),
    #     g[-1,0].detach().cpu().numpy(),
    #     g[-1,-1].detach().cpu().numpy(),
    #     float(g.mean()), float(g.std()))


    # infer Hb,Wb from trav or hits_z if present
    if "trav" in ds.f:
        Hb, Wb = ds.f["trav"].shape[1], ds.f["trav"].shape[2]
    elif "hits_z" in ds.f:
        Hb, Wb = ds.f["hits_z"].shape[1], ds.f["hits_z"].shape[2]
    else:
        raise RuntimeError("Need trav or hits_z in HDF5 to infer (Hb,Wb).")

    # BEV meta
    # res = ds.res if ds.res is not None else args.ray_res
    # L = ds.L if ds.L is not None else args.ray_L
    # W = ds.W if ds.W is not None else args.ray_W

    res = 0.02
    L = 1.2
    W = 0.6

    n_val = int(round(N * args.val_frac))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)


    def print_loader_size(name, loader):
        ds = loader.dataset
        n = len(ds) if hasattr(ds, "__len__") else None
        nb = len(loader) if hasattr(loader, "__len__") else None
        bs = getattr(loader, "batch_size", None)
        print(f"[DATA] {name}: n_samples={n}  n_batches={nb}  batch_size={bs}")

    print_loader_size("train", train_loader)
    print_loader_size("val",   val_loader)



    use_proprio = not args.no_proprio and (ds.foot_pos_b is not None) and (ds.root is not None)
    # model = StudentNet(Hb, Wb, use_trav=args.use_trav, use_proprio=use_proprio).to(device)
    Hb_out, Wb_out = 61, 31  # 旧のまま（H5から取ってOK）
    # model = StudentNetInBEV(Hb_out, Wb_out, use_proprio=use_proprio).to(device)
    # model = StudentNetInBEV_FiLM2D_OS8(Hb_out, Wb_out, use_proprio=True, pretrained_backbone=True).to(device)
    model = SimpleConcatFootNet(Hb_out, Wb_out, pretrained_backbone=True).to(device)

    model.freeze_backbone(True)   # backbone凍結（BNもevalに）


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

    # x_coords, y_coords = make_bev_coords(Hb, Wb, L, W, device=device)

    x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
    y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)





    # ---- main() の中で固定メタを用意（あなたのcfgを反映） ----
    # カメラ intrinsics（あなたの PinholeCameraCfg 由来）
    K = intrinsics_from_pinhole(
        width=64, height=64,
        focal_length=10.5,
        horizontal_aperture=20.955,
        device=device
    )

    # カメラ offset（あなたの OffsetCfg）
  

    # BEV 定義（raycasterと合わせる）
    # z_plane = 0.0  # world ground
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


            # moved が無いなら、その場で推定する
            # if moved is None:
            # 今の足位置（yaw-base）: 収集と同様に "現時刻" を使う
            # foot_pos_b があるならそれを優先（無いなら foot_hist の末尾）
            


            rgb = batch["rgb"].to(device)  # (B,64,64,3) uint8

            # foot_xy = batch["foot_xy"].to(device)  # (B,4,2)

            if "tgt_xy" in batch:
                foot_xy = batch["tgt_xy"].to(device, non_blocking=True)     # (B,4,2)
            elif "cmd_xy" in batch:
                foot_xy = batch["cmd_xy"].to(device, non_blocking=True)   

            foot_pos_b = batch.get("foot_pos_b", None)
            root = batch.get("root_state_w", None)   # ★キー注意

            if use_proprio:
                foot_pos_b = foot_pos_b.to(device)
                root = root.to(device)

            base_pos_w = root[:, 0:3]
            base_quat_wxyz = root[:, 3:7]



        #     bev_grid_in, valid = build_bev_grid_in(
        #     root_state_w=root[:, :13],  # バッチのroot
        #     t_bc=t_bc, q_bc=q_bc_wxyz,
        #     fx=fx, fy=fy, cx=cx, cy=cy,
        #     IMG_H=64, IMG_W=64,
        #     X_MIN=X_MIN_IN, X_MAX=X_MAX_IN, Y_WIDTH=Y_WIDTH_IN,
        #     HB=HB_IN, WB=WB_IN,
        #     world_ground_z=0.0,
        #     v_flip=False,   # BEVが上下逆なら True
        # )



          



            root_hist = batch["root_hist"].to(device, non_blocking=True).float()  # (B,T,13)
            foot_hist = batch["foot_hist"].to(device, non_blocking=True).float()  # (B,T,4,3)
            cont_hist = batch["contact_hist"].to(device, non_blocking=True).float()


            # ---- DBG: cmd と現在足の距離分布（1epochにつき最初の1バッチだけ）----
            if (step == 0) and (not train):
                with torch.no_grad():
                    # 現在足（proxy）
                    if foot_pos_b is not None:
                        foot_now_xy = foot_pos_b[..., :2]          # (B,4,2)
                    else:
                        foot_now_xy = foot_hist[:, -1, :, :2]      # (B,4,2)

                    d_cm = torch.norm(foot_xy - foot_now_xy, dim=-1) * 100.0  # (B,4) cm
                    d_max = d_cm.max(dim=1).values                             # (B,) cm

                    p95 = torch.quantile(d_max, 0.95).item()
                    # print(f"[DBG-D] d_max cm: mean={d_max.mean().item():.2f}  p95={p95:.2f}  max={d_max.max().item():.2f}")

                    # “強制argmaxで1本立てる”前の moved_rate（真の更新っぽさ）
                    THR = 5.0  # cm
                    moved_true = (d_cm > THR)                    # (B,4)
                    moved_rate_true = moved_true.any(dim=1).float().mean().item()
                    print(f"[DBG] moved_rate_true={moved_rate_true:.3f}")



            # B, T, _ = root_hist.shape
            # foot_hist_flat = foot_hist.reshape(B, T, -1)   # (B,T,12)

            # root_feat = root13_hist_to_feat(root_hist, keep_z=True)  # (B,T,9)  ※keep_z=FalseでもOK



            # # cont_hist: (B,T,4) float(0/1) or uint8
            # ch = (cont_hist > 0.5).to(torch.float32)           # (B,T,4) 0/1

            # # 遷移（t=1..T-1 に定義）
            # liftoff   = (ch[:, :-1] * (1.0 - ch[:, 1:]))       # 1->0  (B,T-1,4)
            # touchdown = ((1.0 - ch[:, :-1]) * ch[:, 1:])       # 0->1  (B,T-1,4)

            # # 長さをTに合わせる（先頭に0をpad）
            # pad = torch.zeros((B, 1, 4), device=ch.device, dtype=ch.dtype)
            # liftoff   = torch.cat([pad, liftoff], dim=1)       # (B,T,4)
            # touchdown = torch.cat([pad, touchdown], dim=1)     # (B,T,4)


            fsm_hist = batch.get("fsm_hist", None)
            if fsm_hist is not None:
                fsm_hist = fsm_hist.to(device, non_blocking=True).float()   # (B,T,D)


            # # if step == 0:
            # #     print("liftoff sum", liftoff.sum().item(), "touchdown sum", touchdown.sum().item())

           

            # # proprio_hist = torch.cat([foot_hist_flat, root_feat], dim=-1)  # (B,T,12+9=21)
            # proprio_hist = torch.cat([foot_hist_flat, root_feat, cont_hist], dim=-1) 
            # proprio_flat = proprio_hist.reshape(B, -1)                     # (B, T*21)


            B, T, _ = root_hist.shape
            foot_hist_flat = foot_hist.reshape(B, T, -1)   # (B,T,12)
            root_feat = root13_hist_to_feat(root_hist, keep_z=True)  # (B,T,9)

            # cont_hist: (B,T,4) 0/1 float
            ch = (cont_hist > 0.5).to(torch.float32)

            # # ---- ★fsm があるなら正規化して足す（超重要：スケール合わせ） ----
            # if fsm_hist is not None:
            #     # fsm = [cmd_age, hold_t, bad_t, armed, near4, contact4, stableC4, stableN4]
            #     #       1        1      1     1      4      4        4         4   = 20
            #     f = fsm_hist

            #     # 時間系は「秒」なのでそのままだとスケールがでかい/広い。雑でいいので丸める。
            #     # 例：0〜2秒にクリップして 0〜1 に正規化
            #     f_time = torch.clamp(f[..., 0:3], 0.0, 2.0) / 2.0        # cmd_age, hold, bad
            #     f_armed = f[..., 3:4]                                   # 0/1
            #     f_bin   = torch.clamp(f[..., 4:12], 0.0, 1.0)            # near4, contact4
            #     f_stab  = torch.clamp(f[..., 12:20], 0.0, 2.0) / 2.0     # stableC4, stableN4
            #     fsm_feat = torch.cat([f_time, f_armed, f_bin, f_stab], dim=-1)  # (B,T,20)
            # else:
            #     fsm_feat = None

            # # ---- concat ----
            # if fsm_feat is None:
            #     proprio_hist = torch.cat([foot_hist_flat, root_feat, ch], dim=-1)      # (B,T,12+9+4)
            # else:
            #     proprio_hist = torch.cat([foot_hist_flat, root_feat, ch, fsm_feat], dim=-1)  # (B,T,12+9+4+20)

            # proprio_flat = proprio_hist.reshape(B, -1)   # (B, T*D_total)


            # swing = batch["moved"].to(device, non_blocking=True).float()  # (B,4) one-hot想定
            # # 念のためonehot化（壊れてても落ちないように）
            # swing = (swing > 0.5).float()
            # # 履歴に入れるなら T に複製
            # # swing_hist = swing[:, None, :].expand(B, T, 4)               # (B,T,4)

            # proprio_hist = torch.cat([foot_hist_flat, root_feat, ch], dim=-1)
            # proprio_flat = proprio_hist.reshape(B, -1)


            # # proprio_flat = proprio_hist.reshape(B, -1)          # (B, T*D)
            # # swing = batch["swing_onehot"].to(device).float()    # (B,4)
            # proprio_flat = torch.cat([proprio_flat, swing], dim=-1)  # (B, T*D + 4)




            B, T, _ = root_hist.shape
            foot_hist_flat = foot_hist.reshape(B, T, -1)                      # (B,T,12)
            root_feat = root13_hist_to_feat(root_hist, keep_z=True)           # (B,T,9)
            # ch = (cont_hist > 0.5).to(torch.float32)                          # (B,T,4)

            proprio_parts = [foot_hist_flat, root_feat]                   # 基本はこれ

            # ---- ★FSM最小8次元を追加 ----
            if fsm_hist is not None:
                f = fsm_hist.to(torch.float32)                                # (B,T,20)

                # time系(秒)は軽く正規化（雑でOK、でも必須）
                # 例: 0〜2秒を 0〜1 に
                cmd_age = torch.clamp(f[..., 0:1], 0.0, 2.0) / 2.0            # (B,T,1)
                hold_t  = torch.clamp(f[..., 1:2], 0.0, 2.0) / 2.0            # (B,T,1)
                bad_t   = torch.clamp(f[..., 2:3], 0.0, 2.0) / 2.0            # (B,T,1)

                armed   = torch.clamp(f[..., 3:4], 0.0, 1.0)                  # (B,T,1)

                # contact_state は fsmの contact4 を使う（※あなたのfsm定義だと 8:12）
                contact4 = torch.clamp(f[..., 8:12], 0.0, 1.0)                # (B,T,4)

                fsm_min = torch.cat([cmd_age, hold_t, bad_t, armed, contact4], dim=-1)  # (B,T,8)

                proprio_parts.append(fsm_min)

            proprio_hist = torch.cat(proprio_parts, dim=-1)                   # (B,T,12+9+8)
            # proprio_flat = proprio_hist.reshape(B, -1)                        # (B,T*)

            # swing(=moved onehot)はそのまま最後に付ける
            # swing = batch["moved"].to(device, non_blocking=True).float()      # (B,4)
            # swing = (swing > 0.5).float()
            # proprio_flat = torch.cat([proprio_flat, swing], dim=-1)           # (B,T*33 + 4)


            swing_rep = swing[:, None, :].expand(B, T, 4)          # (B,T,4)
            proprio_hist = torch.cat([proprio_hist, swing_rep], -1) # (B,T,)
            proprio_flat = proprio_hist.reshape(B, -1)              # (B,T* )







            # if moved is None:
            # 今の足位置（yaw-base）の proxy を作る：foot_pos_b があれば優先、無ければ foot_hist末尾
            if foot_pos_b is not None:
                foot_now_xy = foot_pos_b[..., :2]          # (B,4,2)
            else:
                foot_now_xy = foot_hist[:, -1, :, :2]      # (B,4,2)  ※foot_histは既にdevice上

            d = torch.norm(foot_xy - foot_now_xy, dim=-1)  # (B,4) [m]
            THR = 0.05
            moved = (d > THR)

            # 1本も立たないサンプルは最大の脚を1本だけ moved 扱いにする
            # none = (moved.sum(dim=1) == 0)
            # if none.any():
            #     arg = d[none].argmax(dim=1)
            #     moved[none] = False
            #     moved[none, arg] = True

            

            moved = batch.get("moved", None)
            if moved is not None:
                moved = moved.to(device, non_blocking=True).float()  


            
            
      
            
            
            



            bev = rgb_to_bev_in(rgb, bev_grid_in)  # (B,3,128,64)

          

            # targets (教師): foot_xy はすでに yaw-base で保存してる想定
            # foot_xy = batch["foot_xy"].to(device, non_blocking=True)  # (B,4,2)
            tgt_idx = xy_to_index(foot_xy, Hb_out, Wb_out, L, W, res).long()  # (B,4)

            foot_pos_b = batch.get("foot_pos_b", None)
            if use_proprio:
                if foot_pos_b is None:
                    raise RuntimeError("use_proprio=True but foot_pos_b missing in batch.")
                foot_pos_b = foot_pos_b.to(device, non_blocking=True)

            # ---- forward & loss ----
            # with torch.set_grad_enabled(train):
            #     with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            #         foot_logits, trav_logits = model(bev, foot_pos_b=foot_pos_b, root=root)

            #         # 4脚 CE（平均にするのが基本）
            #         foot_loss = 0.0
            #         for leg in range(4):
            #             foot_loss = foot_loss + F.cross_entropy(foot_logits[:, leg, :], tgt_idx[:, leg])
            #         foot_loss = foot_loss / 4.0

            #         loss = foot_loss


            # ---- forward & loss ----
            # with torch.set_grad_enabled(train):
            #     with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            #         # foot_logits, trav_logits = model(bev, foot_pos_b=foot_pos_b, root=root)  # (B,4,K)
            #         # foot_logits, trav_logits = model(bev, proprio=proprio_flat) # using history 
            #         foot_logits, moved_logits = model(bev, proprio=proprio_flat) # using history 



            #         # H, W = Hb_out, Wb_out   # ここはあなたの出力グリッドサイズ
            #         target_prob = gaussian_targets_from_index(tgt_idx, Hb_out, Wb_out, sigma_pix=1.5)  # (B,4,K)

            #         # KL( target || student )：inputはlog-prob、targetはprob
            #         # AMP下で安定させるため float() に寄せるのが無難
            #         logp = F.log_softmax(foot_logits.float(), dim=-1)  # (B,4,K)
            #         foot_loss = F.kl_div(logp, target_prob.to(logp.dtype), reduction="batchmean")

            #         loss = foot_loss



            #         mv = batch.get("moved", None)
            #         if mv is not None:
            #             # mv = mv.to(device, non_blocking=True).float()  # (B,4) 0/1
            #             # loss_mv = F.binary_cross_entropy_with_logits(moved_logits, mv)
            #             # loss = loss + 0.2 * loss_mv   # 0.1〜0.5で調整


            #             mv = batch["moved"].to(device, non_blocking=True).float()  # (B,4) one-hot
            #             pos_weight = torch.full((4,), 3.0, device=device)          # だいたい3
            #             loss_mv = F.binary_cross_entropy_with_logits(moved_logits, mv, pos_weight=pos_weight)
            #             loss = loss + 0.3 * loss_mv


            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    # foot_logits, moved_logits = model(bev, proprio=proprio_flat)  # (B,4,K), (B,4)

                    # target_prob = gaussian_targets_from_index(tgt_idx, Hb_out, Wb_out, sigma_pix=1.5)  # (B,4,K)

                    # logp = F.log_softmax(foot_logits.float(), dim=-1)  # (B,4,K)

                    # # ★脚マスクするため reduction="none"
                    # kl_elem = F.kl_div(
                    #     logp,
                    #     target_prob.to(logp.dtype),
                    #     reduction="none",       # (B,4,K)
                    #     log_target=False
                    # )

                    # # (B,4) に落とす：K方向を合計（= 各脚のKL）
                    # kl_leg = kl_elem.sum(dim=-1)  # (B,4)

                    # # swing onehot (B,4)
                    # mv = batch.get("moved", None)
                    # if mv is None:
                    #     raise RuntimeError("Need swing one-hot in batch['moved'] for masked training.")
                    # mv = mv.to(device, non_blocking=True).float()

                    # # 念のため：各行が1本になるように（壊れてたら修正）
                    # # （すでにonehotなら不要だけど安全）
                    # mv = (mv > 0.5).float()
                    # row_sum = mv.sum(dim=1, keepdim=True).clamp(min=1.0)
                    # mv = mv / row_sum

                    # # ★脚マスクして平均（分母は有効脚数）
                    # foot_loss = (kl_leg * mv).sum() / mv.sum().clamp(min=1.0)

                    # loss = foot_loss



                    # ---- select leg from swing onehot ----
                    swing = batch["moved"].to(device, non_blocking=True).float()          # (B,4)
                    leg = swing.argmax(dim=-1)                                            # (B,)
                    bidx = torch.arange(B, device=device)

                    tgt_idx_sel = tgt_idx[bidx, leg]                                  # (B,)

                    # gaussian_targets_from_index が (B,4) 入力前提なら (B,1) にして squeeze
                    target_prob_sel = gaussian_targets_from_index(
                        tgt_idx_sel.view(B, 1), Hb_out, Wb_out, sigma_pix=1.5
                    ).squeeze(1).to(torch.float32)                                        # (B,K)

                    # ---- forward ----
                    logits = model(bev, proprio=proprio_flat)                             # (B,1,K)
                    logits_sel = logits.squeeze(1)                                        # (B,K)

                    logp_sel = F.log_softmax(logits_sel.float(), dim=-1)                  # (B,K)

                    # KL(target || student)
                    foot_loss = F.kl_div(logp_sel, target_prob_sel.to(logp_sel.dtype), reduction="batchmean")

                    loss = foot_loss



                    # # ===== moved脚だけ CE を計算（ここに置き換え）=====
                    # per_leg_ce = []
                    # for leg in range(4):
                    #     ce = F.cross_entropy(
                    #         foot_logits[:, leg, :], tgt_idx[:, leg],
                    #         reduction="none"
                    #     )  # (B,)
                    #     per_leg_ce.append(ce)
                    # per_leg_ce = torch.stack(per_leg_ce, dim=1)  # (B,4)

                    # eps = 1e-6
                    # foot_loss_per_sample = (per_leg_ce * moved.float()).sum(dim=1) / (moved.sum(dim=1).float() + eps)  # (B,)
                    # foot_loss = foot_loss_per_sample.mean()
                    # loss = foot_loss
                    # # ================================================

                    # if args.use_trav and ("trav" in batch):
                    #     trav = batch["trav"].to(device, non_blocking=True).float()  # (B,Hb,Wb)
                    #     bce = F.binary_cross_entropy_with_logits(trav_logits, trav, pos_weight=trav_pos_weight)
                    #     loss = loss + 0.5 * bce


            
            

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
            # with torch.no_grad():
            #     pred_idx = foot_logits.argmax(dim=-1)                 # (B,4)
            #     # x_coords, y_coords は1回だけ作って外に持つの推奨（ここでは仮に毎回でも可）
            #     x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
            #     y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)
            #     pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)  # (B,4,2)
            #     err = torch.norm(pred_xy - foot_xy, dim=-1).mean()

            

            # with torch.no_grad():
            #     pred_idx = foot_logits.argmax(dim=-1)  # (B,4)

            #     x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
            #     y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)
            #     pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)  # (B,4,2)

            #     per_leg_err = torch.norm(pred_xy - foot_xy, dim=-1)  # (B,4)

            #     # envごとの「最大誤差」と「その脚index」
            #     max_err_env, max_leg_env = per_leg_err.max(dim=1)  # (B,), (B,)

            #     # ログ用：最大誤差の平均（=各envのworst footの誤差を平均）
            #     err_max_mean = max_err_env.mean()

            #     # 参考：バッチ内で一番悪いサンプルを1つ特定
            #     worst_env = max_err_env.argmax()
            #     worst_leg = max_leg_env[worst_env]
            #     worst_err = max_err_env[worst_env]

            #     err = err_max_mean  # ← これで total_foot_m が「最大脚誤差」の平均になる


            def quantize_xy_from_xy(xy, Hb, Wb, L, W, res, x_coords, y_coords):
                # xy: (B,4,2) continuous
                idx = xy_to_index(xy, Hb, Wb, L, W, res).long()               # (B,4)
                xy_q = index_to_xy(idx, x_coords, y_coords, Wb)               # (B,4,2)
                return xy_q, idx

            def pred_xy_from_logits_argmax(foot_logits, x_coords, y_coords, Wb):
                pred_idx = foot_logits.argmax(dim=-1)                         # (B,4)
                pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb)      # (B,4,2)
                return pred_xy, pred_idx



            # with torch.no_grad():
            #     pred_idx = foot_logits.argmax(dim=-1)
            #     pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)

            #     per_leg_err = torch.norm(pred_xy - foot_xy, dim=-1)  # (B,4) [m]

            #     # worst（あなたの今の指標）
            #     err_max_mean = per_leg_err.max(dim=1).values.mean()

            #     # moved脚だけ（play側に近い）
            #     eps = 1e-6
            #     err_moved = (per_leg_err * moved.float()).sum(dim=1) / (moved.sum(dim=1).float() + eps)
            #     err_moved_mean = err_moved.mean()

            #     moved_rate = (moved.sum(dim=1) > 0).float().mean()

            #     # 好きな方を total_foot_m に入れる
            #     err = err_moved_mean

            #     if step == 0 and (not train):
            #         print(f"[METRIC] err_max={err_max_mean.item()*100:.2f}cm  "
            #             f"err_moved={err_moved_mean.item()*100:.2f}cm  moved_rate={moved_rate.item():.2f}")
                    

                    # THR = 0.05  # 5cm
                    # moved_true = (d/100.0 > THR)              # (B,4) bool  ※dはcmなので戻す
                    # moved_rate_true = moved_true.any(dim=1).float().mean().item()
                    # print(f"[DBG] moved_rate_true={moved_rate_true:.3f}")


            
            # with torch.no_grad():
            #     # --- pred: argmax -> xy (cell center) ---
            #     pred_xy, pred_idx = pred_xy_from_logits_argmax(foot_logits, x_coords, y_coords, Wb_out)  # (B,4,2)

            #     # --- teacher: quantize to cell center (onlineと同じ) ---
            #     # foot_xy は連続値の教師（yaw-base）
            #     teach_xy_q, teach_idx = quantize_xy_from_xy(foot_xy, Hb_out, Wb_out, L, W, res, x_coords, y_coords)

            #     # --- error (aligned definition) ---
            #     diff = pred_xy - teach_xy_q                      # (B,4,2) [m]
            #     err_all = diff.norm(dim=-1) * 100.0              # (B,4) [cm]
            #     err_x   = diff[...,0].abs() * 100.0              # (B,4) [cm]
            #     err_y   = diff[...,1].abs() * 100.0              # (B,4) [cm]

            #     # moved脚だけ平均（movedがあるならそれを使う）
            #     eps = 1e-6
            #     err_moved = (err_all * moved.float()).sum(dim=1) / (moved.sum(dim=1).float() + eps)  # (B,) [cm]
            #     err_moved_mean = err_moved.mean() / 100.0  # ← total_foot_m は [m] で持ちたいなら /100

            #     # all4平均も欲しければ
            #     err_all_mean = err_all.mean() / 100.0

            #     if step == 0 and (not train):
            #         print(f"[METRIC] all4={err_all.mean().item():.2f}cm  "
            #             f"x={err_x.mean().item():.2f}cm  y={err_y.mean().item():.2f}cm  "
            #             f"moved={err_moved.mean().item():.2f}cm  moved_rate={(moved.any(dim=1).float().mean().item()):.2f}")

            #     # どれを tr_err/va_err に採用するか（オンラインに寄せるなら moved でも all4でもOK）
            #     err = err_moved_mean   # [m] で保存したいなら



            # with torch.no_grad():
            #     # -----------------------------
            #     # (A) pred: argmax -> xy (cell center)
            #     # -----------------------------
            #     pred_idx = logits.argmax(dim=-1)                                # (B,4)
            #     pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)         # (B,4,2) [m]

            #     # -----------------------------
            #     # (B) teacher: quantize -> xy (cell center)  ※オンラインと同じ定義
            #     # -----------------------------
            #     tgt_idx  = xy_to_index(foot_xy, Hb_out, Wb_out, L, W, res).long()    # (B,4)
            #     tgt_xy_q = index_to_xy(tgt_idx, x_coords, y_coords, Wb_out)          # (B,4,2) [m]

            #     # -----------------------------
            #     # (C) errors (cm)
            #     # -----------------------------
            #     diff   = pred_xy - tgt_xy_q                                          # (B,4,2) [m]
            #     err_cm = diff.norm(dim=-1) * 100.0                                   # (B,4) [cm]
            #     err_x  = diff[..., 0].abs() * 100.0                                  # (B,4) [cm]
            #     err_y  = diff[..., 1].abs() * 100.0                                  # (B,4) [cm]

            #     all4_mean = err_cm.mean().item()
            #     x_mean    = err_x.mean().item()
            #     y_mean    = err_y.mean().item()

            #     # -----------------------------
            #     # (D) "true moved" + "hard update" filter (オンライン寄せ)
            #     #   - true moved: 現在足から 5cm 以上動く脚
            #     #   - hard sample: moved脚が1本でもあるサンプル
            #     # -----------------------------
            #     d_now_cm = torch.norm(foot_xy - foot_now_xy, dim=-1) * 100.0         # (B,4) [cm]
            #     moved_true = (d_now_cm > 5.0)                                        # (B,4) bool
            #     hard = moved_true.any(dim=1)                                         # (B,) bool
            #     hard_rate = hard.float().mean().item()                               # 0..1

            #     # moved脚だけの平均誤差（サンプルごと→平均）
            #     eps = 1e-6
            #     err_moved_per_sample = (err_cm * moved_true.float()).sum(dim=1) / (moved_true.sum(dim=1).float() + eps)  # (B,) [cm]

            #     # hard サンプルだけ
            #     if hard.any():
            #         moved_mean_hard = err_moved_per_sample[hard].mean().item()
            #         x_mean_hard     = err_x[hard].mean().item()
            #         y_mean_hard     = err_y[hard].mean().item()
            #     else:
            #         moved_mean_hard = float("nan")
            #         x_mean_hard     = float("nan")
            #         y_mean_hard     = float("nan")

            #     # -----------------------------
            #     # (E) cell-accuracy (任意だが超便利)
            #     # -----------------------------
            #     acc_all = (pred_idx == tgt_idx).float().mean().item()
            #     acc_moved = (pred_idx[moved_true] == tgt_idx[moved_true]).float().mean().item() if moved_true.any() else float("nan")

            #     # -----------------------------
            #     # (F) print (valの先頭バッチだけ等、条件は外側で制御)
            #     #   例: if (step == 0) and (not train):
            #     # -----------------------------
            #     print(
            #         f"[METRIC] all4={all4_mean:.2f}cm  x={x_mean:.2f}cm  y={y_mean:.2f}cm  "
            #         f"| HARD(rate={hard_rate:.3f}) moved={moved_mean_hard:.2f}cm  x={x_mean_hard:.2f}cm  y={y_mean_hard:.2f}cm  "
            #         f"| ACC all={acc_all:.3f} moved={acc_moved:.3f}"
            #     )

            #     # -----------------------------------------
            #     # 返り値（epoch集計）に使う誤差を選ぶならここ
            #     # 例: オンライン寄せなら hard moved を採用（ただしhardが0ならall4にfallback）
            #     # -----------------------------------------
            #     if hard.any():
            #         err_for_log_cm = err_moved_per_sample[hard].mean()               # tensor scalar [cm]
            #     else:
            #         err_for_log_cm = err_cm.mean()                                   # [cm]

            #     # 既存の total_foot_m が [m] 想定なら cm->m に戻す
            #     err = err_for_log_cm / 100.0                                         # [m]
            # ======== end PRINT / METRIC block ========




            with torch.no_grad():
                B = logits.shape[0]
                bidx = torch.arange(B, device=logits.device)

                # -----------------------------
                # (0) select leg from swing onehot
                # -----------------------------
                swing = batch["moved"].to(logits.device, non_blocking=True).float()   # (B,4) one-hot想定
                swing = (swing > 0.5).float()
                bad = (swing.sum(dim=1) == 0)
                if bad.any():
                    swing[bad, 0] = 1.0
                leg = swing.argmax(dim=-1)                                            # (B,)

                # -----------------------------
                # (A) pred: argmax -> xy (cell center)
                #   logits: (B,1,K) なので squeeze
                # -----------------------------
                pred_idx = logits.squeeze(1).argmax(dim=-1)                           # (B,)
                pred_xy  = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)          # (B,2)

                # -----------------------------
                # (B) teacher: 4脚 -> 選択脚だけ抜く -> xy (cell center)
                # -----------------------------
                tgt_idx_all = xy_to_index(foot_xy, Hb_out, Wb_out, L, W, res).long()  # (B,4)
                tgt_idx     = tgt_idx_all[bidx, leg]                                  # (B,)
                tgt_xy_q    = index_to_xy(tgt_idx, x_coords, y_coords, Wb_out)        # (B,2)

                # -----------------------------
                # (C) errors (cm)
                # -----------------------------
                diff   = pred_xy - tgt_xy_q                                           # (B,2)
                err_cm = diff.norm(dim=-1) * 100.0                                    # (B,)
                err_x  = diff[:, 0].abs() * 100.0                                     # (B,)
                err_y  = diff[:, 1].abs() * 100.0                                     # (B,)

                all_mean = err_cm.mean().item()
                x_mean   = err_x.mean().item()
                y_mean   = err_y.mean().item()

                # -----------------------------
                # (D) "true moved" を選択脚だけで定義（任意）
                #   - 現在足から 5cm以上動くなら moved_true
                # -----------------------------
                if foot_pos_b is not None:
                    foot_now_xy = foot_pos_b[..., :2]               # (B,4,2)
                else:
                    foot_now_xy = foot_hist[:, -1, :, :2]           # (B,4,2)

                foot_now_sel = foot_now_xy[bidx, leg]               # (B,2)
                d_now_cm = torch.norm(foot_xy[bidx, leg] - foot_now_sel, dim=-1) * 100.0  # (B,)
                moved_true = (d_now_cm > 5.0)                       # (B,)
                hard_rate  = moved_true.float().mean().item()

                moved_mean = err_cm[moved_true].mean().item() if moved_true.any() else float("nan")

                # -----------------------------
                # (E) cell-accuracy
                # -----------------------------
                acc = (pred_idx == tgt_idx).float().mean().item()

                print(
                    f"[METRIC-1LEG] err={all_mean:.2f}cm  x={x_mean:.2f}cm  y={y_mean:.2f}cm  "
                    f"| moved_true(rate={hard_rate:.3f}) err_moved={moved_mean:.2f}cm  | ACC={acc:.3f}"
                )

                # epoch集計に使う誤差（オンライン寄せなら moved_true のみ、なければ全体）
                if moved_true.any():
                    err_for_log_cm = err_cm[moved_true].mean()
                else:
                    err_for_log_cm = err_cm.mean()

                err = err_for_log_cm / 100.0   # [m]



                    
                

                    

    


            # with torch.no_grad():
            #     pred_idx = foot_logits.argmax(dim=-1)  # (B,4)

            #     # ★Hb_out/Wb_out を使う
            #     x_coords = torch.linspace(-L/2, L/2, Hb_out, device=device)
            #     y_coords = torch.linspace(-W/2, W/2, Wb_out, device=device)
            #     pred_xy = index_to_xy(pred_idx, x_coords, y_coords, Wb_out)  # (B,4,2)

            #     per_leg_err = torch.norm(pred_xy - foot_xy, dim=-1)  # (B,4)

            #     err_mean = per_leg_err.mean()

            #     eps = 1e-6
            #     err_moved = (per_leg_err * moved.float()).sum(dim=1) / (moved.sum(dim=1).float() + eps)
            #     err_moved = err_moved.mean()

            #     err_max = per_leg_err.max(dim=1).values.mean()

            B = rgb.shape[0]
            total_loss += float(loss.item()) * B
            total_foot_m += float(err.item()) * B
            # total_foot_m += float(err_mean.item()) * B

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

    plt.ylim(bottom=0)          # 縦軸の下限を0に固定


    plt.figure()
    plt.plot(epochs, hist["train_err_cm"], label="train err (cm)")
    plt.plot(epochs, hist["val_err_cm"], label="val err (cm)")
    plt.xlabel("epoch")
    plt.ylabel("error [cm]")
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=0)          # 縦軸の下限を0に固定





    # まとめて保存（2枚保存したければ別ファイルにしてOK）
    # plt.savefig(png_path, dpi=150)
    # print("saved:", csv_path)
    # print("saved:", png_path)

    # GUIがあるなら表示したければ
    plt.show()



if __name__ == "__main__":
    main()
