# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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








import h5py
import numpy as np
import torch

# class TeacherH5Writer:
#     def __init__(self, path: str, rgb_hw=(64, 64), bev_hw=(61, 31), compress=True):
#         H, W = rgb_hw
#         Hb, Wb = bev_hw
#         self.f = h5py.File(path, "w")

#         comp = "gzip" if compress else None

#         self.d_rgb = self.f.create_dataset(
#             "rgb", shape=(0, H, W, 3), maxshape=(None, H, W, 3),
#             dtype=np.uint8, chunks=True, compression=comp
#         )
#         self.d_hits = self.f.create_dataset(
#             "hits_z", shape=(0, Hb, Wb), maxshape=(None, Hb, Wb),
#             dtype=np.float32, chunks=True, compression=comp
#         )
#         self.d_trav = self.f.create_dataset(
#             "trav", shape=(0, Hb, Wb), maxshape=(None, Hb, Wb),
#             dtype=np.uint8, chunks=True, compression=comp
#         )
#         self.d_foot = self.f.create_dataset(
#             "foot_xy", shape=(0, 4, 2), maxshape=(None, 4, 2),
#             dtype=np.float32, chunks=True, compression=comp
#         )
#         self.d_footpos = self.f.create_dataset(
#             "foot_pos_b", shape=(0, 4, 3), maxshape=(None, 4, 3),
#             dtype=np.float32, chunks=True, compression=comp
#         )
#         self.d_root = self.f.create_dataset(
#             "root_state_w", shape=(0, 13), maxshape=(None, 13),
#             dtype=np.float32, chunks=True, compression=comp
#         )


#         self.d_moved = self.f.create_dataset(
#         "moved", shape=(0, 4), maxshape=(None, 4),
#         dtype=np.uint8, chunks=True, compression=comp
#         )

       

#     def set_attrs(self, **kwargs):
#         for k, v in kwargs.items():
#             self.f.attrs[k] = v

#     def _append(self, dset, arr_np):
#         n = dset.shape[0]
#         dset.resize(n + arr_np.shape[0], axis=0)
#         dset[n:n + arr_np.shape[0]] = arr_np

#     @torch.no_grad()
#     def append_batch(self, rgb_u8, hits_z_f32, trav_u8, foot_xy_f32, foot_pos_b_f32, root_state_w_f32, moved_u8=None):
#         """
#         全部 torch Tensor でも numpy でもOK。
#         shape:
#           rgb:      (B,H,W,3)
#           hits_z:   (B,Hb,Wb)
#           trav:     (B,Hb,Wb)
#           foot_xy:  (B,4,2)
#           foot_pos: (B,4,3)
#           root:     (B,13)
#         """
#         def to_np(x, dtype=None):
#             if isinstance(x, np.ndarray):
#                 out = x
#             else:
#                 out = x.detach().cpu().numpy()
#             if dtype is not None:
#                 out = out.astype(dtype, copy=False)
#             return out

#         self._append(self.d_rgb,     to_np(rgb_u8, np.uint8))
#         self._append(self.d_hits,    to_np(hits_z_f32, np.float32))
#         self._append(self.d_trav,    to_np(trav_u8, np.uint8))
#         self._append(self.d_foot,    to_np(foot_xy_f32, np.float32))
#         self._append(self.d_footpos, to_np(foot_pos_b_f32, np.float32))
#         self._append(self.d_root,    to_np(root_state_w_f32, np.float32))


#         if moved_u8 is not None:
#             mv = to_np(moved_u8, np.uint8)
#             # 形を安全にチェック（ミスると silent に壊れるので）
#             assert mv.ndim == 2 and mv.shape[1] == 4, f"moved shape must be (B,4), got {mv.shape}"
#             self._append(self.d_moved, mv)

#     def close(self):
#         self.f.flush()
#         self.f.close()


    


class TeacherH5Writer:
    def __init__(self, path: str, rgb_hw=(64, 64), bev_hw=(61, 31), compress=True, T_hist: int = 16):
        H, W = rgb_hw
        Hb, Wb = bev_hw
        self.f = h5py.File(path, "w")
        comp = "gzip" if compress else None

        self.d_rgb = self.f.create_dataset("rgb", (0,H,W,3), maxshape=(None,H,W,3),
                                           dtype=np.uint8, chunks=True, compression=comp)
        self.d_hits = self.f.create_dataset("hits_z", (0,Hb,Wb), maxshape=(None,Hb,Wb),
                                            dtype=np.float32, chunks=True, compression=comp)
        self.d_trav = self.f.create_dataset("trav", (0,Hb,Wb), maxshape=(None,Hb,Wb),
                                            dtype=np.uint8, chunks=True, compression=comp)

        # ★教師：次の目標（コマンド）
        self.d_cmd = self.f.create_dataset("cmd_xy", (0,4,2), maxshape=(None,4,2),
                                           dtype=np.float32, chunks=True, compression=comp)

        # ★入力：決定時点の実足（3D）
        self.d_footpos = self.f.create_dataset("foot_pos_b", (0,4,3), maxshape=(None,4,3),
                                               dtype=np.float32, chunks=True, compression=comp)

        self.d_root = self.f.create_dataset("root_state_w", (0,13), maxshape=(None,13),
                                            dtype=np.float32, chunks=True, compression=comp)

        self.d_moved = self.f.create_dataset("moved", (0,4), maxshape=(None,4),
                                             dtype=np.uint8, chunks=True, compression=comp)
        
        self.n = 0
        self.T_hist = int(T_hist)


        # =========================
        # ★追加: proprio history datasets
        # =========================
        T = self.T_hist

        # root_hist: (N, T, 13)
        self.ds_root_hist = self.f.create_dataset(
            "root_hist",
            shape=(0, T, 13),
            maxshape=(None, T, 13),
            dtype=np.float32,
            chunks=(256, T, 13),     # chunkは適当でOK（Kの典型に合わせる）
            compression="lzf",       # gzipでもOK。lzfは速い
            shuffle=False,
        )

        # foot_hist: (N, T, 4, 3)
        self.ds_foot_hist = self.f.create_dataset(
            "foot_hist",
            shape=(0, T, 4, 3),
            maxshape=(None, T, 4, 3),
            dtype=np.float32,
            chunks=(256, T, 4, 3),
            compression="lzf",
            shuffle=False,
        )

        chunk0 = 256


        self.d_contact_hist = self.f.create_dataset(
            "contact_hist",
            shape=(chunk0, T, 4),           # ★最初は 0 じゃなく chunk0 で作る
            maxshape=(None, T, 4),
            dtype=np.uint8,
            chunks=(chunk0, T, 4),
            compression="lzf",
            shuffle=False,
        )

        # ★空に戻す（以降は append で増える）
        self.d_contact_hist.resize(0, axis=0)




        self.D_fsm = 20  # 上と一致

        self.d_fsm_hist = self.f.create_dataset(
            "fsm_hist",
            shape=(0, self.T_hist, self.D_fsm),
            maxshape=(None, self.T_hist, self.D_fsm),
            dtype="f4",
            chunks=(64, self.T_hist, self.D_fsm),
            compression="gzip",
        )




    def set_attrs(self, **kwargs):
        for k, v in kwargs.items():
            self.f.attrs[k] = v

        self.f.attrs["T_hist"] = np.int32(self.T_hist)


    def _append(self, dset, arr_np):
        n = dset.shape[0]
        dset.resize(n + arr_np.shape[0], axis=0)
        dset[n:n + arr_np.shape[0]] = arr_np

    @torch.no_grad()
    def append_batch(self, *, rgb_u8, hits_z_f32, trav_u8, cmd_xy_f32, foot_pos_b_f32, root_state_w_f32,  root_hist_f32, foot_hist_f32, contact_hist_u8=None, fsm_hist_f32=None, moved_u8=None):
        def to_np(x, dtype=None):
            if isinstance(x, np.ndarray):
                out = x
            else:
                out = x.detach().cpu().numpy()
            if dtype is not None:
                out = out.astype(dtype, copy=False)
            return out

        self._append(self.d_rgb,  to_np(rgb_u8, np.uint8))
        self._append(self.d_hits, to_np(hits_z_f32, np.float32))
        self._append(self.d_trav, to_np(trav_u8, np.uint8))
        self._append(self.d_cmd,  to_np(cmd_xy_f32, np.float32))
        self._append(self.d_footpos, to_np(foot_pos_b_f32, np.float32))
        self._append(self.d_root, to_np(root_state_w_f32, np.float32))

        if moved_u8 is not None:
            mv = to_np(moved_u8, np.uint8)
            assert mv.ndim == 2 and mv.shape[1] == 4, f"moved shape must be (B,4), got {mv.shape}"
            self._append(self.d_moved, mv)


        # ===== history (必須) =====
        rh = to_np(root_hist_f32, np.float32)
        fh = to_np(foot_hist_f32, np.float32)
        ch = to_np(contact_hist_u8, np.uint8)     # (B,T,4)


        assert rh.ndim == 3 and rh.shape[1:] == (self.T_hist, 13), f"root_hist must be (B,T,13), got {rh.shape}"
        assert fh.ndim == 4 and fh.shape[1:] == (self.T_hist, 4, 3), f"foot_hist must be (B,T,4,3), got {fh.shape}"

        self._append(self.ds_root_hist, rh)
        self._append(self.ds_foot_hist, fh)
        self._append(self.d_contact_hist, ch)


         # ---- fsm_hist (追加) ----
        if fsm_hist_f32 is None:
            # 「必須にしたい」なら raise の方がデバッグしやすい
            # raise RuntimeError("fsm_hist_f32 is None but fsm_hist dataset exists.")
            fsm = np.zeros((1, self.T_hist, self.D_fsm), dtype=np.float32)
        else:
            fsm = to_np(fsm_hist_f32, np.float32)
            assert fsm.ndim == 3 and fsm.shape[1:] == (self.T_hist, self.D_fsm), \
                f"fsm_hist must be (B,T,{self.D_fsm}), got {fsm.shape}"
        self._append(self.d_fsm_hist, fsm)

        # self.n を「真実の長さ」に同期（ズレ防止）
        self.n = self.d_rgb.shape[0]


    def close(self):
        self.f.flush()
        self.f.close()








import torch
import torch.nn.functional as F

def binary_erode(mask_b1hw: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask_b1hw
    k = 2 * r + 1
    inv = (~mask_b1hw).float()
    dil = F.max_pool2d(inv, k, stride=1, padding=r)
    return (dil <= 0.0)

def make_trav_from_hits_z(hits_z_bhw: torch.Tensor,
                          top_band: float,
                          foot_radius_m: float,
                          resolution: float) -> torch.Tensor:
    z = torch.nan_to_num(hits_z_bhw, nan=-1e6, neginf=-1e6, posinf=-1e6)
    zmax = z.amax(dim=(1,2), keepdim=True)
    safe = (z > (zmax - top_band))                      # (B,H,W)
    r = int(round(foot_radius_m / resolution))
    safe = binary_erode(safe[:, None], r)[:, 0]         # (B,H,W)
    return safe

def pick_nearest_safe_xy(trav_bhw: torch.Tensor,
                         x_coords_h: torch.Tensor,
                         y_coords_w: torch.Tensor,
                         ref_xy_b42: torch.Tensor,
                         reach_a=0.25, reach_b=0.18) -> torch.Tensor:
    B,H,W = trav_bhw.shape
    xx = x_coords_h.view(1,H,1).expand(B,H,W)
    yy = y_coords_w.view(1,1,W).expand(B,H,W)

    out = torch.zeros((B,4,2), device=trav_bhw.device, dtype=torch.float32)
    for leg in range(4):
        x0 = ref_xy_b42[:,leg,0].view(B,1,1)
        y0 = ref_xy_b42[:,leg,1].view(B,1,1)
        reach = ((xx-x0)/reach_a)**2 + ((yy-y0)/reach_b)**2 <= 1.0
        cand = trav_bhw & reach
        score = -((xx-x0)**2 + (yy-y0)**2)
        score = score.masked_fill(~cand, -1e9)
        flat = score.view(B, -1)
        idx = torch.argmax(flat, dim=1)
        ix = idx // W
        iy = idx % W
        out[:,leg,0] = x_coords_h[ix]
        out[:,leg,1] = y_coords_w[iy]
        no = (cand.view(B,-1).sum(dim=1) == 0)
        out[no,leg,:] = ref_xy_b42[no,leg,:]
    return out





@torch.no_grad()
def stable_mask(raw, foot_pos_b, lin_thr=0.05, ang_thr=0.2, foot_thr=0.002):
    # raw.scene["robot"].data.root_state_w: (B,13)
    root = raw.scene["robot"].data.root_state_w
    lin = root[:, 7:10].norm(dim=1)
    ang = root[:,10:13].norm(dim=1)
    # foot_pos_b: (B,4,3)
    # 直前との差分は外で見る（foot_thr）
    ok = (lin < lin_thr) & (ang < ang_thr)
    return ok







import torch


from isaaclab.managers import SceneEntityCfg


LEG_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


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

# @torch.no_grad()
# def collect_teacher_dataset(env, out_path: str, n_samples: int,
#                             settle_steps: int = 20,   # 0.4s @50Hz
#                             num_envs_to_save: int | None = None,
#                             top_band: float = 0.03,
#                             foot_radius_m: float = 0.03,
#                             max_wait_steps=200):
#     """
#     env: gym env (ManagerBasedRLEnv でも wrapper でもOK)
#     """

#     raw = getattr(env, "unwrapped", env)

#     # sensors
#     cam = raw.scene.sensors["camera"]
#     ray = raw.scene.sensors["height_scanner"]

#     # ray grid shape from cfg
#     size_xy = ray.cfg.pattern_cfg.size          # [L, W]
#     res = float(ray.cfg.pattern_cfg.resolution)
#     L, W = float(size_xy[0]), float(size_xy[1])
#     Hb = int(round(L / res)) + 1
#     Wb = int(round(W / res)) + 1

#     # BEV coordinate in base frame (GridPattern is centered at (0,0))
#     x_coords = torch.linspace(-L/2, L/2, Hb, device=raw.device)
#     y_coords = torch.linspace(-W/2, W/2, Wb, device=raw.device)

#     # writer
#     writer = TeacherH5Writer(out_path, rgb_hw=(cam.cfg.height, cam.cfg.width), bev_hw=(Hb, Wb))
#     writer.set_attrs(
#         ray_size_xy=np.array([L, W], np.float32),
#         ray_resolution=np.float32(res),
#         top_band=np.float32(top_band),
#         foot_radius_m=np.float32(foot_radius_m),
#         settle_steps=np.int32(settle_steps),
#     )

#     # zero action (wrapper側の action_dim と raw の action_dim がズレることがあるので raw を使うのが安全)
#     # act_dim = raw.action_manager.action_dim

#     act_dim = raw.action_manager.total_action_dim

#     zero = torch.zeros((raw.num_envs, act_dim), device=raw.device)

#     # 何env保存する？
#     B = raw.num_envs if num_envs_to_save is None else min(raw.num_envs, num_envs_to_save)

#     for k in range(n_samples):
#         env.reset()
#         # raw.reset()  # ★EventTermがここで効く



#         stable_count = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.int32)
#         prev_foot = None

#         for t in range(max_wait_steps):
#             _ = env.step(zero)

#             foot = mdp.ee_pos_base_obs(raw)
#             if foot.ndim == 2: foot = foot.view(foot.shape[0],4,3)

#             if prev_foot is None:
#                 prev_foot = foot.clone()

#             dfoot = (foot - prev_foot).norm(dim=-1).max(dim=1).values  # (B,)
#             ok = stable_mask(raw, foot) & (dfoot < 0.002)              # 2mm/frame 例
#             stable_count = torch.where(ok, stable_count + 1, torch.zeros_like(stable_count))
#             prev_foot = foot

#             if (stable_count >= 5).all():   # ★5フレーム連続で安定
#                 break


#         # --- read sensors ---
#         rgb = cam.data.output["rgb"]            # (N,H,W,3) uint8
#         hits_w = ray.data.ray_hits_w            # (N,R,3)
#         hits_w = hits_w.view(raw.num_envs, Hb, Wb, 3)
#         hits_z = hits_w[..., 2]                 # (N,Hb,Wb)

#         # --- proprio debug ---
#         root_state_w = raw.scene["robot"].data.root_state_w  # (N,13)

#         # 実足位置（base座標）がほしい：あなたのmdp.ee_pos_base_obsが使えるならそれで
#         # 例: foot_pos_b = mdp.ee_pos_base_obs(raw)  # (N,4,3)
#         # ここでは「あなたが既に使ってる関数」を呼ぶ前提にしておく
#         from unitree_rl_lab.tasks.locomotion import mdp
#         foot_pos_b = mdp.ee_pos_base_obs(raw)    # (N,4,3)


#         if foot_pos_b.ndim == 2:
#             B, D = foot_pos_b.shape
#             if D == 12:
#                 foot_pos_b = foot_pos_b.view(B, 4, 3)
#             elif D == 8:
#                 # xy だけ来てる場合は z=0 を足して 4x3 にしておく
#                 foot_xy = foot_pos_b.view(B, 4, 2)
#                 z = torch.zeros((B, 4, 1), device=foot_pos_b.device, dtype=foot_pos_b.dtype)
#                 foot_pos_b = torch.cat([foot_xy, z], dim=-1)
#             else:
#                 raise RuntimeError(f"Unexpected foot_pos_b shape: {foot_pos_b.shape}")

#         elif foot_pos_b.ndim == 3:
#             # 期待: [B,4,3]
#             if foot_pos_b.shape[1:] != (4, 3):
#                 raise RuntimeError(f"Unexpected foot_pos_b shape: {foot_pos_b.shape}")
#         else:
#             raise RuntimeError(f"Unexpected foot_pos_b ndim: {foot_pos_b.ndim}")

#         ref_xy = foot_pos_b[..., :2]  # (B,4,2)

#         # --- teacher label ---
#         trav = make_trav_from_hits_z(hits_z, top_band=top_band,
#                                      foot_radius_m=foot_radius_m, resolution=res)  # bool
#         # ref_xy = foot_pos_b[..., :2]  # (N,4,2) 近傍選択の基準（静止ならこれが最強）
#         foot_xy = pick_nearest_safe_xy(trav, x_coords, y_coords, ref_xy)

#         # save batch（先頭B envだけ）
#         writer.append_batch(
#             rgb[:B],
#             hits_z[:B].float(),
#             trav[:B].to(torch.uint8),
#             foot_xy[:B].float(),
#             foot_pos_b[:B].float(),
#             root_state_w[:B].float(),
#         )

#         if (k+1) % 100 == 0:
#             print(f"{k+1}/{n_samples} samples saved")

#     writer.close()
#     print("saved:", out_path)




# @torch.no_grad()
# def collect_teacher_dataset(
#     env,
#     out_path: str,
#     n_samples: int,
#     settle_steps: int = 0,          # もう固定待ち不要なら 0 でもOK
#     max_wait_steps: int = 300,      # ★追加
#     num_envs_to_save: int | None = None,
#     top_band: float = 0.03,
#     foot_radius_m: float = 0.03,
# ):
#     raw = getattr(env, "unwrapped", env)
#     from unitree_rl_lab.tasks.locomotion import mdp  # ★先にimport

#     # sensors
#     cam = raw.scene.sensors["camera"]
#     ray = raw.scene.sensors["height_scanner"]

#     # ray grid shape from cfg
#     size_xy = ray.cfg.pattern_cfg.size
#     res = float(ray.cfg.pattern_cfg.resolution)
#     L, W = float(size_xy[0]), float(size_xy[1])
#     Hb = int(round(L / res)) + 1
#     Wb = int(round(W / res)) + 1

#     x_coords = torch.linspace(-L/2, L/2, Hb, device=raw.device)
#     y_coords = torch.linspace(-W/2, W/2, Wb, device=raw.device)

#     writer = TeacherH5Writer(out_path, rgb_hw=(cam.cfg.height, cam.cfg.width), bev_hw=(Hb, Wb))
#     writer.set_attrs(
#         ray_size_xy=np.array([L, W], np.float32),
#         ray_resolution=np.float32(res),
#         top_band=np.float32(top_band),
#         foot_radius_m=np.float32(foot_radius_m),
#         settle_steps=np.int32(settle_steps),
#         max_wait_steps=np.int32(max_wait_steps),
#     )

#     # # action dim（env側とraw側でズレるなら env.action_space優先が安全）
#     # if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
#     #     act_dim = int(env.action_space.shape[0])
#     # else:
#     #     act_dim = int(raw.action_manager.total_action_dim)

#     # zero = torch.zeros((raw.num_envs, act_dim), device=raw.device)
#     act_dim = int(raw.action_manager.total_action_dim)  # ここは必ず12になるはず
#     zero = torch.zeros((raw.num_envs, act_dim), device=raw.device)

#     Bsave = raw.num_envs if num_envs_to_save is None else min(raw.num_envs, num_envs_to_save)

#     try:
#         saved = 0
#         trials = 0
#         while saved < n_samples:
#             trials += 1
#             _ = env.reset()  # ★raw.reset()は呼ばない

#             # optional 固定待ち
#             for _ in range(settle_steps):
#                 _ = env.step(zero)

#             stable_count = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.int32)
#             prev_foot = None
#             stable = False

#             for t in range(max_wait_steps):
#                 _ = env.step(zero)

#                 foot = mdp.ee_pos_base_obs(raw)
#                 if foot.ndim == 2:
#                     foot = foot.view(foot.shape[0], 4, 3)

#                 if prev_foot is None:
#                     prev_foot = foot.clone()

#                 dfoot = (foot - prev_foot).norm(dim=-1).max(dim=1).values  # (Nenv,)
#                 ok = stable_mask(raw, foot) & (dfoot < 0.002)

#                 stable_count = torch.where(ok, stable_count + 1, torch.zeros_like(stable_count))
#                 prev_foot = foot

#                 # ★保存するenvだけ安定すればOK
#                 if (stable_count[:Bsave] >= 5).all():
#                     stable = True
#                     break

#             if not stable:
#                 # 安定できないサンプルは捨てる（ノイズ源）
#                 continue

#             # --- read sensors (clone推奨) ---
#             rgb = cam.data.output["rgb"]  # (N,H,W,3) uint8
#             rgb = rgb[:Bsave].clone()

#             hits_w = ray.data.ray_hits_w  # (N,R,3)
#             hits_w = hits_w.view(raw.num_envs, Hb, Wb, 3)
#             hits_z = hits_w[..., 2][:Bsave].clone()
#             hits_z = torch.nan_to_num(hits_z, nan=-1.0e6, neginf=-1.0e6, posinf=1.0e6)

#             root_state_w = raw.scene["robot"].data.root_state_w[:Bsave].clone()

#             # foot_pos_b = mdp.ee_pos_base_obs(raw)

#             feet_cfg = SceneEntityCfg(
#                 "robot",
#                 body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # ★順序固定
#             )
#             feet_cfg.resolve(raw.scene)     # ★これで body_ids が入る
#             foot_body_ids = feet_cfg.body_ids  # list[int] or tensor

#             foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)  # (B,4,3)
#             # ref_xy = foot_pos_b[..., :2]    
#             if foot_pos_b.ndim == 2:
#                 B, D = foot_pos_b.shape
#                 foot_pos_b = foot_pos_b.view(B, 4, 3) if D == 12 else foot_pos_b.view(B, 4, 2)  # 必要なら分岐
#                 if foot_pos_b.shape[-1] == 2:
#                     z = torch.zeros((B, 4, 1), device=foot_pos_b.device, dtype=foot_pos_b.dtype)
#                     foot_pos_b = torch.cat([foot_pos_b, z], dim=-1)
#             foot_pos_b = foot_pos_b[:Bsave].clone()

#             ref_xy = foot_pos_b[..., :2]

#             trav = make_trav_from_hits_z(hits_z, top_band=top_band, foot_radius_m=foot_radius_m, resolution=res)
#             foot_xy = pick_nearest_safe_xy(trav, x_coords, y_coords, ref_xy)

#             writer.append_batch(
#                 rgb,                  # torch uint8 OKならこのまま、NGなら .cpu().numpy()
#                 hits_z.float(),
#                 trav.to(torch.uint8),
#                 foot_xy.float(),
#                 foot_pos_b.float(),
#                 root_state_w.float(),
#             )


#             print("[COLLECT DEBUG] ref_xy min/max x:", ref_xy[...,0].min().item(), ref_xy[...,0].max().item(),
#                 "y:", ref_xy[...,1].min().item(), ref_xy[...,1].max().item())

#             print("[COLLECT DEBUG] foot_xy min/max x:", foot_xy[...,0].min().item(), foot_xy[...,0].max().item(),
#                 "y:", foot_xy[...,1].min().item(), foot_xy[...,1].max().item())

#             saved += 1
#             if saved % 100 == 0:
#                 print(f"{saved}/{n_samples} samples saved (trials={trials})")

#     finally:
#         writer.close()
#         print("saved:", out_path)





@torch.no_grad()
def foot_targets_to_actions(
    foot_targets_b,              # (B,4,3) base yaw frame
    nominal_footholds_b,         # (4,3)
    dx_max, dy_max, dz_max
):
    # action = (target - nominal) / max
    B = foot_targets_b.shape[0]
    delta = foot_targets_b - nominal_footholds_b.unsqueeze(0)  # (B,4,3)

    a = torch.zeros((B,4,3), device=foot_targets_b.device, dtype=foot_targets_b.dtype)
    a[...,0] = delta[...,0] / dx_max
    a[...,1] = delta[...,1] / dy_max
    a[...,2] = delta[...,2] / dz_max if dz_max > 1e-6 else 0.0
    a = torch.clamp(a, -1.0, 1.0)
    return a.view(B, 12)  # (B,12)


# @torch.no_grad()
# def collect_teacher_dataset_move_feet(
#     env,
#     out_path: str,
#     n_samples: int,
#     settle_steps: int = 20,
#     move_hold_steps: int = 120,          # 1.2s @50Hz くらいのイメージ
#     reach_tol_m: float = 0.03,          # 2cm以内に近づいたらOK
#     stable_frames: int = 3,
#     max_attempts: int = 10,             # 失敗時にターゲットを引き直す回数
#     target_xy_noise: float = 0.08,      # 8cm（分布を広げる）
#     top_band: float = 0.03,
#     foot_radius_m: float = 0.03,
#     num_envs_to_save: int | None = None,
# ):
#     raw = getattr(env, "unwrapped", env)
#     B = raw.num_envs  # ここは safe_xy のバッチと一致する必要がある


#     cam = raw.scene.sensors["camera"]
#     ray = raw.scene.sensors["height_scanner"]

#     # ray grid
#     size_xy = ray.cfg.pattern_cfg.size
#     res = float(ray.cfg.pattern_cfg.resolution)
#     L, W = float(size_xy[0]), float(size_xy[1])
#     Hb = int(round(L / res)) + 1
#     Wb = int(round(W / res)) + 1

#     x_coords = torch.linspace(-L/2, L/2, Hb, device=raw.device)
#     y_coords = torch.linspace(-W/2, W/2, Wb, device=raw.device)

#     # writer（あなたの既存 writer を使う）
#     writer = TeacherH5Writer(out_path, rgb_hw=(cam.cfg.height, cam.cfg.width), bev_hw=(Hb, Wb))
#     writer.set_attrs(
#         ray_size_xy=np.array([L, W], np.float32),
#         ray_resolution=np.float32(res),
#         top_band=np.float32(top_band),
#         foot_radius_m=np.float32(foot_radius_m),
#         settle_steps=np.int32(settle_steps),
#         move_hold_steps=np.int32(move_hold_steps),
#         reach_tol_m=np.float32(reach_tol_m),
#         target_xy_noise=np.float32(target_xy_noise),
#     )

#     act_dim = raw.action_manager.total_action_dim
#     assert act_dim == 12, f"expected 12D footstep action but got {act_dim}"

#     # Bsave = raw.num_envs if num_envs_to_save is None else min(raw.num_envs, num_envs_to_save)
#     Bsave = 5

#     # ここはあなたのActionTermの nominal / max と一致させる
#     nominal = torch.tensor(
#         [[ 0.25, 0.15, -0.35],
#          [ 0.25,-0.15, -0.35],
#          [-0.25, 0.15, -0.35],
#          [-0.25,-0.15, -0.35]],
#         device=raw.device, dtype=torch.float32
#     )
#     dx_max, dy_max, dz_max = 0.15, 0.1, 0.01

#     # 収集ループ
#     saved = 0
#     env.reset(); raw.reset()

#     stable_count = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.int32)
#     prev_foot = None

#     feet_cfg = SceneEntityCfg(
#         "robot",
#         body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # ★順序固定
#     )
#     feet_cfg.resolve(raw.scene)     # ★これで body_ids が入る
#     foot_body_ids = feet_cfg.body_ids  # list[int] or tensor

#     while saved < n_samples:
#         # --- settle（ベースや足が落ち着くまで） ---
#         zero = torch.zeros((raw.num_envs, act_dim), device=raw.device)
#         for _ in range(settle_steps):
#             _ = env.step(zero)

#         # --- sensors read（trav作成） ---
#         hits_w = ray.data.ray_hits_w.view(raw.num_envs, Hb, Wb, 3)
#         hits_z = hits_w[..., 2]
#         rgb = cam.data.output["rgb"]  # (B,H,W,3)

#         trav = make_trav_from_hits_z(hits_z, top_band=top_band,
#                                      foot_radius_m=foot_radius_m, resolution=res)  # bool (B,Hb,Wb)

#         # --- ターゲットを引いて、低レベルでそこへ寄せる ---
#         ok_env = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)

#         best_good = None
#         best_foot_tgt = None


#         # 失敗時に引き直す
#         for attempt in range(max_attempts):
#             # 1ランダム “狙い” ref_xy（名目 + ノイズ）
#             B = raw.num_envs
#             dev = raw.device
#             env_ids = torch.arange(B, device=dev)

#             # 0:FL,1:FR,2:RL,3:RR
#             # ★順番に：envごとにずらす（分布を速く稼げる）
#             LEG_ORDER = torch.tensor([1, 2, 0, 3], device=dev)  # FR, RL, FL, RR
#             leg_to_move = torch.full((B,), int(LEG_ORDER[saved % 4].item()), device=dev, dtype=torch.long)

#             # もし「全envで同じ脚を動かす」が良いなら：
#             # leg_to_move = torch.full((B,), saved % 4, device=dev, dtype=torch.long)

#             # まず「名目位置を安全セルにスナップ」した固定ターゲットを作る（他脚用）
#             nom_xy = nominal[:, :2].unsqueeze(0).expand(B, -1, -1)        # (B,4,2)
#             safe_nom_xy = pick_nearest_safe_xy(trav, x_coords, y_coords, nom_xy)  # (B,4,2)

#             # 次に「動かす脚だけ」ノイズした ref を作る
#             ref_xy = safe_nom_xy.clone()                                   # (B,4,2) まず固定脚は安全名目
#             noise_one = (torch.rand((B, 2), device=dev) * 2 - 1) * target_xy_noise  # (B,2)
#             ref_xy[env_ids, leg_to_move, :] += noise_one                   # ★動かす脚だけずらす

#             # そのrefを安全セルに射影（動かす脚だけ変わる）
#             safe_xy = pick_nearest_safe_xy(trav, x_coords, y_coords, ref_xy)       # (B,4,2)

#             # 最終ターゲット：他脚は safe_nom_xy で固定、動かす脚だけ safe_xy を採用
#             foot_tgt = nominal.unsqueeze(0).expand(B, -1, -1).clone()       # (B,4,3)
#             foot_tgt[..., :2] = safe_nom_xy
#             foot_tgt[env_ids, leg_to_move, :2] = safe_xy[env_ids, leg_to_move, :2]

#             # foot_tgt[..., 2] は nominal のまま（安定しやすい）

#             # 4 action化してホールド
#             action = foot_targets_to_actions(foot_tgt, nominal, dx_max, dy_max, dz_max)

#             stable_count.zero_()
#             prev_foot = None


#             target_good = min(20, n_samples - saved)

#             success = torch.zeros(B, device=dev, dtype=torch.bool)  # 成功をラッチ
#             stable_count.zero_()
#             prev_foot = None


            

#             for t in range(move_hold_steps):
#                 _ = env.step(action)

#                 # 実足（yaw-base）をあなたの確定版で取得
#                 foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)  # (B,4,3)

#                 # 目標への距離（xy）
#                 # err = torch.norm(foot_pos_b[..., :2] - foot_tgt[..., :2], dim=-1).max(dim=1).values  # (B,)

#                 err = torch.norm(
#                     foot_pos_b[env_ids, leg_to_move, :2] - foot_tgt[env_ids, leg_to_move, :2],
#                     dim=-1
#                 )  # (B,)


#                 # if prev_foot is None:
#                 #     prev_foot = foot_pos_b.clone()
#                 # dfoot = (foot_pos_b - prev_foot).norm(dim=-1).max(dim=1).values
#                 # prev_foot = foot_pos_b


#                 # ok = (err < reach_tol_m) & (dfoot < 0.006)
#                 # stable_count = torch.where(ok, stable_count + 1, torch.zeros_like(stable_count))
#                 # ok_env = stable_count >= stable_frames

#                 # # if ok_env.all():
#                 # #     break


#                 # # ★一度成功したenvは成功として保持（後で少し揺れても消さない）
#                 # success |= ok_env



#                 if prev_foot is None: #only moving leg
#                     prev_foot = foot_pos_b.clone()

#                 dfoot_leg = torch.norm(
#                     foot_pos_b[env_ids, leg_to_move, :2] - prev_foot[env_ids, leg_to_move, :2],
#                     dim=-1
#                 )
#                 prev_foot = foot_pos_b

#                 ok = (err < reach_tol_m) & (dfoot_leg < 0.006)
#                 stable_count = torch.where(ok, stable_count + 1, torch.zeros_like(stable_count))
#                 ok_env = stable_count >= stable_frames

#                 success |= ok_env


#                 # ★成功済みenvは stable_count を固定しておく（揺れで落ちないように）
#                 stable_count = torch.where(success, stable_frames * torch.ones_like(stable_count), stable_count)

#                 if int(success.sum().item()) >= target_good:
#                     break

#                 # if ok_env.any():
#                 #     break


#             # good = success.nonzero(as_tuple=False).squeeze(-1)

#             # # --- hold loop 終了後（attempt内） ---
#             # # good = ok_env.nonzero(as_tuple=False).squeeze(-1)  # (Ngood,)
#             # # 何個貯まったらこのattemptを打ち切るか（効率のため）
#             # need = min(Bsave, n_samples - saved)               # 今回欲しい最大数
#             # # if good.numel() >= max(1, need):                   # 1個でも成功したら保存へ行ける
#             # #     break

#             # if good.numel() >= 1:
#             #     break


#             good = success.nonzero(as_tuple=False).squeeze(-1)

#             # ★成功が1個でもあれば、このattemptは採用
#             if good.numel() > 0:
#                 best_good = good
#                 best_foot_tgt = foot_tgt.detach().clone()

#                 break



            

#             # # 少なくとも保存対象Bsaveが全部OKなら採用
#             # if ok_env[:Bsave].all():
#             #     break

#         # if not ok_env[:Bsave].all():
#         #     # このサイクルは失敗：resetしてやり直し
#         #     env.reset(); raw.reset()
#         #     continue

        
#         # # attemptループを抜けた後（attemptの外）で、最終的にgoodをもう一度評価して保存
#         # good = ok_env.nonzero(as_tuple=False).squeeze(-1)
#         # if good.numel() == 0:
#         #     # このサイクルは全滅 → resetしてやり直し（またはcontinueで次のwhileへ）
#         #     env.reset(); raw.reset()
#         #     continue

#         # # 保存数を決める（成功envが多ければ先頭からK個だけ）
#         # K = min(int(good.numel()), Bsave, n_samples - saved)
#         # good = good[:K]

#         # # ★ここで「動かした後」のセンサを読み直す（重要）
#         # rgb = cam.data.output["rgb"]  # (B,H,W,3)
#         # hits_w = ray.data.ray_hits_w.view(raw.num_envs, Hb, Wb, 3)
#         # hits_z = hits_w[..., 2]
#         # trav = make_trav_from_hits_z(hits_z, top_band=top_band,
#         #                             foot_radius_m=foot_radius_m, resolution=res)

#         # root_state_w = raw.scene["robot"].data.root_state_w  # (B,13)
#         # foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)  # (B,4,3)

#         # # # 成功envだけ保存
#         # # writer.append_batch(
#         # #     rgb[good],
#         # #     hits_z[good].float(),
#         # #     trav[good].to(torch.uint8),
#         # #     foot_tgt[good, :, :2].float(),   # 教師ラベル（ターゲット）
#         # #     foot_pos_b[good].float(),        # 実足（デバッグ）
#         # #     root_state_w[good].float(),
#         # # )

#         if best_good is None:
#             env.reset(); raw.reset()
#             continue

#         good = best_good
#         foot_tgt = best_foot_tgt

#         # 保存数を決める
#         K = min(int(good.numel()), Bsave, n_samples - saved)
#         good = good[:K]

#         # ★動かした後のセンサを読み直す
#         rgb = cam.data.output["rgb"]
#         hits_w = ray.data.ray_hits_w.view(raw.num_envs, Hb, Wb, 3)
#         hits_z = hits_w[..., 2]
#         trav = make_trav_from_hits_z(hits_z, top_band=top_band, foot_radius_m=foot_radius_m, resolution=res)

#         root_state_w = raw.scene["robot"].data.root_state_w
#         foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)


#         for i in good[:K].tolist():
#             writer.append_batch(
#                 rgb[i:i+1],
#                 hits_z[i:i+1].float(),
#                 trav[i:i+1].to(torch.uint8),
#                 foot_tgt[i:i+1, :, :2].float(),
#                 foot_pos_b[i:i+1].float(),
#                 root_state_w[i:i+1].float(),
#             )
#         # saved += K


#         saved += K
#         if saved % 100 == 0:
#             print(f"{saved}/{n_samples} samples saved")


        
#         print("[COLLECT DEBUG] ref_xy min/max x:", ref_xy[...,0].min().item(), ref_xy[...,0].max().item(),
#                 "y:", ref_xy[...,1].min().item(), ref_xy[...,1].max().item())

#         print("[COLLECT DEBUG] foot_xy min/max x:", foot_pos_b[...,0].min().item(), foot_pos_b[...,0].max().item(),
#             "y:", foot_pos_b[...,1].min().item(), foot_pos_b[...,1].max().item())





#         # # --- 保存（この時点で foot_tgt は教師ラベル、foot_pos_b は実達成） ---
#         # root_state_w = raw.scene["robot"].data.root_state_w  # (B,13)
#         # foot_pos_b = get_foot_pos_yawbase(raw, foot_body_ids)  # (B,4,3)

#         # writer.append_batch(
#         #     rgb[:Bsave],
#         #     hits_z[:Bsave].float(),
#         #     trav[:Bsave].to(torch.uint8),
#         #     foot_tgt[:Bsave, :, :2].float(),   # ★教師ラベル（安全セル上ターゲット）
#         #     foot_pos_b[:Bsave].float(),        # 実足（デバッグ/補助）
#         #     root_state_w[:Bsave].float(),
#         # )

        


#         # saved += 1
#         # if saved % 100 == 0:
#         #     print(f"{saved}/{n_samples} samples saved")

#     writer.close()
#     print("saved:", out_path)




import torch.nn.functional as Func



@torch.no_grad()
def collect_teacher_dataset_move_feet(
    env,
    out_path: str,
    n_samples: int,
    settle_steps: int = 20,
    move_hold_steps: int = 240,
    reach_tol_m: float = 0.04,
    stable_frames: int = 3,
    max_attempts: int = 10,
    target_xy_noise: float = 0.08,
    top_band: float = 0.03,
    foot_radius_m: float = 0.03,
    num_envs_to_save: int | None = None,
    dfoot_thresh: float = 0.01,         # ★脚だけdfoot
):
    raw = getattr(env, "unwrapped", env)
    dev = raw.device
    B = raw.num_envs

    cam = raw.scene.sensors["camera"]
    ray = raw.scene.sensors["height_scanner"]

    # ray grid
    size_xy = ray.cfg.pattern_cfg.size
    res = float(ray.cfg.pattern_cfg.resolution)
    L, W = float(size_xy[0]), float(size_xy[1])
    Hb = int(round(L / res)) + 1
    Wb = int(round(W / res)) + 1
    x_coords = torch.linspace(-L/2, L/2, Hb, device=dev)
    y_coords = torch.linspace(-W/2, W/2, Wb, device=dev)

    writer = TeacherH5Writer(out_path, rgb_hw=(cam.cfg.height, cam.cfg.width), bev_hw=(Hb, Wb))
    writer.set_attrs(
        ray_size_xy=np.array([L, W], np.float32),
        ray_resolution=np.float32(res),
        top_band=np.float32(top_band),
        foot_radius_m=np.float32(foot_radius_m),
        settle_steps=np.int32(settle_steps),
        move_hold_steps=np.int32(move_hold_steps),
        reach_tol_m=np.float32(reach_tol_m),
        target_xy_noise=np.float32(target_xy_noise),
        dfoot_thresh=np.float32(dfoot_thresh),
    )

    act_dim = raw.action_manager.total_action_dim
    assert act_dim == 12, f"expected 12D footstep action but got {act_dim}"
    zero = torch.zeros((B, act_dim), device=dev)

    Bsave = B if num_envs_to_save is None else min(B, num_envs_to_save)
    Bsave = 16


    nominal = torch.tensor(
        [[ 0.25, 0.15, -0.35],
         [ 0.25,-0.15, -0.35],
         [-0.25, 0.15, -0.35],
         [-0.25,-0.15, -0.35]],
        device=dev, dtype=torch.float32
    )
    dx_max, dy_max, dz_max = 0.15, 0.10, 0.01

    # feet ids（毎回resolveしない）
    feet_cfg = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    feet_cfg.resolve(raw.scene)
    foot_body_ids = feet_cfg.body_ids

    LEG_ORDER = torch.tensor([1, 2, 0, 3], device=dev)  # FR, RL, FL, RR

    env.reset(); raw.reset()

    saved = 0
    env_ids = torch.arange(B, device=dev)

    while saved < n_samples:
        # --- settle ---
        for _ in range(settle_steps):
            _ = env.step(zero)

        # ここから attempt で「決定→実行→成功なら“決定時点”を保存」
        best_good = None
        best_foot_tgt_label = None

        # --- attempt ---
        for attempt in range(max_attempts):

            # ====== (A) 決定直前のスナップショット（入力として保存するもの） ======
            # センサ（決定前）
            hits_w0 = ray.data.ray_hits_w.view(B, Hb, Wb, 3)
            hits_z0 = hits_w0[..., 2].clone()                    # (B,Hb,Wb)
            rgb0 = cam.data.output["rgb"].clone()                # (B,H,W,3)

            trav0 = make_trav_from_hits_z(
                hits_z0, top_band=top_band, foot_radius_m=foot_radius_m, resolution=res
            )  # bool (B,Hb,Wb)

            root0 = raw.scene["robot"].data.root_state_w.clone()  # (B,13)

            # 決定前の足位置（yaw-base）
            foot0 = get_foot_pos_yawbase(raw, foot_body_ids).clone()  # (B,4,3)
            foot0_xy = foot0[..., :2]                                 # (B,4,2)

            # ====== (B) 4脚ターゲットを決める（ラベル用） ======
            # 基本は「全脚ホールド＝現状維持」
            foot_tgt_label = foot0.clone()  # (B,4,3) これが教師ラベル（保存する）
            leg_to_move = torch.full((B,), int(LEG_ORDER[saved % 4].item()),
                                     device=dev, dtype=torch.long)

            # 動かす脚だけノイズして安全セルへ
            ref_xy = foot0_xy.clone()
            noise = (torch.rand((B, 2), device=dev) * 2 - 1) * target_xy_noise
            ref_xy[env_ids, leg_to_move, :] += noise

            safe_xy = pick_nearest_safe_xy(trav0, x_coords, y_coords, ref_xy)  # (B,4,2)
            foot_tgt_label[env_ids, leg_to_move, :2] = safe_xy[env_ids, leg_to_move, :2]

            # ====== (C) 実行用コマンドは「クリップ後」を使う（成功判定が破綻しないように） ======
            foot_tgt_cmd = foot_tgt_label.clone()
            delta = foot_tgt_cmd - nominal.unsqueeze(0)  # (B,4,3)
            delta[..., 0] = delta[..., 0].clamp(-dx_max, dx_max)
            delta[..., 1] = delta[..., 1].clamp(-dy_max, dy_max)
            delta[..., 2] = delta[..., 2].clamp(-dz_max, dz_max)
            foot_tgt_cmd = nominal.unsqueeze(0) + delta

            action = foot_targets_to_actions(foot_tgt_cmd, nominal, dx_max, dy_max, dz_max)

            # ====== (D) 成功判定（“動かす脚だけ”で判定） ======
            success = torch.zeros(B, device=dev, dtype=torch.bool)
            stable_count = torch.zeros(B, device=dev, dtype=torch.int32)
            prev_foot = None

            for t in range(move_hold_steps):
                _ = env.step(action)

                foot = get_foot_pos_yawbase(raw, foot_body_ids)  # (B,4,3)

                # 動かす脚だけの誤差（cmdに対して）
                err = torch.norm(
                    foot[env_ids, leg_to_move, :2] - foot_tgt_cmd[env_ids, leg_to_move, :2],
                    dim=-1
                )  # (B,)

                if prev_foot is None:
                    prev_foot = foot.clone()

                dfoot_leg = torch.norm(
                    foot[env_ids, leg_to_move, :2] - prev_foot[env_ids, leg_to_move, :2],
                    dim=-1
                )  # (B,)
                prev_foot = foot

                ok = (err < reach_tol_m) & (dfoot_leg < dfoot_thresh)
                stable_count = torch.where(ok, stable_count + 1, torch.zeros_like(stable_count))
                ok_env = stable_count >= stable_frames
                success |= ok_env

                # できるだけ複数集める（Bsaveぶん揃ったら打ち切り）
                target_good = min(Bsave, B, n_samples - saved)
                if int(success.sum().item()) >= target_good:
                    break

            good = success.nonzero(as_tuple=False).squeeze(-1)

            if good.numel() > 0:
                # ★この attempt を採用：スナップショット入力 + ラベルを保存したいので退避
                best_good = good
                best_foot_tgt_label = foot_tgt_label.detach().clone()
                best_foot_tgt_cmd = foot_tgt_cmd.detach().clone() 

                # スナップショット類も “採用attemptのもの” を保持したいので再利用できるように上書き
                best_rgb0 = rgb0
                best_hits_z0 = hits_z0
                best_trav0 = trav0
                best_root0 = root0
                best_foot0 = foot0
                break

        # attempt 全滅
        if best_good is None:
            env.reset(); raw.reset()
            continue

        # --- 保存（決定前スナップショットを保存する） ---
        good = best_good
        K = min(int(good.numel()), Bsave, n_samples - saved)
        good = good[:K]

        # Writerが可変バッチOKならまとめて1回で書くのが速い
        # だめなら下の for 1件ずつへ
        try:
            writer.append_batch(
                best_rgb0[good],
                best_hits_z0[good].float(),
                best_trav0[good].to(torch.uint8),
                # best_foot_tgt_label[good, :, :2].float(),   
                best_foot_tgt_cmd[good, :, :2].float(),# ★教師ラベル（決定時点）
                best_foot0[good].float(),                   # ★決定時点の実足（proprioの一部として）
                best_root0[good].float(),                   # ★決定時点のproprio
            )
        except Exception:
            for i in good.detach().cpu().tolist():
                writer.append_batch(
                    best_rgb0[i:i+1],
                    best_hits_z0[i:i+1].float(),
                    best_trav0[i:i+1].to(torch.uint8),
                    # best_foot_tgt_label[i:i+1, :, :2].float(),
                    best_foot_tgt_cmd[good, :, :2].float(),# ★教師ラベル（決定時点）
                    best_foot0[i:i+1].float(),
                    best_root0[i:i+1].float(),
                )

        saved += K
        if saved % 100 == 0:
            print(f"[COLLECT] {saved}/{n_samples} saved")

        print("[COLLECT DEBUG] ref_xy min/max x:", ref_xy[...,0].min().item(), ref_xy[...,0].max().item(),
        "y:", ref_xy[...,1].min().item(), ref_xy[...,1].max().item())

        # print("[COLLECT DEBUG] foot_xy min/max x:", foot_pos_b[...,0].min().item(), foot_pos_b[...,0].max().item(),
        #     "y:", foot_pos_b[...,1].min().item(), foot_pos_b[...,1].max().item())

    writer.close()
    print("saved:", out_path)







def _unwrap_reset_out(reset_out):
    # gymnasium: (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out

def _unwrap_step_out(step_out):
    # rsl-rl wrapper: (obs, rew, done, info)
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs, rew, done, info = step_out
        return obs, rew, done, info

    # gymnasium style: (obs, rew, terminated, truncated, info)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, rew, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, rew, done, info

    raise RuntimeError(f"Unexpected step() return: type={type(step_out)}, len={getattr(step_out,'__len__',lambda:None)()}")



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

def yawbase_to_base(p_y, root_state_w, yaw_quat_fn, quat_apply_fn):
    """
    p_y: (B,4,3) yaw-base 相対点
    戻り: (B,4,3) base(フル姿勢) 相対点
    """
    base_pos = root_state_w[:, 0:3]
    q_full   = root_state_w[:, 3:7]
    q_yaw    = yaw_quat_fn(q_full)

    # yaw-base -> world
    p_w = base_pos[:, None, :] + quat_apply_fn(q_yaw[:, None, :], p_y)

    # world -> base
    p_b = quat_apply_fn(quat_conjugate(q_full)[:, None, :], p_w - base_pos[:, None, :])
    return p_b









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




@torch.no_grad()
def collect_teacher_dataset_walk(
    env,
    policy,
    out_path: str,
    n_samples: int,
    max_steps: int = 10000000,
    settle_steps: int = 0,
    num_envs_to_save: int | None = None,
    top_band: float = 0.03,
    foot_radius_m: float = 0.03,
    T_hist: int = 64,
    
):
    raw = getattr(env, "unwrapped", None)
    if raw is None:
        raw = getattr(env, "env", None) or getattr(env, "_env", None) or env
    dev = raw.device
    B = raw.num_envs

    cam = raw.scene.sensors["camera"]
    ray = raw.scene.sensors["height_scanner"]

    # 例：コマンドterm取得（名前はあなたの設定に合わせて）
    # cmd_term: MultiLegBaseCommand3 = raw.command_manager._terms["step_fr_to_block"]
    cmd_term = raw.command_manager._terms["step_fr_to_block"]


    # ray grid (あなたの既存と同じ)
    size_xy = ray.cfg.pattern_cfg.size
    res = float(ray.cfg.pattern_cfg.resolution)
    L, W = float(size_xy[0]), float(size_xy[1])
    Hb = int(round(L / res)) + 1
    Wb = int(round(W / res)) + 1

    writer = TeacherH5Writer(out_path, rgb_hw=(cam.cfg.height, cam.cfg.width), bev_hw=(Hb, Wb), T_hist = T_hist)
    writer.set_attrs(
        ray_size_xy=np.array([L, W], np.float32),
        ray_resolution=np.float32(res),
        top_band=np.float32(top_band),
        foot_radius_m=np.float32(foot_radius_m),
        settle_steps=np.int32(settle_steps),
        # move_hold_steps=np.int32(move_hold_steps),
        # reach_tol_m=np.float32(reach_tol_m),
        # target_xy_noise=np.float32(target_xy_noise),
        # dfoot_thresh=np.float32(dfoot_thresh),
    )


    Bsave = B if num_envs_to_save is None else min(B, num_envs_to_save)
    Bsave = min(Bsave, 16)

    # env.reset(); raw.reset()


    feet_cfg = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    feet_cfg.resolve(raw.scene)
    foot_body_ids = feet_cfg.body_ids


    # --- Contact sensor（Rewardと同じ形式で取る）---
    contact_sensor = raw.scene.sensors["contact_forces"]   # あなたの設定名に合わせる
    sensor_body_names = contact_sensor.body_names

    leg_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    sensor_cols = [sensor_body_names.index(leg) for leg in leg_order]

    CONTACT_THRESH = 8.0  # Rewardと同じ値にする




    sensor_cols = []
    for leg in leg_order:
        if leg not in sensor_body_names:
            raise RuntimeError(f"{leg} not in contact_sensor.body_names: {sensor_body_names}")
        sensor_cols.append(sensor_body_names.index(leg))
    sensor_cols = torch.tensor(sensor_cols, device=dev, dtype=torch.long)  # (4,)

    def get_contacts_u8():
        F = contact_sensor.data.net_forces_w          # (B, Ns, 3)
        Fz = F.index_select(1, sensor_cols)[:, :, 2]  # (B,4)
        contacts = (Fz.abs() > contact_threshold)     # (B,4) bool
        return contacts.to(torch.uint8)               # (B,4) u8
    

    # =========================
    # Proprio history ring buffer (post-step)
    # =========================
    T = int(T_hist)
    ptr = 0  # 次に書くスロット（0..T-1）

    # root: (B,T,13) , foot: (B,T,4,3)
    root_hist = torch.zeros((B, T, 13), device=dev, dtype=torch.float32)
    foot_hist = torch.zeros((B, T, 4, 3), device=dev, dtype=torch.float32)
    # (B,T,4) 0/1 を保存したいので uint8 が軽い
    contact_hist = torch.zeros((B, T, 4), device=dev, dtype=torch.uint8)

    


    # def hist_write(root13_b13, foot_b43, contact_b4_u8):
    #     nonlocal ptr
    #     root_hist[:, ptr, :] = root13_b13
    #     foot_hist[:, ptr, :, :] = foot_b43
    #     contact_hist[:, ptr, :] = contact_b4_u8
    #     ptr = (ptr + 1) % T

    # def hist_gather(env_ids):
    #     t = torch.arange(T, device=dev)
    #     idx = (ptr + t) % T
    #     r = root_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,13)
    #     f = foot_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,4,3)
    #     c = contact_hist.index_select(0, env_ids).index_select(1, idx)   # (Bu,T,4)
    #     return r, f, c
    
    

    # def hist_fill(done_ids, root13_now, foot43_now, contact_now_u8):
    #     # done_ids: (Nd,) long
    #     # root13_now: (B,13), foot43_now: (B,4,3), contact_now: (B,4)
    #     contact_now_u8 = contact_now_u8.to(torch.uint8)
    #     for t in range(T_hist):
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
        t = torch.arange(T, device=dev)
        idx = (ptr + t) % T
        r = root_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,13)
        f = foot_hist.index_select(0, env_ids).index_select(1, idx)      # (Bu,T,4,3)
        c = contact_hist.index_select(0, env_ids).index_select(1, idx)   # (Bu,T,4)
        s = fsm_hist.index_select(0, env_ids).index_select(1, idx)       # (Bu,T,20)
        return r, f, c, s

    def hist_fill(done_ids, root13_now, foot43_now, contact_now_u8):
        contact_now_u8 = contact_now_u8.to(torch.uint8)
        # fsm は reset直後はゼロで埋める
        fsm0 = torch.zeros((done_ids.numel(), D_FSM), device=dev, dtype=torch.float32)
        for t in range(T):
            root_hist[done_ids, t, :] = root13_now[done_ids]
            foot_hist[done_ids, t, :, :] = foot43_now[done_ids]
            contact_hist[done_ids, t, :] = contact_now_u8[done_ids]
            fsm_hist[done_ids, t, :] = fsm0




    


    # ===== params (Reward/HoldSimpleと一致させる) =====
    DT = float(raw.step_dt)
    T_HOLD  = 0.2
    DROPOUT = 0.2

    F_ON  = 10.0
    F_OFF = 6.0

    NEAR_ON  = 0.16
    NEAR_OFF = 0.20
    USE_XYZ  = True

    # ===== states =====
    hold_t = torch.zeros((B,), device=dev)
    bad_t  = torch.zeros((B,), device=dev)
    cmd_age = torch.zeros((B,), device=dev)

    contact_state = torch.zeros((B,4), dtype=torch.bool, device=dev)
    near_state    = torch.zeros((B,4), dtype=torch.bool, device=dev)

    stable_t_contact = torch.zeros((B,4), device=dev)
    stable_t_near = torch.zeros((B,4), device=dev)


    # 直前のcmd_version（cmd_age更新用）
    cmd_ver_prev2 = cmd_term.cmd_version.clone()


    # ===== fsm hist =====
    # [cmd_age, hold_t, bad_t, armed, near4, contact4, stableC4, stableN4] = 1+1+1+1+4+4+4+4=20
    D_FSM = 20
    fsm_hist = torch.zeros((B, T_hist, D_FSM), device=dev, dtype=torch.float32)

    def fsm_write(fsm_bD):
        nonlocal ptr
        fsm_hist[:, ptr, :] = fsm_bD








    # obs = env.get_observations()
    # obs = env.reset()
    obs = _unwrap_reset_out(env.reset()) # vec env

    root13 = raw.scene["robot"].data.root_state_w[:, :13].clone()
    foot43 = get_foot_pos_yawbase(raw, foot_body_ids).clone()
    F = contact_sensor.data.net_forces_w[:, sensor_cols, :]
    Fz = F[..., 2].abs()
    contact = (Fz > CONTACT_THRESH).to(torch.uint8)

    # contact = torch.zeros((B,4), device=dev, dtype=torch.uint8)


    hist_fill(torch.arange(B, device=dev), root13, foot43, contact)
    # hist_fill(torch.arange(B, device=dev), root13, foot43)




    # 例：goal_w を作れるならループ前に初期化
    goal_w = torch.stack([cmd_term._debug_goal_w_cache[leg] for leg in LEG_ORDER], dim=1)  # (B,4,3)
    goal_w_prev = goal_w.detach().clone()




    # cmd_ver_prev = cmd_term.cmd_version.clone()

    cmd_prev_update_y = base_cmd_to_yawbase(
        cmd_term.command.view(B,4,3).clone(),
        raw.scene["robot"].data.root_state_w.clone(),
        yaw_quat, quat_apply
    ).detach()
    cmd_ver_prev = cmd_term.cmd_version.clone()
    # cmd_prev_update = cmd_term.command.view(B,4,3).clone()   # 「前回更新時」のcmd
    cmd_prev_update_b = cmd_term.command.view(B,4,3).detach().clone()






    # settle（下位ポリシーで歩行を安定させる / あるいはゼロでも）
    for _ in range(settle_steps):
        # ここは「下位ポリシーのaction」を入れるのが本筋
        # action = low_level_policy(obs)
        # obs, rew, done, info = env.step(action)

        # obs = raw.get_observations()
        # obs = env.reset()
        # act = policy(get_policy_obs(obs))
        # obs, rew, done, info = env.step(act)
        # _ = env.step(raw.action_manager.action)  # ←仮。あなたの実際の流れに合わせて差し替え

        act = policy(obs)
        obs, _, _, _ = _unwrap_step_out(env.step(act))

    phase_prev = cmd_term.phase.clone()
    cmd_prev = cmd_term.command.view(B, 4, 3).clone()

    saved = 0

    for step in range(max_steps):
        # ---- 低レベルで1step進める（ここをあなたの下位実行に合わせる） ----
        # action = low_level_policy(obs)
        # obs, rew, done, info = env.step(action)
        # obs = raw.get_observations()
        # obs = env.reset()
        # act = policy(get_policy_obs(obs))
        # obs, rew, done, info = env.step(act)
        # _ = env.step(raw.action_manager.action)  # 仮
        
        


        act = policy(obs)
        obs, rew, done, info = _unwrap_step_out(env.step(act))


        # post-stepの状態を履歴に積む（保存タイミングと揃える）
        root13 = raw.scene["robot"].data.root_state_w[:, :13].clone()
        foot43 = get_foot_pos_yawbase(raw, foot_body_ids).clone()

        F = contact_sensor.data.net_forces_w[:, sensor_cols, :]   # (B,4,3)
        Fz = F[..., 2].abs()
        contact_b4 = (Fz > CONTACT_THRESH).to(torch.uint8)        # (B,4) 0/1

        # 履歴にpush
        # hist_write(root13, foot43, contact_b4)
        # hist_write(root13, foot43)




        # --- cmd_updated / cmd_age ---
        cmd_ver_now2 = cmd_term.cmd_version.clone()
        # cmd_updated2 = (cmd_ver_now2 != cmd_ver_prev2) & (~done)  # (B,) bool

        # cmd_age = torch.where(cmd_updated2, torch.zeros_like(cmd_age), cmd_age + DT)

        cmd_updated2 = ((cmd_ver_now2 != cmd_ver_prev2) & (~done)).bool()
        cmd_age = torch.where(cmd_updated2, torch.zeros_like(cmd_age), cmd_age + DT)
        cmd_ver_prev2 = cmd_ver_now2

        


        # --- contact raw -> hysteresis contact_state ---
        F = contact_sensor.data.net_forces_w[:, sensor_cols, :]   # (B,4,3)
        Fz = F[..., 2].abs()                                      # (B,4)
        contact_on  = (Fz > F_ON)
        contact_off = (Fz < F_OFF)
        contact_state_new = torch.where(contact_state, ~contact_off, contact_on)
        same_c = (contact_state_new == contact_state)
        stable_t_contact = torch.where(same_c, stable_t_contact + DT, torch.zeros_like(stable_t_contact))
        contact_state = contact_state_new

        # --- feet in base(full) で dist を計算（Rewardの定義に合わせるのが重要） ---
        feet_b = get_foot_pos_basefull(raw, foot_body_ids)        # (B,4,3)
        cmd_b  = cmd_term.command.view(B,4,3)                     # (B,4,3) base(full)

        dist = (feet_b[...,:2] - cmd_b[...,:2]).norm(dim=-1)  # (B,4)

        # --- hysteresis near_state ---
        near_on  = (dist < NEAR_ON)
        near_off = (dist > NEAR_OFF)
        near_state_new = torch.where(near_state, ~near_off, near_on)

        same_n = (near_state_new == near_state)
        stable_t_near = torch.where(same_n, stable_t_near + DT, torch.zeros_like(stable_t_near))
        near_state = near_state_new

        # --- good_all + hold/bad/dropout ---
        good_legs = contact_state & near_state            # (B,4)
        good_all  = good_legs.all(dim=-1)                 # (B,)

        bad_t  = torch.where(good_all, torch.zeros_like(bad_t), bad_t + DT)
        hold_t = torch.where(good_all, hold_t + DT, hold_t)
        hold_t = torch.where(bad_t > DROPOUT, torch.zeros_like(hold_t), hold_t)

        armed = (hold_t >= T_HOLD).float()                # (B,) 0/1（phase無しの「更新直前サイン」）



        # --- fsm feature vector ---
        fsm = torch.cat([
            cmd_age.unsqueeze(-1),
            hold_t.unsqueeze(-1),
            bad_t.unsqueeze(-1),
            armed.unsqueeze(-1),
            near_state.float(),
            contact_state.float(),
            stable_t_contact,
            stable_t_near,
        ], dim=-1)  # (B,20)


        contact_u8 = contact_state.to(torch.uint8)  # (B,4)


        # fsm_write(fsm)
        hist_write(root13, foot43, contact_u8, fsm)








        # done: [B] bool
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            # reset由来の変化を「フェーズ進行」として数えない
            # phase_prev[done_ids] = cmd_term.phase[done_ids]
            # cmd_prev[done_ids] = cmd_term.command.view(B,4,3)[done_ids]

            # cmd_prev_update[done_ids] = cmd_term.command.view(B,4,3)[done_ids]


            root_r = raw.scene["robot"].data.root_state_w.clone()
            cmd_b_r = cmd_term.command.view(B,4,3).clone()
            cmd_y_r = base_cmd_to_yawbase(cmd_b_r, root_r, yaw_quat, quat_apply)

            cmd_prev_update_y[done_ids] = cmd_y_r[done_ids].detach()
            cmd_ver_prev[done_ids]      = cmd_term.cmd_version[done_ids]
            cmd_ver_prev2[done_ids]     = cmd_term.cmd_version[done_ids]
            cmd_prev_update_b[done_ids] = cmd_term.command.view(B,4,3)[done_ids].detach()




            hold_t[done_ids] = 0.0
            bad_t[done_ids]  = 0.0
            cmd_age[done_ids]= 0.0

            contact_state[done_ids] = False
            near_state[done_ids]    = False

            stable_t_contact[done_ids] = 0.0
            stable_t_near[done_ids]    = 0.0

            # cmd_version基準も同期（cmd_ageの誤検知防止）


            # reset後に履歴を同値埋め
            root13_r = raw.scene["robot"].data.root_state_w[:, :13].clone()
            foot43_r = get_foot_pos_yawbase(raw, foot_body_ids).clone()
            # hist_fill(done_ids, root13_r, foot43_r)

            # F = contact_sensor.data.net_forces_w[:, sensor_cols, :]
            # Fz = F[..., 2].abs()
            # contact_r = (Fz > CONTACT_THRESH).to(torch.uint8)

            # hist_fill(done_ids, root13_r, foot43_r, contact_r)

            # contact_r = contact_state.to(torch.uint8)
            contact_r = torch.zeros((B,4), device=dev, dtype=torch.uint8)

            hist_fill(done_ids, root13_r, foot43_r, contact_r)




        # ---- step後：フェーズ進行を検出 ----
        # phase_now = cmd_term.phase
        # advanced = (phase_now != phase_prev)  # [B] bool


        # ---- ここが新トリガ ----
        cmd_ver_now = cmd_term.cmd_version
        cmd_updated = (cmd_ver_now != cmd_ver_prev) & (~done)     # [B]

        upd_ids = cmd_updated.nonzero(as_tuple=False).squeeze(-1)


        # cmd_now = cmd_term.command.view(B, 4, 3)

        # # 例：XYだけで「どれだけ変わったか」を測る（脚ごと→最大）
        # d = torch.norm(cmd_now[:, :, :2] - cmd_prev[:, :, :2], dim=-1)      # [B,4]
        # cmd_changed = d.max(dim=-1).values > 4e-2                            # [B]

        # # done は保存しない（resetを誤検知しない）
        # advanced = cmd_changed & (~done)
        # # もし step_out が terminated/truncated の場合は done を合成して作る


        # if advanced.any():
        # if cmd_updated.any():
        if upd_ids.numel() > 0:


            # idx = advanced.nonzero(as_tuple=False).squeeze(-1)
            # idx = cmd_updated.nonzero(as_tuple=False).squeeze(-1)
            idx = upd_ids


            # 保存数調整
            K = min(int(idx.numel()), Bsave, n_samples - saved)
            idx = idx[:K]

            # 入力スナップショット（この時点のセンサ・proprio）
            hits_w = ray.data.ray_hits_w.view(B, Hb, Wb, 3)
            hits_z = hits_w[..., 2].clone()
            rgb0   = cam.data.output["rgb"].clone()
            root0  = raw.scene["robot"].data.root_state_w.clone()
            foot0 = get_foot_pos_yawbase(raw, foot_body_ids).clone()
            # foot0  = get_foot_pos_yawbase(raw, SceneEntityCfg("robot", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]).resolve(raw.scene) or None)  # ←あなたの既存関数を使ってOK

            # 教師ラベル：この時点の “次フェーズ目標”
            # cmd_now = cmd_term.command.view(B, 4, 3)
            cmd_now_b = cmd_term.command.view(B,4,3)              # base(フル姿勢)のはず
            root0     = raw.scene["robot"].data.root_state_w.clone()

            cmd_now = base_cmd_to_yawbase(cmd_now_b, root0, yaw_quat, quat_apply)   # (B,4,3)


            # 変更脚マスク（cmd_term側に moved_mask があればそれを使うのが正確）
            moved = None
            # if hasattr(cmd_term, "moved_mask"):
            #     moved = cmd_term.moved_mask
            # else:
            #     moved = (torch.norm(cmd_now - cmd_prev, dim=-1) > 2e-2)  # [B,4]


            cmd_b  = cmd_term.command.view(B,4,3)
            cmd_y  = base_cmd_to_yawbase(cmd_b, root0, yaw_quat, quat_apply)   # (B,4,3)

            # moved = (torch.norm(cmd_y[:, :, :2] - cmd_prev_update_y[:, :, :2], dim=-1) > 2e-2)  # 


            cmd_now_b = cmd_term.command.view(B,4,3)
            # moved = (torch.norm(cmd_now_b[:, :, :2] - cmd_prev_update_b[:, :, :2], dim=-1) > 6e-2)
            moved = Func.one_hot(cmd_term.swing_leg, num_classes=4).to(torch.uint8)   # (B,4) 0/1



            # cmd_term._debug_goal_w_cache[leg] が (B,3) の想定
            # goal_w = torch.stack([cmd_term._debug_goal_w_cache[leg] for leg in LEG_ORDER], dim=1)  # (B,4,3)

            # d = torch.norm(goal_w[:, :, :2] - goal_w_prev[:, :, :2], dim=-1)  # (B,4)
            # moved = d > 1e-3  # 1mm
            # goal_w_prev = goal_w.clone()


            trav_u8 = torch.zeros((B, Hb, Wb), device=dev, dtype=torch.uint8)  # 使わないならダミーでOK


            # 保存（例：append_batchに moved/phase を増やすなら writer 側も拡張）
           

            # writer.append_batch(
            #     rgb0[idx],
            #     hits_z[idx].float(),
            #     trav_u8[idx],                  # ★(K,Hb,Wb)
            #     cmd_now[idx, :, :2].float(),   # ★(K,4,2)
            #     foot0[idx].float(),
            #     root0[idx].float(),
            #     moved[idx],
            # )

            # root_seq, foot_seq = hist_gather(idx)  # (K,T,13), (K,T,4,3)
            # root_seq, foot_seq, cont_seq = hist_gather(idx)  # (K,T,13),(K,T,4,3),(K,T,4)
            root_seq, foot_seq, cont_seq, fsm_seq = hist_gather(idx)


            swing_leg_u8 = cmd_term.swing_leg[idx].to(torch.uint8)          # (K,)
            swing_onehot_f32 = cmd_term.swing_onehot_vec[idx].to(torch.float32)  # (K,4) こっちでも可


            # fsm_seq = fsm_hist.index_select(0, idx).index_select(1, idx_time)  # (K,T,D_FSM)




            writer.append_batch(
                rgb_u8=rgb0[idx],
                hits_z_f32=hits_z[idx].float(),
                trav_u8=trav_u8[idx],
                cmd_xy_f32=cmd_now[idx, :, :2].float(),   # ★教師
                foot_pos_b_f32=foot0[idx].float(),        # ★入力（実足）
                root_state_w_f32=root0[idx].float(),
                moved_u8=moved[idx],
                root_hist_f32=root_seq,      # ★追加 (K,T,13)
                foot_hist_f32=foot_seq,      # ★追加 (K,T,4,3)
                contact_hist_u8=cont_seq,
                fsm_hist_f32=fsm_seq, 
                # swing_leg_u8=swing_leg_u8,   
            )

            saved += K
            if saved >= n_samples:
                break

            cmd_ver_prev[upd_ids] = cmd_ver_now[upd_ids]
            # cmd_prev_update[upd_ids] = cmd_now[upd_ids].detach()
            cmd_prev_update_y[upd_ids] = cmd_y[upd_ids].detach()
            cmd_prev_update_b[upd_ids] = cmd_now_b[upd_ids].detach()




            if saved % 100 == 0:
                print(f"[COLLECT] {saved}/{n_samples} saved")

            print("[COLLECT DEBUG] ref_xy min/max x:")

            ph = cmd_term.phase[idx]
            print(f"[COLLECT DEBUG] saved_env={idx.tolist()} phase={ph.tolist()} done={done[idx].tolist()}")


        # prev更新（advancedの有無に関係なく）
        # phase_prev = phase_now.clone()
        # cmd_prev = cmd_term.command.view(B, 4, 3).clone()

    writer.close()






def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)



    test_cfg = agent_cfg.to_dict()
    # # 2. 次に、その「辞書」の中身をクラスオブジェクトで上書きします
    # #    (辞書なので、アクセスは[]を使います)
    # test_cfg["policy"]["class_name"] = VisionMLPActorCritic
    # test_cfg["policy"]["class_name"] = LocoTransformerActorCritic
    # test_cfg["policy"]["class_name"] = LocoTransformerFinetune
    # test_cfg["policy"]["class_name"] = LocoTransformerHFP
    test_cfg["policy"]["class_name"] = MlpHFP
    # test_cfg["policy"]["class_name"] = VisionHighLevelAC
    # test_cfg["policy"]["class_name"] = VisionHighRNN



    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner = OnPolicyRunner(env, test_cfg, log_dir=log_dir, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # ★これが「policy（callable）」になる
    inference_policy = ppo_runner.get_inference_policy(device=agent_cfg.device)
    # 以降 inference_policy(obs) -> actions が返る



    # ===== Teacher data collection mode =====
    # if getattr(args_cli, "collect_teacher", False):
    #     collect_teacher_dataset_move_feet(
    #         env,
    #         out_path=getattr(args_cli, "teacher_out", "teacher_static.h5"),
    #         n_samples=getattr(args_cli, "teacher_samples", 100),
    #         settle_steps=getattr(args_cli, "teacher_settle_steps", 20),
    #         num_envs_to_save=getattr(args_cli, "teacher_save_envs", 64),
    #         top_band=getattr(args_cli, "teacher_top_band", 0.03),
    #         foot_radius_m=getattr(args_cli, "teacher_foot_radius", 0.03),
    #     )
    #     env.close()
    #     return
    # =======================================


    if getattr(args_cli, "collect_walk", True):
        collect_teacher_dataset_walk(
            env,
            out_path=getattr(args_cli, "teacher_out", "teacher_walk_his.h5"),
            n_samples=getattr(args_cli, "teacher_samples", 2000),
            settle_steps=getattr(args_cli, "teacher_settle_steps", 0),
            num_envs_to_save=getattr(args_cli, "teacher_save_envs", 64),
            policy = inference_policy,
            top_band=getattr(args_cli, "teacher_top_band", 0.03),
            foot_radius_m=getattr(args_cli, "teacher_foot_radius", 0.03),
            T_hist=48,
        )
        env.close()
        return




    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    raw = env.unwrapped


    print("sensors keys =", list(raw.scene.sensors.keys()))
    raw.reset()
    print("after reset sensors keys =", list(raw.scene.sensors.keys()))




    # if getattr(args_cli, "collect_teacher", True):
    # # ※video wrapper や RslRlVecEnvWrapper を付ける前にやるのが重要
    #     collect_teacher_dataset(
    #         raw,
    #         out_path=getattr(args_cli, "teacher_out", "teacher_static.h5"),
    #         n_samples=getattr(args_cli, "teacher_samples", 5000),
    #         settle_steps=getattr(args_cli, "teacher_settle_steps", 20),
    #         num_envs_to_save=getattr(args_cli, "teacher_save_envs", 1),
    #         top_band=getattr(args_cli, "teacher_top_band", 0.03),
    #         foot_radius_m=getattr(args_cli, "teacher_foot_radius", 0.03),
    #     )
    #     env.close()
    #     return

    


    print(">>> COLLECT MODE <<<")



    








    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    #for paper2

    test_cfg = agent_cfg.to_dict()
    # # 2. 次に、その「辞書」の中身をクラスオブジェクトで上書きします
    # #    (辞書なので、アクセスは[]を使います)
    # test_cfg["policy"]["class_name"] = VisionMLPActorCritic
    # test_cfg["policy"]["class_name"] = LocoTransformerActorCritic
    # test_cfg["policy"]["class_name"] = LocoTransformerFinetune
    # test_cfg["policy"]["class_name"] = LocoTransformerHFP
    # test_cfg["policy"]["class_name"] = MlpHFP
    # test_cfg["policy"]["class_name"] = VisionHighLevelAC
    test_cfg["policy"]["class_name"] = VisionHighRNN



    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner = OnPolicyRunner(env, test_cfg, log_dir=log_dir, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # runnerにobs_normalizerが存在するかを安全にチェックし、なければNoneを取得
    obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)

    # JITモデルをエクスポート
    export_policy_as_jit(policy_nn, obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    export_policy_as_onnx(
        policy_nn, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment 
    # obs, _ = env.get_observations() #before update
    obs = env.get_observations()


    # --- ループ前にセンサ参照を取る ---
    # cam = raw_env.scene.sensors["camera"]
    # ray = raw_env.scene.sensors["height_scanner"]

    # # RayCasterのグリッド形状（あなたの設定だと 1.2/0.2 -> 7, 0.6/0.2 -> 4 で 7x4）
    # L, W = ray.cfg.pattern_cfg.size
    # res = float(ray.cfg.pattern_cfg.resolution)
    # Hb = int(round(L / res)) + 1
    # Wb = int(round(W / res)) + 1


    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # obs, _, _, _ = env.step(torch.zeros_like(env.action_space.sample()))

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        
        # ★ センサは raw_env 側から読む
        # rgb = cam.data.output["rgb"]            # (B,64,64,3)
        # hits = ray.data.ray_hits_w              # (B, Hb*Wb, 3)
        # hits = hits.view(raw_env.num_envs, Hb, Wb, 3)
        # z = hits[..., 2]   


        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
