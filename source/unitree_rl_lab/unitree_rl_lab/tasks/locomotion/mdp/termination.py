


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_apply_inverse


from .helpers_single_block import _block_pos_w, _block_quat_w, _block_ang_vel_w, _yaw_from_quat
from .helpers_single_block import _base_pos_xy, _base_yaw


from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import ManagerTermBase



def _block_center_xy(env, key="stone2"):
    return _block_pos_w(env, key)[..., :2]  # [B,2]

def _block_yaw_w(env, key="stone2"):
    return _yaw_from_quat(_block_quat_w(env, key))      # [B]

def _block_theta(env, key="stone2"):
    q = _block_quat_w(env, key)                         # [B,4]
    w,x,y,z = q.unbind(-1)
    zc = 1.0 - 2.0*(x*x + y*y)
    return torch.arccos(torch.clamp(zc, -1.0, 1.0))     # [B]

def _block_wmag(env, key="stone2"):
    w = _block_ang_vel_w(env, key)                      # [B,3]
    return torch.linalg.norm(w, dim=-1)


    


# def success_hold_fr_single_block(env, T_hold_s=0.5):
#     return (env._buf.hold_t >= T_hold_s)






def _yaw_from_quat_wxyz(q):  # q=(w,x,y,z)
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def _rot2d(yaw):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([cy, -sy], dim=-1),
                        torch.stack([sy,  cy], dim=-1)], dim=-2)



# class HoldFROnBlockWithContact(ManagerTermBase):
#     def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         P = cfg.params
#         self.block_name = P.get("block_name", "stone2")
#         self.T_hold_s   = P.get("T_hold_s", 1.0)
#         self.half_x     = P.get("half_x", 0.10)
#         self.half_y     = P.get("half_y", 0.10)
#         self.margin     = P.get("margin", 0.02)
#         self.robot = env.scene.articulations["robot"]
#         self.fr_id = self.robot.body_names.index("FR_foot")
#         self.block = env.scene.rigid_objects[self.block_name]
#         # ここで foot 用 ContactSensor を scene に用意しておく前提（cfg 側）
#         self.sensor = env.scene.sensors["contact_forces"]  # ContactSensor
#     def __call__(self, env):
#         # 矩形内判（省略可）

#         # --- ブロック姿勢 (world) ---
#         p = self.block.data.root_pos_w
#         blk_pos_w = p[:, 0, :] if p.ndim == 3 else p        # [B,3]
#         q = self.block.data.root_quat_w
#         blk_quat_w = q[:, 0, :] if q.ndim == 3 else q       # [B,4] (wxyz)
#         yaw = _yaw_from_quat_wxyz(blk_quat_w)               # [B]
#         Rz = _rot2d(yaw)                                    # [B,2,2] block→world
#         R_wb2 = Rz.transpose(-1, -2)                        # world→block (2×2)

#         # --- FR 足先 (world) ---
#         if hasattr(self.robot.data, "body_link_pose_w"):
#             fr_w = self.robot.data.body_link_pose_w[:, self.fr_id, :3]
#         else:
#             fr_w = self.robot.data.body_pos_w[:, self.fr_id, :3]      # [B,3]

#         # --- world→block でXYを変換して矩形内判定 ---
#         d_xy_w   = fr_w[..., :2] - blk_pos_w[..., :2]                   # [B,2]
#         d_xy_blk = (R_wb2 @ d_xy_w.unsqueeze(-1)).squeeze(-1)           # [B,2]

#         hx = self.half_x - self.margin
#         hy = self.half_y - self.margin
#         inside = (d_xy_blk[..., 0].abs() <= hx) & (d_xy_blk[..., 1].abs() <= hy)  # [B]

#         # current_contact_time: [N_sensors, B_bodies] → 足1枚なら [B]
#         ctime = self.sensor.data.current_contact_time[:, 0]      # [B] 
#         done = (ctime >= self.T_hold_s) & inside
#         # リセット時にセンサ側は勝手にリセットされるので明示ゼロ化不要
#         return done




class HoldFROnBlockWithContact(ManagerTermBase):
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        self.block_name = P.get("block_name", "stone2")
        self.T_hold_s   = P.get("T_hold_s", 1.2)
        self.half_x     = P.get("half_x", 0.10)
        self.half_y     = P.get("half_y", 0.10)
        self.margin     = P.get("margin", 0.02)

        self.robot = env.scene.articulations["robot"]
        self.fr_id = self.robot.body_names.index("FR_foot")
        self.block = env.scene.rigid_objects[self.block_name]

        # ContactSensor（例: "contact_forces"）
        self.sensor = env.scene.sensors["contact_forces"]

        # --- センサ中の「FR列」を特定 ---、指定あｓ市の時間のみみたい
        fr_col = None
        # 1) body_ids で一致列を探す
        body_ids = getattr(self.sensor.data, "body_ids", None)
        if body_ids is not None:
            ids0 = body_ids[0] if body_ids.ndim == 2 else body_ids  # [M]
            hits = (ids0 == self.fr_id).nonzero(as_tuple=True)[0]
            if hits.numel() == 1:
                fr_col = int(hits.item())
        # 2) 名前で探す（センサが body_names を持っていれば）
        if fr_col is None:
            names = getattr(self.sensor, "body_names", None)
            if names is not None and "FR_foot" in names:
                fr_col = names.index("FR_foot")
        # 3) 最後の手段：ユーザーが列を固定している前提（非推奨）
        if fr_col is None:
            raise RuntimeError(
                "ContactSensor から FR 列を特定できません。"
                "センサcfgで bodies=['FR_foot'] として単独登録にするか、"
                "body_ids/body_names を expose してください。"
            )
        self.fr_col = fr_col



    def __call__(self, env):
        # 矩形内判（省略可）

        # --- ブロック姿勢 (world) ---
        p = self.block.data.root_pos_w
        blk_pos_w = p[:, 0, :] if p.ndim == 3 else p        # [B,3]
        q = self.block.data.root_quat_w
        blk_quat_w = q[:, 0, :] if q.ndim == 3 else q       # [B,4] (wxyz)
        yaw = _yaw_from_quat_wxyz(blk_quat_w)               # [B]
        Rz = _rot2d(yaw)                                    # [B,2,2] block→world
        R_wb2 = Rz.transpose(-1, -2)                        # world→block (2×2)

        # --- FR 足先 (world) ---
        if hasattr(self.robot.data, "body_link_pose_w"):
            fr_w = self.robot.data.body_link_pose_w[:, self.fr_id, :3]
        else:
            fr_w = self.robot.data.body_pos_w[:, self.fr_id, :3]      # [B,3]

        # --- world→block でXYを変換して矩形内判定 ---
        d_xy_w   = fr_w[..., :2] - blk_pos_w[..., :2]                   # [B,2]
        d_xy_blk = (R_wb2 @ d_xy_w.unsqueeze(-1)).squeeze(-1)           # [B,2]

        hx = self.half_x - self.margin
        hy = self.half_y - self.margin
        inside = (d_xy_blk[..., 0].abs() <= hx) & (d_xy_blk[..., 1].abs() <= hy)  # [B]


        # ★ ここがポイント：FR の列だけ抜く
        ctime_all = getattr(self.sensor.data, "current_contact_time", None)
        if ctime_all is not None:
            # 形状はおおむね [B, M] 想定（環境により [B, S, M] 等もあるので適宜 squeeze）
            ctime_fr = ctime_all[..., self.fr_col]   # [B]
        else:
            # 一部の版は current_contact_time が無いので、自前で積算
            is_contact = self.sensor.data.is_contact[..., self.fr_col]  # [B]
            if not hasattr(self, "_ct_accum"):
                self._ct_accum = torch.zeros(env.num_envs, device=env.device)
            self._ct_accum = torch.where(is_contact, self._ct_accum + env.step_dt, torch.zeros_like(self._ct_accum))
            ctime_fr = self._ct_accum

        done = (ctime_fr >= self.T_hold_s) & inside


        env.extras["fr_hold_ok_mask"]  = done
        env.extras["fr_hold_ok_count"] = int(done.sum().item())
        return done





# 失敗：ブロックが大きく傾いたら True（学習を切り上げ）
def block_overtilt_single(env, limit_angle=0.30):  # 例: 0.30rad ≈ 17°
    theta = _block_theta(env)
    return (theta > limit_angle)

# （任意）ブロックの角速度が高すぎたら失敗
def block_high_angvel_single(env, limit_w=2.0):    # rad/s
    return (_block_wmag(env) > limit_w)