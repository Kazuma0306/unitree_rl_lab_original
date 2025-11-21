


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





def _block_theta(env, key="stone2"):
    q = _block_quat_w(env, key)                         # [B,4]
    w,x,y,z = q.unbind(-1)
    zc = 1.0 - 2.0*(x*x + y*y)
    return torch.arccos(torch.clamp(zc, -1.0, 1.0))     # [B]

def _block_wmag(env, key="stone2"):
    w = _block_ang_vel_w(env, key)                      # [B,3]
    return torch.linalg.norm(w, dim=-1)

# 失敗：ブロックが大きく傾いたら True（学習を切り上げ）
def block_overtilt_single(env, limit_angle=0.30):  # 例: 0.30rad ≈ 17°
    theta = _block_theta(env)
    return (theta > limit_angle)

# （任意）ブロックの角速度が高すぎたら失敗
def block_high_angvel_single(env, limit_w=2.0):    # rad/s
    return (_block_wmag(env) > limit_w)








def _rot3_from_quat_wxyz(q): # [B,3,3]
    w,x,y,z = q.unbind(-1)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    r00 = 1 - 2*(yy+zz); r01 = 2*(xy-wz);   r02 = 2*(xz+wy)
    r10 = 2*(xy+wz);     r11 = 1 - 2*(xx+zz); r12 = 2*(yz-wx)
    r20 = 2*(xz-wy);     r21 = 2*(yz+wx);   r22 = 1 - 2*(xx+yy)
    return torch.stack([torch.stack([r00,r01,r02], -1),
                        torch.stack([r10,r11,r12], -1),
                        torch.stack([r20,r21,r22], -1)], -2)

# class HoldAllFeetWithContact(ManagerTermBase):
#     """
#     終了条件:
#       - FR がブロック上面矩形内にあり、
#       - かつ 4 脚すべてが contact sensor の current_contact_time >= T_hold_s
#       - かつ FL/RL/RR は MultiLegBaseCommand で出したベース座標目標の周辺にいる
#     を満たしたときに done=True を返す。

#     ※ cmd_name を None にすると「コマンド目標の周辺」は見ず、
#       4 脚の contact_time + FR ブロック上だけで判定する。
#     """

#     def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         P = cfg.params
#         self.env = env

#         # --- ブロック関連 ---
#         self.block_name = P.get("block_name", "stone2")
#         self.T_hold_s   = P.get("T_hold_s", 1.2)
#         self.half_x     = P.get("half_x", 0.10)
#         self.half_y     = P.get("half_y", 0.10)
#         self.margin     = P.get("margin", 0.02)

#         # --- 接触＆コマンド関連 ---
#         self.contact_sensor_name = P.get("contact_sensor_name", "contact_forces")
#         self.contact_threshold   = P.get("contact_threshold", 0)  # |Fz| > これで接触とみなす
#         self.cmd_name            = P.get("cmd_name", "step_fr_to_block")  # MultiLegBaseCommand の名前
#         self.near_radius_cmd     = P.get("near_radius_cmd", 0.05)       # ベースXY距離の許容

#         # --- 足順序と FR 名 ---
#         self.leg_names = P.get("leg_names", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
#         self.fr_leg_name = P.get("fr_leg_name", "FR_foot")

#         # --- シーン参照 ---
#         scene = env.scene
#         self.robot = scene.articulations["robot"]
#         self.block = scene.rigid_objects[self.block_name]
#         self.sensor = env.scene.sensors["contact_forces"]

#         self.num_legs = len(self.leg_names)
#         self.foot_ids = [self.robot.body_names.index(n) for n in self.leg_names]
#         self.name_to_idx = {n: i for i, n in enumerate(self.leg_names)}
#         assert self.fr_leg_name in self.name_to_idx, \
#             f"fr_leg_name {self.fr_leg_name} が leg_names にありません"
#         self.fr_idx = self.name_to_idx[self.fr_leg_name]

#         # --- ContactSensor 中の各足の列を特定 ---
#         self.leg_cols = []
#         body_ids = getattr(self.sensor.data, "body_ids")
#         sensor_names = getattr(self.sensor, "body_names")
#         for leg in self.leg_names:
#             col = None
#             # 1) body_ids から探す
#             if body_ids is not None:
#                 ids0 = body_ids[0] if body_ids.ndim >= 2 else body_ids  # [M]
#                 body_id = self.robot.body_names.index(leg)
#                 hits = (ids0 == body_id).nonzero(as_tuple=True)[0]
#                 if hits.numel() >= 1:
#                     col = int(hits[0].item())
#             # 2) センサ側の body_names から探す
#             if col is None and sensor_names is not None and leg in sensor_names:
#                 col = sensor_names.index(leg)
#             if col is None:
#                 raise RuntimeError(
#                     f"ContactSensor から {leg} の列を特定できません。"
#                     "センサcfgで bodies=[...] を適切に設定するか、"
#                     "body_ids/body_names を expose してください。"
#                 )
#             self.leg_cols.append(col)

#         # --- contact_time 自前積算用（current_contact_time が無い環境向け） ---
#         self._ct_accum = None

#     def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
#         device = env.device
#         B = env.num_envs
#         dt = env.step_dt
#         scene = env.scene

#         # ==============================
#         # 1. ブロック姿勢（yaw, 中心）
#         # ==============================
#         p = self.block.data.root_pos_w
#         blk_pos_w = p[:, 0, :] if p.ndim == 3 else p  # [B,3]

#         q = self.block.data.root_quat_w
#         blk_quat_w = q[:, 0, :] if q.ndim == 3 else q # [B,4] (wxyz)

#         yaw = _yaw_from_quat_wxyz(blk_quat_w)         # [B]
#         Rz = _rot2d(yaw)                              # [B,2,2] block→world
#         R_wb2 = Rz.transpose(-1, -2)                  # [B,2,2] world→block

#         # ==============================
#         # 2. 足先位置（world） [B, num_legs, 3]
#         # ==============================
#         if hasattr(self.robot.data, "body_link_pose_w"):
#             feet_w = self.robot.data.body_link_pose_w[:, self.foot_ids, :3]
#         else:
#             feet_w = self.robot.data.body_pos_w[:, self.foot_ids, :3]
#         feet_xy_w = feet_w[..., :2]                  # [B,num_legs,2]

#         # FR の world 座標
#         fr_xy_w = feet_xy_w[:, self.fr_idx, :]       # [B,2]

#         # --- FR がブロック矩形内か？ ---
#         d_xy_w   = fr_xy_w - blk_pos_w[..., :2]      # [B,2]
#         d_xy_blk = (R_wb2 @ d_xy_w.unsqueeze(-1)).squeeze(-1)  # [B,2]

#         hx = self.half_x - self.margin
#         hy = self.half_y - self.margin
#         inside_fr = (d_xy_blk[..., 0].abs() <= hx) & (d_xy_blk[..., 1].abs() <= hy)  # [B]

#         # ==============================
#         # 3. 接触判定（net_forces_w）
#         # ==============================
#         # [B, num_bodies, 3] 想定
#         F = self.env.scene.sensors["contact_forces"].data.net_forces_w
#         Fz = torch.stack([F[:, bid, 2] for bid in self.foot_ids], dim=-1)  # [B,num_legs]
#         contacts = (Fz.abs() > self.contact_threshold)                     # [B,num_legs] bool
     
#         # ==============================
#         # 4. contact_time の取得（current_contact_time or 自前積算）
#         # ==============================
#         ctime_all = getattr(self.sensor.data, "current_contact_time", None)
#         if ctime_all is not None:
#             # [B,M] 想定（[B,S,M] の場合も最後の次元が列）
#             ctimes = torch.stack(
#                 [ctime_all[..., col] for col in self.leg_cols],
#                 dim=-1,
#             )  # [B,num_legs]
#         else:
#             # 自前で is_contact から積算
#             is_contact_all = getattr(self.sensor.data, "is_contact", None)
#             if is_contact_all is None:
#                 # 最悪 0 にしておく
#                 ctimes = torch.zeros(B, self.num_legs, device=device)
#             else:
#                 if (self._ct_accum is None) or (self._ct_accum.shape != (B, self.num_legs)):
#                     self._ct_accum = torch.zeros(B, self.num_legs, device=device)
#                 new_cols = []
#                 for j, col in enumerate(self.leg_cols):
#                     is_c = is_contact_all[..., col]        # [B]
#                     acc_j = self._ct_accum[:, j]           # [B]
#                     acc_j = torch.where(is_c, acc_j + dt, torch.zeros_like(acc_j))
#                     new_cols.append(acc_j)
#                 self._ct_accum = torch.stack(new_cols, dim=-1)  # [B,num_legs]
#                 ctimes = self._ct_accum

#         # FR の contact_time
#         ctime_fr = ctimes[:, self.fr_idx]                   # [B]

#         # ==============================
#         # 5. コマンド目標（ベース座標）との距離
#         # ==============================
#         if self.cmd_name is not None:
#             # base 姿勢
#             base_p = self.robot.data.root_pos_w            # [B,3]
#             base_q = self.robot.data.root_quat_w           # [B,4]
#             R_wb3 = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # [B,3,3] world→base

#             # feet_w を base 座標へ
#             diff_w = feet_w - base_p.unsqueeze(1)          # [B,num_legs,3]
#             feet_b = torch.matmul(
#                 R_wb3.unsqueeze(1),                         # [B,1,3,3]
#                 diff_w.unsqueeze(-1),                       # [B,num_legs,3,1]
#             ).squeeze(-1)                                   # [B,num_legs,3]
#             feet_xy_b = feet_b[..., :2]                     # [B,num_legs,2]

#             # コマンド（MultiLegBaseCommand 出力）[B,3*num_legs]
#             cmd = env.command_manager.get_command(self.cmd_name)
#             tgt_b = cmd.view(B, self.num_legs, 3)           # [B,num_legs,3]
#             tgt_xy_b = tgt_b[..., :2]                       # [B,num_legs,2]

#             diff_xy = feet_xy_b - tgt_xy_b
#             dist_xy = diff_xy.norm(dim=-1)                  # [B,num_legs]
#             near_cmd = (dist_xy <= self.near_radius_cmd)    # [B,num_legs] bool
#         else:
#             near_cmd = torch.ones(B, self.num_legs, dtype=torch.bool, device=device)

#         # ==============================
#         # 6. 各脚の条件 & 全体の done 判定
#         # ==============================
#         # FR: ブロック矩形内 & contact_time >= T_hold_s
#         cond_fr = inside_fr & (ctime_fr >= self.T_hold_s)

#         # 他脚: contact_time >= T_hold_s & near_cmd & 接触
#         cond_others = torch.ones(B, dtype=torch.bool, device=device)
#         for j, name in enumerate(self.leg_names):
#             if name == self.fr_leg_name:
#                 continue
#             cond_j = (ctimes[:, j] >= self.T_hold_s)
#             cond_j &= near_cmd[:, j]
#             cond_j &= contacts[:, j]
#             cond_others &= cond_j

#         done = cond_fr & cond_others  # [B]

#         # デバッグ用
#         # env.extras["all_feet_hold_mask"]  = done
#         # env.extras["all_feet_hold_count"] = int(done.sum().item())
#         # env.extras["all_feet_contact_times"] = ctimes

#         return done


    

class HoldAllFeetWithContact(ManagerTermBase):
    """
    終了条件:
      - FR がブロック上面矩形内にあり、
      - かつ 4 脚すべての contact_time >= T_hold_s で接触しており、
      - かつ FL/RL/RR は MultiLegBaseCommand で出したベース座標目標の周辺にいる
    を満たしたときに done=True を返す。
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        self.env = env

        # --- ブロック関連 ---
        self.block_name = P.get("block_name", "stone2")
        self.T_hold_s   = P.get("T_hold_s", 0.1)
        self.half_x     = P.get("half_x", 0.10)
        self.half_y     = P.get("half_y", 0.10)
        self.margin     = P.get("margin", 0.01)

        # --- 接触＆コマンド関連 ---
        self.contact_sensor_name = P.get("contact_sensor_name", "contact_forces")
        self.contact_threshold   = P.get("contact_threshold", 0.0)  # |Fz| > これで接触とみなす
        self.cmd_name            = P.get("cmd_name", "step_fr_to_block")
        self.near_radius_cmd     = P.get("near_radius_cmd", 0.05)

        # --- 足順序と FR 名 ---
        self.leg_names   = P.get("leg_names", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        self.fr_leg_name = P.get("fr_leg_name", "FR_foot")

        scene = env.scene
        self.robot = scene.articulations["robot"]
        self.block = scene.rigid_objects[self.block_name]

        # ContactSensor 本体
        if self.contact_sensor_name not in scene.sensors:
            raise RuntimeError(
                f"scene.sensors に '{self.contact_sensor_name}' が見つかりません。"
            )
        self.sensor = scene.sensors[self.contact_sensor_name]

        # contact_time を使うので track_air_time 必須
        if not self.sensor.cfg.track_air_time:
            raise RuntimeError(
                f"ContactSensor '{self.contact_sensor_name}' の cfg.track_air_time が False です。"
                " current_contact_time を使うには True にしてください。"
            )

        # --- ロボット側 foot ID（位置取得用） ---
        self.num_legs = len(self.leg_names)
        self.foot_ids = [self.robot.body_names.index(n) for n in self.leg_names]

        if self.fr_leg_name not in self.leg_names:
            raise RuntimeError(
                f"fr_leg_name='{self.fr_leg_name}' が leg_names={self.leg_names} に含まれていません。"
            )
        self.fr_idx = self.leg_names.index(self.fr_leg_name)

        # --- ContactSensor 内の列 index を名前から引く ---
        sensor_body_names = self.sensor.body_names  # 1env分の body 名リスト
        self.leg_cols: list[int] = []
        for leg in self.leg_names:
            if leg not in sensor_body_names:
                raise RuntimeError(
                    "ContactSensor の body_names に "
                    f"'{leg}' が含まれていません。\n"
                    f"  sensor.body_names = {sensor_body_names}\n"
                    "ContactSensorCfg.prim_path が足リンクにマッチしているか確認してください。"
                )
            self.leg_cols.append(sensor_body_names.index(leg))

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        device = env.device
        B = env.num_envs
        dt = env.step_dt

        # ==============================
        # 1. ブロック姿勢（yaw, 中心）
        # ==============================
        p = self.block.data.root_pos_w
        blk_pos_w = p[:, 0, :] if p.ndim == 3 else p  # [B,3]

        q = self.block.data.root_quat_w
        blk_quat_w = q[:, 0, :] if q.ndim == 3 else q  # [B,4] (wxyz)

        yaw = _yaw_from_quat_wxyz(blk_quat_w)         # [B]
        Rz = _rot2d(yaw)                              # [B,2,2] block→world
        R_wb2 = Rz.transpose(-1, -2)                  # [B,2,2] world→block

        # ==============================
        # 2. 足先位置（world） [B, num_legs, 3]
        # ==============================
        if hasattr(self.robot.data, "body_link_pose_w"):
            feet_w = self.robot.data.body_link_pose_w[:, self.foot_ids, :3]
        else:
            feet_w = self.robot.data.body_pos_w[:, self.foot_ids, :3]  # [B,num_legs,3]
        feet_xy_w = feet_w[..., :2]

        # FR の world XY
        fr_xy_w = feet_xy_w[:, self.fr_idx, :]       # [B,2]

        # --- FR がブロック矩形内か？ ---
        d_xy_w   = fr_xy_w - blk_pos_w[..., :2]                     # [B,2]
        d_xy_blk = (R_wb2 @ d_xy_w.unsqueeze(-1)).squeeze(-1)       # [B,2]

        hx = self.half_x - self.margin
        hy = self.half_y - self.margin
        inside_fr = (d_xy_blk[..., 0].abs() <= hx) & (d_xy_blk[..., 1].abs() <= hy)  # [B]

        # ==============================
        # 3. 接触判定（net_forces_w）
        # ==============================
        F = self.sensor.data.net_forces_w              # 期待形状: [B, num_sensor_bodies, 3]
        if F is None:
            raise RuntimeError(
                f"ContactSensor '{self.contact_sensor_name}' の data.net_forces_w が None です。"
                " update_period や asset の activate_contact_sensors を確認してください。"
            )

        # 各脚の Fz を sensor.body_names の順に抜き出す
        Fz = torch.stack(
            [F[:, col, 2] for col in self.leg_cols], dim=-1
        )  # [B, num_legs]
        contacts = (Fz.abs() > self.contact_threshold)  # [B,num_legs] bool

        # ==============================
        # 4. contact_time の取得（current_contact_time）
        # ==============================
        ctime_all = self.sensor.data.current_contact_time  # 期待形状: [B, num_sensor_bodies]
        if ctime_all is None:
            raise RuntimeError(
                f"ContactSensor '{self.contact_sensor_name}' の current_contact_time が None です。"
                " cfg.track_air_time=True になっているか確認してください。"
            )

        ctimes = torch.stack(
            [ctime_all[:, col] for col in self.leg_cols], dim=-1
        )  # [B,num_legs]

        # FR の contact_time
        ctime_fr = ctimes[:, self.fr_idx]  # [B]

        # ==============================
        # 5. コマンド目標（ベース座標）との距離
        # ==============================
        if self.cmd_name is not None:
            # base 姿勢
            base_p = self.robot.data.root_pos_w            # [B,3]
            base_q = self.robot.data.root_quat_w           # [B,4]
            R_wb3 = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # [B,3,3] world→base

            # feet_w を base 座標へ
            diff_w = feet_w - base_p.unsqueeze(1)          # [B,num_legs,3]
            feet_b = torch.matmul(
                R_wb3.unsqueeze(1),                        # [B,1,3,3]
                diff_w.unsqueeze(-1),                      # [B,num_legs,3,1]
            ).squeeze(-1)                                  # [B,num_legs,3]
            feet_xy_b = feet_b[..., :2]                    # [B,num_legs,2]

            # コマンド（MultiLegBaseCommand 出力）[B,3*num_legs]
            cmd = env.command_manager.get_command(self.cmd_name)
            tgt_b = cmd.view(B, self.num_legs, 3)          # [B,num_legs,3]
            tgt_xy_b = tgt_b[..., :2]                      # [B,num_legs,2]

            diff_xy = feet_xy_b - tgt_xy_b
            dist_xy = diff_xy.norm(dim=-1)                 # [B,num_legs]
            near_cmd = (dist_xy <= self.near_radius_cmd)   # [B,num_legs] bool
        else:
            near_cmd = torch.ones(B, self.num_legs, dtype=torch.bool, device=device)

        # ==============================
        # 6. 各脚の条件 & 全体の done 判定
        # ==============================
        # FR: ブロック矩形内 & contact_time >= T_hold_s & 接触
        cond_fr = inside_fr & (ctime_fr >= self.T_hold_s) & contacts[:, self.fr_idx]

        # 他脚: contact_time >= T_hold_s & near_cmd & 接触
        cond_others = torch.ones(B, dtype=torch.bool, device=device)
        for j, name in enumerate(self.leg_names):
            if name == self.fr_leg_name:
                continue
            cond_j = (ctimes[:, j] >= self.T_hold_s)
            cond_j = cond_j & near_cmd[:, j]
            cond_j = cond_j & contacts[:, j]
            cond_others = cond_others & cond_j

        done = cond_fr & cond_others  # [B]


        env.extras["fr_hold_ok_mask"]  = done
        env.extras["fr_hold_ok_count"] = int(done.sum().item())

        # デバッグ用に見たければここをコメントアウト解除
        # env.extras["all_feet_hold_mask"]       = done
        # env.extras["all_feet_contact_times"]   = ctimes
        # env.extras["all_feet_contacts_bool"]   = contacts
        # env.extras["all_feet_near_cmd_bool"]   = near_cmd

        return done




class AllBlocksMovedTermination(ManagerTermBase):
    """
    指定した複数のブロックについて、エピソード開始時の「位置」または「姿勢」から
    閾値以上変化したら終了（失敗）とする。
    """
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        
        # 対象ブロック
        default_blocks = [f"stone{i}" for i in range(1, 7)]
        self.block_names = P.get("block_names", default_blocks)
        
        # 閾値
        self.pos_limit = P.get("pos_limit", 0.20)  # 20cm動いたらアウト
        self.ori_limit = P.get("ori_limit", 0.50)  # 0.5rad(約28度)回転したらアウト
                                                   # (傾きだけでなくYaw回転も含む)

        # シーンからオブジェクト取得
        self.blocks = []
        for name in self.block_names:
            if name in env.scene.rigid_objects:
                self.blocks.append(env.scene.rigid_objects[name])

        num_envs = env.num_envs
        num_blocks = len(self.blocks)
        
        # 初期状態保存用 [B, N, 3] / [B, N, 4]
        self.init_pos = torch.zeros(num_envs, num_blocks, 3, device=env.device)
        self.init_quat = torch.zeros(num_envs, num_blocks, 4, device=env.device)
        self.is_first = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # 1. 現在の状態取得
        curr_pos_list = []
        curr_quat_list = []
        for b in self.blocks:
            # pos
            p = b.data.root_pos_w
            if p.ndim == 3: p = p[:, 0, :]
            curr_pos_list.append(p)
            # quat
            q = b.data.root_quat_w
            if q.ndim == 3: q = q[:, 0, :]
            curr_quat_list.append(q)

        curr_pos = torch.stack(curr_pos_list, dim=1)   # [B, N, 3]
        curr_quat = torch.stack(curr_quat_list, dim=1) # [B, N, 4]

        # 2. リセット時の初期状態保存
        # episode_length_buf == 0 (リセット直後) の環境を特定
        reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        
        if len(reset_ids) > 0:
            self.init_pos[reset_ids] = curr_pos[reset_ids]
            self.init_quat[reset_ids] = curr_quat[reset_ids]
        
        if self.is_first:
            self.init_pos[:] = curr_pos
            self.init_quat[:] = curr_quat
            self.is_first = False

        # 3. 判定A: 位置のズレ (Euclidean Distance)
        dist = torch.norm(curr_pos - self.init_pos, dim=-1) # [B, N]
        fail_pos = (dist > self.pos_limit).any(dim=1)

        # 4. 判定B: 姿勢のズレ (Quaternion Angle)
        # 2つのクォータニオン q1, q2 の間の角度 theta は
        # theta = 2 * acos( |dot(q1, q2)| )
        # dot積の絶対値をとるのは、qと-qが同じ回転を表すため
        dot = torch.sum(curr_quat * self.init_quat, dim=-1).abs() # [B, N]
        # 誤差で1.0を超えることがあるのでclamp
        dot = torch.clamp(dot, min=0.0, max=1.0)
        angle = 2.0 * torch.acos(dot) # [B, N] (rad)
        
        fail_ori = (angle > self.ori_limit).any(dim=1)

        # 5. どちらかでもアウトならTrue
        return fail_pos | fail_ori