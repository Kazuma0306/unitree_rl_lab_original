from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase




def contact_ft_stack(
    env, 
    sensor_cfg,                # SceneEntityCfg("contact_forces", body_names=[...]) を渡す
    normalize: str = "mg",     # "mg" | "none"
    mass_kg: float | None = None,
    return_shape: str = "flat" # "flat"=[B, K*L*3] / "bkl3"=[B, K, L, 3]
):
    """
    ContactSensor の履歴から、sensor_cfg.body_ids（例: 足4本）の力履歴を返す。
    """
    contact = env.scene.sensors[sensor_cfg.name]
    data = contact.data

    # 1) 履歴テンソル（最新→過去）を取得
    X = getattr(data, "net_forces_w_history", None)         # [B, T, bodies, 3]
    if X is None:
        # フォールバック（履歴なしの環境でも一応動く）
        X = data.net_forces_w.unsqueeze(1)                  # [B, 1, bodies, 3]

    # 2) SceneEntityCfg が解決済みの body_ids で選択（名前処理は SceneEntityCfg に任せる）
    keep = sensor_cfg.body_ids                              # 例: 足4本のインデックス
    X = X[:, :, keep, :]                                    # -> [B, T, L, 3]

    # 3) 正規化（任意だが推奨）
    if normalize == "mg":
        if mass_kg is None:
            # 最初のアーティキュレーションから per-env の総質量を取得
            art = next(iter(env.scene.articulations.values()))
            mass_per_env = art.data.body_masses.sum(dim=1)  # [B]
        else:
            mass_per_env = torch.full((env.num_envs,), float(mass_kg), device=X.device)
        X = X / (mass_per_env.view(-1, 1, 1, 1) * 9.81)
        X = torch.clamp(X, -3.0, 3.0)
    elif normalize == "none":
        X = torch.clamp(X, -500.0, 500.0)

    B, K, L, D = X.shape
    return X if return_shape == "bkl3" else X.view(B, K * L * D)



from .helpers_single_block import _block_pos_w, _block_quat_w, _yaw_from_quat

from .helpers_single_block import _base_pos_xy, _base_yaw




def fr_target_xy_rel_single_block(env, cmd_name="step_fr_to_block", block_key="stone2"):
    # コマンド: [ux, uy]
    t_local = env.command_manager.get_command(cmd_name)  # [B,2]
    ux, uy = t_local[..., 0], t_local[..., 1]

    # ブロック中心 (world) と yaw
    blk_pos_w = _block_pos_w(env, key=block_key)             # [B,3]
    yaw_blk   = _yaw_from_quat(_block_quat_w(env, key=block_key))  # [B]

    # ブロック座標(ux,uy) → world
    cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
    R = torch.stack([torch.stack([cy, -sy], dim=-1),
                     torch.stack([sy,  cy], dim=-1)], dim=-2)      # [B,2,2]
    t_xy = torch.stack([ux, uy], dim=-1)                           # [B,2]
    tgt_xy_w = (R @ t_xy.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]  # [B,2]

    # world → base座標（相対）
    yaw_base = _base_yaw(env)                                      # [B]
    cyb, syb = torch.cos(yaw_base), torch.sin(yaw_base)
    Rinv = torch.stack([torch.stack([cyb, syb], dim=-1),
                        torch.stack([-syb, cyb], dim=-1)], dim=-2) # [B,2,2]
    rel = (Rinv @ (tgt_xy_w - _base_pos_xy(env)).unsqueeze(-1)).squeeze(-1)  # [B,2]
    return rel






# CANON_LEG = ["FL","FR","RL","RR"]  # 観測の脚順はこれで統一

# def _foot_ids(robot):
#     return [robot.body_names.index(f"{n}_foot") for n in CANON_LEG]

# def _rotmat_body_to_world_from_quat_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
#     """q=(w,x,y,z) 前提の回転行列（body→world）"""
#     w, x, y, z = q_wxyz.unbind(-1)
#     xx, yy, zz = x*x, y*y, z*z
#     xy, xz, yz = x*y, x*z, y*z
#     wx, wy, wz = w*x, w*y, w*z
#     r00 = 1 - 2*(yy + zz); r01 = 2*(xy - wz);     r02 = 2*(xz + wy)
#     r10 = 2*(xy + wz);     r11 = 1 - 2*(xx + zz); r12 = 2*(yz - wx)
#     r20 = 2*(xz - wy);     r21 = 2*(yz + wx);     r22 = 1 - 2*(xx + yy)
#     return torch.stack([
#         torch.stack([r00, r01, r02], dim=-1),
#         torch.stack([r10, r11, r12], dim=-1),
#         torch.stack([r20, r21, r22], dim=-1),
#     ], dim=-2)  # [...,3,3]


# def ee_pos_base_obs(env, scale=0.5, quat_order="wxyz"):
#     """
#     各脚末端の位置を 胴体基準で返す。
#     return: [B, 12] = (FL,FR,RL,RR)  (x,y,z)
#     """
#     robot = env.scene.articulations["robot"]
#     B = robot.data.root_pos_w.shape[0]
#     ids = _foot_ids(robot)                                      # [4]

#     # 足位置（world）
#     if hasattr(robot.data, "body_link_pose_w"):
#         feet_w = robot.data.body_link_pose_w[:, ids, :3]        # [B,4,3]
#     else:
#         feet_w = robot.data.body_pos_w[:, ids, :3]              # [B,4,3]

#     # 胴体姿勢・位置（world）
#     base_pos_w  = robot.data.root_pos_w                         # [B,3]
#     base_quat_w = getattr(robot.data, "root_quat_w", robot.data.root_orient_w)  # [B,4]

#     # 必要なら xyzw→wxyz に並べ替え
#     if quat_order == "xyzw":
#         x, y, z, w = base_quat_w.unbind(-1)
#         base_quat_w = torch.stack([w, x, y, z], dim=-1)

#     R_bw = _rotmat_body_to_world_from_quat_wxyz(base_quat_w)    # [B,3,3]
#     R_wb = R_bw.transpose(1, 2)                                 # world→body

#     # 胴体基準へ変換（平行移動を引いて回転）
#     rel_w  = feet_w - base_pos_w.unsqueeze(1)                   # [B,4,3]
#     feet_b = torch.einsum('bij,bkj->bki', R_wb, rel_w)          # [B,4,3]

#     # スケールで無次元化（例: 脚長~0.5m）
#     feet_b = (feet_b / scale).clamp(-3.0, 3.0)                  # 任意のクリップ

#     return feet_b.reshape(B, 12)            





def _rotmat_body_to_world_from_quat_wxyz(q):  # q: [B,4], (w,x,y,z)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    r00 = 1 - 2*(yy + zz); r01 = 2*(xy - wz);     r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);     r11 = 1 - 2*(xx + zz); r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);     r21 = 2*(yz + wx);     r22 = 1 - 2*(xx + yy)
    return torch.stack([torch.stack([r00, r01, r02], -1),
                        torch.stack([r10, r11, r12], -1),
                        torch.stack([r20, r21, r22], -1)], -2)

def _get_root_pos_w(data):
    if hasattr(data, "root_pos_w"):   return data.root_pos_w          # [B,3]
    if hasattr(data, "root_state"):   return data.root_state[..., :3] # [B,3]
    raise AttributeError("root position not found")

def _get_root_quat_w(data):
    q = None
    if hasattr(data, "root_quat_w"):     q = data.root_quat_w         # [B,4]
    elif hasattr(data, "root_orient_w"): q = data.root_orient_w       # [B,4]
    elif hasattr(data, "root_state"):     q = data.root_state[..., 3:7]
    else:
        raise AttributeError("root quaternion not found")
    return q  # order handled by caller

def _get_feet_pos_w(data, robot, leg_names=("FL","FR","RL","RR")):
    idxs = [robot.body_names.index(f"{n}_foot") for n in leg_names]
    if hasattr(data, "body_link_pose_w"): return data.body_link_pose_w[:, idxs, :3]  # [B,4,3]
    if hasattr(data, "body_pos_w"):       return data.body_pos_w[:, idxs, :3]        # [B,4,3]
    if hasattr(data, "link_pos_w"):       return data.link_pos_w[:, idxs, :3]        # [B,4,3]
    raise AttributeError("foot positions not found")

def ee_pos_base_obs(env, scale=0.5, quat_order="wxyz"):
    """
    各脚末端の位置を base 座標に変換して返す。
    return: [B, 12]  (順序: FL, FR, RL, RR)  (x,y,z)
    """
    robot = env.scene.articulations["robot"]
    B = robot.data.root_pos_w.shape[0]

    feet_w   = _get_feet_pos_w(robot.data, robot)          # [B,4,3]
    base_p_w = _get_root_pos_w(robot.data)                 # [B,3]
    base_q   = _get_root_quat_w(robot.data)                # [B,4]

    # 必要なら xyzw→wxyz に並べ替え
    if quat_order == "xyzw":
        x, y, z, w = base_q.unbind(-1)
        base_q = torch.stack([w, x, y, z], dim=-1)

    R_bw = _rotmat_body_to_world_from_quat_wxyz(base_q)    # [B,3,3]
    R_wb = R_bw.transpose(1, 2)                            # world→body

    rel_w  = feet_w - base_p_w.unsqueeze(1)                # [B,4,3]
    feet_b = torch.einsum('bij,bkj->bki', R_wb, rel_w)     # [B,4,3]

    feet_b = (feet_b / scale).clamp(-3.0, 3.0)             # 任意のスケール/クリップ
    return feet_b.reshape(B, 12)







def leg_xy_err(
    env,
    leg_name="FR_foot",
    cmd_name="step_fr_to_block",
    block_key="stone2",
):
    """
    返り値: [B,2] = 目標XY(ベース座標) - 指定脚末端XY(ベース座標)
    正ならその軸の正方向へ脚を動かせば目標に近づくことを意味する
    """
    # --- 1) ブロック座標のコマンド -> world -> base（あなたのロジックを流用） ---
    t_local = env.command_manager.get_command(cmd_name)  # [B,2] (ux,uy) in block frame
    ux, uy = t_local[..., 0], t_local[..., 1]

    blk_pos_w = _block_pos_w(env, key=block_key)                    # [B,3]
    yaw_blk   = _yaw_from_quat(_block_quat_w(env, key=block_key))   # [B]
    cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
    R_bw = torch.stack([torch.stack([cy, -sy], dim=-1),
                        torch.stack([sy,  cy], dim=-1)], dim=-2)    # [B,2,2] block->world

    t_xy_blk = torch.stack([ux, uy], dim=-1)                        # [B,2]
    tgt_xy_w = (R_bw @ t_xy_blk.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]  # [B,2]

    # world -> base（2D）
    yaw_base = _base_yaw(env)                                       # [B]
    cyb, syb = torch.cos(yaw_base), torch.sin(yaw_base)
    R_wb = torch.stack([torch.stack([cyb, syb], dim=-1),
                        torch.stack([-syb, cyb], dim=-1)], dim=-2)  # [B,2,2] world->base
    base_xy_w = _base_pos_xy(env)                                   # [B,2]
    tgt_xy_b = (R_wb @ (tgt_xy_w - base_xy_w).unsqueeze(-1)).squeeze(-1)  # [B,2]

    # --- 2) 指定脚末端の位置を base 座標へ ---
    robot = env.scene.articulations["robot"]
    leg_id = robot.body_names.index(leg_name)
    foot_xy_w = robot.data.body_pos_w[:, leg_id, :2]                 # [B,2]
    foot_xy_b = (R_wb @ (foot_xy_w - base_xy_w).unsqueeze(-1)).squeeze(-1)  # [B,2]

    # --- 3) 誤差（目標 − 脚） ---
    err_xy_b = tgt_xy_b - foot_xy_b                                  # [B,2]
    return err_xy_b



def leg_xy_err2(env, leg_name="FR_foot",
                         cmd_name="step_fr_to_block", block_key="stone2",
                         ):
    """
    返り値: [B,3] = 目標位置(world) - 指定脚末端位置(world)
    """
    # --- ブロック座標(ux,uy) → world のターゲット ---
    cmd = env.command_manager.get_command(cmd_name)            # [B,2]
    ux, uy = cmd[..., 0], cmd[..., 1]

    blk_pos_w  = _block_pos_w(env, key=block_key)              # [B,3]
    yaw_blk    = _yaw_from_quat(_block_quat_w(env, key=block_key))  # [B]
    cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
    R = torch.stack([torch.stack([cy, -sy], dim=-1),
                     torch.stack([sy,  cy], dim=-1)], dim=-2)  # [B,2,2]

    tgt_xy_w = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
    tgt_z_w  = blk_pos_w[..., 2] + 0.15
    tgt_w    = torch.cat([tgt_xy_w, tgt_z_w.unsqueeze(-1)], dim=-1)  # [B,3]

    # --- 指定脚の world 位置 ---
    robot = env.scene.articulations["robot"]
    i = robot.body_names.index(leg_name)
    if hasattr(robot.data, "body_link_pose_w"):
        foot_w = robot.data.body_link_pose_w[:, i, :3]
    else:
        foot_w = robot.data.body_pos_w[:, i, :3]

    # --- 誤差（目標 - 脚） in world ---
    err_w = tgt_w - foot_w                                     # [B,3]
    return err_w