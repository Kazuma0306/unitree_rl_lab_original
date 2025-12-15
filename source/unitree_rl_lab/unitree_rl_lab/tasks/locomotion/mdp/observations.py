from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import os
import matplotlib.pyplot as plt



def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase




# def contact_ft_stack(
#     env, 
#     sensor_cfg,                # SceneEntityCfg("contact_forces", body_names=[...]) を渡す
#     normalize: str = "mg",     # "mg" | "none"
#     mass_kg: float | None = None,
#     return_shape: str = "flat" # "flat"=[B, K*L*3] / "bkl3"=[B, K, L, 3]
# ):
#     """
#     ContactSensor の履歴から、sensor_cfg.body_ids（例: 足4本）の力履歴を返す。
#     """
#     contact = env.scene.sensors[sensor_cfg.name]
#     data = contact.data

#     # 1) 履歴テンソル（最新→過去）を取得
#     X = getattr(data, "net_forces_w_history", None)         # [B, T, bodies, 3]
#     if X is None:
#         # フォールバック（履歴なしの環境でも一応動く）
#         X = data.net_forces_w.unsqueeze(1)                  # [B, 1, bodies, 3]

#     # 2) SceneEntityCfg が解決済みの body_ids で選択（名前処理は SceneEntityCfg に任せる）
#     keep = sensor_cfg.body_ids                              # 例: 足4本のインデックス
#     X = X[:, :, keep, :]                                    # -> [B, T, L, 3]

#     # 3) 正規化（任意だが推奨）
#     if normalize == "mg":
#         if mass_kg is None:
#             # 最初のアーティキュレーションから per-env の総質量を取得
#             art = next(iter(env.scene.articulations.values()))
#             mass_per_env = art.data.body_masses.sum(dim=1)  # [B]
#         else:
#             mass_per_env = torch.full((env.num_envs,), float(mass_kg), device=X.device)
#         X = X / (mass_per_env.view(-1, 1, 1, 1) * 9.81)
#         X = torch.clamp(X, -3.0, 3.0)
#     elif normalize == "none":
#         X = torch.clamp(X, -500.0, 500.0)

#     B, K, L, D = X.shape
#     return X if return_shape == "bkl3" else X.view(B, K * L * D)




def contact_ft_stack(
    env,
    sensor_cfg,                   # SceneEntityCfg("contact_forces", body_names=[...])
    normalize: str = "mg",        # "mg" | "none"
    mass_kg: float | None = None,
    return_shape: str = "flat",   # "flat" | "bkl3" | "bkl"
    components: str = "xyz",      # "xyz" | "z"
):
    """
    ContactSensor の履歴から指定 body の力履歴を返す。

    - components="xyz":
        return_shape="flat"  -> [B, T*L*3]
        return_shape="bkl3"  -> [B, T, L, 3]

    - components="z":
        return_shape="flat"  -> [B, T*L]      (Fzのみ)
        return_shape="bkl"   -> [B, T, L]     (Fzのみ)
    """
    contact = env.scene.sensors[sensor_cfg.name]
    data = contact.data

    # 1) 履歴テンソル [B, T, bodies, 3]
    X = getattr(data, "net_forces_w_history", None)
    if X is None:
        # 履歴がない場合は T=1 として扱う
        X = data.net_forces_w.unsqueeze(1)   # [B, 1, bodies, 3]

    # 2) 対象 body（足など）だけ抜く -> [B, T, L, 3]
    keep = sensor_cfg.body_ids
    X = X[:, :, keep, :]

    # 3) 必要な成分だけ選択
    if components == "z":
        # Fz のみ -> [B, T, L, 1]
        X = X[..., 2:3]
    elif components == "xyz":
        # 何もしない -> [B, T, L, 3]
        pass
    else:
        raise ValueError(f"Unsupported components={components}. Use 'xyz' or 'z'.")

    # 4) 正規化
    if normalize == "mg":
        if mass_kg is None:
            art = next(iter(env.scene.articulations.values()))
            mass_per_env = art.data.body_masses.sum(dim=1)  # [B]
        else:
            mass_per_env = torch.full((env.num_envs,), float(mass_kg), device=X.device)

        # [B, 1, 1, 1] で broadcast
        X = X / (mass_per_env.view(-1, 1, 1, 1) * 9.81)
        X = torch.clamp(X, -3.0, 3.0)
    elif normalize == "none":
        X = torch.clamp(X, -500.0, 500.0)

    B, T, L, D = X.shape  # D=3 か 1

    # 5) 返り値の形を整形
    if components == "xyz":
        if return_shape == "bkl3":
            return X                         # [B, T, L, 3]
        elif return_shape == "flat":
            return X.view(B, T * L * D)      # [B, T*L*3]
        else:
            raise ValueError(
                f"return_shape='{return_shape}' is invalid for components='xyz'. "
                "Use 'flat' or 'bkl3'."
            )

    elif components == "z":
        # D=1 なので最後の次元は落としてしまう
        X = X.squeeze(-1)                    # [B, T, L]
        if return_shape == "bkl":
            return X                         # [B, T, L]
        elif return_shape == "flat":
            return X.view(B, T * L)          # [B, T*L]
        else:
            raise ValueError(
                f"return_shape='{return_shape}' is invalid for components='z'. "
                "Use 'flat' or 'bkl'."
            )






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



# def leg_xy_err2(env, leg_name="FR_foot",
#                          cmd_name="step_fr_to_block", block_key="stone2",
#                          ):
#     """
#     返り値: [B,3] = 目標位置(world) - 指定脚末端位置(world)
#     """
#     # --- ブロック座標(ux,uy) → world のターゲット ---
#     cmd = env.command_manager.get_command(cmd_name)            # [B,2]
#     ux, uy = cmd[..., 0], cmd[..., 1]

#     blk_pos_w  = _block_pos_w(env, key=block_key)              # [B,3]
#     yaw_blk    = _yaw_from_quat(_block_quat_w(env, key=block_key))  # [B]
#     cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
#     R = torch.stack([torch.stack([cy, -sy], dim=-1),
#                      torch.stack([sy,  cy], dim=-1)], dim=-2)  # [B,2,2]

#     tgt_xy_w = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
#     tgt_z_w  = blk_pos_w[..., 2] + 0.15
#     tgt_w    = torch.cat([tgt_xy_w, tgt_z_w.unsqueeze(-1)], dim=-1)  # [B,3]

#     # --- 指定脚の world 位置 ---
#     robot = env.scene.articulations["robot"]
#     i = robot.body_names.index(leg_name)
#     if hasattr(robot.data, "body_link_pose_w"):
#         foot_w = robot.data.body_link_pose_w[:, i, :3]
#     else:
#         foot_w = robot.data.body_pos_w[:, i, :3]

#     # --- 誤差（目標 - 脚） in world ---
#     err_w = tgt_w - foot_w                                     # [B,3]
#     return err_w



LEG_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

def _rotmat_body_to_world_from_quat_wxyz(q):  # 既存のあなたの関数
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

def _rotmat_world_to_body_from_quat_wxyz(q):
    # body->world の転置が world->body
    R_bw = _rotmat_body_to_world_from_quat_wxyz(q)
    return R_bw.transpose(-1, -2)  # [B,3,3]

def _get_root_pos_w(data):
    if hasattr(data, "root_pos_w"):   return data.root_pos_w          # [B,3]
    if hasattr(data, "root_state"):   return data.root_state[..., :3] # [B,3]
    raise AttributeError("root position not found")

def _get_root_quat_w(data):
    if hasattr(data, "root_quat_w"):     return data.root_quat_w      # [B,4] (wxyz)
    if hasattr(data, "root_orient_w"):   return data.root_orient_w    # [B,4] (wxyz)
    if hasattr(data, "root_state"):      return data.root_state[..., 3:7]  # [B,4] (wxyz)
    raise AttributeError("root quaternion not found")

def _get_feet_pos_w(data, robot, leg_names=LEG_ORDER):
    idxs = [robot.body_names.index(n) for n in leg_names]
    if hasattr(data, "body_link_pose_w"):
        return data.body_link_pose_w[:, idxs, :3]  # [B,4,3]
    if hasattr(data, "body_pos_w"):
        return data.body_pos_w[:, idxs, :3]        # [B,4,3]
    if hasattr(data, "link_pos_w"):
        return data.link_pos_w[:, idxs, :3]        # [B,4,3]
    raise AttributeError("foot positions not found")

def feet_pos_base(env, leg_names=LEG_ORDER):
    """全脚の足先位置をベース座標系で返す: [B, 4, 3]"""
    robot = env.scene.articulations["robot"]
    base_p_w = _get_root_pos_w(robot.data)                 # [B,3]
    base_q_w = _get_root_quat_w(robot.data)                # [B,4] (wxyz)
    R_wb3    = _rotmat_world_to_body_from_quat_wxyz(base_q_w)  # [B,3,3]
    feet_w   = _get_feet_pos_w(robot.data, robot, leg_names)   # [B,4,3]
    # world -> base
    diff     = feet_w - base_p_w.unsqueeze(1)              # [B,4,3]
    feet_b   = (R_wb3 @ diff.transpose(-1, -2)).transpose(-1, -2)  # [B,4,3]
    return feet_b


def targets_base_from_command(env, cmd_name="step_fr_to_block", leg_names=LEG_ORDER):
    """
    CommandManager から各脚の目標(ベース座標)を取得。
    返り値: [B, 4, 3] (順序は leg_names)
    """
    cmd = env.command_manager.get_command(cmd_name)
    if cmd.dim() == 2 and cmd.shape[1] == 12:
        tgt_b = cmd.view(cmd.shape[0], 4, 3)  # [B,4,3]
    elif cmd.dim() == 3 and cmd.shape[1:] == (4, 3):
        tgt_b = cmd
    else:
        raise ValueError(f"Unexpected command shape for '{cmd_name}': {tuple(cmd.shape)}. Expect [B,12] or [B,4,3].")
    # 並びが環境の body_names と異なる場合は leg_names に合わせて並べ替える処理をここに挟んでください。
    return tgt_b


def legs_err_base_all(env, cmd_name="step_fr_to_block", use_xyz=True, leg_names=LEG_ORDER):
    """
    返り値:
      use_xyz=True  -> [B, 4, 3]（各脚の [dx,dy,dz]_base
      use_xyz=False -> [B, 4, 2]（各脚の [dx,dy]_base
    """
    feet_b = feet_pos_base(env, leg_names)           # [B,4,3]
    tgt_b  = targets_base_from_command(env, cmd_name)  # [B,4,3]
    err_b  = tgt_b - feet_b                           # [B,4,3]
    if use_xyz:
        B = err_b.shape[0]
        return err_b.reshape(B, -1) 
    else:
        return err_b[..., :2]



# 例: シーン上のブロック名
BLOCK_KEYS = ["stone1", "stone2", "stone3", "stone4",
              "stone5", "stone6", "stone7", "stone8"]

def all_blocks_state_base(
    env,
    block_keys=BLOCK_KEYS,
):
    """
    返り値: [B, 8 * F]
        各ブロックについて
        [pos_b(xyz), yaw_rel_cos, yaw_rel_sin] を並べたもの
    """
    device = env.device
    B = env.num_envs
    N = len(block_keys)

    robot = env.scene.articulations["robot"]
    base_p = robot.data.root_pos_w      # [B,3]
    base_q = robot.data.root_quat_w     # [B,4]

    # world -> base の回転 3x3
    R_wb3 = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # [B,3,3]

    feats = []

    # base の yaw（相対yawを取りたい場合）
    base_yaw = _yaw_from_quat(base_q)  # [B]

    for key in block_keys:
        robj = env.scene.rigid_objects[key]

        # root_state_w: [B, num_instances, 13] みたいな形なら [:,0,:] でOK
        state_w = robj.data.root_state_w[:, 0, :]  # [B,13] pos(3)+quat(4)+vel...
        pos_w   = state_w[:, :3]                  # [B,3]
        quat_w  = state_w[:, 3:7]                 # [B,4]

        # --- 位置: world -> base ---
        diff_w = pos_w - base_p                  # [B,3]
        pos_b  = torch.matmul(R_wb3, diff_w.unsqueeze(-1)).squeeze(-1)  # [B,3]

        # --- yaw: baseに対する相対yaw (cos,sin) ---
        yaw_blk  = _yaw_from_quat(quat_w)        # [B]
        yaw_rel  = yaw_blk - base_yaw            # [B]
        yaw_rel  = (yaw_rel + torch.pi) % (2*torch.pi) - torch.pi  # [-pi,pi] に wrap
        cy, sy   = torch.cos(yaw_rel), torch.sin(yaw_rel)

        feat = torch.cat(
            [pos_b, cy.unsqueeze(-1), sy.unsqueeze(-1)], dim=-1
        )  # [B, 5]
        feats.append(feat)

    # [B, N, 5] -> [B, N*5]
    all_feats = torch.cat(feats, dim=-1)
    return all_feats  # Critic側にだけ食わせる







#視覚観測（Depth 2　Heightmap）

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera, RayCasterCamera
from isaaclab.envs import ManagerBasedEnv
# from isaaclab.envs.mdp.observations import (
#     generic_io_descriptor, record_shape, record_dtype,
# )


# @generic_io_descriptor(
#     units="m",
#     axes=["Heightmap"],
#     observation_type="Sensor",
#     on_inspect=[record_shape, record_dtype],
# )
def depth_heightmap(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # base 座標系で見たときの XY 範囲 [m]
    x_range: tuple[float, float] = (-1.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    # Heightmap 解像度
    grid_shape: tuple[int, int] = (64, 64),
    # 高さの初期値（何も hit が無いセル用）
    default_height: float = -8.0,
) -> torch.Tensor:
    """Depth 画像から base フレームの Heightmap を作る観測。

    返り値 shape: (num_envs, H_map, W_map, 1)
    """

    # --- 1. センサとロボットを取得 ---
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    # depth 画像を取得（distance_to_image_plane を想定）
    depth = sensor.data.output["depth"]  # (N, H, W, 1) or (N, H, W)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]  # (N, H, W)

    # 無限距離を 0 に
    depth = depth.clone()
    depth[depth == float("inf")] = 3.0

    num_envs, H_cam, W_cam = depth.shape

    # --- 2. Depth を 3D 点群（カメラ座標系）へ投影 ---
    # IsaacLab 公式の unproject_depth/transform_points の使い方と同じです
    points_cam = math_utils.unproject_depth(
        depth,  # (N, H, W)
        sensor.data.intrinsic_matrices,  # (N, 3, 3)
        # is_ortho=False がデフォルト（perspective depth）
    )  # (N, H, W, 3)

    # (N, H*W, 3) に reshape
    points_cam = points_cam.view(num_envs, -1, 3)

    # =======================================================
    # 【修正箇所】 World座標を使わず、Robot基準で直接計算する
    # =======================================================
    
    # 1. カメラの取り付け位置（Base基準）を定義
    cam_pos_b = torch.tensor([0.25, 0.0, 0.25], device=env.device).repeat(num_envs, 1)


    cam_quat_b = sensor.data.quat_w_ros  # (N, 4) 
    
    # 3. カメラ座標(P_cam) -> ベース座標(P_base) へ変換
    # P_base = Rot * P_cam + Trans
    points_base = math_utils.transform_points(
        points_cam,
        cam_pos_b,   # 平行移動 (Offset)
        cam_quat_b,  # 回転 (Rotation)
    )



    X = points_base[..., 0]
    Y = points_base[..., 1]
    Z = points_base[..., 2]  # base 原点から見た高さ（Z）


    print(f"DEBUG: X range: {X.min().item():.2f} ~ {X.max().item():.2f}")
    print(f"DEBUG: Y range: {Y.min().item():.2f} ~ {Y.max().item():.2f}")
    print(f"DEBUG: Z range: {Z.min().item():.2f} ~ {Z.max().item():.2f}")

    print("DEBUG base_pos_w[0]:   ", robot.data.root_pos_w[0])
    print("DEBUG cam_pos_w[0]:    ", sensor.data.pos_w[0])
    print("DEBUG cam-base (env0): ", sensor.data.pos_w[0] - robot.data.root_pos_w[0])

    # --- 5. XY 範囲でフィルタリング ---
    xmin, xmax = x_range
    ymin, ymax = y_range

    # depth>0 かつ XY 範囲内だけを有効に
    # valid = (depth.view(num_envs, -1) > 0)
    # valid &= (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)


    # depth_flat = depth.view(num_envs, -1)

    # # Z<0（base より下）だけを足場候補とする
    # valid_depth = depth_flat > 0
    # valid_xy    = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    # valid_z     = (Z < 0.3)
    # valid       = valid_depth & valid_xy & valid_z


    valid = (depth.view(num_envs, -1) > 0) # Inf/0以外


    # --- 6. XY を Heightmap グリッド index に量子化 ---
    H_map, W_map = grid_shape

    gx = ((X - xmin) / (xmax - xmin) * (H_map - 1)).long()
    gy = ((Y - ymin) / (ymax - ymin) * (W_map - 1)).long()
    gx = gx.clamp(0, H_map - 1)
    gy = gy.clamp(0, W_map - 1)

    # --- 7. 各セルごとに高さを集約（ここでは最小 Z = 地面側 を採用） ---
    # 注意: Z は base 原点からの高さなので、地面は負の値になるはず。
    heightmap = torch.full(
        (num_envs, H_map, W_map),
        fill_value=default_height,
        device=env.device,
        dtype=Z.dtype,
    )

    # まず全セルを default_height にしておき、hit があるところだけ上書き/上書き最小値
    for env_id in range(num_envs):
        mask = valid[env_id]
        if not torch.any(mask):
            continue

        ix = gx[env_id][mask]
        iy = gy[env_id][mask]
        z  = Z[env_id][mask]

        flat_idx = ix * W_map + iy                   # (P_valid,)
        hm_flat = heightmap[env_id].view(-1)         # (H_map*W_map,)

        # PyTorch 1.12+ なら index_reduce_ が使えます
        hm_flat.index_reduce_(
            dim=0,
            index=flat_idx,
            source=z,
            reduce="amax",   # or "amax" で「一番高い」面を取る
        )

    # shape を image() と揃えて (N, H, W, 1) で返す
    # return heightmap.unsqueeze(-1)

    # 代表として env 0 の現在ステップ
    step0 = int(env.episode_length_buf[0].item()) if hasattr(env, "episode_length_buf") else 0
    if step0 % 100 == 0:
        # dump_heightmap は (B, H*W) でも動くようにしておけばよい
        dump_heightmap(
            heightmap.view(num_envs, -1),  # [N, H*W]
            H_map,
            W_map,
            save_dir="/tmp/heightmaps",     # 好きなパスに変更
            env_index=0,
            step=step0,
        )

    

    return heightmap.view(num_envs, -1)




def dump_heightmap(heightmap: torch.Tensor, H: int, W: int,
                   save_dir: str, env_index: int = 0, step: int = 0):
    os.makedirs(save_dir, exist_ok=True)
    hm = heightmap[env_index].view(H, W).detach().cpu().numpy()

    plt.figure()
    im = plt.imshow(hm, origin="lower")
    plt.colorbar(im, label="height [m]")
    plt.title(f"env {env_index}, step {step}")
    plt.savefig(os.path.join(save_dir, f"heightmap_e{env_index}_s{step:06d}.png"),
                bbox_inches="tight")
    plt.close()





def depth_heightmap2(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # base 座標系で見たときの XY 範囲 [m]
    x_range: tuple[float, float] = (-1.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    # Heightmap 解像度
    grid_shape: tuple[int, int] = (64, 64),
    # 高さの初期値（何も hit が無いセル用）
    default_height: float = 0.0,
) -> torch.Tensor:
    """Depth 画像から base フレームの Heightmap + 観測マスクを作る観測。

    返り値 shape: (num_envs, 2 * H_map * W_map)
      [ 前半 H_map*W_map: height, 後半 H_map*W_map: mask(0/1) ]
    """

    # --- 1. センサとロボットを取得 ---
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    # depth 画像を取得（distance_to_image_plane を想定）
    depth = sensor.data.output["depth"]  # (N, H, W, 1) or (N, H, W)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]  # (N, H, W)

    # 元の depth を保存（valid 判定用）
    raw_depth = depth
    # invalid: inf / nan / 0以下
    invalid = torch.isinf(raw_depth) | torch.isnan(raw_depth) | (raw_depth <= 0.0)

    # unproject 用に invalid を 0 に潰す
    depth = raw_depth.clone()
    depth[invalid] = 0.0

    num_envs, H_cam, W_cam = depth.shape

    # --- 2. Depth を 3D 点群（カメラ座標系）へ投影 ---
    points_cam = math_utils.unproject_depth(
        depth,  # (N, H, W)
        sensor.data.intrinsic_matrices,  # (N, 3, 3) か (1, 3, 3)
    )  # (N, H, W, 3)

    # (N, H*W, 3) に reshape
    points_cam = points_cam.view(num_envs, -1, 3)

     # 1. カメラの取り付け位置（Base基準）を定義
    cam_pos_b = torch.tensor([0.25, 0.0, 0.25], device=env.device).repeat(num_envs, 1)


    cam_quat_b = sensor.data.quat_w_ros  # (N, 4) 
    
    # 3. カメラ座標(P_cam) -> ベース座標(P_base) へ変換
    # P_base = Rot * P_cam + Trans
    points_base = math_utils.transform_points(
        points_cam,
        cam_pos_b,   # 平行移動 (Offset)
        cam_quat_b,  # 回転 (Rotation)
    )

    X = points_base[..., 0]
    Y = points_base[..., 1]
    Z = points_base[..., 2]  # base 原点から見た高さ（Z）

    # --- 4. マスク条件を作る ---
    xmin, xmax = x_range
    ymin, ymax = y_range

    # # flatten して depth と合わせる
    # depth_flat   = raw_depth.view(num_envs, -1)
    # invalid_flat = invalid.view(num_envs, -1)

    # valid_depth = ~invalid_flat
    # valid_xy    = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    # # Z<0.3 など、足場候補にしたい高さ範囲に合わせて調整
    # valid_z     = (Z < 0.3)

    # valid = valid_depth & valid_xy & valid_z  # (N, H*W)

    valid = (raw_depth.view(num_envs, -1) > 0) # Inf/0以外

    # --- 5. XY を Heightmap グリッド index に量子化 ---
    H_map, W_map = grid_shape

    gx = ((X - xmin) / (xmax - xmin) * (H_map - 1)).long()
    gy = ((Y - ymin) / (ymax - ymin) * (W_map - 1)).long()
    gx = gx.clamp(0, H_map - 1)
    gy = gy.clamp(0, W_map - 1)

    # --- 6. 高さマップ & マスクマップを初期化 ---
    heightmap = torch.full(
        (num_envs, H_map, W_map),
        fill_value=default_height,
        device=env.device,
        dtype=Z.dtype,
    )
    # True = このセルは「観測済み」
    maskmap = torch.zeros(
        (num_envs, H_map, W_map),
        device=env.device,
        dtype=torch.bool,
    )

    # --- 7. 各セルごとに高さを集約 & マスク更新 ---
    for env_id in range(num_envs):
        mask_pts = valid[env_id]  # この env の有効な点
        if not torch.any(mask_pts):
            continue

        ix = gx[env_id][mask_pts]
        iy = gy[env_id][mask_pts]
        z  = Z[env_id][mask_pts]

        flat_idx = ix * W_map + iy  # (P_valid,)

        # 高さ: 「一番高い面」を採用（Z は base 原点からの高さ）
        hm_flat = heightmap[env_id].view(-1)
        hm_flat.index_reduce_(
            dim=0,
            index=flat_idx,
            source=z,
            reduce="amax",
        )

        # マスク: 1点でも入ったセルを True に
        mm_flat = maskmap[env_id].view(-1)
        mm_flat[flat_idx] = True

    # --- 8. flatten & 結合 ---
    h_flat = heightmap.view(num_envs, -1)             # (N, H*W)
    m_flat = maskmap.view(num_envs, -1).float()       # (N, H*W) 0/1

    obs = torch.cat([h_flat, m_flat], dim=-1)         # (N, 2*H*W)

    # （お好みで）デバッグ保存
    step0 = int(env.episode_length_buf[0].item()) if hasattr(env, "episode_length_buf") else 0
    if step0 % 1000 == 0:
        dump_heightmap(
            h_flat,          # 高さだけ可視化
            H_map,
            W_map,
            save_dir="/tmp/heightmaps",
            env_index=0,
            step=step0,
        )

    return obs





def depth_heightmap3(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # base 座標系で見たときの XY 範囲 [m]
    x_range: tuple[float, float] = (-1.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    # Heightmap 解像度
    grid_shape: tuple[int, int] = (64, 64),
    # 高さの初期値（★修正: 死角を穴として認識させるため、低い値を推奨）
    default_height: float = -2.0,
) -> torch.Tensor:
    """
    修正版 Heightmap 生成関数
    - 機能: マスク出力、デバッグ保存機能は元のまま維持
    - 修正: forループを排除(高速化)、座標変換ロジックを修正(正確化)
    """

    # --- 1. センサとロボットを取得 ---
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]

    # depth 画像を取得
    depth = sensor.data.output["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]  # (N, H, W)

    num_envs, H_cam, W_cam = depth.shape
    device = env.device

    # --- 2. 座標変換 (Camera -> World -> Base) [高速化・修正箇所] ---
    
    # (A) Depth -> Camera座標系 (点群)
    points_cam = math_utils.unproject_depth(depth, sensor.data.intrinsic_matrices)
    points_cam = points_cam.view(num_envs, -1, 3)  # (N, P, 3)

    # (B) Camera -> World座標系
    # センサーの正確な位置姿勢を使用
    cam_pos_w = sensor.data.pos_w
    cam_quat_w = sensor.data.quat_w_ros 
    points_world = math_utils.transform_points(points_cam, cam_pos_w, cam_quat_w)

    # (C) World -> Base座標系
    # ロボットのベース位置姿勢
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    # ベース基準への変換: P_base = R_inv * (P_world - T_base)
    # 平行移動
    rel_pos = points_world - base_pos_w.unsqueeze(1)
    # 回転 (World->Base なので、Baseの回転の逆回転をかける)
    base_quat_inv = math_utils.quat_inv(base_quat_w)
    
    # 回転適用 (transform_pointsの平行移動成分を0にして回転だけ適用)
    points_base = math_utils.transform_points(
        rel_pos, 
        torch.zeros_like(base_pos_w), 
        base_quat_inv
    )

    X = points_base[..., 0]
    Y = points_base[..., 1]
    Z = points_base[..., 2]

    # --- 3. マスク条件とグリッド化 [高速化箇所] ---
    xmin, xmax = x_range
    ymin, ymax = y_range
    H_map, W_map = grid_shape

    # 有効な点: 範囲内 かつ depthが有効(>0) かつ Zが異常値でない
    # ※ Z < 0.3 などの条件が必要であればここに追加してください (例: & (Z < 0.5))
    valid_mask = (X >= xmin) & (X < xmax) & \
                 (Y >= ymin) & (Y < ymax) & \
                 (depth.view(num_envs, -1) > 0) & \
                 (~torch.isinf(Z))

    # グリッドインデックス計算
    gx = ((X - xmin) / (xmax - xmin) * H_map).long()
    gy = ((Y - ymin) / (ymax - ymin) * W_map).long()
    gx = gx.clamp(0, H_map - 1)
    gy = gy.clamp(0, W_map - 1)

    # --- 4. 集約処理 (Scatter Reduce) [forループ除去] ---
    
    # バッチ一括処理用のフラットインデックスを作成
    # index = env_id * (H*W) + gx * W + gy
    batch_ids = torch.arange(num_envs, device=device).unsqueeze(1).expand_as(gx)
    flat_indices = batch_ids * (H_map * W_map) + gx * W_map + gy

    # validな点だけを抽出
    valid_indices = flat_indices[valid_mask]
    valid_z       = Z[valid_mask]

    # (A) Heightmap の作成
    # 全環境分を1次元配列で確保
    heightmap_flat = torch.full(
        (num_envs * H_map * W_map,), 
        default_height, 
        device=device, 
        dtype=Z.dtype
    )
    # 最大値プーリングで書き込み
    heightmap_flat.index_reduce_(
        dim=0,
        index=valid_indices,
        source=valid_z,
        reduce="amax",
        include_self=True
    )

    # (B) Maskmap の作成 (元の機能の維持)
    # 全環境分を1次元配列で確保 (0.0で初期化)
    maskmap_flat = torch.zeros(
        (num_envs * H_map * W_map,), 
        device=device, 
        dtype=torch.float32 # catするためにfloat
    )
    # データがある場所を 1.0 にする
    # (値は何でもいいので、valid_indices の場所に 1.0 を書き込む)
    maskmap_flat.index_fill_(0, valid_indices, 1.0)

    # --- 5. 整形と結合 (元の出力形式に合わせる) ---
    h_out = heightmap_flat.view(num_envs, -1)  # (N, H*W)
    m_out = maskmap_flat.view(num_envs, -1)    # (N, H*W)

    obs = torch.cat([h_out, m_out], dim=-1)    # (N, 2*H*W)

    # --- 6. デバッグ保存 (元の機能の維持) ---
    if hasattr(env, "episode_length_buf"):
        step0 = int(env.episode_length_buf[0].item())
        if step0 % 1000 == 0:
            # dump_heightmap 関数が定義されている前提
            try:
                # dump_heightmap は元の実装に合わせて呼び出す
                # (h_flat を渡していたようなので、h_out を渡す)
                dump_heightmap(
                    h_out,          
                    H_map,
                    W_map,
                    save_dir="/tmp/heightmaps",
                    env_index=0,
                    step=step0,
                )
            except NameError:
                pass # dump_heightmapがない場合はスキップ

    return obs



def depth_heightmap4(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (-1.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    grid_shape: tuple[int, int] = (64, 64),
    default_height: float = -4.0,  # ユーザー様の環境に合わせて -4.0 に設定
    # 固定オフセット (Base座標系)
    cam_pos_offset: list = [0.25, 0.0, 0.25], 
) -> torch.Tensor:
    
    # --- 1. データ取得 ---
    sensor = env.scene.sensors[sensor_cfg.name]
    
    depth = sensor.data.output["depth"]
    if depth.ndim == 4: depth = depth[..., 0]
    
    # invalid (Inf/NaN/負) の処理 (元のコードの再現)
    # 元コード: depth[invalid] = 0.0
    # ここで 0.0 にしておかないと unproject で Inf が発生し、計算がおかしくなります
    depth = torch.nan_to_num(depth, posinf=0.0, neginf=0.0)
    depth[depth <= 0.0] = 0.0

    N, H_cam, W_cam = depth.shape
    device = env.device

    # --- 2. 座標変換 (元のロジックを再現) ---
    
    # (A) Unproject
    points_cam = math_utils.unproject_depth(depth, sensor.data.intrinsic_matrices)
    points_cam = points_cam.view(N, -1, 3)

    # (B) Transform: "固定位置" + "センサー回転(World)"
    # ※ 元のコード通りの組み合わせです
    
    # 位置: 固定値
    pos_offset = torch.tensor(cam_pos_offset, device=device).unsqueeze(0).expand(N, -1)
    
    # 回転: センサーの値をそのまま使う (元のコードに合わせる)
    # これによりカメラのチルト角などが反映されます
    rot_quat = sensor.data.quat_w_ros

    # 変換実行
    points_base = math_utils.transform_points(points_cam, pos_offset, rot_quat)

    X = points_base[..., 0]
    Y = points_base[..., 1]
    Z = points_base[..., 2]

    # --- 3. グリッド化 (元のロジックを再現) ---
    xmin, xmax = x_range
    ymin, ymax = y_range
    H_map, W_map = grid_shape

    # グリッドインデックス計算
    gx = ((X - xmin) / (xmax - xmin) * (H_map - 1)).long()
    gy = ((Y - ymin) / (ymax - ymin) * (W_map - 1)).long()

    # ★重要: 元のコード同様、範囲チェックで捨てずに clamp で端に寄せます
    # これにより、範囲外の点もマップの縁に書き込まれます（-4にならなくなります）
    gx = gx.clamp(0, H_map - 1)
    gy = gy.clamp(0, W_map - 1)

    # --- 4. 集約処理 (Scatter Reduce) ---
    
    # 有効な点: depth > 0 のみ (X,Yの範囲チェックはしない)
    valid_mask = (depth.view(N, -1) > 0)
    
    batch_ids = torch.arange(N, device=device).unsqueeze(1).expand_as(gx)
    flat_indices = batch_ids * (H_map * W_map) + gx * W_map + gy

    valid_indices = flat_indices[valid_mask]
    valid_z       = Z[valid_mask]

    # Heightmap
    heightmap_flat = torch.full((N * H_map * W_map,), default_height, device=device, dtype=Z.dtype)
    heightmap_flat.index_reduce_(0, valid_indices, valid_z, reduce="amax", include_self=True)

    # Maskmap
    maskmap_flat = torch.zeros((N * H_map * W_map,), device=device, dtype=torch.float32)
    maskmap_flat.index_fill_(0, valid_indices, 1.0)

    obs = torch.cat([heightmap_flat.view(N, -1), maskmap_flat.view(N, -1)], dim=-1)


    # --- 6. デバッグ保存 (元の機能の維持) ---
    if hasattr(env, "episode_length_buf"):
        step0 = int(env.episode_length_buf[0].item())
        if step0 % 1000 == 0:
            # dump_heightmap 関数が定義されている前提
            try:
                # dump_heightmap は元の実装に合わせて呼び出す
                # (h_flat を渡していたようなので、h_out を渡す)
                dump_heightmap(
                    heightmap_flat.view(N, -1),          
                    H_map,
                    W_map,
                    save_dir="/tmp/heightmaps",
                    env_index=0,
                    step=step0,
                )
            except NameError:
                pass # dump_heightmapがない場合はスキップ
    
    return obs









import torch
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.utils.math import quat_apply_inverse  # ← こっちを使う


def get_block_top_pos_base(env, block_names, block_height=0.3):
    """各ブロック上面中心の base 座標系での位置 [num_envs, num_blocks, 3] を返す。"""
    device = env.device
    robot = env.scene.articulations["robot"]

    # base の world pose
    base_state = robot.data.root_state_w  # [num_envs, 13]
    base_pos_w = base_state[:, 0:3]       # [N, 3]
    base_quat_w = base_state[:, 3:7]      # [N, 4]

    block_pos_list = []
    for name in block_names:
        block = env.scene.rigid_objects[name]
        state = block.data.root_state_w   # [num_envs, 13]
        pos_w = state[:, 0:3].clone()

        # 上面中心に補正 (z方向に半分だけ上げる)
        pos_w[:, 2] += block_height * 0.5

        # base 座標系へ変換
        rel_w = pos_w - base_pos_w                      # world での相対
        # pos_b = quat_rotate_inverse(base_quat_w, rel_w) # base 系に回転
        pos_b = quat_apply_inverse(base_quat_w, rel_w)

        block_pos_list.append(pos_b)

    # [num_blocks, num_envs, 3] → [num_envs, num_blocks, 3]
    block_pos_b = torch.stack(block_pos_list, dim=1)
    return block_pos_b




def select_near_blocks(block_pos_b, x_max=2.5, y_max=1.2, max_blocks=6):
    """
    block_pos_b: [N_env, N_block, 3] (base座標系)
    から、前方の近いブロックだけ max_blocks 個選んで返す。
    戻り値: [N_env, max_blocks, 3]
    """
    x = block_pos_b[..., 0]
    y = block_pos_b[..., 1]

    # 条件: base 前方 & ある程度近い
    mask = (x > -0.25) & (x < x_max) & (y.abs() < y_max)  # [N_env, N_block]

    # 距離でソート
    dist2 = x**2 + y**2
    # マスクされてるところは大きな距離にして除外
    big = 1e6
    dist2_masked = torch.where(mask, dist2, big * torch.ones_like(dist2))

    # 距離の小さい順に index を取る
    # topk の代わりに sort して先頭 max_blocks を取る
    sorted_dist, sorted_idx = torch.sort(dist2_masked, dim=1)



    # 先頭 max_blocks の index
    idx = sorted_idx[:, :max_blocks]  # [N_env, max_blocks]
    dist_top = sorted_dist[:, :max_blocks]


    # gather で [N_env, max_blocks, 3] を作る
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
    near_blocks = torch.gather(block_pos_b, dim=1, index=idx_expanded)

    # 「ブロックが存在しない」ケース：全部 big になっているので、
    # sorted_dist[:, 0] > big/2 ならその env はブロックなしと見なして 0 埋め等もできる
    # return near_blocks


    # 「本当に有効なブロックか？」を距離で判定
    valid = dist_top < (big * 0.5)   # [N_env, max_blocks], True ならブロックあり

    # 無効なところは 0 埋め
    near_blocks = torch.where(
        valid.unsqueeze(-1),
        near_blocks,
        torch.zeros_like(near_blocks)
    )

    near_mask = valid.float()
    return near_blocks, near_mask


def obs_near_blocks(env: ManagerBasedEnv):
    block_names = [f"stone{i}" for i in range(1, 17)]
    block_pos_b = get_block_top_pos_base(env, block_names, block_height=0.3)
    near_blocks, near_mask = select_near_blocks(block_pos_b, x_max=2.5, y_max=1.2, max_blocks=5)

    # if env.env_id == 0:   # 適当な env だけ
    # print("blocks:", near_blocks[0])
    # print("mask  :", near_mask[0])
    obs_blocks = torch.cat(
    [near_blocks.reshape(env.num_envs, -1), near_mask], dim=-1
    )  # [N, 4*3 + 4]
    # return near_blocks.reshape(env.num_envs, -1)
    return obs_blocks








# def get_block_top_pos_base_collection(
#     env,
#     collection_name: str = "stones",
#     block_height: float = 0.3,
# ):
#     """
#     RigidObjectCollection 版:
#     各ブロック上面中心の base 座標系での位置 [N_env, N_block, 3] を返す
#     """
#     device = env.device
#     robot = env.scene.articulations["robot"]

#     # base の world pose
#     # すでに root_state_w を使っているならそのままでもOK
#     base_state = robot.data.root_state_w        # [N_env, 13]
#     base_pos_w  = base_state[:, 0:3]           # [N_env, 3]
#     base_quat_w = base_state[:, 3:7]           # [N_env, 4]

#     # コレクションからブロック一括取得
#     stones = env.scene.rigid_object_collections[collection_name]

#     # 位置だけで良いので object_pos_w を使う
#     block_pos_w = stones.data.object_pos_w     # [N_env, N_block, 3]
#     block_pos_w = block_pos_w.clone()

#     # 上面中心に補正 (z方向に半分だけ上げる)
#     block_pos_w[..., 2] += block_height * 0.5

#     # base 座標系へ変換（ブロードキャスト）
#     rel_w = block_pos_w - base_pos_w[:, None, :]    # [N_env, N_block, 3]
#     pos_b = quat_apply_inverse(base_quat_w[:, None, :], rel_w)

#     return pos_b  # [N_env, N_block, 3]




def get_block_top_pos_base_collection(env, collection_name="stones", block_height=0.3):
    """RigidObjectCollection 版: [N_env, N_block, 3] を返す"""
    robot = env.scene.articulations["robot"]
    stones = env.scene.rigid_object_collections[collection_name]

    # base の world pose
    base_state = robot.data.root_state_w            # [N_env, 13]
    base_pos_w  = base_state[:, 0:3]                # [N_env, 3]
    base_quat_w = base_state[:, 3:7]                # [N_env, 4]

    # ブロック位置 [N_env, N_block, 3] と仮定
    block_pos_w = stones.data.object_pos_w.clone()  # [N_env, N_block, 3]

    # 上面中心に補正
    block_pos_w[..., 2] += block_height * 0.5

    # base 基準に平行移動
    rel_w = block_pos_w - base_pos_w[:, None, :]    # [N_env, N_block, 3]

    n_env, n_block, _ = rel_w.shape

    # クォータニオンをブロック数ぶん複製して flatten
    quat_flat = base_quat_w[:, None, :].expand(-1, n_block, -1).reshape(-1, 4)  # [N_env*N_block, 4]
    vec_flat  = rel_w.reshape(-1, 3)                                            # [N_env*N_block, 3]

    # まとめて base 座標系へ
    pos_b_flat = quat_apply_inverse(quat_flat, vec_flat)                        # [N_env*N_block, 3]
    pos_b = pos_b_flat.view(n_env, n_block, 3)                                  # [N_env, N_block, 3]

    return pos_b



def select_near_blocks2(block_pos_b, x_max=2.5, y_max=1.2, max_blocks=6):
    """
    block_pos_b: [N_env, N_block, 3] (base座標系)
    戻り値:
      near_blocks: [N_env, max_blocks, 3]
      near_mask  : [N_env, max_blocks] (1=有効, 0=ダミー)
    """
    x = block_pos_b[..., 0]
    y = block_pos_b[..., 1]

    N_env, N_block = x.shape

    # 条件フィルタ
    mask = (x > -0.25) & (x < x_max) & (y.abs() < y_max)  # [N_env, N_block]

    dist2 = x**2 + y**2
    big = 1e6
    dist2_masked = torch.where(mask, dist2, big * torch.ones_like(dist2))

    # 距離小さい順にソート
    sorted_dist, sorted_idx = torch.sort(dist2_masked, dim=1)

    # 実際に使えるのは max_blocks か N_block の小さい方
    k = min(max_blocks, N_block)

    idx_k = sorted_idx[:, :k]       # [N_env, k]
    dist_k = sorted_dist[:, :k]     # [N_env, k]

    idx_expanded = idx_k.unsqueeze(-1).expand(-1, -1, 3)
    near_k = torch.gather(block_pos_b, dim=1, index=idx_expanded)   # [N_env, k, 3]

    # 有効フラグ
    valid_k = dist_k < (big * 0.5)  # [N_env, k]

    # ここからパディングして [N_env, max_blocks, ...] にそろえる
    if k < max_blocks:
        pad_blocks = torch.zeros(
            (N_env, max_blocks - k, 3),
            device=block_pos_b.device,
            dtype=block_pos_b.dtype,
        )
        pad_mask = torch.zeros(
            (N_env, max_blocks - k),
            device=block_pos_b.device,
            dtype=valid_k.dtype,
        )
        near_blocks = torch.cat([near_k, pad_blocks], dim=1)  # [N_env, max_blocks, 3]
        near_mask = torch.cat([valid_k.float(), pad_mask], dim=1)  # [N_env, max_blocks]
    else:
        near_blocks = near_k
        near_mask = valid_k.float()

    # 無効なところは 0 埋め（念のため）
    near_blocks = torch.where(
        near_mask.unsqueeze(-1) > 0.5,
        near_blocks,
        torch.zeros_like(near_blocks),
    )

    return near_blocks, near_mask





def obs_near_blocks_col(env):
    # コレクションから取得
    block_pos_b = get_block_top_pos_base_collection(
        env,
        collection_name="stones",
        block_height=0.3,
    )
    # ここから先は元のまま
    near_blocks, near_mask = select_near_blocks2(
        block_pos_b,
        x_max=2.5,
        y_max=1.2,
        max_blocks=5,
    )

    obs_blocks = torch.cat(
        [near_blocks.reshape(env.num_envs, -1), near_mask],
        dim=-1,
    )
    return obs_blocks