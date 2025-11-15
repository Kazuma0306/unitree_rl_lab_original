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



