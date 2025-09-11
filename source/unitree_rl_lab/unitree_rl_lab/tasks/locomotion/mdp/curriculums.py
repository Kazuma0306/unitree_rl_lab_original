from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg



def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


# def terrain_level_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
#     """地形の難易度を直接操作して調整する、最終修正版のカリキュラム。"""
    
#     # [修正点] .terrain_generator を使わず、存在する terrain_levels 属性を直接使用する
#     terrain_levels_tensor = env.scene.terrain.terrain_levels

#     if len(env_ids) == 0:
#         return torch.mean(terrain_levels_tensor.float())

#     # パフォーマンスを評価
#     robot = env.scene["robot"]
#     # target_pos = env.command_manager.get_command("goal_position")[env_ids]
#     # robot_pos = robot.data.root_pos_w[env_ids]
#     # distance_to_target = torch.norm(target_pos[:, :2] - robot_pos[:, :2], dim=1)

#     command = env.command_manager.get_command("goal_position")[env_ids]
#     des_pos_b = command[:, :3]
#     distance = torch.norm(des_pos_b, dim=1)

#     # 成功・失敗を判定
#     success_threshold = 0.4
#     successes = distance < success_threshold

#     # エピソードの長さを取得（どれだけ長く生き残ったか）
#     episode_lengths = env.episode_length_buf[env_ids]
#     max_episode_steps = env.cfg.episode_length_s / (env.cfg.decimation * env.sim.get_physics_dt())


#     # 昇格/降格対象のIDを取得
#     promote_ids = env_ids[successes]


#     # 失敗の中でも、「すぐに死んでしまった」壊滅的な失敗の場合のみ降格
#     # 例：最大エピソード長の25%も生き残れなかった場合
#     catastrophic_failure = ~successes & (episode_lengths < max_episode_steps * 0.25)
#     demote_ids = env_ids[catastrophic_failure]
#     max_level = env.scene.terrain.max_terrain_level

#     levels_before_update = terrain_levels_tensor.clone()


#     # terrain_levels テンソルを直接更新
#     if len(promote_ids) > 0:
#         new_levels = terrain_levels_tensor[promote_ids] + 1
#         terrain_levels_tensor[promote_ids] = torch.clamp(new_levels, max=max_level)
#     if len(demote_ids) > 0:
#         new_levels = terrain_levels_tensor[demote_ids] - 1
#         terrain_levels_tensor[demote_ids] = torch.clamp(new_levels, min=0)


#     changed_indices = (levels_before_update != terrain_levels_tensor).nonzero(as_tuple=True)[0]

#     # [修正点] 返り値も .terrain_generator を経由せず、平均値を返す
#     return torch.mean(terrain_levels_tensor.float())


# def terrain_levels_vel_proj(
#     env: ManagerBasedRLEnv, done_env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     promote=0.7, demote=0.4
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     terrain: TerrainImporter = env.scene.terrain
#     cmd = env.command_manager.get_command("base_velocity")  # [N,3] を想定

#     # dt を安全に取得
#     dt = env.max_episode_length_s / max(env.max_episode_length, 1)

#     # バッファ初期化（per-env）
#     dev = env.device
#     if not hasattr(env, "_cur_prog"):
#         env._cur_prog    = torch.zeros(env.num_envs, device=dev)  # 実前進距離（指令方向への射影）
#         env._cur_reqdist = torch.zeros(env.num_envs, device=dev)  # 要求距離（|cmd|の積分）

#     # ---- 毎ステップ更新（呼び出し側で毎stepこのブロックを先に呼ぶか、ここに含めてもOK） ----
#     v_xy = asset.data.root_lin_vel_w[:, :2]                 # 実速度 [m/s]
#     cmd_xy = cmd[:, :2]
#     cmd_spd = torch.linalg.norm(cmd_xy, dim=1)              # |cmd| [m/s]
#     dir_xy = torch.where(cmd_spd[:, None] > 1e-6, cmd_xy / cmd_spd[:, None], torch.zeros_like(cmd_xy))
#     proj = (v_xy * dir_xy).sum(dim=1).clamp(min=0.0)        # 指令方向への前進 [m/s]（逆走は0扱い）
#     env._cur_prog    += proj * dt
#     env._cur_reqdist += cmd_spd * dt

#     # ---- エピソード終了時のみ昇降判定 ----
#     if done_env_ids.numel() > 0:
#         # 進捗率 = 実前進距離 / 要求距離（0割回避）
#         req = torch.clamp(env._cur_reqdist[done_env_ids], min=1e-6)
#         ratio = env._cur_prog[done_env_ids] / req

#         move_up   = ratio >= promote
#         move_down = (ratio <= demote) & (~move_up)

#         terrain.update_env_origins(done_env_ids, move_up, move_down)

#         # リセット
#         env._cur_prog[done_env_ids]    = 0.0
#         env._cur_reqdist[done_env_ids] = 0.0

#     # ログ用（平均レベル）
#     return torch.mean(terrain.terrain_levels.float())

def terrain_levels_vel_proj(
    env: ManagerBasedRLEnv, done_env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    promote=0.70, demote=0.40, req_min_m=0.10  # ← 小さすぎる指令は判定から除外
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    cmd = env.command_manager.get_command("base_velocity")  # [N,3]

    # dt は一定なら比率では相殺されますが、明示でOK
    dt = env.max_episode_length_s / max(env.max_episode_length, 1)

    dev = env.device
    if not hasattr(env, "_cur_prog"):
        env._cur_prog    = torch.zeros(env.num_envs, device=dev)  # 実前進距離（指令方向への射影）
        env._cur_reqdist = torch.zeros(env.num_envs, device=dev)  # 要求距離（|cmd|の積分）

    # --- 毎ステップ更新：ボディ座標の速度で投影 ---
    if hasattr(asset.data, "root_lin_vel_b"):
        v_xy = asset.data.root_lin_vel_b[:, :2]          # [m/s] body frame
    else:
        # 代替：必要ならワールド→ボディ回転を自作（yaw回転でも可）
        v_xy = asset.data.root_lin_vel_w[:, :2]          # TODO: 必ずボディ系に変換してください

    cmd_xy  = cmd[:, :2]                                 # body frame command
    cmd_spd = torch.linalg.norm(cmd_xy, dim=1)           # |cmd| [m/s]
    dir_xy  = torch.where(cmd_spd[:, None] > 1e-6, cmd_xy / cmd_spd[:, None], torch.zeros_like(cmd_xy))

    fwd = (v_xy * dir_xy).sum(dim=1).clamp(min=0.0)      # 逆走は0扱い（必要なら外しても可）
    env._cur_prog    += fwd * dt
    env._cur_reqdist += cmd_spd * dt

    # --- エピソード終了時のみ昇降 ---
    if done_env_ids.numel() > 0:
        req   = env._cur_reqdist[done_env_ids]
        ratio = env._cur_prog[done_env_ids] / torch.clamp(req, min=1e-6)

        # ★ 小さすぎる指令（standing / yaw-only）は昇降対象外に
        valid = req >= req_min_m

        move_up   = (ratio >= promote) & valid
        move_down = (ratio <= demote)  & valid & (~move_up)

        terrain.update_env_origins(done_env_ids, move_up, move_down)

        # 必ずリセット（昇降の有無に関わらず）
        env._cur_prog[done_env_ids]    = 0.0
        env._cur_reqdist[done_env_ids] = 0.0

    return torch.mean(terrain.terrain_levels.float())
