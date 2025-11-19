from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

import math


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


def terrain_levels_nav(
    env,
    env_ids,
    goal_key: str = "goal_position",
    success_radius: float = 0.3,          # 到達半径[m]
    demote_radius: float = 1.3,           # まだこれより遠ければ降格（地形/タイルに合わせ調整）
    check_after_frac: float = 0.5,        # エピソード半分経過で降格判定を許可
    ):
    terrain = env.scene.terrain
    if len(env_ids) == 0:
        return terrain.terrain_levels.float().mean()

    # 相対ゴール（body frame）: XYのみの残距離
    cmd  = env.command_manager.get_command(goal_key)[env_ids]
    dist = torch.norm(cmd[:, :3], dim=1)          # 残距離 d_t

    # 成功（到達）
    move_up = dist < success_radius

    # 降格（時間が半分以上経過 & まだ遠い & 未達）
    max_steps = getattr(env, "max_episode_length", None)
    if max_steps is None:
        ctrl_dt   = env.sim.get_physics_dt() * env.cfg.decimation
        max_steps = int(env.cfg.episode_length_s / ctrl_dt)
    elapsed_enough = env.episode_length_buf[env_ids] > int(check_after_frac * max_steps)

    move_down = (~move_up) & elapsed_enough & (dist > demote_radius)

    # 公式と同じAPI
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()


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



class schedule_reward_weight(ManagerTermBase):
    """Curriculum that smoothly modifies a reward term's weight during training.

    - 公式の構成と同じく `__init__(cfg, env)` と `__call__(env, env_ids, term_name, weight, num_steps)` を提供。
    - `num_steps` 期間で現在の重みから `weight` へ滑らかに遷移します（線形/余弦スケジュール）。
    - しきい値を超えた瞬間に一発で置き換える挙動にしたい場合は、cfg側の `schedule="step"` を指定してください。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration (公式と同じ流れ)
        term_name = cfg.params["term_name"]
        self._term_name = term_name
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

        # 追加：開始時の重みをキャプチャ（滑らか遷移の始点）
        self._start_weight = float(self._term_cfg.weight)


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # 呼び出し毎に現在stepから α を計算し、開始重み→目標重み へ補間
        # 公式と同じ署名・戻り値（現在の設定重み）を維持


        t = int(env.common_step_counter)
        target_w = float(weight)
        dur = max(0, int(num_steps))

        # α（0→1）計算
      
        # 滑らか遷移（linear / cosine）
        if dur <= 0:
            alpha = 1.0
        else:
            x = max(0.0, min(1.0, t / float(dur))) 
            alpha = 0.5 - 0.5 * math.cos(math.pi * x)

        # 補間（始点＝初期重み、終点＝引数のweight）
        new_w = (1.0 - alpha) * self._start_weight + alpha * target_w

        # 反映（idempotent）
        if self._term_cfg.weight != new_w:
            self._term_cfg.weight = new_w
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight






import isaaclab.envs.mdp as mdp
from types import SimpleNamespace




# def mass_curriculum(
#     env, env_ids, old_range, *,
#     stages,
#     up_successes=64,  min_eps=100,  up_rate=0.7,
#     down_rate=0.25,   down_min_eps=100,
#     cooldown_steps=0,
# ):
#     ctx = getattr(env, "_curr", None)
#     if ctx is None:
#         ctx = env._curr = SimpleNamespace(stage=0, succ=0, eps=0, last=-10**9)

#     # ✅ extras から今ステップの成功数を回収（なければ0）
#     n_succ_step = int(env.extras.get("fr_hold_ok_count", 0))
#     if n_succ_step:
#         ctx.succ += n_succ_step
#         # 取り込み後に 0 へ戻す（連続加算防止）
#         env.extras["fr_hold_ok_count"] = 0

#     # エピソード終了数の更新
#     if hasattr(env, "reset_buf"):
#         ctx.eps += int(env.reset_buf.sum().item())

#     # クールダウン
#     if env.common_step_counter - ctx.last < cooldown_steps:
#         return mdp.modify_term_cfg.NO_CHANGE

#     # 判定
#     eps = max(1, ctx.eps)
#     rate = ctx.succ / eps
#     changed = False
#     if (ctx.eps >= min_eps and ctx.succ >= up_successes and rate >= up_rate
#         and ctx.stage < len(stages)-1):
#         ctx.stage += 1; changed = True
#     elif (ctx.eps >= down_min_eps and rate <= down_rate and ctx.stage > 0):
#         ctx.stage -= 1; changed = True

#     print(changed)
#     if not changed:
#         return mdp.modify_term_cfg.NO_CHANGE

#     # ステージ更新
#     ctx.last = env.common_step_counter
#     ctx.succ = 0
#     ctx.eps  = 0
#     lo, hi = stages[ctx.stage]
    
#     return (float(lo), float(hi))


def shared_mass_curriculum(
    env, env_ids, old_range, *,
    stages,
    master=False,
    up_successes=64,  min_eps=100,  up_rate=0.7,
    down_rate=0.25,   down_min_eps=100,
    cooldown_steps=0,
    alpha=1.0,
):
    # --- 初回だけコンテキストを作成 ---
    # hasattr を使えば「既にあるのに作り直す」ことはありえない
    if not hasattr(env, "_stone_mass_curr"):
        # ★ ここを stages[0] ではなく old_range から取る
        lo0, hi0 = old_range
        env._stone_mass_curr = SimpleNamespace(
            stage=0,
            succ=0,
            eps=0,
            last=-10**9,
            lo=float(lo0),
            hi=float(hi0),
        )

    ctx = env._stone_mass_curr

    # ----- master 側だけが stage/統計を更新 -----
    if master:
        n_succ_step = int(env.extras.get("fr_hold_ok_count", 0))
        if n_succ_step:
            ctx.succ += n_succ_step
            env.extras["fr_hold_ok_count"] = 0

        if hasattr(env, "reset_buf"):
            ctx.eps += int(env.reset_buf.sum().item())

        if env.common_step_counter - ctx.last >= cooldown_steps:
            eps  = max(1, ctx.eps)
            rate = ctx.succ / eps

            changed = False
            if (ctx.eps >= min_eps and
                ctx.succ >= up_successes and
                rate >= up_rate and
                ctx.stage < len(stages) - 1):
                ctx.stage += 1
                changed = True
            elif (ctx.eps >= down_min_eps and
                  rate <= down_rate and
                  ctx.stage > 0):
                ctx.stage -= 1
                changed = True

            if changed:
                ctx.last = env.common_step_counter
                ctx.succ = 0
                ctx.eps  = 0

        tgt_lo, tgt_hi = stages[ctx.stage]
        ctx.lo = (1 - alpha) * ctx.lo + alpha * float(tgt_lo)
        ctx.hi = (1 - alpha) * ctx.hi + alpha * float(tgt_hi)

    # master でも slave でも、共有レンジをそのまま返す
    return (float(ctx.lo), float(ctx.hi))
