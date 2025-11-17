from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg

from isaaclab.utils.math import quat_apply_inverse

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# def stand_still(
#     env: ManagerBasedRLEnv,distance_threshold: float, command_name: str = "goal_position", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]

#     root_pos_w = asset.data.root_pos_w

#     command = env.command_manager.get_command(command_name)
#     des_pos_b = command[:, :3]
#     distance = torch.norm(des_pos_b, dim=1)

#     reward = -torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
#     return reward * (distance < distance_threshold)


def stand_still(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    heading_threshold: float,
    time_threshold: float,
    command_name: str = "goal_position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """論文に基づき、目標達成後の静止を促す罰則関数。"""
    
    # 1. 必要な情報を取得
    asset: Articulation = env.scene[asset_cfg.name]
    robot_pos_w = asset.data.root_pos_w
    robot_quat_w = asset.data.root_quat_w
    command = env.command_manager.get_command(command_name)
    
    # -- 罰則の計算 --
    # 速度が大きいほどペナルティが大きくなる
    linear_speed = torch.norm(asset.data.root_lin_vel_b, dim=1)
    angular_speed = torch.norm(asset.data.root_ang_vel_b, dim=1)
    # 論文の式 [2.5 * ||v_t|| + 1 * ||w_t||]
    penalty_expression = 2.5 * linear_speed + 1.0 * angular_speed

    # -- 罰則を有効にする3つの条件を判定 --
    # 条件1: 目標地点に近いか？
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    position_mask = (distance < distance_threshold)

    # 条件2: 目標の向きに近いか？
    heading_error = torch.abs(command[:, 3])
    heading_mask = (heading_error < heading_threshold)

    # 条件3: エピソードの終盤か？
    current_time = env.episode_length_buf * env.physics_dt
    max_time = env.cfg.episode_length_s
    time_mask = (current_time > max_time - time_threshold)
    
    # 3つの条件がすべてTrueの場合のみ、罰則を適用する
    final_mask = position_mask & heading_mask & time_mask
    
    # 罰則なので、マイナスの値を返す
    return -penalty_expression * final_mask.float()


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)




def joint_position_penalty_nav(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    distance_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """ナビゲーションタスクに合わせて修正された関節位置ペナルティ。"""
    asset: Articulation = env.scene[asset_cfg.name]

    # [修正点] ワールド座標系に統一して、目標までの正確な距離を計算
    # 1. ロボットのワールド座標を取得
    robot_pos_w = asset.data.root_pos_w
    # 2. コマンド（環境原点からの相対座標）を取得
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    
    # 「動くべきか」の判定ロジック
    should_be_moving = distance > distance_threshold
    
    # 「実際に動いているか」の判定 (ここはローカル速度の大きさなので修正不要)
    is_moving = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) > velocity_threshold
    
    # 関節のズレに対するペナルティ (修正不要)
    penalty = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    # ペナルティを適用
    return torch.where(torch.logical_or(should_be_moving, is_moving), penalty, stand_still_scale * penalty)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward























def base_accel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """論文の"Base Accel. Penalty"を再現します。

    胴体の急な線形加速と角加速を罰します。
    """
    robot: Articulation = env.scene["robot"]
    dt = env.sim.get_physics_dt()

    # envオブジェクトに前ステップの速度バッファが存在するかチェック
    # 存在しない場合（初回）、現在の速度で初期化し、加速度は0とする
    if not hasattr(env, "last_base_lin_vel_b"):
        env.last_base_lin_vel_b = robot.data.root_lin_vel_b.clone()
        env.last_base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        # 初回ステップの加速度は0として扱う
        return torch.zeros(env.num_envs, device=env.device)

    # 加速度と角加速度を計算 (v_t - v_{t-1}) / dt
    lin_accel = (robot.data.root_lin_vel_b - env.last_base_lin_vel_b) / dt
    ang_accel = (robot.data.root_ang_vel_b - env.last_base_ang_vel_b) / dt

    # 論文の式(Table IV)に基づいてペナルティを計算
    # ||lin_accel||^2 + 0.02 * ||ang_accel||^2
    penalty = torch.sum(torch.square(lin_accel), dim=1) + 0.02 * torch.sum(torch.square(ang_accel), dim=1)

    # 次のステップのために現在の速度を保存
    env.last_base_lin_vel_b[:] = robot.data.root_lin_vel_b
    env.last_base_ang_vel_b[:] = robot.data.root_ang_vel_b

    return penalty





@torch.no_grad()
def delta_T_mask(env, last_seconds: float) -> torch.Tensor:
    """残り `last_seconds` 秒だけ1になる時間マスク δ_T."""
    # 1エピソードの総ステップ数を取得
    max_steps = getattr(env, "max_episode_length", None)
    if max_steps is None:
        ctrl_dt   = env.sim.get_physics_dt() * env.cfg.decimation
        max_steps = int(env.cfg.episode_length_s / ctrl_dt)
    steps_left = max_steps - env.episode_length_buf   # (N,)
    # 閾値（秒→ステップ）
    ctrl_dt = env.sim.get_physics_dt() * env.cfg.decimation
    last_steps = int(last_seconds / ctrl_dt + 1e-6)
    return (steps_left <= last_steps).float()         # (N,)

@torch.no_grad()
def delta_p_mask(env, command_name: str = "goal_position", radius: float = 0.3) -> torch.Tensor:
    """ゴールから半径 `radius` m 以内で1になる空間マスク δ_p（相対コマンド想定, XYのみ）."""
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    dist = des_pos_b.norm(dim=1)
    return (dist < radius).float()



@torch.no_grad()
def position_endgame_reward(env, command_name="goal_position",
                            sigma: float = 1.0, last_seconds: float = 2.0) -> torch.Tensor:
    """
    位置追従 r_pos = 1 / (1 + ||d/sigma||^2) を δ_T で終盤だけ有効化。
    相対コマンド (body) を想定。XY距離で評価。
    """
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    d = des_pos_b.norm(dim=1) / (sigma + 1e-8)
    r = 1.0 / (1.0 + d * d)
    return r * delta_T_mask(env, last_seconds)

@torch.no_grad()
def heading_endgame_reward(env, command_name="goal_position",
                           last_seconds: float = 2.0, activate_radius: float = 2.5) -> torch.Tensor:
    """
    ヘディング合わせ（前方 x 軸とゴール方向の内積）を
    “近づいた時だけ” & “終盤だけ”有効化： δ_p · δ_T。
    """
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    dir_b = des_pos_b / (des_pos_b.norm(dim=1, keepdim=True) + 1e-8)
    fwd_b = torch.tensor([1.0, 0.0], device=env.device).expand_as(dir_b)
    cos = (fwd_b * dir_b).sum(dim=1)              # [-1,1]
    r = 0.5 * (cos + 1.0)                         # [0,1]
    return r * delta_T_mask(env, last_seconds) * delta_p_mask(env, command_name, activate_radius)


# def move_in_direction_early(env, command_name="goal_position") -> torch.Tensor:
#     """
#     学習初期だけ使う“方向合わせのshaping”（目標方向速度）。
#     後で外しやすいように単体関数化。
#     """
#     des_pos_b = env.command_manager.get_command(command_name)[:, :2]
#     dir_b = des_pos_b / (des_pos_b.norm(dim=1, keepdim=True) + 1e-8)
#     v_b = env.scene["robot"].data.root_lin_vel_b[:, :2]
#     v_par = (v_b * dir_b).sum(dim=1)
#     return torch.clamp(v_par, min=0.0)




def move_in_direction_early(
    env,
    command_name: str = "goal_position",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Move-in-Direction (論文準拠):
    cos⟨v_b, dir_b⟩ をそのまま返す（[-1,1]）。
    - 距離ゲートなし
    - 学習初期のみランナー側で weight>0 にする
    """
    # 目標方向（body座標の相対コマンド）
    des = env.command_manager.get_command(command_name)[:, :2]             # (N,2)
    dir_b = des / (des.norm(dim=1, keepdim=True) + eps)                    # (N,2)

    # 機体速度（body）
    v_b = env.scene["robot"].data.root_lin_vel_b[:, :2]                    # (N,2)
    v_n = v_b.norm(dim=1) + eps                                            # (N,)

    # コサイン整合（ゼロ平均のshaping）
    cos_align = (v_b * dir_b).sum(dim=1) / v_n                             # (N,)
    return cos_align  # [-1, 1]


def progress_potential_rel(env, command_name="goal_position", clip: float = 0.05) -> torch.Tensor:
    """
    一歩で縮めた距離 d_{t-1}-d_t を褒める（潜在報酬）。相対コマンド(body)前提。
    clip は1ステップの最大ご褒美（m）で、ノイズ抑制用。
    """
    des = env.command_manager.get_command(command_name)[:, :2]       # (N,2) 相対ゴール in body
    d_t = des.norm(dim=1)                                            # (N,)

    # 初期化
    if not hasattr(env, "_prev_dist") or env._prev_dist.shape[0] != env.num_envs:
        env._prev_dist = d_t.clone()

    prog = env._prev_dist - d_t                                      # 縮めた距離（負なら後退）
    env._prev_dist = d_t                                             # 更新

    return torch.clamp(prog, min=-clip, max=clip)  


def heading_alignment_rel(env, command_name: str = "goal_position") -> torch.Tensor:
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    dir_b = des_pos_b / (des_pos_b.norm(dim=1, keepdim=True) + 1e-8)
    fwd_b = torch.tensor([1.0, 0.0], device=env.device).expand_as(dir_b)  # 機体前方 = x軸
    cos = (fwd_b * dir_b).sum(dim=1)                                      # [-1,1]
    return 0.5 * (cos + 1.0)                                              # [0,1]




def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()



def stand_still_negative(env, last_seconds: float = 1.0,
                         lin_coef: float = 2.5, ang_coef: float = 1.0,
                         command_name: str = "goal_position", reach_radius: float = 0.1) -> torch.Tensor:
    """
    到達域 終盤(最後の1s)で “動くこと”を罰する（＝静止させる）。
    論文の Stand Still を負項として実装。
    """
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    reached = (des_pos_b.norm(dim=1) < reach_radius).float()
    mT = delta_T_mask(env, last_seconds)
    v_lin = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=1)
    w_z   = env.scene["robot"].data.root_ang_vel_b[:, 2].abs()
    return (lin_coef * v_lin + ang_coef * w_z) * reached * mT


def dont_wait(
    env: ManagerBasedRLEnv,
    velocity_threshold: float,
    distance_threshold: float,
    command_name: str , # 使用するコマンド名を変更
) -> torch.Tensor:
    """
    目標から遠いにも関わらず、低速でいる場合にペナルティを与える
    """
    # 1. 現在の速度を取得
    robot: Articulation = env.scene["robot"]
    base_lin_vel_b = robot.data.root_lin_vel_b

    # 2. 誤差ベクトルから目標までの距離を、速度から速さを計算
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    speed = torch.norm(base_lin_vel_b[:, :2], dim=1)

    # 3. 条件を計算
    is_waiting = (distance > distance_threshold) & (speed < velocity_threshold)
    
    return is_waiting.float()



def dont_wait_rel(env, command_name: str = "goal_position",
                           distance_threshold: float = 0.8, velocity_threshold: float = 0.1) -> torch.Tensor:
    des_pos_b = env.command_manager.get_command(command_name)[:, :2]
    dist  = des_pos_b.norm(dim=1)
    speed = env.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=1)
    # 遠い & 遅い ほど強い罰（連続値）
    a = torch.sigmoid(10*(dist - distance_threshold))
    b = torch.sigmoid(10*(velocity_threshold - speed))
    return a * b 






def _contacts_bool(env, sensor_cfg, min_contact_time: float, force_z_thresh: float|None):
    """ContactSensor から安定接地のboolを作る（時間,力の併用でチャタリング抑制）"""
    cs = env.scene.sensors[sensor_cfg.name]

    time_ok = None
    if hasattr(cs.data, "current_contact_time"):
        time_ok = cs.data.current_contact_time[:, sensor_cfg.body_ids] > (min_contact_time - 1e-9)

    force_ok = None
    if hasattr(cs.data, "net_forces_w_history") and force_z_thresh is not None:
        fz_max = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].abs().max(dim=1)[0]
        force_ok = fz_max > force_z_thresh

    if time_ok is None and force_ok is None:
        nf = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        return nf > 1.0
    if time_ok is None:
        return force_ok
    if force_ok is None:
        return time_ok
    return time_ok | force_ok


def feet_on_stone_height_only(
    env,
    sensor_cfg,
    asset_cfg=SceneEntityCfg("robot"),
    *,
    hole_z: float = -5.0,
    stone_eps: float = 4.7,
    min_contact_time: float = 0.04,
    force_z_thresh: float | None = None,
    foot_sole_offset: float = 0.0,
    normalize_by_feet: bool = True,
) -> torch.Tensor:
    """
    最小版: (接地) ∧ (足底Z > hole_z + stone_eps) の脚を数えて加点。
    地形／スキャナ依存なし。まずは学習を前に進めるためのシグナルとして。
    """
    contact = _contacts_bool(env, sensor_cfg, min_contact_time, force_z_thresh)  # (B,F) bool
    robot   = env.scene[asset_cfg.name]

    pos_w   = robot.data.body_pos_w[:, asset_cfg.body_ids, :]   # (B,F,3)
    foot_z  = pos_w[..., 2] - foot_sole_offset

    ok = contact & (foot_z > (hole_z + stone_eps))
    r  = ok.float().sum(dim=1)

    v = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2].norm(dim=-1)  # 実速度[m/s]
    gate = ((v - 0.05) / (0.30 - 0.05)).clamp(0.0, 1.0)  # 0→1へ滑らか遷移
    return r* gate / ok.shape[1] if normalize_by_feet else r



def feet_gap_contact_penalty(
    env,
    sensor_cfg,
    asset_cfg=SceneEntityCfg("robot"),
    *,
    hole_z: float = -5.0,       # 穴面の固定Z
    gap_tol: float = 4.7,      # この高さ帯以下で接地→減点
    min_contact_time: float = 0.04,
    force_z_thresh: float | None = None,
    foot_sole_offset: float = 0.0,
    normalize_by_feet: bool = True,
) -> torch.Tensor:
    """
    条件: (接地) ∧ (足底Z < hole_z + gap_tol)
    接地中に“穴面近傍”で踏んだ脚を数えて減点します。
    """
    contact = _contacts_bool(env, sensor_cfg, min_contact_time, force_z_thresh)  # (B,F) bool
    robot   = env.scene[asset_cfg.name]

    pos_w   = robot.data.body_pos_w[:, asset_cfg.body_ids, :]   # (B,F,3)
    foot_z  = pos_w[..., 2] - foot_sole_offset

    bad = contact & (foot_z < (hole_z + gap_tol))
    r = bad.float().sum(dim=1)
    return r / bad.shape[1] if normalize_by_feet else r


def feet_gap_discrete_penalty(
    env,
    sensor_cfg,
    asset_cfg=SceneEntityCfg("robot"),
    *,
    stone_top_z: float = 0.0,      # 石の天面Z（均一前提）
    edge_band: float = 0.03,       # 天面から下へ2cmを“危険帯”とみなす
    min_contact_time: float = 0.01,
    foot_sole_offset: float = 0.0,
    normalize_by_feet: bool = False,  # ★希釈しない
) -> torch.Tensor:
    # 接地（ヒステリシス弱め）
    cs = env.scene.sensors[sensor_cfg.name]
    contact = cs.data.current_contact_time[:, sensor_cfg.body_ids] > (min_contact_time - 1e-9)  # (B,F)
    # 足底高さ
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_sole_offset  # (B,F)
    # 天面より edge_band 以上低い場所での接地＝危険
    bad = contact & (foot_z < (stone_top_z - edge_band))  # (B,F)
    r = bad.float().sum(dim=1)  # 1脚1カウント
    return r / bad.shape[1] if normalize_by_feet else r














#single block env


from .helpers_single_block import _block_pos_w, _block_quat_w, _block_ang_vel_w, _yaw_from_quat

# from .helpers_single_block import _base_pos_xy, _base_yaw



def _fr_foot_pos_w(env):
    robot = env.scene.articulations["robot"]
    fr_id = robot.body_names.index("FR_foot")
    return robot.data.body_pos_w[:, fr_id, :3]  # [B,3]




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




def fr_on_block_rect(env: ManagerBasedRLEnv, margin=0.02):
    """FRが上面矩形(内側にmargin)に入っていれば＋"""
    fr_p = _fr_foot_pos_w(env)                   # [B,3]
    c_xy = _block_center_xy(env)                 # [B,2]
    yaw  = _block_yaw_w(env)                     # [B]
    # 上面矩形半辺（[hx,hy]）：env側で保持（各env同一なら [B,2] へtile）
    hx = 0.1 - margin   # [B]  x方向
    hy = 0.1 - margin   # [B]  y方向

    cy, sy = torch.cos(-yaw), torch.sin(-yaw)  # world→block
    Rinv = torch.stack([torch.stack([cy, -sy], dim=-1),
                        torch.stack([sy,  cy], dim=-1)], dim=-2)  # [B,2,2]
    d_xy = (Rinv @ (fr_p[..., :2]-c_xy).unsqueeze(-1)).squeeze(-1)  # [B,2]
    inside = (d_xy[...,0].abs() <= hx) & (d_xy[...,1].abs() <= hy)
    return inside.float()  # [B]







# def _rotmat_body_to_world_from_quat_wxyz(q):  # q=(w,x,y,z)
#     w,x,y,z = q.unbind(-1)
#     xx,yy,zz = x*x, y*y, z*z
#     xy,xz,yz = x*y, x*z, y*z
#     wx,wy,wz = w*x, w*y, w*z
#     r00 = 1 - 2*(yy+zz); r01 = 2*(xy - wz);  r02 = 2*(xz + wy)
#     r10 = 2*(xy + wz);   r11 = 1 - 2*(xx+zz); r12 = 2*(yz - wx)
#     r20 = 2*(xz - wy);   r21 = 2*(yz + wx);  r22 = 1 - 2*(xx+yy)
#     return torch.stack([torch.stack([r00,r01,r02],-1),
#                         torch.stack([r10,r11,r12],-1),
#                         torch.stack([r20,r21,r22],-1)], -2)

# def _yaw_from_quat_wxyz(q):  # q=(w,x,y,z)
#     w,x,y,z = q.unbind(-1)
#     return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

# def _rot2d(yaw):
#     cy, sy = torch.cos(yaw), torch.sin(yaw)
#     return torch.stack([torch.stack([cy, -sy], dim=-1),
#                         torch.stack([sy,  cy], dim=-1)], dim=-2)  # [...,2,2]

# # --- ベース座標で使えるユーティリティ ---
# def _base_pose(env):
#     robot = env.scene.articulations["robot"]
#     p = robot.data.root_pos_w
#     q = getattr(robot.data, "root_quat_w", robot.data.root_state[..., 3:7])  # wxyz
#     R_bw = _rotmat_body_to_world_from_quat_wxyz(q)  # body->world
#     R_wb = R_bw.transpose(1, 2)                     # world->body (base)
#     return p, q, R_wb

# def _fr_foot_pos_b(env):  # [B,3] (base座標)
#     robot = env.scene.articulations["robot"]
#     p_w, _, R_wb = _base_pose(env)
#     i = robot.body_names.index("FR_foot")
#     if hasattr(robot.data, "body_link_pose_w"):
#         foot_w = robot.data.body_link_pose_w[:, i, :3]
#     else:
#         foot_w = robot.data.body_pos_w[:, i, :3]
#     return torch.einsum('bij,bi->bj', R_wb, foot_w - p_w)

# def _block_center_xy_b(env, key="stone2"):  # [B,2] (base座標XY)
#     p_w, _, R_wb = _base_pose(env)
#     blk = env.scene.rigid_objects[key]
#     bp = blk.data.root_pos_w
#     bp = bp[:, 0, :] if bp.ndim == 3 else bp
#     c_b = torch.einsum('bij,bi->bj', R_wb, bp - p_w)
#     return c_b[..., :2]

# def _block_yaw_rel_base(env, key="stone2"):  # ブロックYaw - ベースYaw（相対yaw）
#     _, q_base, _ = _base_pose(env)
#     yaw_base = _yaw_from_quat_wxyz(q_base)
#     blk = env.scene.rigid_objects[key]
#     q = blk.data.root_quat_w
#     q = q[:, 0, :] if q.ndim == 3 else q
#     yaw_blk = _yaw_from_quat_wxyz(q)
#     # 相対yaw（base基準で見たブロック面の向き）
#     return (yaw_blk - yaw_base)

# # --- 矩形内判定（ベース座標版） ---
# def fr_on_block_rect_base(env, margin=0.02, key="stone2", half_x=0.1, half_y=0.1):
#     """
#     FR足がブロック上面の矩形（内側にmargin）に入っていれば1（ベース座標系で判定）
#     """
#     fr_b_xy = _fr_foot_pos_b(env)[..., :2]     # [B,2]
#     c_b_xy  = _block_center_xy_b(env, key)     # [B,2]
#     yaw_rel = _block_yaw_rel_base(env, key)    # [B]

#     # base座標XY での相対位置を、ブロック軸に合わせて回転
#     # R_base->block = rot2d(-(yaw_rel))
#     R = _rot2d(-yaw_rel)                        # [B,2,2]
#     d_xy_b = fr_b_xy - c_b_xy                   # [B,2]
#     d_xy_block = (R @ d_xy_b.unsqueeze(-1)).squeeze(-1)  # [B,2]

#     hx = half_x - margin
#     hy = half_y - margin
#     inside = (d_xy_block[..., 0].abs() <= hx) & (d_xy_block[..., 1].abs() <= hy)
#     return inside.float()










def fr_on_block_bonus(env: ManagerBasedRLEnv, margin=0.02, block_key="stone2"):
    """
    FR足がブロック上面の矩形内にあり、かつ高さが適合している場合にボーナスを与える。
    """
    # --- 1. 必要なデータの取得 ---
    fr_p = _fr_foot_pos_w(env)                   # [B,3]
    c_xy = _block_center_xy(env)                 # [B,2]
    yaw  = _block_yaw_w(env)                     # [B]
    
    # ブロックの寸法（環境設定に合わせて調整してください）
    block_half_width = 0.1 
    block_half_length = 0.1
    # ★重要: ブロックの上面の高さ（Z）
    # 中心Z + 半分の高さ = 上面
    block_pos = _block_pos_w(env, block_key)     # [B,3]
    block_top_z = block_pos[..., 2] + 0.15       # 例: 厚みが0.3なら半分は0.15

    # --- 2. XY平面の矩形判定（元のロジック） ---
    hx = block_half_width - margin
    hy = block_half_length - margin

    cy, sy = torch.cos(-yaw), torch.sin(-yaw)
    Rinv = torch.stack([torch.stack([cy, -sy], dim=-1),
                        torch.stack([sy,  cy], dim=-1)], dim=-2) # [B,2,2]
    
    # ローカル座標への変換
    vec_xy = fr_p[..., :2] - c_xy
    d_xy = (Rinv @ vec_xy.unsqueeze(-1)).squeeze(-1) # [B,2]
    
    is_xy_inside = (d_xy[..., 0].abs() <= hx) & (d_xy[..., 1].abs() <= hy)

    # --- 3. ★追加: 高さ（Z）の判定 ---
    # 足のZ座標が、ブロック上面付近にあるか (誤差 ±2cm以内など)
    fr_z = fr_p[..., 2]
    z_err = torch.abs(fr_z - block_top_z)
    is_z_correct = z_err < 0.03  # 3cm以内の誤差なら「乗っている」とみなす

    # --- 4. ★追加: 接触/安定性の判定 ---

    ft = env.scene.sensors["contact_forces"].data.net_forces_w
    robot = env.scene.articulations["robot"]
    # leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}[leg]
    fr_id = robot.body_names.index("FR_foot")
    fz = ft[:, fr_id, 2]

    # contact_forces = env.scene.net_contact_forces[:, body_idx, :]
    is_touching = torch.norm(fz, dim=-1) > 1.0

    # --- 5. 総合判定 ---
    # XYが入っていて、かつ 高さも合っていて、かつ 
    success = is_xy_inside & is_z_correct & is_touching

    # ボーナス値を返す (bool -> float -> value)
    return success.float() * 5.0  # 大きめのボーナス（例: +5.0）








def _yaw_from_quat_wxyz(q):
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def _rot2d(yaw):
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([cy,-sy],-1),
                        torch.stack([sy, cy],-1)], -2)

class FROnBlockBonusOnce(ManagerTermBase):
    """
    FR がブロック上面矩形内(+接触)を T_hold_s 連続で満たした瞬間だけボーナス支払い。
    RewardManager が *dt を掛ける前提なので、支払いステップのみ bonus/dt を返す。
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        self.block_name = P.get("block_name", "stone2")
        self.half_x     = P.get("half_x", 0.10)
        self.half_y     = P.get("half_y", 0.10)
        self.margin     = P.get("margin", 0.02)
        self.T_hold_s   = P.get("T_hold_s", 0.50)
        self.bonus      = P.get("bonus", 0.03)
        self.require_contact = P.get("require_contact", True)
        self.contact_sensor  = P.get("contact_sensor_name", "contact_forces")

        self.robot = env.scene.articulations["robot"]
        self.fr_id = self.robot.body_names.index("FR_foot")
        self.block = env.scene.rigid_objects[self.block_name]

        # per-env 状態
        self.hold_t = torch.zeros(env.num_envs, device=env.device)
        self.paid   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # 任意: 接触センサ（あれば使う）
        self.sensor = env.scene.sensors.get(self.contact_sensor, None)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # --- ブロック yaw / 中心 ---
        p = self.block.data.root_pos_w; blk_pos = p[:,0,:] if p.ndim==3 else p
        q = self.block.data.root_quat_w; blk_q = q[:,0,:] if q.ndim==3 else q
        yaw = _yaw_from_quat_wxyz(blk_q)
        Rz = _rot2d(yaw); R_wb2 = Rz.transpose(-1,-2)

        # --- FR 足先 ---
        if hasattr(self.robot.data, "body_link_pose_w"):
            fr_w = self.robot.data.body_link_pose_w[:, self.fr_id, :3]
        else:
            fr_w = self.robot.data.body_pos_w[:, self.fr_id, :3]

        # --- 矩形内判定（ブロック座標XY） ---
        d_xy_blk = (R_wb2 @ (fr_w[...,:2] - blk_pos[...,:2]).unsqueeze(-1)).squeeze(-1)
        hx = self.half_x - self.margin; hy = self.half_y - self.margin
        inside = (d_xy_blk[...,0].abs() <= hx) & (d_xy_blk[...,1].abs() <= hy)

        # --- 接触（あれば使う、無ければ inside のみ） ---

        ft = env.scene.sensors["contact_forces"].data.net_forces_w
        robot = env.scene.articulations["robot"]
        # leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}[leg]
        fr_id = robot.body_names.index("FR_foot")
        fz = ft[:, fr_id, 2]

        # contact_forces = env.scene.net_contact_forces[:, body_idx, :]
        on_contact = torch.norm(fz, dim=-1) > 1.0

        good = inside & on_contact

        # if self.require_contact and self.sensor is not None:
        #     # センサによって shape が違うので is_contact があればそれを使うのが楽
        #     if hasattr(self.sensor.data, "is_contact"):
        #         on_contact = self.sensor.data.is_contact[:, 0]  # [B] を想定
        #     else:
        #         # フォールバック: net_forces_w のノルムで判定
        #         F = getattr(self.sensor.data, "net_forces_w", None)
        #         if F is not None:
        #             fmag = F[:, 0].norm(dim=-1)                       # [B]
        #             on_contact = fmag > 1.0
        #         else:
        #             on_contact = torch.ones_like(inside, dtype=torch.bool)
        #     good = inside & on_contact
        # else:
        #     good = inside

        # --- タイマ更新＆一回払い ---
        dt = env.step_dt
        self.hold_t = torch.where(good, self.hold_t + dt, torch.zeros_like(self.hold_t))

        # just reached
        reached = (self.hold_t >= self.T_hold_s) & (~self.paid)
        payout  = torch.where(reached, torch.full_like(self.hold_t, self.bonus / dt),
                              torch.zeros_like(self.hold_t))

        # 状態更新
        self.paid  |= reached
        if hasattr(env, "reset_buf"):
            m = env.reset_buf > 0
            if m.any():
                self.hold_t[m] = 0.0
                self.paid[m]   = False

        return payout








# def fr_fz(env):
#     ft = env.scene.sensors["contact_forces"].data.net_forces_w  # [B,4,3]
#     return ft[:, 1, 2]  # FRのZ

# def impact_spike_penalty_fr(env, dfz_thresh=0.15):
#     fz_hist = env._buf.ft_stack[..., 2]  # [B,K,4] (あなたのft_stackをバッファ化しておく)

#     print(fz_hist)

#     FR = 1  # ["FL","FR","RL","RR"] の順想定
#     dfz = fz_hist[:, 1:, FR] - fz_hist[:, :-1, FR]  # 時系列差分 [B, K-1]
#     spike = torch.clamp(dfz, min=0.0).amax(dim=1)   # 上方向スパイクの最大値
#     excess = torch.clamp(spike - dfz_thresh, min=0.0)

#     return excess


#holr alive
# def support_reg(env):
#     return env._buf.is_holding.float()


def ang_vel(env):
    return (_block_wmag(env)**2)

def block_tilt_margin(env):
    return (torch.clamp(_block_theta(env)/0.20, 0, 1.0)**2.5)


# def support_force_band_reward(env: ManagerBasedRLEnv, sensor_cfg,
#                               fz_min=0.03, fz_max=0.20,
#                               leg="FR", mass_kg=15.0, normalize=True):
#     # 接触センサ（ワールド座標の合力）: [B,4,3]  [FL,FR,RL,RR]想定
#     ft = env.scene.sensors[sensor_cfg.name].data.net_forces_w
#     leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}[leg]
#     fz = ft[:, leg_index, 2]  # [B] Newton

#     if normalize:
#         fz = fz / (mass_kg * 9.81)  # → 体重比

#     inside = (fz >= fz_min) & (fz <= fz_max)
#     return inside.float()  # [B]




def fr_target_progress_reward2(env, d_clip=0.05, cmd_name="step_fr_to_block", block_key="stone2"):
    # 目標XY（world）
    cmd = env.command_manager.get_command(cmd_name)  # [B,2] = (ux,uy)
    ux, uy = cmd[..., 0], cmd[..., 1]
    c_xy = _block_pos_w(env, block_key)[..., :2]
    yaw  = _yaw_from_quat(_block_quat_w(env, block_key))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([torch.stack([cy, -sy], dim=-1),
                     torch.stack([sy,  cy], dim=-1)], dim=-2)      # [B,2,2]
    tgt_xy = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + c_xy  # [B,2]

    # FR 足の位置と速度（world）
    robot = env.scene.articulations["robot"]
    fr_id = robot.body_names.index("FR_foot")
    fr_xy = robot.data.body_pos_w[:, fr_id, :2]          # [B,2]
    fr_vxy = robot.data.body_vel_w[:, fr_id, :2]         # [B,2]

    # 方向ベクトル（FR→目標）
    diff = tgt_xy - fr_xy                                # [B,2]
    dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
    dir_to_tgt = diff / dist.unsqueeze(-1)               # [B,2]

    # 接近速度 = dir_to_tgt · v  （正なら近づいている）
    approach_speed = (dir_to_tgt * fr_vxy).sum(dim=-1)   # [B], m/s

    # 1 ステップの距離進捗 ≈ approach_speed * dt
    # dt = getattr(env, "dt", 0.005)
    dt  = env.step_dt
    r = (approach_speed * dt).clamp(min=0.0, max=d_clip) # [B]  ← 常に >=0
    return r




def fr_target_progress_reward3(env, d_clip=0.05, cmd_name="step_fr_to_block", block_key="stone2"):
    # --- 目標XY（world）: あなたのコードそのまま ---
    cmd = env.command_manager.get_command(cmd_name)  # [B,2]
    ux, uy = cmd[..., 0], cmd[..., 1]
    c_xy = _block_pos_w(env, block_key)[..., :2]
    yaw  = _yaw_from_quat(_block_quat_w(env, block_key))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([torch.stack([cy, -sy], dim=-1),
                     torch.stack([sy,  cy], dim=-1)], dim=-2)  # [B,2,2]
    tgt_xy = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + c_xy  # [B,2]

    # --- FRの位置・速度（world） ---
    robot = env.scene.articulations["robot"]
    fr_id = robot.body_names.index("FR_foot")
    fr_pos_w = robot.data.body_pos_w[:, fr_id, :3]      # [B,3]
    fr_v_w   = robot.data.body_vel_w[:, fr_id, :3]      # [B,3]

    # --- 胴体（base）の運動（world） ---
    base_pos_w = robot.data.root_pos_w                  # [B,3]
    base_v_w   = robot.data.root_lin_vel_w              # [B,3]
    base_om_w  = robot.data.root_ang_vel_w              # [B,3]

    r_base_to_fr = fr_pos_w - base_pos_w                # [B,3]
    rigid_v_at_fr = base_v_w + torch.cross(base_om_w, r_base_to_fr, dim=-1)  # [B,3]

    # --- 相対速度（world） ---
    v_rel_w = fr_v_w - rigid_v_at_fr                    # [B,3]
    v_rel_xy = v_rel_w[:, :2]                           # [B,2]

    # --- 方向ベクトル（FR→目標, world） ---
    diff = tgt_xy - fr_pos_w[:, :2]                     # [B,2]
    dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
    dir_to_tgt = diff / dist.unsqueeze(-1)              # [B,2]

    # --- 進歩（符号付き; 往復は相殺） ---
    dt = env.step_dt
    r = (dir_to_tgt * v_rel_xy).sum(dim=-1) * dt        # [B]
    return r.clamp(min=-d_clip, max=d_clip)



# def fr_target_distance_reward(env, cmd_name="step_fr_to_block", block_key="stone2"):
#     """
#     目標地点との「距離」そのものを評価する報酬。
#     近づくほど指数関数的に高くなる (Kernel function)。
#     """
#     # --- 1. 目標座標 (tgt_xy) の計算 (Progressと同じ) ---
#     cmd = env.command_manager.get_command(cmd_name)
#     ux, uy = cmd[..., 0], cmd[..., 1]
#     c_xy = _block_pos_w(env, block_key)[..., :2]
#     yaw  = _yaw_from_quat(_block_quat_w(env, block_key))
#     cy, sy = torch.cos(yaw), torch.sin(yaw)
#     R = torch.stack([torch.stack([cy, -sy], dim=-1),
#                      torch.stack([sy,  cy], dim=-1)], dim=-2)
#     tgt_xy = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + c_xy

#     # --- 2. FR足の現在位置 ---
#     robot = env.scene.articulations["robot"]
#     fr_id = robot.body_names.index("FR_foot")
#     fr_xy = robot.data.body_pos_w[:, fr_id, :2]

#     # --- 3. 距離の計算 ---
#     dist = torch.linalg.norm(tgt_xy - fr_xy, dim=-1)
    
#     # --- 4. 報酬変換 (距離0で1.0, 離れると減衰) ---
#     # sigma が小さいほど「高精度」を要求する (例: 0.1m 〜 0.2m)
#     sigma = 0.1
#     return torch.exp(- (dist / sigma)**2 )




# def fr_target_distance_reward3d(env, cmd_name="step_fr_to_block", block_key="stone2"):
#     """
#     目標地点との「3次元距離」を評価する報酬。
#     XYが合っていても、高さ(Z)が合わなければ報酬が最大にならないようにする。
#     """
#     # --- 1. 目標座標 (tgt_xy) の計算 ---
#     cmd = env.command_manager.get_command(cmd_name)
#     ux, uy = cmd[..., 0], cmd[..., 1]
    
#     block_pos = _block_pos_w(env, block_key) # [B, 3]
#     c_xy = block_pos[..., :2]
#     c_z  = block_pos[..., 2]
    
#     yaw  = _yaw_from_quat(_block_quat_w(env, block_key))
#     cy, sy = torch.cos(yaw), torch.sin(yaw)
#     R = torch.stack([torch.stack([cy, -sy], dim=-1),
#                      torch.stack([sy,  cy], dim=-1)], dim=-2)
    
#     # XYの目標 (石上のローカル座標をワールドへ)
#     tgt_xy = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + c_xy

#     # ★ Zの目標 (重要)
#     # 石の高さ(厚み)が  なので、中心から  が上面。
#     # そこに足の半径(例: 0.02)やめり込みを考慮した目標高さを設定。
#     # ここでは「石の上面」をターゲットにします。
#     stone_half_height = 0.15 
#     tgt_z = c_z + stone_half_height

#     # [B, 3] の目標座標結合
#     tgt_pos = torch.cat([tgt_xy, tgt_z.unsqueeze(-1)], dim=-1)

#     # --- 2. FR足の現在位置 (3D) ---
#     robot = env.scene.articulations["robot"]
#     fr_id = robot.body_names.index("FR_foot")
#     fr_pos = robot.data.body_pos_w[:, fr_id, :3] # [B, 3]

#     # --- 3. 3次元距離の計算 ---
#     dist = torch.linalg.norm(tgt_pos - fr_pos, dim=-1)
    
#     # --- 4. 報酬変換 ---
#     # これにより、空中に浮いていると dist が 0 にならず、報酬が減るため、
#     # ロボットは足を着地させようとします。
#     sigma = 0.1
#     # return torch.exp(- (dist / sigma)**2 )
#     return 1.0 / (1.0 + dist * 5.0)




# def fr_target_distance_reward_3d2(env, cmd_name="step_fr_to_block", block_key="stone2"):
#     # --- 1. 目標座標 (tgt_xy) の計算 ---
#     cmd = env.command_manager.get_command(cmd_name)
#     ux, uy = cmd[..., 0], cmd[..., 1]
    
#     block_pos = _block_pos_w(env, block_key) # [B, 3]
#     blk_pos_w = p[:, 0, :] if p.ndim == 3 else p

#     c_xy = block_pos[..., :2]
#     c_z  = block_pos[..., 2]
    
#     yaw  = _yaw_from_quat(_block_quat_w(env, block_key))
#     cy, sy = torch.cos(yaw), torch.sin(yaw)
#     R = torch.stack([torch.stack([cy, -sy], dim=-1),
#                      torch.stack([sy,  cy], dim=-1)], dim=-2)
    
#     # XYの目標 (石上のローカル座標をワールドへ)
#     tgt_xy = (R @ torch.stack([ux, uy], dim=-1).unsqueeze(-1)).squeeze(-1) + c_xy

#     # ★ Zの目標 (重要)
#     # 石の高さ(厚み)が  なので、中心から  が上面。
#     # そこに足の半径(例: 0.02)やめり込みを考慮した目標高さを設定。
#     # ここでは「石の上面」をターゲットにします。
#     stone_half_height = 0.15 
#     tgt_z = c_z + stone_half_height
#     # ... (目標座標 tgt_xy, tgt_z の計算までは同じ) ...
    
#     # 現在位置
#     robot = env.scene.articulations["robot"]
#     fr_id = robot.body_names.index("FR_foot")

#     # 可視化コードと同様の分岐処理
#     if hasattr(robot.data, "body_link_pose_w"):
#         fr_pose_w = robot.data.body_link_pose_w[:, fr_id] # [B,7]
#         curr_pose = fr_pose_w[:, :3]
#     else:
#         curr_pose = robot.data.body_pos_w[:, fr_id, :3]

#     curr_pos = robot.data.body_pos_w[:, fr_id, :3]
    
#     dist_xy = torch.linalg.norm(tgt_xy - curr_pos[..., :2], dim=-1)
#     dist_z  = torch.abs(tgt_z - curr_pos[..., 2]) # L1 distance for Z is often better

#     # --- ★ 修正ポイント ---
    
#     # 1. XY距離報酬: 近づくことへの報酬
#     rew_xy = 1.0 / (1.0 + dist_xy * 5.0)
    
#     # 2. 足上げ（クリアランス）報酬
#     # 「まだXY的に遠い(例えば10cm以上)なら、足はターゲットより高くあってほしい」
#     # ブロックの高さより少し上を維持するように促す
#     clearance_height = tgt_z + 0.05  # 目標より5cm上
#     is_far = (dist_xy > 0.06).float()
    
#     # 遠い時は「高さ」をターゲットにするのではなく、「クリアランス高さ」周辺にいることを報酬にする
#     # 近い時は「ターゲット高さ(着地)」に合うことを報酬にする
#     target_z_dynamic = is_far * clearance_height + (1.0 - is_far) * tgt_z
    
#     dist_z_dynamic = torch.abs(target_z_dynamic - curr_pos[..., 2])
#     rew_z = 1.0 / (1.0 + dist_z_dynamic * 10.0)

#     # 3. 衝突ペナルティ（オプション）
#     # 足の高さがブロックより低いのに、XYがブロックに近い場合はペナルティ
    
#     # 最終報酬: XYとZの積、あるいは重み付け和
#     return rew_xy * rew_z




def fr_target_distance_reward_3d3(env, cmd_name="step_fr_to_block", block_name="stone2"):
    """
    可視化コードと全く同じロジックで計算する距離報酬
    """
    # --- 1. ブロック位置の取得（可視化コード準拠） ---
    block = env.scene.rigid_objects[block_name]
    
    # 位置: 次元を確認して [B, 3] に整形
    p = block.data.root_pos_w
    blk_pos_w = p[:, 0, :] if p.ndim == 3 else p
    
    # 回転: 次元を確認して [B, 4] -> yaw -> R
    q = block.data.root_quat_w
    blk_quat_w = q[:, 0, :] if q.ndim == 3 else q
    
    w, x, y, z = blk_quat_w.unbind(-1)
    yaw_blk = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
    R = torch.stack(
        [torch.stack([cy, -sy], dim=-1), torch.stack([sy, cy], dim=-1)],
        dim=-2,
    ) # [B,2,2]

    # --- 2. ターゲット座標の計算（可視化コード準拠） ---
    cmd = env.command_manager.get_command(cmd_name) # [B,2] or [B,3]
    ux, uy = cmd[..., 0], cmd[..., 1]
    t_xy = torch.stack([ux, uy], dim=-1) # [B,2]

    # XY: ローカル座標 ux,uy をワールドへ
    tgt_xy_w = (R @ t_xy.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
    
    # ★ Z: 可視化コードと完全に同じオフセットにする
    # もし可視化でマーカーがちょうどいい位置にあるなら、これに従う
    tgt_z_w = blk_pos_w[..., 2] + 0.15
    
    # 目標位置結合 [B,3]
    tgt_pos_w = torch.cat([tgt_xy_w, tgt_z_w.unsqueeze(-1)], dim=-1)

    # --- 3. FR足の位置（可視化コード準拠） ---
    robot = env.scene.articulations["robot"]
    fr_id = robot.body_names.index("FR_foot")
    
    # 可視化コードと同様の分岐処理
    if hasattr(robot.data, "body_link_pose_w"):
        fr_pose_w = robot.data.body_link_pose_w[:, fr_id] # [B,7]
        fr_pos_w = fr_pose_w[:, :3]
    else:
        fr_pos_w = robot.data.body_pos_w[:, fr_id, :3]

    # --- 4. 距離計算と報酬 ---
    # XYとZを分ける（推奨）
    dist_xy = torch.linalg.norm(tgt_pos_w[..., :2] - fr_pos_w[..., :2], dim=-1)
    dist_z  = torch.abs(tgt_pos_w[..., 2] - fr_pos_w[..., 2])
    
    # (A) まずXYを近づける
    rew_xy = 1.0 / (1.0 + dist_xy * 5.0)
    
    # (B) Zに関しては「クリアランス」を考慮
    # ターゲットより遠い(>10cm)ときは、ターゲット高さ+5cm(回避高さ)を目指す
    # ターゲットに近い(<10cm)ときは、ターゲット高さそのもの(着地)を目指す
    is_far = (dist_xy > 0.10).float()
    target_z_adaptive = is_far * (tgt_z_w + 0.05) + (1.0 - is_far) * tgt_z_w
    
    dist_z_adaptive = torch.abs(target_z_adaptive - fr_pos_w[..., 2])
    rew_z = 1.0 / (1.0 + dist_z_adaptive * 10.0)

    return rew_xy * rew_z










def _get_root_pos_w(data):
    if hasattr(data, "root_pos_w"):   return data.root_pos_w
    if hasattr(data, "root_state"):   return data.root_state[..., :3]
    raise AttributeError("root position not found on ArticulationData")

def _get_root_quat_w(data):
    if hasattr(data, "root_quat_w"):     return data.root_quat_w          # [B,4] wxyz
    if hasattr(data, "root_orient_w"):   return data.root_orient_w        # [B,4] wxyz（環境による）
    if hasattr(data, "root_state"):      return data.root_state[..., 3:7] # [B,4] wxyz
    raise AttributeError("root quaternion not found on ArticulationData")

def _rotmat_body_to_world_from_quat_wxyz(q):  # q=(w,x,y,z)
    w,x,y,z = q.unbind(-1)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    r00 = 1 - 2*(yy+zz); r01 = 2*(xy - wz);  r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);   r11 = 1 - 2*(xx+zz); r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);   r21 = 2*(yz + wx);  r22 = 1 - 2*(xx+yy)
    return torch.stack([torch.stack([r00,r01,r02],-1),
                        torch.stack([r10,r11,r12],-1),
                        torch.stack([r20,r21,r22],-1)], -2)

def _yaw_from_quat_wxyz(q):
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def fr_target_distance_reward_3d4(
    env,
    cmd_name="step_fr_to_block",
    block_name="stone2",
    top_offset=0.15,
    far_xy_thresh=0.10,
    z_clearance=0.05,
    sigma_xy=0.10,
    sigma_z =0.05,
):
    """
    ベース座標で (target - FR) の 3D 誤差を取り、異方性ガウシアン核で評価。
    RewardManager 側で *dt が掛かる実装なので、ここでは dt を掛けない。
    返り値: [B]
    """
    # --- ブロック姿勢（world） ---
    block = env.scene.rigid_objects[block_name]
    p = block.data.root_pos_w
    blk_pos_w = p[:, 0, :] if p.ndim == 3 else p                # [B,3]
    q = block.data.root_quat_w
    blk_quat_w = q[:, 0, :] if q.ndim == 3 else q               # [B,4] (wxyz)

    yaw = _yaw_from_quat_wxyz(blk_quat_w)                       # [B]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    Rz = torch.stack([torch.stack([cy,-sy],-1),
                      torch.stack([sy, cy],-1)], -2)            # [B,2,2] block→world

    # --- ターゲット（world） ---
    cmd = env.command_manager.get_command(cmd_name)             # [B,2] (ux,uy)
    tgt_xy_w = (Rz @ cmd.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
    tgt_z_w  = blk_pos_w[..., 2] + top_offset
    tgt_w    = torch.cat([tgt_xy_w, tgt_z_w.unsqueeze(-1)], dim=-1)  # [B,3]

    # --- FR足（world） ---
    robot = env.scene.articulations["robot"]
    fr_id = robot.body_names.index("FR_foot")
    if hasattr(robot.data, "body_link_pose_w"):
        fr_w = robot.data.body_link_pose_w[:, fr_id, :3]
    else:
        fr_w = robot.data.body_pos_w[:, fr_id, :3]             # [B,3]

    # --- base 姿勢（world）→ 回転行列 & 平行移動取得（安全版） ---
    base_p_w = _get_root_pos_w(robot.data)                     # [B,3]
    base_q_w = _get_root_quat_w(robot.data)                    # [B,4] wxyz
    R_bw = _rotmat_body_to_world_from_quat_wxyz(base_q_w)      # [B,3,3]
    R_wb = R_bw.transpose(1, 2)                                # world→base

    # --- world→base へ変換 ---
    tgt_b = torch.einsum('bij,bi->bj', R_wb, tgt_w - base_p_w) # [B,3]
    fr_b  = torch.einsum('bij,bi->bj', R_wb, fr_w  - base_p_w) # [B,3]

    # 遠いときは Z にクリアランス（base で判定＆付与）
    dist_xy_b = (tgt_b[:, :2] - fr_b[:, :2]).norm(dim=-1)      # [B]
    tgt_b_z   = torch.where(dist_xy_b > far_xy_thresh,
                            tgt_b[:, 2] + z_clearance,
                            tgt_b[:, 2])
    tgt_b = torch.stack([tgt_b[:,0], tgt_b[:,1], tgt_b_z], dim=-1)

    # --- ガウシアン核（異方性） ---
    err = tgt_b - fr_b
    e_xy2 = (err[:,0]**2 + err[:,1]**2) / (sigma_xy**2 + 1e-12)
    e_z2  = (err[:,2]**2)               / (sigma_z**2  + 1e-12)
    q = 0.5 * (e_xy2 + e_z2)
    r = torch.exp(-q)                                         # [B]
    return r










from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.envs import ManagerBasedRLEnv

class FRProgressToStone(ManagerTermBase):
    """FR足の target までの3D距離の差分（prev - curr）を返す。
       ManagerTermBase として実装し、prev_dist を内部に保持する。"""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        self.block_name = P.get("block_name", "stone2")
        self.cmd_name   = P.get("cmd_name", "step_fr_to_block")
        self.top_offset = P.get("top_offset", 0.15)
        self.d_clip     = P.get("d_clip", 0.05)

        self.robot = env.scene.articulations["robot"]
        self.fr_id = self.robot.body_names.index("FR_foot")
        self.block = env.scene.rigid_objects[self.block_name]

        # per-env バッファ（このクラスの“状態”）
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # --- 目標 (world) ---
        p = self.block.data.root_pos_w
        blk_pos = p[:, 0, :] if p.ndim == 3 else p      # [B,3]

        q = self.block.data.root_quat_w                 # wxyz
        q = q[:, 0, :] if q.ndim == 3 else q
        w,x,y,z = q.unbind(-1)
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R = torch.stack([torch.stack([cy,-sy],-1), torch.stack([sy,cy],-1)], -2)

        cmd = env.command_manager.get_command(self.cmd_name)        # [B,2]
        tgt_xy = (R @ cmd.unsqueeze(-1)).squeeze(-1) + blk_pos[:, :2]
        tgt_z  = blk_pos[:, 2] + self.top_offset
        tgt    = torch.cat([tgt_xy, tgt_z.unsqueeze(-1)], dim=-1)   # [B,3]

        # --- FR足の位置 (world) ---
        if hasattr(self.robot.data, "body_link_pose_w"):
            fr = self.robot.data.body_link_pose_w[:, self.fr_id, :3]
        else:
            fr = self.robot.data.body_pos_w[:, self.fr_id, :3]

        # --- 距離と進歩 ---
        diff    = tgt - fr
        dist_xy = diff[:, :2].norm(dim=-1)
        dist_z  = diff[:,  2].abs()
        dist    = torch.sqrt(dist_xy**2 + dist_z**2)                # [B]

        # エピソードリセットで prev を初期化
        if hasattr(env, "reset_buf"):
            m = env.reset_buf > 0
            if m.any():
                self.prev_dist[m] = dist[m].detach()

        # 初回 or 未初期化は 0 を返して初期化
        if not torch.isfinite(self.prev_dist).all():
            self.prev_dist[:] = dist.detach()
            return torch.zeros_like(dist)

        delta = (self.prev_dist - dist).clamp(-self.d_clip, self.d_clip)
        self.prev_dist = dist.detach()
        return delta  





def _rotmat_body_to_world_from_quat_wxyz(q):  # q=(w,x,y,z)
    w,x,y,z = q.unbind(-1)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    r00 = 1 - 2*(yy+zz); r01 = 2*(xy - wz);  r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);   r11 = 1 - 2*(xx+zz); r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);   r21 = 2*(yz + wx);  r22 = 1 - 2*(xx+yy)
    return torch.stack([torch.stack([r00,r01,r02],-1),
                        torch.stack([r10,r11,r12],-1),
                        torch.stack([r20,r21,r22],-1)], -2)

class FRProgressToStoneBase(ManagerTermBase):
    """FR足の target までの3D距離（base座標）の差分（prev - curr）を返す。"""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        P = cfg.params
        self.block_name = P.get("block_name", "stone2")
        self.cmd_name   = P.get("cmd_name", "step_fr_to_block")
        self.top_offset = P.get("top_offset", 0.15)
        self.d_clip     = P.get("d_clip", 0.1)

        self.robot = env.scene.articulations["robot"]
        self.fr_id = self.robot.body_names.index("FR_foot")
        self.block = env.scene.rigid_objects[self.block_name]

        # per-env バッファ（base距離の前回値）
        self.prev_dist = torch.full((env.num_envs,), float("nan"), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # --- ブロック姿勢（world） ---
        p = self.block.data.root_pos_w
        blk_pos = p[:, 0, :] if p.ndim == 3 else p                    # [B,3]
        q = self.block.data.root_quat_w
        q = q[:, 0, :] if q.ndim == 3 else q                          # [B,4] (wxyz)
        w,x,y,z = q.unbind(-1)
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))           # [B]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R2 = torch.stack([torch.stack([cy,-sy],-1),
                          torch.stack([sy, cy],-1)], -2)              # [B,2,2] block→world (yaw)

        # --- ターゲット（world） ---
        cmd = env.command_manager.get_command(self.cmd_name)          # [B,2] (ux,uy)
        tgt_xy = (R2 @ cmd.unsqueeze(-1)).squeeze(-1) + blk_pos[:, :2]
        tgt_z  = blk_pos[:, 2] + self.top_offset
        tgt_w  = torch.cat([tgt_xy, tgt_z.unsqueeze(-1)], dim=-1)     # [B,3]

        # --- FR足（world） ---
        if hasattr(self.robot.data, "body_link_pose_w"):
            fr_w = self.robot.data.body_link_pose_w[:, self.fr_id, :3]
        else:
            fr_w = self.robot.data.body_pos_w[:, self.fr_id, :3]      # [B,3]

        # --- base姿勢（world） ---
        base_p = self.robot.data.root_pos_w                           # [B,3]
        if hasattr(self.robot.data, "root_quat_w"):
            base_q = self.robot.data.root_quat_w                      # [B,4] (wxyz)
        else:
            base_q = self.robot.data.root_state[..., 3:7]             # [B,4] (wxyz想定)

        R_bw = _rotmat_body_to_world_from_quat_wxyz(base_q)           # [B,3,3]
        R_wb = R_bw.transpose(1, 2)                                    # world→base

        # --- world→base 変換 ---
        tgt_b = torch.einsum('bij,bi->bj', R_wb, tgt_w - base_p)      # [B,3]
        fr_b  = torch.einsum('bij,bi->bj', R_wb, fr_w  - base_p)      # [B,3]

        # --- base座標の3D距離 & 進歩 ---
        dist = (tgt_b - fr_b).norm(dim=-1)                            # [B]

        # リセット時は prev を再初期化
        if hasattr(env, "reset_buf"):
            m = env.reset_buf > 0
            if m.any():
                self.prev_dist[m] = dist[m].detach()

        # 初回は0返して初期化
        if not torch.isfinite(self.prev_dist).all():
            self.prev_dist[:] = dist.detach()
            return torch.zeros_like(dist)

        delta = (self.prev_dist - dist).clamp(-self.d_clip, self.d_clip)
        self.prev_dist = dist.detach()
        return delta