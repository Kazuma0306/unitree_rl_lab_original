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
