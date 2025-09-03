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


def stand_still(
    env: ManagerBasedRLEnv,distance_threshold: float, command_name: str = "goal_position", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)

    reward = -torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (distance < distance_threshold)


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


# def joint_position_penalty(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
# ) -> torch.Tensor:
#     """Penalize joint position error from default on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


# def joint_position_penalty(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     stand_still_scale: float,
#     velocity_threshold: float,
#     distance_threshold: float, # <-- 新しい引数を追加
#     command_name: str,
# ) -> torch.Tensor:
#     """ナビゲーションタスクに合わせて修正された関節位置ペナルティ。"""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]

#     # [修正点 1] ロボットの現在位置と目標位置から、目標までの距離を計算
#     robot_pos = asset.data.root_pos_w
#     target_pos = env.command_manager.get_command(command_name)
#     distance_to_target = torch.linalg.norm(target_pos[:, :2] - robot_pos[:, :2], dim=1)
#     # [修正点 2] 「動くべきか」の判定ロジックを変更
#     #             コマンドのノルム > 0  ->  目標までの距離 > 閾値
#     should_be_moving = distance_to_target > distance_threshold
#     # ロボットが実際に動いているかの判定はそのまま
#     is_moving = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) > velocity_threshold
#     # 関節のズレに対するペナルティの大きさは同じ
#     penalty = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     # 「動くべき」または「動いている」場合は通常のペナルティ、
#     # 「止まるべき（目標に到達済み）」かつ「止まっている」場合はより強いペナルティを適用
#     return torch.where(torch.logical_or(should_be_moving, is_moving), penalty, stand_still_scale * penalty)


def joint_position_penalty(
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