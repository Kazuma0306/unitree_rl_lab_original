from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING




# # --- 設定クラス: CommandTermCfg を継承 ---
# @configclass
# class StepFRToBlockCommandCfg(CommandTermCfg):
#     """FR用 単一ブロックターゲット (ux, uy) を出すコマンド設定"""
#     class_type: type = StepFRToBlockCommand
#     # リサンプリング周期 [s]
#     resampling_time_range: tuple[float, float] = (2.0, 3.0)
#     debug_vis: bool = False
#     # ブロック中心からのローカル・オフセット範囲（m）
#     local_offset_range: tuple[float, float] = (-0.02, 0.02)




# @configclass
# class StepTargetCommandCfg(CommandTermCfg):
#     """Configuration for the uniform velocity command generator."""

#     class_type: type = StepTargetCommandCfg

#     asset_name: str = MISSING
#     """Name of the asset in the environment for which the commands are generated."""

#     heading_command: bool = False
#     """Whether to use heading command or angular velocity command. Defaults to False.

#     If True, the angular velocity command is computed from the heading error, where the
#     target heading is sampled uniformly from provided range. Otherwise, the angular velocity
#     command is sampled uniformly from provided range.
#     """

#     heading_control_stiffness: float = 1.0
#     """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

#     rel_standing_envs: float = 0.0
#     """The sampled probability of environments that should be standing still. Defaults to 0.0."""

#     rel_heading_envs: float = 1.0
#     """The sampled probability of environments where the robots follow the heading-based angular velocity command
#     (the others follow the sampled angular velocity command). Defaults to 1.0.

#     This parameter is only used if :attr:`heading_command` is True.
#     """

#     @configclass
#     class Ranges:
#         """Uniform distribution ranges for the velocity commands."""

#         lin_vel_x: tuple[float, float] = MISSING
#         """Range for the linear-x velocity command (in m/s)."""

#         lin_vel_y: tuple[float, float] = MISSING
#         """Range for the linear-y velocity command (in m/s)."""

#         ang_vel_z: tuple[float, float] = MISSING
#         """Range for the angular-z velocity command (in rad/s)."""

#         heading: tuple[float, float] | None = None
#         """Range for the heading command (in rad). Defaults to None.

#         This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
#         """

#     ranges: Ranges = MISSING
#     """Distribution ranges for the velocity commands."""

#     goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
#         prim_path="/Visuals/Command/velocity_goal"
#     )
#     """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

#     current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
#         prim_path="/Visuals/Command/velocity_current"
#     )
#     """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

#     # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
#     goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
#     current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
