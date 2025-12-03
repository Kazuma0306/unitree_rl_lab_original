
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def randomize_multiple_stones(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stone_names: list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
):
    """複数のブロック(Stone)をまとめて reset_root_state_uniform でランダム化するイベント."""
    # env_ids が None の場合は全 env を対象
    if env_ids is None:
        import torch
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    for name in stone_names:
        mdp.reset_root_state_uniform(
            env=env,
            env_ids=env_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=SceneEntityCfg(name),
        )