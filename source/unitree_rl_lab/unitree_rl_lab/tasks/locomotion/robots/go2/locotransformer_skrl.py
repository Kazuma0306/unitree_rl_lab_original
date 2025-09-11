from skrl.models.torch import Model
import torch

class VisionMLPPolicy(Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        proprio_dim = observation_space["base_lin_vel"].shape[0] \
                    + observation_space["base_ang_vel"].shape[0] \
                    + observation_space["projected_gravity"].shape[0] \
                    + observation_space["position_commands"].shape[0]\
                    + observation_space["joint_pos_rel"].shape[0]\
                    + observation_space["joint_vel_rel"].shape[0]\
                    + observation_space["last_action"].shape[0]
               
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 8, stride=4),  # (C, H, W) → features
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        cnn_out_dim = 64 * 6 * 6  # 実際の出力次元に合わせて

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(proprio_dim + cnn_out_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_space.shape[0])
        )

    def compute(self, inputs, role):
        obs = inputs["states"]
        proprio = torch.cat([
            obs["base_lin_vel"],
            obs["base_ang_vel"],
            obs["projected_gravity"],
            obs["joint_pos_rel"],
            obs["joint_vel_rel"],
            obs["last_action"],
            obs["position_commands"],
        ], dim=-1)

        camera = obs["camera_image"].permute(0, 3, 1, 2)  # (B, C, H, W)
        features = self.cnn(camera)

        x = torch.cat([proprio, features], dim=-1)
        return self.mlp(x), {}
