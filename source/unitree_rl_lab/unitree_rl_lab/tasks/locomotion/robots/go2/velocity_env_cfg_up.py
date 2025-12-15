#Upper Model  Environment

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers



from isaaclab.utils.math import quat_apply_inverse
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd


from isaaclab.envs.mdp import StepFRToBlockCommandCfg, MultiLegBaseCommandCfg


from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg7 import RobotEnvCfg

LOW_LEVEL_ENV_CFG = RobotEnvCfg()







COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "descrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #      proportion=0.1, border_width=0.25, obstacle_width_range =(0.8,  0.9) , obstacle_height_range = (0.01, 0.05), num_obstacles = 80
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)



STEPPING_STONES_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),

        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=0.2, border_width=0.25,  horizontal_scale = 0.01, stone_height_max = 0.01, stone_width_range = (1.0, 1.5), stone_distance_range = (0.05, 0.08),  holes_depth = -5.0, platform_width = 1.5,

        # ),

        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
             proportion=0.7, border_width=0.25,  horizontal_scale = 0.01, stone_height_max = 0.01, stone_width_range = (0.7, 1.5), stone_distance_range = (0.05, 0.09),  holes_depth = -5.0, platform_width = 1.5,

        ),


        #difficult version
        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=0.9, border_width=0.25,  horizontal_scale = 0.05, stone_height_max = 0.01, stone_width_range = (0.5, 0.9), stone_distance_range = (0.06, 0.09),  holes_depth = -5.0, platform_width = 1.5,

        # ),

        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=1.0, border_width=0.25,  stone_height_max = 0.05, stone_width_range = (0.2, 0.9), stone_distance_range = (0.08, 0.26),  holes_depth = -5.0, platform_width = 1.5,

        # ),
    },
)

BLOCK_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    # border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={

        "block_terrain": terrain_gen.MeshBlockTerrainCfg(
             proportion=0.7, 

        ),

    },
)



MOAT_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    # border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={

        "moat_terrain": terrain_gen.MeshMoatTerrainCfg(
             proportion=0.7, platform_width = 1.0,

        ),

    },
)



sizex = 0.2
sizey = 0.4
MASS_B = 100.0

# MASS_B = 0.1


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        # terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # terrain_generator=DESCRETE_OBSTACLES_CFG,
        # terrain_generator= STEPPING_STONES_CFG, 
        terrain_generator= MOAT_CFG, # proposed env
        # terrain_generator= BLOCK_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #for proposed terrain
    # stones: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    #         rigid_objects=stones_dict
    #     )



    # #左前
    stone1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_1",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.4, 0.2, -0.14)
            pos=(1.2, 0.2, -0.14)
        )
    )

    #右前
    stone2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_2",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.4, -0.2, -0.14)
            pos=(1.2, -0.2, -0.14)
        )
    )

    #ML
    stone3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_3",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.8, 0.2, -0.14)
        )
    )


    #MR
    stone4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_4",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.8, -0.2, -0.14)
        )
    )


    #FL
    stone5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_5",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.2, 0.2, -0.14)
            pos=(1.0, 0.2, -0.14)
        )
    )


    #FR
    stone6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_6",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.2, -0.2, -0.14)
            pos=(1.0, -0.2, -0.14)
        )
    )

    #RL
    stone7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_7",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(-0.2, 0.2, -0.14)
            pos=(0.6, 0.2, -0.14)
        )
    )

    #RR
    stone8 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_8",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(-0.2, -0.2, -0.14)
            pos=(0.6, -0.2, -0.14)
        )
    )

    #左前2
    stone9 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_9",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.6, 0.2, -0.14)
            pos=(1.4, 0.2, -0.14)
        )
    )

    #右前2
    stone10 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_10",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            # pos=(0.6, -0.2, -0.14)
            pos=(1.4, -0.2, -0.14)
        )
    )


    
    #左前3
    stone11 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_11",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(1.6, 0.2, -0.14)
        )
    )

    #右前3
    stone12 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_12",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(1.6, -0.2, -0.14)
        )
    )


    #左前4
    stone13 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_13",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(1.8, 0.2, -0.14)
        )
    )


    #右前4
    stone14= RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_14",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(1.8, -0.2, -0.14)
        )
    )



    #左前4
    stone15 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_15",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(2.0, 0.2, -0.14)
        )
    )


    #右前4
    stone16= RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_16",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(2.0, -0.2, -0.14)
        )
    )









   


    

    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.1, 3.1]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     debug_vis = True,
    #     update_period=0.02, #50Hz, Same as Low Policy, Ideal 30Hz TODO
    #     height=64,
    #     width=64,
    #     data_types=["depth"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
    #     # ),
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=16.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
    #     # ),

    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=10.5, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.1, 3.0)
    #     ),
    #     depth_clipping_behavior = "max",
    #     # depth_clipping_behavior = "zero",
    #     # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.32, 0.0, 0.15), rot=(0.2418, -0.6645,  0.6645, -0.2418), convention="ros"),
    #     offset=CameraCfg.OffsetCfg(pos=(0.37, 0.0, 0.15), rot=(0.0616, -0.7044, 0.7044, -0.0616), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 1), rot = (0.0, -0.7071, 0.7071, 0.0), convention="ros"),



    # )

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     debug_vis = True,
    #     update_period=0.0,
    #     height=64,
    #     width=64,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )



    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=12 , track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )





@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.FootstepPolicyActionCfg = mdp.FootstepPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt", #TODO
        policy_path=f"/home/digital/isaac_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_proposed4/2025-12-12_13-10-05/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction, #lower's action
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy, # lower's observation
    )



@configclass
class HighLevelPolicyObsCfg(ObsGroup):

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100)) #last command

    leg_position = ObsTerm(func = mdp.ee_pos_base_obs)#ベース座標系での脚位置


    # front_depth = ObsTerm(
    #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    # )

    # heightmap = ObsTerm(
    # func=mdp.depth_heightmap4,
    # params=dict(
    #     sensor_cfg=SceneEntityCfg("camera"),  # 定義したカメラ名
    #     asset_cfg=SceneEntityCfg("robot"),
    #     x_range=(0.1, 3.0),
    #     y_range=(-2.0, 2.0),
    #     grid_shape=(64, 64),
    #     default_height=-4,
    # ),
    # )

    # heightmap = ObsTerm(
    #     func=mdp.image,
    #     # env.scene.sensors["camera"] に合わせて名前を変える
    #     params={
    #         "sensor_cfg": SceneEntityCfg("camera"),
    #         "data_type": "depth",   # TiledCameraCfg.data_types=["depth"] と合わせる
    #         "normalize": True,      # depthの inf を0にするなどの処理をしてくれる
    #     },
    #     # ↓ここがキモ：このtermだけ履歴を持たせる
    #     history_length=2,           # 4フレーム分スタック
    #     flatten_history_dim=True,   # (B,4,H,W,C) → (B, 4*H*W*C)にflatten
    # )

    heightmap = ObsTerm(
        func = mdp.obs_near_blocks

    )

    ft_stack = ObsTerm(
            func=mdp.contact_ft_stack,   # 下の関数
            params=dict(
                sensor_cfg=SceneEntityCfg("contact_forces",
                # body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                body_names=".*_foot"),
                mass_kg=15.0,
                return_shape="flat",
                components="z"
            ),
            clip=(-3.0, 3.0),
        )


    def __post_init__(self):
            # self.history_length = 3
            # self.enable_corruption = True
            self.concatenate_terms = False




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: mdp.ObservationGroupCfg = HighLevelPolicyObsCfg()
    critic: mdp.ObservationGroupCfg = HighLevelPolicyObsCfg()


    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity)
    #     pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    #     # front_depth = ObsTerm(
    #     #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    #     # )

    #     heightmap = ObsTerm(
    #     func=mdp.depth_heightmap,
    #     params=dict(
    #         sensor_cfg=SceneEntityCfg("tiled_camera"),  # 定義したカメラ名
    #         asset_cfg=SceneEntityCfg("robot"),
    #         x_range=(-1.0, 1.0),
    #         y_range=(-0.8, 0.8),
    #         grid_shape=(32, 32),
    #         default_height=0.0,
    #     ),
    #     )


    #     # def __post_init__(self):
    #     #     # self.history_length = 3
    #     #     self.enable_corruption = True
    #     #     self.concatenate_terms = False


    # # observation groups
    # policy: PolicyCfg = PolicyCfg()


    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for critic group."""
    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity)
    #     pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    #     # front_depth = ObsTerm(
    #     #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    #     # )

    #     heightmap = ObsTerm(
    #     func=mdp.depth_heightmap,
    #     params=dict(
    #         sensor_cfg=SceneEntityCfg("tiled_camera"),  # 定義したカメラ名
    #         asset_cfg=SceneEntityCfg("robot"),
    #         x_range=(-1.0, 1.0),
    #         y_range=(-0.8, 0.8),
    #         grid_shape=(32, 32),
    #         default_height=0.0,
    #     ),
    #     )

    #     # def __post_init__(self):
    #     #     # self.history_length = 3
    #     #     # self.enable_corruption = True
    #     #     self.concatenate_terms = False


    # critic: CriticCfg = CriticCfg()






@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.3,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.3,
        params={"std": 1.6, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.6,
        params={"command_name": "pose_command"},
    )

    distance_progress = RewTerm(func= mdp.BaseProgressToTargetRel, weight = 15)#ベース座標系での進捗, all legs weighted sum



    # joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # energy = RewTerm(func=mdp.energy, weight=-3e-5)

    # -- robot
    # dont_wait = RewTerm(
    #     func=mdp.dont_wait_rel, weight=-1.0, # 前回実装した自作関数 -2.0 for simple walking
    #     params={"velocity_threshold": 0.2, "distance_threshold":0.8, "command_name": "pose_command"}
    # )

    dont_wait = RewTerm(
        func=mdp.dont_wait_rel3, weight=-1, 
        params={"distance_threshold": 0.2, "max_distance":0.8, "command_name": "pose_command"}
    )




    # base_acc = RewTerm(func = mdp.base_accel_l2, weight = -0.0005)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0) 

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )








@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(24.0, 24.0),
        debug_vis=True,
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.5, 1.8), pos_y=(-0.0, 0.0), heading=(-0, 0)),
    )

    step_fr_to_block = mdp.FootstepFromHighLevelCfg(
        debug_vis = True
    )







@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


    # schedule_lin = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "position_tracking_fine_grained",
    #         "weight": 0.1,
    #         "num_steps": 25000
    #     }
    # )

    schedule_lin = CurrTerm(
        func = mdp.schedule_reward_weight,
        params = {
            "term_name" : "distance_progress",
            "weight": 1,
            "num_steps": 25000
        }
    )





@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the walking environment."""

    # environment settings
    # scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    scene: SceneEntityCfg = RobotSceneCfg(num_envs=2048, env_spacing=2.5)

    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=256, env_spacing=2.5)
    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=2, env_spacing=2.5)

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5#TODO　５Hz
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

       
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 2
        #     self.scene.terrain.terrain_generator.num_cols = 2





class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False