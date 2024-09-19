from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures.state import StateArray, StateTensor





def main():
    train_dataset = UnifiedDataset(
        desired_data=["nusc_trainval-train", "deepurban_trainval-train_SanFrancisco"],
        centric="scene",
        desired_dt=0.1,
        only_types=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        state_format="x,y,z,xd,yd,h",
        obs_format="x,y,z,xd,yd,h",
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        standardize_data=False,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=1,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "deepurban_trainval": "~/DeepUrban/deepurban_scenarios",
            "nusc_trainval": "~/nuscenes",
            
        },
    )
    val_dataset = UnifiedDataset(
        desired_data=["nusc_trainval-val","deepurban_trainval-val_SanFrancisco"],
        centric="scene",
        desired_dt=0.1,
        only_types=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        state_format="x,y,z,xd,yd,h",
        obs_format="x,y,z,xd,yd,h",
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        standardize_data=False,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=1,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "deepurban_trainval": "~/DeepUrban/deepurban_scenarios",
            "nusc_trainval": "~/nuscenes",
            
        },
    )
    
    
    print(f"# Data Samples: {len(train_dataset):,}")
    print(f"# Data Samples: {len(val_dataset):,}")
    


if __name__ == "__main__":
    main()
