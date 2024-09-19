import glob
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Final, Generator, Iterable, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

import numpy as np
import pandas as pd
import dill
import json
from tqdm import tqdm
import time

from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps import TrafficLightStatus, VectorMap

from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)

def get_map_metadata(map_name: str, map_path: str) -> VectorMap:
    """Get the metadata for the given map.
    Args:
        map_name: The name of the map.
        map_path: The path to the map.
    Returns:
        The metadata for the map.
    """
    # Load map_meta.json
    with open(map_path + "/location_meta.json", "r") as f:
        map_meta = json.load(f)
        
    latitude = map_meta[map_name]["latitude"]
    longitude = map_meta[map_name]["longitude"]
    
    return (latitude, longitude)

def get_locations(data_dir: str) -> List[str]:
    """Get the locations in the given directory.
    Args:
        data_dir: The directory containing the dataset.
    Returns:
        A tuple of locations.
    """
    # Get the locations
    location_path = os.path.join(os.path.dirname(data_dir), "maps", "location_meta.json")
    with open(location_path, "r") as f:
        location_data = json.load(f)
    locations = tuple(location_data.keys())
    
    return locations

def get_dt(data_dir: str) -> float:
    """
    Get the delta time for the given directory.
    Args:
        data_dir: The directory containing the dataset.
    Returns:
        The delta time for the dataset.
    """
    # Get the delta time
    dt_path = os.path.join(os.path.dirname(data_dir), "maps", "location_meta.json")
    with open(dt_path, "r") as f:
        dt_data = json.load(f)
    dt = dt_data["dt"]
    return dt


DEEPURBAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}


def deepurban_type_to_unified_type(deepurban_type: int) -> AgentType:
    if deepurban_type == 1:
        return AgentType.VEHICLE
    elif deepurban_type == 2:
        return AgentType.PEDESTRIAN
    elif deepurban_type == 3:
        return AgentType.BICYCLE
    elif deepurban_type == 5:
        return AgentType.MOTORCYCLE
    elif deepurban_type == 4:
        return AgentType.TRAILER
    elif deepurban_type == 6:
        return AgentType.TRUCK
    elif deepurban_type == 7:
        return AgentType.BUS
    elif deepurban_type == 8:
        return AgentType.SCOOTER
    elif deepurban_type == 9:
        return AgentType.TRAIN
    elif deepurban_type == 10:
        return AgentType.ANIMAL
    elif deepurban_type == 11:
        return AgentType.BICYCLERACK
    elif deepurban_type == 12:
        return AgentType.MOVABLEOBJECT
    elif deepurban_type == 13:
        return AgentType.AGRICULTURALVEHICLE
    elif deepurban_type == 14:
        return AgentType.CONSTRUCTIONVEHICLE
    elif deepurban_type == 15:
        return AgentType.PICKUP
    elif deepurban_type == 16:
        return AgentType.VAN
    else:
        return AgentType.UNKNOWN
    
    
    
def load_scenario_data(data_dir: str, location: str, name: str, scene_dt: float) -> List[Dict[str, str]]:
    """Load the dataset from the given directory.
    Args:
        data_dir: The directory containing the dataset.
        name: The name of the scenario.
    Returns:
        A list of dictionaries containing the information for the scenario.
    """
    
    # Get the dill file name
    fname = glob.glob(str(data_dir) + "/" + location + "/" + name + ".dill")[0]
    
    # Load the dill file
    with open(fname, "rb") as f:
        loaded_dataset = dill.load(f)
        
    
    # the loaded dataset is a pandas dataframe with a dictionary and axe[0] as the agent_id and axe[1] as the scene_ts
    df_dill_data = []
    for scene_ts, scene_data in loaded_dataset.items():
        for agent_id, agent_data in scene_data.items():
            # check if agent_data is nan
            if isinstance(agent_data, float):
                continue
            category_id = deepurban_type_to_unified_type(agent_data["category_id"])
            
            if agent_data["dimension"] is None:
                agent_data["dimension"] = [0, 0, 0]
            
            df_dill_data.append({
                "agent_id": int(agent_id),
                "scene_ts": int(scene_ts),
                "x": agent_data["translation"][0],
                "y": agent_data["translation"][1],
                "z": agent_data["translation"][2],
                "length": agent_data["dimension"][0],
                "width": agent_data["dimension"][1],
                "height": agent_data["dimension"][2],
                "vx": agent_data["velocity"][0],
                "vy": agent_data["velocity"][1],
                "vz": agent_data["velocity"][2],
                "ax": agent_data["acceleration"][0],
                "ay": agent_data["acceleration"][1],
                "az": agent_data["acceleration"][2],
                "category_id": AgentType(category_id),
                "ego_vehicle": agent_data["ego_vehicle"],
                
            })
            
    df = pd.DataFrame(df_dill_data)
    
    # # print min and max value of each column
    # print("Min and Max value of each column")
    # for col in df.columns:
    #     print(f"{col}: {df[col].min()} - {df[col].max()}")
        
    
    # Sort the DataFrame by agent_id and scene_ts
    df = df.sort_values(by=["agent_id", "scene_ts"])
    df = df.reset_index(drop=True)
    
    return df

# function to read in log_scenario_split.json file and return a tuple of two dictionaries (inverted_dict, loaded_dataset)
def read_split_log(data_dir: str, env_name: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Read in the log_scenario_split.json file and return a list of dictionaries.
    Args:
        data_dir: The directory containing the dataset.
    Returns:
        A list of dictionaries containing the information for the scenario.
    """
    name = env_name.split("_")[1]
    # Get the json file name
    # Read in all json files in split folder and name in filename
    pattern = f"{data_dir}/../split/*{name}*.json"
    json_files = glob.glob(pattern)
    
    loaded_dataset = {}
    for fname in json_files:
        # Load the json file
        with open(fname, "r") as f:
            data = json.load(f)
            loaded_dataset.update(data)
    
     
    return loaded_dataset


# Define a function to create an AgentMetadata object for each group
def create_agent_metadata(group):
    agent_info = AgentMetadata(
        name=group.name,
        agent_type=AgentType(group["category_id"].iloc[0]),
        first_timestep=group["scene_ts"].min(),
        last_timestep=group["scene_ts"].max(),
        extent=FixedExtent(
            length=group["length"].iloc[0],
            width=group["width"].iloc[0],
            height=group["height"].iloc[0],
        ),
    )
    return agent_info

def create_agent_metadata_for_presence(agent_df, agent_id):
    group = agent_df.loc[agent_id]
    agent_info = AgentMetadata(
        name=agent_id,
        agent_type=AgentType(group["category_id"].iloc[0]),
        first_timestep=group["scene_ts"].min(),
        last_timestep=group["scene_ts"].max(),
        extent=FixedExtent(
            length=group["length"].iloc[0],
            width=group["width"].iloc[0],
            height=group["height"].iloc[0],
        ),
    )
    return agent_info
    
        
    
    
    