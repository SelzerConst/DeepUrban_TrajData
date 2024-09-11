from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import os

from itertools import chain


import numpy as np
import pandas as pd
from tqdm import tqdm
import lanelet2

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    AgentMetadata,
    AgentType,
    FixedExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.deepurban import deepurban_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import DeepUrbanSceneRecord
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane
import time


class DeepUrbanDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        """
        Computes the metadata for the DeepUrban dataset
        
        Args:
            env_name: Name of the environment (e.g."train")
            data_dir: Path to the data directory
        Returns:
            EnvMetadata object: Metadata object containing information about the environment
        """
        scene_split_map = deepurban_utils.read_split_log(
        data_dir=data_dir, env_name=env_name
        )
        locations = deepurban_utils.get_locations(data_dir)
        if env_name == "deepurban_trainval":
            # combine deepurban_utils.DEEPURBAN_LOCATIONS with train and val as "train_location", "val_location"
            dataset_parts: List[Tuple[str, ...]] = [
            tuple(chain.from_iterable(('train_' + location, 'val_' + location) for location in locations))
            ]
        elif env_name == "deepurban_test":
            # combine deepurban_utils.DEEPURBAN_LOCATIONS with test as "test_location"
            dataset_parts: List[Tuple[str, ...]] = [
            tuple(chain.from_iterable(('test_' + location) for location in locations))
            ]
        
        DEEPURBAN_DT = deepurban_utils.get_dt(data_dir)
        
        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=DEEPURBAN_DT,
            parts=dataset_parts,
            map_locations=locations,
            scene_split_map=scene_split_map,
        )
    
    
    
    def load_dataset_obj(self, verbose: bool = False) -> None:
        """
        Loads the dataset object
        
        Args:
            verbose: Boolean value if the loading process should be printed
        """
        
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)
        
        
        self.dataset_obj: Dict[str, pd.DataFrame] = dict()
        
        # invert the scene split map (-> e.g. "train": "munichtal_1_0_250, munichtal_2_0_250, ...")      
        inverse_scene_split_map = {}
        
        for key, value in self.metadata.scene_split_map.items():
            if value not in inverse_scene_split_map:
                inverse_scene_split_map[value] = [key]
            else:
                inverse_scene_split_map[value].append(key)
        if self.name == "deepurban_trainval":
            # Get all scene names for train and val
            TRAIN_SCENES = inverse_scene_split_map["train"]
            VAL_SCENES = inverse_scene_split_map["val"]
            ALL_SCENES = TRAIN_SCENES + VAL_SCENES
        
        if self.name == "deepurban_test":
            # Get all scene names for test
            ALL_SCENES = inverse_scene_split_map["test"]
        
        # start tqdm progress bar
        pbar = tqdm(total=len(ALL_SCENES))

        # Load the dataset with scene_split_map
        for scene_name in ALL_SCENES:
            # Load the dataset
            self.dataset_obj [ scene_name ] = deepurban_utils.load_scenario_data(
                data_dir=self.metadata.data_dir, location=scene_name.split('_')[0], name=scene_name, scene_dt=self.metadata.dt
            )
            pbar.update(1)
                
        
    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        """
        Get matching scenes from dataset object
        
        Args:
            scene_tag: SceneTag object
            scene_desc_contains: List of strings that the scene description must contain
            env_cache: EnvCache object
        Returns:
            List[SceneMetadata]: List of SceneMetadata objects
        """
        all_scenes_list: List[DeepUrbanSceneRecord] = list()
        # Implement how to get matching scenes from dataset object
        # You need to create a SceneMetadata object for each matching scene
        scenes_list: List[SceneMetadata] = list()
            
        inverse_scene_split_map = {}
        
        for key, value in self.metadata.scene_split_map.items():
            if value not in inverse_scene_split_map:
                inverse_scene_split_map[value] = [key]
            else:
                inverse_scene_split_map[value].append(key)
        
        if self.name == "deepurban_trainval":
            # Get all scene names for train
            TRAIN_SCENES = inverse_scene_split_map["train"]
            # Get all scene names for val
            VAL_SCENES = inverse_scene_split_map["val"]
            ALL_SCENES = TRAIN_SCENES + VAL_SCENES
            
        if self.name == "deepurban_test":
            # Get all scene names for test
            ALL_SCENES = inverse_scene_split_map["test"]
            
        # Iterate over all scenes in the dataset
        for idx, scene_name in enumerate(ALL_SCENES):
            
            scene_length = self.dataset_obj[scene_name]["scene_ts"].max().item() + 1
            scene_location = scene_name.split("_")[0]
            scene_split: str = self.metadata.scene_split_map[scene_name]
            
            all_scenes_list.append(
                DeepUrbanSceneRecord(
                    scene_name, 
                    scene_location,
                    scene_length,
                    scene_split,
                    idx,
                )
            )
            full_scene_split = f"{scene_split}_{scene_location}"
            if (
                scene_location in str(scene_tag)
                and full_scene_split in str(scene_tag)
                and scene_desc_contains is None
            ):
                scenes_list.append(
                    SceneMetadata(
                        # Environment name (e.g. "ds_small")
                        env_name=self.metadata.name,
                        # Scene name (e.g. "munichtal_0_0_250")
                        name= scene_name,
                        # Frame period (e.g. 0.08)
                        dt=self.metadata.dt,
                        # Data idx (e.g. 0)
                        raw_data_idx=idx,
                    )
            )
        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list
    
    
    
    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        """
        Get matching scenes from cache
        
        Args:
            scene_tag: SceneTag object
            scene_desc_contains: List of strings that the scene description must contain
            env_cache: EnvCache object
        Returns:
            List[Scene]: List of Scene objects
        """
        all_scenes_list: List[DeepUrbanSceneRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                data_idx,
            ) = scene_record
            
            full_scene_split = f"{scene_split}_{scene_location}"
            if (
                scene_location in str(scene_tag)
                and full_scene_split in str(scene_tag)
                and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)
        return scenes_list
    
    
    
    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        """
        Get information about a specific scene
        
        Args:
            scene_info: SceneMetadata object
        Returns:
            Scene object: Scene object containing information about the scene
        """
        
        # Implement how to get information about a specific scene
        _ , scene_name, _, data_idx = scene_info
        
        scene_data: pd.DataFrame = self.dataset_obj[scene_name]
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_data["scene_ts"].max().item() + 1
        scene_location: str = scene_name.split("_")[0]
        
        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info if we're not using the cache.
        )
        
    
    
    
    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """
        Get information about the agents in a specific scene
        
        Args:
            scene: Scene object
            cache_path: Path to the cache directory
            cache_class: SceneCache class
        Returns:
            Tuple[List[AgentMetadata], List[List[AgentMetadata]]]: Tuple of agent_list and agent_presence
        """
        
            
        # Copy self.dataset_obj[scene.name] dictionary to scene_data
        scene_data: pd.DataFrame = self.dataset_obj[scene.name].copy()
        
        # Add heading column which is the yaw column for standardization
        scene_data["heading"] = np.arctan2(scene_data["vy"], scene_data["vx"])
        
        # Change agend_id column to string
        scene_data["agent_id"] = scene_data["agent_id"].astype(str)
        
        #Change agend_id column at the ego agent to "ego"
        scene_data.loc[scene_data["ego_vehicle"] == 1, "agent_id"] = "ego"
        
        # Delete all agents with only one timestep
        scene_data = scene_data.groupby("agent_id").filter(lambda x: len(x) > 1)     
        
        
        # Agent dataframes (besides ego agent)
        agent_df = scene_data[scene_data["ego_vehicle"] == 0]
        
        # Ego agent dataframes
        ego_df = scene_data[scene_data["ego_vehicle"] == 1]
        
        test = AgentType(ego_df["category_id"].iloc[0])
        test2 = AgentType.VEHICLE
        
        
        
        # Get ego agent metadata dictionary with type, length, width, height       
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            #agent_type=AgentType(ego_df["category_id"].iloc[0]),
            first_timestep=ego_df["scene_ts"].min(),
            last_timestep=ego_df["scene_ts"].max(),
            extent=FixedExtent(
                length=ego_df["length"].iloc[0],
                width=ego_df["width"].iloc[0],
                height=ego_df["height"].iloc[0],
            ),
        )
        
        # Create agent presence list
        # Initialize agent_presence with ego_agent_info for each timestep
        agent_presence: List[List[AgentMetadata]] = [[ego_agent_info] for _ in range(scene_data["scene_ts"].max() + 1)]
        
        # Create agent list  apply the meta data function to each group
        agent_list: List[AgentMetadata] = [ego_agent_info] + agent_df.groupby("agent_id").apply(deepurban_utils.create_agent_metadata).tolist()
        
        # Create a dictionary where the keys are the timesteps and the values are the agents present at that timestep
        agent_presence_dict = agent_df.groupby("scene_ts")["agent_id"].apply(list).to_dict()
        
        
        # Set the index of agent_df to agent_id
        agent_df.set_index("agent_id", inplace=True)

        # Convert this dictionary to a list of lists
        for ts, agents in agent_presence_dict.items():
            # Get the metadata for each agent present at the timestep
            agent_presence[ts] += [agent for agent in agent_list if agent.name in agents]
                
        
        # Restructure the scene_data dataframe
        scene_data = scene_data[["agent_id", "scene_ts", "x", "y", "vx", "vy", "ax", "ay", "heading", "z"]]#, "length", "width", "height", "z", "yaw", "pitch", "roll", "vz", "wx", "wy", "wz", "az", "category_id", "road_id", "road_position_s", "road_position_t", "orientation_h", "orientation_p", "orientation_r", "orientation_type", "ego_vehicle", "original_agent_id"]]
             
        # Set the index of the scene_data dataframe to agent_id and scene_ts
        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)
        # Sort the scene_data dataframe by agent_id and scene_ts
        scene_data.sort_index(inplace=True)
        
        # Save the actual agent data to the cache
        cache_class.save_agent_data(
            scene_data,
            cache_path,
            scene,
        )
        
        # Returns a list of AgentMetadata objects
        return agent_list, agent_presence
    
    
    def cache_map(
        self,
        map_path: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Transform the map from the DeepUrban format to the VectorMap format and cache it
        
        Args:
            map_path: Path to the map file
            cache_path: Path to the cache directory
            map_cache_class: SceneCache class
            map_params: Dictionary containing map parameters
        """     
        
        
        map_path = str(map_path)
        # get the last part of the map_path
        actual_map_name = map_path.split('/')[-1].split('.')[0]
        
        vector_map = VectorMap(
            map_id=f"{self.name}:{actual_map_name}"
        )
        # join list elements to a string with '/' as separator
        meta_map_path = '/'.join(map_path.split('/')[:-1])
        
        # get lateral and longitudinal coordinates of the map
        lat_lot = deepurban_utils.get_map_metadata(actual_map_name, meta_map_path)
        
        map_projector = lanelet2.projection.LocalCartesianProjector(lanelet2.io.Origin(lat_lot[0], lat_lot[1]))
        
        laneletmap = lanelet2.io.load(map_path, map_projector)
        traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        lane_graph = lanelet2.routing.RoutingGraph(laneletmap, traffic_rules)
        
        maximum_bound: np.ndarray = np.full((3,), np.nan)
        minimum_bound: np.ndarray = np.full((3,), np.nan)
        
        
        # Set to zero as z is zero in the map but there seem to be some problems with the projector
        for point in laneletmap.pointLayer:
            point.z = 0
            
        for lanelet in tqdm(
            laneletmap.laneletLayer, desc=f"Creating Vectorized Map", leave=False
        ):
            left_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.leftBound]
            )
            right_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.rightBound]
            )
            center_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.centerline]
            )

            # Adding the element to the map.
            new_lane = RoadLane(
                id=str(lanelet.id),
                center=Polyline(center_pts),
                left_edge=Polyline(left_pts),
                right_edge=Polyline(right_pts),
            )

            new_lane.next_lanes.update(
                [str(l.id) for l in lane_graph.following(lanelet)]
            )

            new_lane.prev_lanes.update(
                [str(l.id) for l in lane_graph.previous(lanelet)]
            )

            left_lane_change = lane_graph.left(lanelet)
            if left_lane_change:
                new_lane.adj_lanes_left.add(str(left_lane_change.id))

            right_lane_change = lane_graph.right(lanelet)
            if right_lane_change:
                new_lane.adj_lanes_right.add(str(right_lane_change.id))

            vector_map.add_map_element(new_lane)

            # Computing the maximum and minimum map coordinates.
            maximum_bound = np.fmax(maximum_bound, left_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, left_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, right_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, right_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, center_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, center_pts.min(axis=0))
        
        
        # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
        vector_map.extent = np.concatenate((minimum_bound, maximum_bound))

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
    
    
    
    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Cache all maps in the DeepUrban dataset
        
        Args:
            cache_path: Path to the cache directory
            map_cache_class: SceneCache class
            map_params: Dictionary containing map parameters          
        """
        
        data_dir_path = Path(self.metadata.data_dir)
        file_paths = list(data_dir_path.glob("../maps/*.osm"))
        for map_path in tqdm(
            file_paths,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_path, cache_path, map_cache_class, map_params)

