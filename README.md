# DeepUrban: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The following is an extension of the [trajdata](https://github.com/NVlabs/trajdata/tree/main) dataloader to easily access our dataset [DeepUrban](https://iv.ee.hm.edu/deepurban/).

### Package Developer Installation

First, in whichever environment you would like to use (conda, venv, ...), make sure to install all required dependencies with (for easier version control only nuscenes dependencies are installed, if other dependencies are needed please uncomment them in requirements.txt)
```
pip install -r requirements.txt
```
Then, install trajdata itself in editable mode with
```
pip install -e .
```
Then, download the raw datasets (nuScenes, Lyft Level 5, ETH/UCY, etc.) in case you do not already have them. For more information about how to structure dataset folders/files, please see [`DATASETS.md`](./DATASETS.md).

### DeepUrban Changes
The DeepUrban was added as suggested by the original trajdata. Changes needed to be done, if you want to expand the original trajdata yourself, can be looked up  [here](https://github.com/NVlabs/trajdata/tree/main).
An additional change have been made in `/src/trajdata/data_structures/agent.py` by extending the AgentTypes as well as in `/src/trajdata/visualization/vis.py` to visualize all the different AgentTypes accordingly.

## Example

Please see the `examples/` folder for more examples, below is just one demonstration for the new dataset DeepUrban.

### DeepUrban Usage
The following will load data from both the nuScenes mini dataset as well as the DeepUrban scenarios from SanFrancisco.
A more extended example can be found in `examples/deepurban_example.py`.

```py
dataset = UnifiedDataset(
    desired_data=["nusc_mini", "deepurban_trainval-val_SanFrancisco"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes",
        "deepurban": "~/datasets/DeepUrban/deepurban_scenarios",
    },
    desired_dt=0.1,
)
```

## Supported Datasets with DeepUrban Extension
Currently, the dataloader supports interfacing with the following datasets:

| Dataset | ID | Splits | Locations | Description | dt | Maps |
|---------|----|--------|------------|-------------|----|------|
| nuScenes Train/TrainVal/Val | `nusc_trainval` | `train`, `train_val`, `val` | `boston`, `singapore` | nuScenes prediction challenge training/validation/test splits (500/200/150 scenes) | 0.5s (2Hz) | :white_check_mark: |
| nuScenes Test | `nusc_test` | `test` | `boston`, `singapore` | nuScenes test split, no annotations (150 scenes) | 0.5s (2Hz) | :white_check_mark: |
| nuScenes Mini | `nusc_mini` | `mini_train`, `mini_val` | `boston`, `singapore` | nuScenes mini training/validation splits (8/2 scenes) | 0.5s (2Hz) | :white_check_mark: |
| nuPlan Train | `nuplan_train` | N/A | `boston`, `singapore`, `pittsburgh`, `las_vegas` | nuPlan training split (947.42 GB) | 0.05s (20Hz) | :white_check_mark: |
| nuPlan Validation | `nuplan_val` | N/A | `boston`, `singapore`, `pittsburgh`, `las_vegas` | nuPlan validation split (90.30 GB) | 0.05s (20Hz) | :white_check_mark: |
| nuPlan Test | `nuplan_test` | N/A | `boston`, `singapore`, `pittsburgh`, `las_vegas` | nuPlan testing split (89.33 GB) | 0.05s (20Hz) | :white_check_mark: |
| nuPlan Mini | `nuplan_mini` | `mini_train`, `mini_val`, `mini_test` | `boston`, `singapore`, `pittsburgh`, `las_vegas` | nuPlan mini training/validation/test splits (942/197/224 scenes, 7.96 GB) | 0.05s (20Hz) | :white_check_mark: |
| Waymo Open Motion Training | `waymo_train` | `train` | N/A | Waymo Open Motion Dataset `training` split | 0.1s (10Hz) | :white_check_mark: |
| Waymo Open Motion Validation | `waymo_val` | `val` | N/A | Waymo Open Motion Dataset `validation` split | 0.1s (10Hz) | :white_check_mark: |
| Waymo Open Motion Testing | `waymo_test` | `test` | N/A | Waymo Open Motion Dataset `testing` split | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Train | `lyft_train` | `train` | `palo_alto` | Lyft Level 5 training data - part 1/2 (8.4 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Train Full | `lyft_train_full` | `train` | `palo_alto` | Lyft Level 5 training data - part 2/2 (70 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Validation | `lyft_val` | `val` | `palo_alto` | Lyft Level 5 validation data (8.2 GB) | 0.1s (10Hz) | :white_check_mark: |
| Lyft Level 5 Sample | `lyft_sample` | `mini_train`, `mini_val` | `palo_alto` | Lyft Level 5 sample data (100 scenes, randomly split 80/20 for training/validation) | 0.1s (10Hz) | :white_check_mark: |
| INTERACTION Dataset Single-Agent | `interaction_single` | `train`, `val`, `test`, `test_conditional` | `usa`, `china`, `germany`, `bulgaria` | Single-agent split of the INTERACTION Dataset (where the goal is to predict one target agents' future motion) | 0.1s (10Hz) | :white_check_mark: |
| INTERACTION Dataset Multi-Agent | `interaction_multi` | `train`, `val`, `test`, `test_conditional` | `usa`, `china`, `germany`, `bulgaria` | Multi-agent split of the INTERACTION Dataset (where the goal is to jointly predict multiple agents' future motion) | 0.1s (10Hz) | :white_check_mark: |
| ETH - Univ | `eupeds_eth` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The ETH (University) scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| ETH - Hotel | `eupeds_hotel` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `zurich` | The Hotel scene from the ETH BIWI Walking Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Univ | `eupeds_univ` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The University scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara1 | `eupeds_zara1` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara1 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| UCY - Zara2 | `eupeds_zara2` | `train`, `val`, `train_loo`, `val_loo`, `test_loo` | `cyprus` | The Zara2 scene from the UCY Pedestrians dataset | 0.4s (2.5Hz) | |
| Stanford Drone Dataset | `sdd` | `train`, `val`, `test` | `stanford` | Stanford Drone Dataset (60 scenes, randomly split 42/9/9 (70%/15%/15%) for training/validation/test) | 0.0333...s (30Hz) | |
| DeepUrban Dataset | deepurban_trainval | `train_<location>` `val_<location>` e.g. `train_MunichTal`| `MunichTal` `SanFrancisco` `SindelfingenBreuningerland` `StuttgartUniversitaetsstrasse` | DeepUrban Dataset (80/10/10 split) | 0.1s (10Hz) | :white_check_mark: | 
|





## Citation

If you use this software, please cite it as follows:
```
@Inproceedings{selzer2024deepurban,
  author = {Selzer, Constantin and Flohr, Fabian},
  title = {{DeepUrban}: Interaction-aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery },
  booktitle = {{IEEE International Conference on Intelligent Transportation Systems (ITSC)}},
  month = sept,
  year = {2024},
  address = {Edmonton, Canada},
}
```
