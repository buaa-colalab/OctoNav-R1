#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.tasks.nav.nav import (
    ShortestPathPoint,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectViewLocation,
)
from habitat.tasks.nav.octonav_task import (
    OctoNavEpisode,
    OctoGoal,
    ExtendedInstructionData
)
from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
from habitat.datasets.utils import VocabDict

if TYPE_CHECKING:
    from omegaconf import DictConfig


CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"


@registry.register_dataset(name="OctoNav-v1")
class OctoNavDatasetV1(Dataset):
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    episodes: List[OctoNavEpisode] = []
    # ObjectNav
    category_to_task_category_id: Dict[str, Dict[str, int]]
    category_to_scene_annotation_category_id: Dict[str, Dict[str, int]]
    goals_by_category: Dict[str, Dict[str,Sequence[OctoGoal]]]
    # InstanceImageNav
    image_goals: Dict[str, OctoGoal]
    # VLN-CE
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = OctoNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        flag = True
        for dataset in config.datasets:
            flag = flag and os.path.exists(dataset.data_path.format(split=dataset.split)) and os.path.exists(dataset.scenes_dir)
        return flag

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        """Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls.scene_from_scene_path(e.scene_id) for e in dataset.episodes}
        )

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def _load_from_file(self, fname: str, scenes_dir: str, dataset_name: str, task:str, overrides: Optional[Dict] = None) -> None:
        """
        Load the data from a file into `self.episodes`. This can load `.pickle`
        or `.json.gz` file formats.
        """
        if fname.endswith(".pickle"):
            # NOTE: not implemented for pointnav
            with open(fname, "rb") as f:
                self.from_binary(pickle.load(f), scenes_dir=scenes_dir)
        else:
            with gzip.open(fname, "rt") as f:
                self.from_json(f.read(), dataset_name, task, overrides=overrides, scenes_dir=scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        self.category_to_task_category_id = {}
        self.category_to_scene_annotation_category_id = {}
        self.episodes = []
        self.image_goals = {}

        if config is None:
            return
        
        total_episodes = []
        for dataset in config.datasets:
            overrides = dataset.get('overrides')
            scenes_dir = dataset.get('scenes_dir', 'data/scene_datasets/')
            if 'VLN-CE' in dataset.task:
                if dataset.task == 'RxR-VLN-CE':
                    for role in dataset.roles: 
                        datasetfile_path = dataset.data_path.format(split=dataset.split, role=role)
                        self._load_from_file(datasetfile_path, scenes_dir, dataset.dataset_name, dataset.task, overrides)

                    if ALL_LANGUAGES_MASK not in dataset.languages:
                        languages_to_load = set(dataset.languages)
                        self.episodes = [
                            episode
                            for episode in self.episodes
                            if episode.instruction.language in languages_to_load
                        ]
                else:
                    datasetfile_path = dataset.data_path.format(split=dataset.split)
                    self._load_from_file(datasetfile_path, scenes_dir, dataset.dataset_name, dataset.task, overrides)
                self.episodes = list(
                    filter(self.build_content_scenes_filter(config), self.episodes)
                )
            else:
                datasetfile_path = dataset.data_path.format(split=dataset.split)

                self._load_from_file(datasetfile_path, scenes_dir, dataset.dataset_name, dataset.task, overrides)

                # Read separate file for each scene
                dataset_dir = os.path.dirname(datasetfile_path)
                has_individual_scene_files = os.path.exists(
                    self.content_scenes_path.split("{scene}")[0].format(
                        data_path=dataset_dir
                    )
                )
                if has_individual_scene_files:
                    scenes = config.content_scenes
                    if ALL_SCENES_MASK in scenes:
                        scenes = self._get_scenes_from_folder(
                            content_scenes_path=self.content_scenes_path,
                            dataset_dir=dataset_dir,
                        )
                    else:
                        cur_scenes = self._get_scenes_from_folder(
                            content_scenes_path=self.content_scenes_path,
                            dataset_dir=dataset_dir,
                        )
                        scenes = [scene for scene in scenes if scene in cur_scenes]

                    for scene in scenes:
                        scene_filename = self.content_scenes_path.format(
                            data_path=dataset_dir, scene=scene
                        )

                        self._load_from_file(scene_filename, scenes_dir, dataset.dataset_name, dataset.task, overrides)
                else:
                    self.episodes = list(
                        filter(self.build_content_scenes_filter(config), self.episodes)
                    )
            total_episodes.extend(self.episodes)
            self.episodes = []
        self.episodes = list(total_episodes)

    
    def from_json(
        self, json_str: Union[str, Dict], dataset_name: str, task: str, overrides: Optional[Dict] = None, scenes_dir: Optional[str] = None
    ) -> None:
        if isinstance(json_str, str):
            deserialized = json.loads(json_str)
        else:
            deserialized = json_str
        
        if task == "ObjectNav":
            if CONTENT_SCENES_PATH_FIELD in deserialized:
                self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

            if "category_to_task_category_id" in deserialized:
                self.category_to_task_category_id[dataset_name] = deserialized[
                    "category_to_task_category_id"
                ]

            if "category_to_scene_annotation_category_id" in deserialized:
                self.category_to_scene_annotation_category_id[dataset_name] = deserialized[
                    "category_to_scene_annotation_category_id"
                ]

            if "category_to_mp3d_category_id" in deserialized:
                self.category_to_scene_annotation_category_id[dataset_name] = deserialized[
                    "category_to_mp3d_category_id"
                ]

            assert len(self.category_to_task_category_id[dataset_name]) == len(
                self.category_to_scene_annotation_category_id[dataset_name]
            )

            assert set(self.category_to_task_category_id[dataset_name].keys()) == set(
                self.category_to_scene_annotation_category_id[dataset_name].keys()
            ), "category_to_task and category_to_mp3d must have the same keys"

            if len(deserialized["episodes"]) == 0:
                return

            if "goals_by_category" not in deserialized:
                deserialized = self.dedup_goals(deserialized)

            if dataset_name not in self.goals_by_category.keys():
                self.goals_by_category[dataset_name] = {}
            for k, v in deserialized["goals_by_category"].items():
                self.goals_by_category[dataset_name][k] = [self.__deserialize_goal(g) for g in v]

            for i, episode in enumerate(deserialized["episodes"]):
                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **episode)
                episode.overrides = overrides

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                episode.goals = self.goals_by_category[dataset_name][episode.goals_key]

                if episode.shortest_paths is not None:
                    for path in episode.shortest_paths:
                        for p_index, point in enumerate(path):
                            if point is None or isinstance(point, (int, str)):
                                point = {
                                    "action": point,
                                    "rotation": None,
                                    "position": None,
                                }

                            path[p_index] = ShortestPathPoint(**point)

                self.episodes.append(episode)  # type: ignore [attr-defined]
        elif task == "PointNav" or task == "ImageNav":
            for i, episode in enumerate(deserialized["episodes"]):
                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **episode)
                episode.overrides = overrides

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = OctoGoal(**goal)
                if episode.shortest_paths is not None:
                    for path in episode.shortest_paths:
                        for p_index, point in enumerate(path):
                            path[p_index] = ShortestPathPoint(**point)
                self.episodes.append(episode)
        elif task == "InstanceImageNav":
            if len(deserialized["episodes"]) == 0:
                return
            for k, g in deserialized["goals"].items():
                self.image_goals[k] = self.__deserialize_goal(g)

            for i, episode in enumerate(deserialized["episodes"]):
                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **episode)
                episode.overrides = overrides

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                episode.goals = [self.image_goals[episode.goal_key]]
                self.episodes.append(episode)  # type: ignore[attr-defined]
        elif task == "R2R-VLN-CE":
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )

            for i, episode in enumerate(deserialized["episodes"]):
                episode["episode_id"] = str(episode["episode_id"])
                episode["trajectory_id"] = str(episode["trajectory_id"])

                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **episode)
                episode.overrides = overrides

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                episode.instruction = ExtendedInstructionData(**episode.instruction)
                if episode.goals is not None:
                    for g_index, goal in enumerate(episode.goals):
                        episode.goals[g_index] = OctoGoal(**goal)
                self.episodes.append(episode)
        elif task == 'RxR-VLN-CE':
            for i, episode in enumerate(deserialized["episodes"]):
                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **episode)
                episode.overrides = overrides

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                episode.instruction = ExtendedInstructionData(**episode.instruction)
                if episode.goals is not None:
                    for g_index, goal in enumerate(episode.goals):
                        episode.goals[g_index] = OctoGoal(**goal)
                self.episodes.append(episode)
        elif task == 'OctoNav':
            if len(deserialized["episodes"]) == 0:
                return
            task_episodes = {}
            for key, value in deserialized.items():
                if key == 'episodes':
                    continue
                tmp_episodes = self.episodes
                self.episodes = []
                self.from_json(value, dataset_name, key, overrides, scenes_dir)
                task_episodes[key] = self.episodes
                self.episodes = tmp_episodes
            for i, origin_episodes in enumerate(deserialized["episodes"]):
                episode = OctoNavEpisode(task_name=task, dataset_name=dataset_name, **origin_episodes)
                episode.overrides = overrides
                episode.task_episodes = []
                for item in origin_episodes['task_episodes']:
                    episode.task_episodes.append(task_episodes[item['task']][item['id']])
                episode.shortest_path = origin_episodes['info']['shortest_path']
                episode.steps = origin_episodes['info']['shortest_path_steps']
                self.episodes.append(episode)


            
    
    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> OctoGoal:
        g = OctoGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        if 'image_goals' in serialized_goal:
            for iidx, params in enumerate(g.image_goals):
                g.image_goals[iidx] = InstanceImageParameters(**params)  # type: ignore[arg-type]

        return g