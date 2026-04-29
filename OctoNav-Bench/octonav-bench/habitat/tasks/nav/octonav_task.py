# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import attr
import cv2
import habitat_sim
import numpy as np
import quaternion
from gym import Space, spaces
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (RGBSensor, Sensor, SensorSuite,
                                    SensorTypes, Simulator, VisualObservation)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
from habitat.tasks.nav.nav import (NavigationEpisode, NavigationGoal,
                                   NavigationTask)
from habitat.tasks.nav.object_nav_task import ObjectViewLocation
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (quaternion_from_coeff,
                                          quaternion_rotate_vector)
from habitat.utils.visualizations import fog_of_war, maps
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose

try:
    from habitat.datasets.octonav.octonav_dataset import OctoNavDatasetV1
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

MAP_THICKNESS_SCALAR: int = 128


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str,
                                          Union[float,
                                                str]]]] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class OctoNavEpisode(NavigationEpisode):
    # common
    dataset_name: str
    task_name: str
    overrides: Optional[Dict] = None
    # ObjectNav
    object_category: Optional[str] = None
    # ImageNav
    goal_object_id: str = None
    goal_image_id: int = None
    # VLN-CE
    reference_path: Optional[List[List[float]]] = None
    instruction: ExtendedInstructionData = None
    trajectory_id: Optional[Union[int, str]] = None

    # Multi-Task
    task_episodes: Optional[List["OctoNavEpisode"]] = None
    shortest_path: Optional[List[float]] = None
    steps: Optional[List[Dict]] = None

    @property
    def goals_key(self) -> str:  # objectnav
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"

    @property
    def goal_key(self) -> str:  # instance_imagenav
        """The key to retrieve the instance goal"""
        sid = os.path.basename(self.scene_id)
        for x in [".glb", ".basis"]:
            sid = sid[:-len(x)] if sid.endswith(x) else sid
        return f"{sid}_{self.goal_object_id}"


@attr.s(auto_attribs=True, kw_only=True)
class OctoGoal(NavigationGoal):
    # ObjectNav
    object_id: str = None
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None
    # InstanceImageNav
    image_goals: List[InstanceImageParameters] = None
    object_surface_area: Optional[float] = None


@registry.register_sensor
class OctoObjectGoalSensor(Sensor):
    cls_uuid: str = "octogoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "OctoNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1, )
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = 0
            for mp in self._dataset.category_to_task_category_id.values():
                max_value = max(max_value, max(mp.values()))

        return spaces.Box(low=0,
                          high=max_value,
                          shape=sensor_shape,
                          dtype=np.int64)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OctoNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}.")
            return None
        if not isinstance(episode.goals[0], OctoGoal):
            logger.error(f"First goal should be ObjectGoal, \
episode {episode.episode_id}.")
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [
                    self._dataset.category_to_task_category_id[
                        episode.dataset_name][category_name]
                ],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, OctoGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor.")


@registry.register_sensor
class OctoInstanceImageGoalSensor(RGBSensor):
    cls_uuid: str = "instance_imagegoal"
    _current_image_goal: Optional[VisualObservation]
    _current_episode_id: Optional[str]

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "OctoNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        from habitat.datasets.octonav.octonav_dataset import OctoNavDatasetV1

        assert isinstance(dataset, OctoNavDatasetV1), (
            "Provided dataset needs to be OctoNavDatasetV1")

        self._dataset = dataset
        self._sim = sim
        super().__init__(config=config)
        self._current_episode_id = None
        self._current_image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

    def _add_sensor(self, img_params: InstanceImageParameters,
                    sensor_uuid: str) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = img_params.image_dimensions
        spec.hfov = img_params.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    sensor_uuid:
                    SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s for s in agent.agent_config.sensor_specifications
            if s.uuid != sensor_uuid
        ]

    def _get_instance_image_goal(
            self, img_params: InstanceImageParameters) -> VisualObservation:
        sensor_uuid = f"{self.cls_uuid}_sensor"
        self._add_sensor(img_params, sensor_uuid)

        self._sim._sensors[sensor_uuid].draw_observation()
        img = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)
        return img

    def get_observation(
        self,
        *args: Any,
        episode: OctoNavEpisode,
        **kwargs: Any,
    ) -> Optional[VisualObservation]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}.")
            return None
        if not isinstance(episode.goals[0], OctoGoal):
            logger.error(
                f"First goal should be OctoGoal, episode {episode.episode_id}."
            )
            return None

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        img_params = episode.goals[0].image_goals[episode.goal_image_id]
        self._current_image_goal = self._get_instance_image_goal(img_params)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class OctoInstanceImageGoalHFOVSensor(Sensor):
    cls_uuid: str = "instance_imagegoal_hfov"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0.0, high=360.0, shape=(1, ), dtype=np.float32)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(self, *args: Any, episode: OctoNavEpisode,
                        **kwargs: Any) -> np.ndarray:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}.")
            return None
        if not isinstance(episode.goals[0], OctoGoal):
            logger.error(
                f"First goal should be OctoGoal, episode {episode.episode_id}."
            )
            return None

        img_params = episode.goals[0].image_goals[episode.goal_image_id]
        return np.array([img_params.hfov], dtype=np.float32)


@registry.register_sensor(name="OctoNavInstructionSensor")
class OctoNavInstructionSensor(Sensor):

    def __init__(self, sim: Simulator, **kwargs):
        self._sim = sim
        self.uuid = "instruction"
        self.observation_space = spaces.Dict()
        self._current_episode_id = None
        self._current_instruction = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def get_imagegoal(self, episode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        hash_obj = hashlib.sha256(episode.episode_id.encode("utf-8"))
        hash_bytes = hash_obj.digest()
        seed = abs(int.from_bytes(hash_bytes, byteorder="big")) % (2**32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation)
        return goal_observation["rgb"]

    def _add_sensor(self, img_params: InstanceImageParameters,
                    sensor_uuid: str) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = img_params.image_dimensions
        spec.hfov = img_params.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    sensor_uuid:
                    SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s for s in agent.agent_config.sensor_specifications
            if s.uuid != sensor_uuid
        ]

    def get_instanceimagegoal(self, episode):
        img_params = episode.goals[0].image_goals[episode.goal_image_id]
        sensor_uuid = f"{self.uuid}_sensor"
        self._add_sensor(img_params, sensor_uuid)

        self._sim._sensors[sensor_uuid].draw_observation()
        img = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)
        return img

    def _get_observation(self, episode: OctoNavEpisode, **kwargs):
        if episode.task_name == "PointNav":
            agent_position = np.array(episode.start_position).astype(
                np.float32)
            rotation_world_agent = quaternion.quaternion(
                *episode.start_rotation[::-1])
            goal_position = np.array(episode.goals[0].position,
                                     dtype=np.float32)
            direction_vector = goal_position - agent_position
            direction_vector_agent = quaternion_rotate_vector(
                rotation_world_agent.inverse(), direction_vector)
            rho, phi = cartesian_to_polar(-direction_vector_agent[2],
                                          direction_vector_agent[0])
            degree = abs(phi * 180 / np.pi)
            if phi > 0:
                rotate_description = (
                    f"a clockwise angle of {degree:.2f} degrees")
            else:
                rotate_description = (
                    f"an anti-clockwise angle of {degree:.2f} degrees")
            return {
                "text":
                f"Move towards the target point located at a distance \
{rho:.2f}m and {rotate_description}; after that, stop. ",
            }
        elif episode.task_name == "ObjectNav":
            return {
                "text":
                f"Find the {episode.object_category} and then head straight \
to the {episode.object_category}; after that, stop. ",
            }
        elif episode.task_name == "ImageNav":
            return {
                "text":
                "Align your trajectory using the reference image {ImageNav} \
to approach the scene matching the visual perspective; after that, stop.",
                "ImageNav": self.get_imagegoal(episode),
            }
        elif episode.task_name == "InstanceImageNav":
            return {
                "text": "Locate the object instance in the reference image \
{InstanceImageNav} and verify its presence in the current view; after that, \
stop.",
                "InstanceImageNav": self.get_instanceimagegoal(episode),
            }
        elif "VLN" in episode.task_name:
            return {
                "text": episode.instruction.instruction_text,
                "tokens": episode.instruction.instruction_tokens,
                "trajectory_id": episode.trajectory_id,
            }
        elif episode.task_name == "OctoNav":
            data_dict = {"text": episode.instruction}
            for task_episode in episode.task_episodes:
                if task_episode.task_name == "ImageNav":
                    data_dict["ImageNav"] = self.get_imagegoal(task_episode)
                elif task_episode.task_name == "InstanceImageNav":
                    data_dict["InstanceImageNav"] = self.get_instanceimagegoal(
                        task_episode)
            return data_dict
        else:
            return None

    def get_observation(self, episode: OctoNavEpisode, **kwargs):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_instruction
        self._current_instruction = self._get_observation(episode=episode,
                                                          **kwargs)
        self._current_episode_id = episode_uniq_id
        return self._current_instruction


@registry.register_measure
class OctoNavDistanceToGoal(Measure):
    cls_uuid: str = "octonav_distance_to_goal"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[List[Tuple[float, float,
                                                       float]]] = None
        self._distance_to = self._config.distance_to

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._episode_view_points = []
        for task_episode in episode.task_episodes:
            task_name = task_episode.task_name
            if self._distance_to[task_name] == "VIEW_POINTS":
                self._episode_view_points.append([
                    view_point.agent_state.position
                    for goal in task_episode.goals
                    for view_point in goal.view_points
                ])
            elif self._distance_to[task_name] == "POINT":
                self._episode_view_points.append(
                    [goal.position for goal in task_episode.goals])
            else:
                logger.error(f"Non valid distance_to parameter was provided\
: {self._distance_to[task_name]}")
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode, *args: Any,
                      **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
                self._previous_position, current_position, atol=1e-4):
            distance_to_targets = []
            for i in range(len(self._episode_view_points)):
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points[i])
                distance_to_targets.append(distance_to_target)

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_targets


@registry.register_measure
class OctoNavTaskSuccess(Measure):
    cls_uuid: str = "octonav_task_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._sim = sim
        self._config = config
        self._success_distance = self._config.success_distance

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OctoNavDistanceToGoal.cls_uuid])
        self.task_num = len(episode.task_episodes)
        self.success_distance = []
        for task_episode in episode.task_episodes:
            self.success_distance.append(
                self._success_distance[task_episode.task_name])
        self.update_metric(episode=episode, task=task, *args,
                           **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        distance_to_target = task.measurements.measures[
            OctoNavDistanceToGoal.cls_uuid].get_metric()

        self._metric = [
            1.0 if distance_to_target[i] < self.success_distance[i] else 0.0
            for i in range(self.task_num)
        ]


@registry.register_measure
class OctoNavSuccess(Measure):
    cls_uuid: str = "octonav_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OctoNavTaskSuccess.cls_uuid])
        self.current_task = 0
        self.task_num = len(episode.task_episodes)
        self.update_metric(episode=episode, task=task, *args,
                           **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        task_success = task.measurements.measures[
            OctoNavTaskSuccess.cls_uuid].get_metric()

        if (self.current_task < self.task_num
                and task_success[self.current_task]):
            self.current_task += 1
        self._metric = [1.0] * self.current_task + [0.0] * (self.task_num -
                                                            self.current_task)


@registry.register_measure
class OctoNavOracleSuccess(Measure):
    cls_uuid: str = "octonav_oracle_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OctoNavTaskSuccess.cls_uuid])
        self.task_num = len(episode.task_episodes)
        self.succeed = [0.0] * self.task_num
        self.update_metric(episode=episode, task=task, *args,
                           **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        task_success = task.measurements.measures[
            OctoNavTaskSuccess.cls_uuid].get_metric()

        self.succeed = [
            max(self.succeed[i], task_success[i]) for i in range(self.task_num)
        ]
        self._metric = self.succeed


@registry.register_measure
class OctoNavFinalSuccess(Measure):
    cls_uuid: str = "octonav_final_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OctoNavTaskSuccess.cls_uuid])
        self.update_metric(episode=episode, task=task, *args,
                           **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        task_success = task.measurements.measures[
            OctoNavTaskSuccess.cls_uuid].get_metric()

        if (hasattr(task, "is_stop_called")
                and task.is_stop_called  # type: ignore
                and task_success[-1] == 1.0):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class OctoNavSPL(Measure):
    cls_uuid: str = "octonav_spl"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[List[float]] = None
        self._agent_episode_distance: Optional[float] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid,
                                                     [OctoNavSuccess.cls_uuid])

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self.task_num = len(episode.task_episodes)
        self._start_end_episode_distance = episode.shortest_path
        self.success_task = 0
        self.record_metric = [0.0] * self.task_num
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        cur_success_task = task.measurements.measures[
            OctoNavSuccess.cls_uuid].current_task

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position)

        self._previous_position = current_position

        while self.success_task < cur_success_task:
            i = self.success_task
            self.record_metric[i] = self._start_end_episode_distance[i] / max(
                self._start_end_episode_distance[i],
                self._agent_episode_distance,
            )
            self.success_task += 1
        self._metric = self.record_metric


@registry.register_measure
class OctoNavLocalSPL(Measure):
    cls_uuid: str = "octonav_localspl"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any,
                 **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[List[float]] = None
        self._agent_episode_distance: Optional[float] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid,
                                                     [OctoNavSuccess.cls_uuid])

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self.task_num = len(episode.task_episodes)
        self._start_end_episode_distance = [episode.shortest_path[0]]
        for i in range(1, self.task_num):
            self._start_end_episode_distance.append(episode.shortest_path[i] -
                                                    episode.shortest_path[i -
                                                                          1])
        self.success_task = 0
        self.last_success_dis = 0.0
        self.record_metric = [0.0] * self.task_num
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any,
                      **kwargs: Any):
        cur_success_task = task.measurements.measures[
            OctoNavSuccess.cls_uuid].current_task

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position)

        self._previous_position = current_position

        while self.success_task < cur_success_task:
            i = self.success_task
            dis = self._agent_episode_distance - self.last_success_dis
            self.record_metric[i] = self._start_end_episode_distance[i] / max(
                self._start_end_episode_distance[i], dis)

            self.last_success_dis = self._agent_episode_distance
            self.success_task += 1
        self._metric = self.record_metric


@registry.register_measure
class OctoNavTopDownMap(Measure):
    cls_uuid: str = "octonav_top_down_map"

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: List[Optional[Tuple[int, int]]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR))
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR))
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding:t_x + self.point_padding + 1,
            t_y - self.point_padding:t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self,
                              episode,
                              color=maps.MAP_TARGET_POINT_INDICATOR):
        if self._config.draw_goal_positions:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(goal.position, color)
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]) == int(
                            goal.object_id
                        ), (f"Object_id doesn't correspond to id in semantic \
scene objects dictionary for episode: {episode}")

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0)
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z]) for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ] if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        ) for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(self, episode: NavigationEpisode,
                            agent_position: AgentState):
        if self._config.draw_shortest_path:
            self._shortest_path_points = [
                maps.to_grid(
                    p["position"][2],
                    p["position"][0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                ) for p in episode.steps
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(self,
                          height,
                          ref_floor_height=None,
                          ceiling_height=2.0):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1] - 0.5
        return ref_floor_height <= height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid,
                                                     [OctoNavSuccess.cls_uuid])
        self._top_down_map = self.get_original_map()
        self._step_count = 0
        self._sim.get_agent_state().position
        self._previous_xy_location = [
            None for _ in range(len(self._sim.habitat_config.agents))
        ]
        self.cur_task = 0
        self.task_names = [
            task_episode.task_name for task_episode in episode.task_episodes
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            for task_episode in episode.task_episodes:
                # self._draw_goals_view_points(task_episode)
                self._draw_goals_positions(
                    task_episode, maps.TASK_COLOR[task_episode.task_name])

        if self._config.draw_source:
            self._draw_point(episode.start_position,
                             maps.MAP_SOURCE_POINT_INDICATOR)

        self.update_metric(episode, task)
        self._step_count = 0

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        success = task.measurements.measures[
            OctoNavSuccess.cls_uuid].get_metric()
        self._step_count += 1
        map_positions: List[Tuple[float]] = []
        map_angles = []
        for agent_index in range(len(self._sim.habitat_config.agents)):
            agent_state = self._sim.get_agent_state(agent_index)
            map_positions.append(self.update_map(agent_state, agent_index))
            map_angles.append(OctoNavTopDownMap.get_polar_angle(agent_state))
        while (self.cur_task + 1 < len(success)
               and success[self.cur_task] == 1.0):
            self.cur_task += 1
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
        }

    @staticmethod
    def get_polar_angle(agent_state):
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(ref_rotation.inverse(),
                                                  np.array([0, 0, -1]))
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        return np.array(phi)

    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = maps.TASK_COLOR[self.task_names[self.cur_task]]

            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = OctoNavTopDownMap.get_polar_angle(agent_state)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_fog_of_war_mask(self, agent_position, angle):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                angle,
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist /
                maps.calculate_meters_per_pixel(self._map_resolution,
                                                sim=self._sim),
            )


@registry.register_task(name="OctoNav-v1")
class OctoNavTask(NavigationTask):
    multitask_episode: OctoNavEpisode = None

    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        from habitat.core.registry import registry

        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._physics_target_sps = config.physics_target_sps
        assert self._physics_target_sps > 0, (
            "physics_target_sps must be positive")

        self.measurements = Measurements(
            self._init_entities(
                entities_configs=config.measurements,
                register_func=registry.get_measure,
            ).values())

        self.sensor_suite = SensorSuite(
            self._init_entities(
                entities_configs=config.lab_sensors,
                register_func=registry.get_sensor,
            ).values())

        self.actions = self._init_entities(
            entities_configs=config.actions,
            register_func=registry.get_task_action,
        )
        self._action_keys = list(self.actions.keys())

        self._is_episode_active = False

    def get_multi_metrics(self):
        distance_to_goal = self.measurements.measures[
            OctoNavDistanceToGoal.cls_uuid].get_metric()
        success = self.measurements.measures[
            OctoNavSuccess.cls_uuid].get_metric()
        oracle_success = self.measurements.measures[
            OctoNavOracleSuccess.cls_uuid].get_metric()
        final_success = self.measurements.measures[
            OctoNavFinalSuccess.cls_uuid].get_metric()
        spl = self.measurements.measures[OctoNavSPL.cls_uuid].get_metric()
        localspl = self.measurements.measures[
            OctoNavLocalSPL.cls_uuid].get_metric()
        success[-1] * final_success
        result = {
            "OctoNav": {
                "distance_to_goal": distance_to_goal[-1],
                "success": success[-1],
                "oracle_success": oracle_success[-1],
                "spl": spl[-1] * success[-1],
            }
        }
        for idx, task_episode in enumerate(
                self.multitask_episode.task_episodes):
            task_name = task_episode.task_name
            if "VLN-CE" in task_name:
                task_name = "VLN-CE"
            result[task_name] = {
                "success": success[idx],
                "oracle_success": oracle_success[idx],
                "spl": localspl[idx],
            }
        if OctoNavTopDownMap.cls_uuid in self.measurements.measures:
            result["OctoNav"]["top_down_map"] = self.measurements.measures[
                OctoNavTopDownMap.cls_uuid].get_metric()
        return result

    def reset(self, episode: Episode):
        observations = self._sim.reset()
        self.multitask_episode = episode

        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                task=self,
                should_time=True,
            ))

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        self._is_episode_active = True

        return observations
