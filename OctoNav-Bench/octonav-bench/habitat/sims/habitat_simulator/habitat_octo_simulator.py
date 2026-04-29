#!/usr/bin/env python3

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import habitat_sim
import magnum as mn
import numpy as np
from gym import spaces
from habitat.config.default import get_agent_config
from habitat.core.batch_rendering.env_batch_renderer_constants import (
    KEYFRAME_OBSERVATION_KEY, KEYFRAME_SENSOR_PREFIX)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (AgentState, Observations, SensorSuite,
                                    ShortestPathPoint, Simulator)
from habitat.core.spaces import Space
from habitat.sims.habitat_simulator.habitat_simulator import (HabitatSimSensor,
                                                              overwrite_config)
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    pass


@registry.register_simulator(name="OctoSim-v0")
class HabitatOctoSim(habitat_sim.Simulator, Simulator):

    def __init__(self, config: DictConfig, current_episode: Episode) -> None:
        self.habitat_config = config

        sim_sensors = []
        for agent_config in self.habitat_config.agents.values():
            for sensor_cfg in agent_config.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)

                assert sensor_type is not None, (
                    "invalid sensor type {}".format(sensor_cfg.type))
                sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.rewrite_config(current_episode.overrides)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        super().__init__(self.sim_config)
        # load additional object paths specified by the dataset
        # TODO: Should this be moved elsewhere?
        obj_attr_mgr = self.get_object_template_manager()
        for path in self.habitat_config.additional_object_paths:
            obj_attr_mgr.load_configs(path)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[
                self.habitat_config.default_agent_id].action_space))
        self._prev_sim_obs: Optional[Observations] = None

    def rewrite_config(self, overrides: Optional[Dict] = None):
        if overrides is None:
            return
        self.habitat_config = copy.deepcopy(self.habitat_config)
        OmegaConf.set_readonly(self.habitat_config, False)
        for key, value in overrides.items():
            setattr(self.habitat_config, key, value)

    def create_sim_config(
            self, _sensor_suite: SensorSuite) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError("Incompatible version of Habitat-Sim detected, \
please upgrade habitat_sim")
        overwrite_config(
            config_from=self.habitat_config.habitat_sim_v0,
            config_to=sim_config,
            # Ignore key as it gets propagated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        if "mp3d" in self.habitat_config.scene:
            sim_config.scene_dataset_config_file = (
                "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json")
        elif "hm3d" in self.habitat_config.scene:
            if "hm3d_v0.2" in self.habitat_config.scene:
                sim_config.scene_dataset_config_file = (
                    "data/scene_datasets/hm3d_v0.2/"
                    "hm3d_annotated_basis.scene_dataset_config.json")
            else:
                sim_config.scene_dataset_config_file = (
                    "data/scene_datasets/hm3d/"
                    "hm3d_annotated_basis.scene_dataset_config.json")
        elif "gibson" in self.habitat_config.scene:
            sim_config.scene_dataset_config_file = (
                "data/scene_datasets/gibson/"
                "gibson_semantic.scene_dataset_config.json")
        else:
            sim_config.scene_dataset_config_file = (
                self.habitat_config.scene_dataset)

        sim_config.scene_id = self.habitat_config.scene
        lab_agent_config = get_agent_config(self.habitat_config)
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=lab_agent_config,
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "sim_sensors",
                "start_position",
                "start_rotation",
                "articulated_agent_urdf",
                "articulated_agent_type",
                "joint_start_noise",
                "joint_that_can_control",
                "motion_data_path",
                "ik_arm_urdf",
                "grasp_managers",
                "max_climb",
                "max_slope",
                "joint_start_override",
                "auto_update_sensor_transform",
            },
        )
        # configure default navmesh parameters to match the configured agent
        if self.habitat_config.default_agent_navmesh:
            sim_config.navmesh_settings = habitat_sim.nav.NavMeshSettings()
            sim_config.navmesh_settings.set_defaults()
            sim_config.navmesh_settings.agent_radius = agent_config.radius
            sim_config.navmesh_settings.agent_height = agent_config.height
            sim_config.navmesh_settings.agent_max_climb = (
                lab_agent_config.max_climb)
            sim_config.navmesh_settings.agent_max_slope = (
                lab_agent_config.max_slope)
            sim_config.navmesh_settings.include_static_objects = (
                self.habitat_config.navmesh_include_static_objects)

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec(
            )  # type: ignore[operator]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type":
                    lambda v: getattr(habitat_sim.FisheyeSensorModelType, v),
                    "sensor_subtype":
                    lambda v: getattr(habitat_sim.SensorSubType, v),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2])

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.habitat_config.habitat_sim_v0.gpu_gpu)
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications

        agent_config.action_space = {
            0:
            habitat_sim.ActionSpec("stop"),
            1:
            habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.forward_step_size),
            ),
            2:
            habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle),
            ),
            3:
            habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(
                    amount=self.habitat_config.turn_angle),
            ),
        }

        output = habitat_sim.Configuration(sim_config, [agent_config])
        output.enable_batch_renderer = (
            self.habitat_config.renderer.enable_batch_renderer)
        return output

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, agent_name in enumerate(
                self.habitat_config.agents_order):
            agent_cfg = self.habitat_config.agents[agent_name]
            if agent_cfg.is_set_start_state:
                self.set_agent_state(
                    [float(k) for k in agent_cfg.start_position],
                    [float(k) for k in agent_cfg.start_rotation],
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self) -> Observations:
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def step(self, action: Optional[Union[str, np.ndarray,
                                          int]]) -> Observations:
        if action is None:
            sim_obs = self.get_sensor_observations()
        else:
            sim_obs = super().step(action)
        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        assert not self.config.enable_batch_renderer

        sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def reconfigure(
        self,
        habitat_config: DictConfig,
        ep_info: Optional[Episode] = None,
        should_close_on_new_scene: bool = True,
    ) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = habitat_config.scene == self._current_scene
        self.habitat_config = habitat_config

        self.rewrite_config(ep_info.overrides)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = habitat_config.scene
            if should_close_on_new_scene:
                self.close(destroy=False)
            super().reconfigure(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[Sequence[float], Sequence[Sequence[float]],
                          np.ndarray],
        episode: Optional[Episode] = None,
    ) -> float:
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)])
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self,
        source: AgentState,
        targets: Sequence[AgentState],
        agent_id: int = 0,
    ) -> List[ShortestPathPoint]:
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead")

    @property
    def up_vector(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self) -> np.ndarray:
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self) -> List[float]:
        return self.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]) -> bool:
        return self.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        return self.semantic_scene

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self.get_agent(agent_id)
        new_state = self.get_agent(agent_id).get_state()
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position,
                                           rotation,
                                           reset_sensors=False)

        if success:
            sim_obs = self.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(self,
                                     position: np.ndarray,
                                     max_search_radius: float = 2.0) -> float:
        return self.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius)

    def island_radius(self, position: Sequence[float]) -> float:
        return self.pathfinder.island_radius(position)

    @property
    def previous_step_collided(self):
        return self._prev_sim_obs.get("collided", False)

    def add_keyframe_to_observations(self, observations):
        assert self.config.enable_batch_renderer

        assert KEYFRAME_OBSERVATION_KEY not in observations
        for _sensor_uuid, sensor in self._sensors.items():
            node = sensor._sensor_object.node
            transform = node.absolute_transformation()
            rotation = mn.Quaternion.from_matrix(transform.rotation())
            self.gfx_replay_manager.add_user_transform_to_keyframe(
                KEYFRAME_SENSOR_PREFIX + _sensor_uuid,
                transform.translation,
                rotation,
            )
        observations[KEYFRAME_OBSERVATION_KEY] = (
            self.gfx_replay_manager.extract_keyframe())
