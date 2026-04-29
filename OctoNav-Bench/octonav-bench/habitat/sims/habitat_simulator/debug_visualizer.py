#!/usr/bin/env python3

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import habitat_sim
import magnum as mn
import numpy as np
from habitat.core.logging import logger
from habitat.utils.common import check_make_dir
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject
from PIL import Image, ImageDraw


def project_point(render_camera: habitat_sim.sensor.CameraSensor,
                  point: mn.Vector3) -> mn.Vector2:
    projected_point_3d = render_camera.projection_matrix.transform_point(
        render_camera.camera_matrix.transform_point(point))

    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / render_camera.projection_size()[0]
    point_2d += mn.Vector2(0.5)
    point_2d *= render_camera.viewport
    return mn.Vector2i(point_2d)


def stitch_image_matrix(images: List[Image.Image], num_col: int = 8):
    if len(images) == 0:
        raise ValueError("No images provided.")

    image_mode = images[0].mode
    image_size = images[0].size
    for image in images:
        if image.size != image_size:
            # TODO: allow shrinking/growing images
            raise ValueError("Image sizes must all match.")
    num_rows = math.ceil(len(images) / float(num_col))
    stitched_image = Image.new(image_mode,
                               size=(image_size[0] * num_col,
                                     image_size[1] * num_rows))

    for ix, image in enumerate(images):
        col = ix % num_col
        row = math.floor(ix / num_col)
        coords = (int(col * image_size[0]), int(row * image_size[1]))
        stitched_image.paste(image, box=coords)

    bdo = DebugObservation(np.array(stitched_image))
    bdo.image = stitched_image
    return bdo


class DebugObservation:

    def __init__(self, obs_data: np.ndarray) -> None:
        self.obs_data: np.ndarray = obs_data
        self.image: Image.Image = (
            None  # creation deferred to show or save time
        )

    def create_image(self) -> None:
        from habitat_sim.utils import viz_utils as vut

        self.image = vut.observation_to_image(self.obs_data, "color")

    def get_image(self) -> Image.Image:
        """
        Retrieve the PIL Image.
        """

        if self.image is None:
            self.create_image()
        return self.image

    def show(self) -> None:
        """
        Display the image via PIL.
        """

        if self.image is None:
            self.create_image()
        self.image.show()

    def show_point(self, p_2d: np.ndarray) -> None:
        """
        Show the image with a 2D point marked on it as a blue circle.

        :param p_2d: The 2D pixel point in the image.
        """
        if self.image is None:
            self.create_image()
        point_image = self.image.copy()
        draw = ImageDraw.Draw(point_image)
        circle_rad = 5  # pixels
        draw.ellipse(
            (
                p_2d[0] - circle_rad,
                p_2d[1] - circle_rad,
                p_2d[0] + circle_rad,
                p_2d[1] + circle_rad,
            ),
            fill="blue",
            outline="blue",
        )
        point_image.show()

    def save(self, output_path: str, prefix: str = "") -> str:
        if self.image is None:
            self.create_image()
        from datetime import datetime

        check_make_dir(output_path)

        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S%f")
        file_path = os.path.join(output_path, prefix + date_time + ".png")
        self.image.save(file_path)
        return file_path


def draw_object_highlight(
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    debug_line_render: habitat_sim.gfx.DebugLineRender,
    camera_transform: mn.Matrix4,
    color: mn.Color4 = None,
) -> None:
    if color is None:
        color = mn.Color4.magenta()

    obj_bb = obj.aabb
    obj_center = obj.transformation.transform_point(obj_bb.center())
    obj_size = obj_bb.size().max() / 2

    debug_line_render.draw_circle(
        translation=obj_center,
        radius=obj_size,
        color=color,
        normal=camera_transform.translation - obj_center,
    )


def dblr_draw_bb(
    debug_line_render: habitat_sim.gfx.DebugLineRender,
    bb: mn.Range3D,
    transform: mn.Matrix4 = None,
    color: mn.Color4 = None,
) -> None:
    if color is None:
        color = mn.Color4.magenta()
    if transform is not None:
        debug_line_render.push_transform(transform)
    debug_line_render.draw_box(bb.min, bb.max, color)
    if transform is not None:
        debug_line_render.pop_transform()


class DebugVisualizer:

    def __init__(
        self,
        sim: habitat_sim.Simulator,
        output_path: str = "visual_debug_output/",
        resolution: Tuple[int, int] = (500, 500),
        clear_color: Optional[mn.Color4] = None,
        equirect=False,
    ) -> None:
        self.sim = sim
        self.output_path = output_path
        self.sensor_uuid = "dbv_rgb_sensor"
        self.sensor_resolution = resolution
        self.debug_obs: List[DebugObservation] = []
        self.debug_line_render = sim.get_debug_line_render()
        self.sensor: habitat_sim.simulator.Sensor = None
        self.agent: habitat_sim.simulator.Agent = None

        self.dblr_callback: Callable = None
        self.dblr_callback_params: Dict[str, Any] = None  # kwargs

        self.agent_id = 0
        # default black background
        self.clear_color = (mn.Color4.from_linear_rgb_int(0)
                            if clear_color is None else clear_color)
        self._equirect = equirect

    def __del__(self) -> None:
        """
        When a DBV is removed, it should clean up its agent/sensor.
        """
        self.remove_dbv_agent()

    @property
    def equirect(self) -> bool:
        return self._equirect

    @equirect.setter
    def equirect(self, equirect: bool) -> None:
        """
        Set the equirect mode on or off.
        If dbv is already initialized to a different mode, re-initialize it.
        """

        if self._equirect != equirect:
            # change the value
            self._equirect = equirect
            if self.agent is not None:
                # re-initialize the agent
                self.remove_dbv_agent()
                self.create_dbv_agent(self.sensor_resolution)

    def create_dbv_agent(self, resolution: Tuple[int, int] = (500, 500)):
        self.sensor_resolution = resolution

        debug_agent_config = habitat_sim.agent.AgentConfiguration()

        debug_sensor_spec = (habitat_sim.CameraSensorSpec()
                             if not self._equirect else
                             habitat_sim.EquirectangularSensorSpec())
        debug_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        debug_sensor_spec.position = [0.0, 0.0, 0.0]
        debug_sensor_spec.resolution = [resolution[0], resolution[1]]
        debug_sensor_spec.uuid = self.sensor_uuid
        debug_sensor_spec.clear_color = self.clear_color

        debug_agent_config.sensor_specifications = [debug_sensor_spec]
        self.sim.agents.append(
            habitat_sim.Agent(
                self.sim.get_active_scene_graph().get_root_node().create_child(
                ),
                debug_agent_config,
            ))
        self.agent = self.sim.agents[-1]
        self.agent_id = len(self.sim.agents) - 1
        self.sim._Simulator__sensors.append({})
        self.sim._update_simulator_sensors(self.sensor_uuid, self.agent_id)
        self.sensor = self.sim._Simulator__sensors[self.agent_id][
            self.sensor_uuid]

    def remove_dbv_agent(self) -> None:
        """
        Clean up a previously initialized DBV agent.
        """

        if self.agent is None:
            print("No active dbv agent to remove.")
            return

        if self.agent_id < len(self.sim.agents):
            # remove the agent and sensor from the Simulator instance
            self.agent.close()
            del self.sim._Simulator__sensors[self.agent_id]
            del self.sim.agents[self.agent_id]

        self.agent = None
        self.agent_id = 0
        self.sensor = None

    def look_at(
        self,
        look_at: mn.Vector3,
        look_from: Optional[mn.Vector3] = None,
        look_up: Optional[mn.Vector3] = None,
    ) -> None:
        """
        Point the debug camera at a target.
        Standard look_at function syntax.

        :param look_at: 3D global position to point the camera towards.
        :param look_from: 3D global position of the camera.
        :param look_up: 3D global "up" vector for aligning the camera roll.
        """

        if self.agent is None:
            self.create_dbv_agent(self.sensor_resolution)

        camera_pos = (look_from if look_from is not None else
                      self.agent.scene_node.translation)
        if look_up is None:
            # pick a valid "up" vector.
            look_dir = look_at - camera_pos
            look_up = (mn.Vector3(0, 1.0, 0) if look_dir[0] != 0
                       or look_dir[2] != 0 else mn.Vector3(1.0, 0, 0))
        self.agent.scene_node.rotation = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(camera_pos, look_at, look_up).rotation())
        self.agent.scene_node.translation = camera_pos

    def translate(self,
                  vec: mn.Vector3,
                  local: bool = False,
                  show: bool = True) -> Optional[DebugObservation]:
        if self.agent is None:
            self.create_dbv_agent(self.sensor_resolution)

        if not local:
            self.agent.scene_node.translate(vec)
        else:
            self.agent.scene_node.translate_local(vec)
        if show:
            obs = self.get_observation()
            obs.show()
            return obs
        return None

    def rotate(
        self,
        angle: float,
        axis: Optional[mn.Vector3] = None,
        local: bool = False,
        show: bool = True,
    ) -> Optional[DebugObservation]:
        if self.agent is None:
            self.create_dbv_agent(self.sensor_resolution)

        if axis is None:
            axis = mn.Vector3(0, 1, 0)

        if not local:
            self.agent.scene_node.rotate(mn.Rad(angle), axis)
        else:
            self.agent.scene_node.rotate_local(mn.Rad(angle), axis)
        if show:
            obs = self.get_observation()
            obs.show()
            return obs
        return None

    def get_observation(
        self,
        look_at: Optional[mn.Vector3] = None,
        look_from: Optional[mn.Vector3] = None,
    ) -> DebugObservation:
        """
        Render a debug observation of the current state and return it.
        Optionally configure the camera transform.

        :param look_at: 3D global position to point the camera towards.
        :param look_from: 3D global position of the camera.
        :return: a DebugObservation wrapping the np.ndarray.
        """

        if self.agent is None:
            self.create_dbv_agent(self.sensor_resolution)

        if look_at is not None:
            self.look_at(look_at, look_from)
        if self.dblr_callback is not None:
            self.dblr_callback(**self.dblr_callback_params)
        self.sensor.draw_observation()
        return DebugObservation(self.sensor.get_observation())

    def render_debug_lines(
        self,
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
    ) -> None:
        # support None input to make usage easier elsewhere
        if debug_lines is not None:
            for points, color in debug_lines:
                for p_ix, point in enumerate(points):
                    if p_ix == 0:
                        continue
                    prev_point = points[p_ix - 1]
                    self.debug_line_render.draw_transformed_line(
                        prev_point,
                        point,
                        color,
                    )

    def render_debug_circles(
        self,
        debug_circles: Optional[List[Tuple[mn.Vector3, float, mn.Vector3,
                                           mn.Color4]]] = None,
    ) -> None:
        # support None input to make usage easier elsewhere
        if debug_circles is not None:
            for center, radius, normal, color in debug_circles:
                self.debug_line_render.draw_circle(
                    translation=center,
                    radius=radius,
                    color=color,
                    num_segments=12,
                    normal=normal,
                )

    def render_debug_frame(
        self,
        axis_length: float = 1.0,
        transformation: Optional[mn.Matrix4] = None,
    ) -> None:
        if transformation is None:
            transformation = mn.Matrix4.identity_init()
        origin = mn.Vector3()
        debug_lines = [
            ([origin, mn.Vector3(axis_length, 0, 0)], mn.Color4.red()),
            ([origin, mn.Vector3(0, axis_length, 0)], mn.Color4.green()),
            ([origin, mn.Vector3(0, 0, axis_length)], mn.Color4.blue()),
        ]
        self.debug_line_render.push_transform(transformation)
        self.render_debug_lines(debug_lines)
        self.debug_line_render.pop_transform()

    def peek(
        self,
        subject=Union[
            habitat_sim.physics.ManagedArticulatedObject,
            habitat_sim.physics.ManagedRigidObject,
            str,
            int,
        ],
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[List[Tuple[mn.Vector3, float, mn.Vector3,
                                           mn.Color4]]] = None,
    ) -> DebugObservation:
        subject_bb = None
        subject_transform = mn.Matrix4.identity_init()

        if isinstance(subject, int):
            from habitat.sims.habitat_simulator.sim_utilities import \
                get_obj_from_id

            subject_obj = get_obj_from_id(self.sim, subject)
            if subject_obj is None:
                raise AssertionError(f"The integer subject, '{subject}', \
is not a valid object_id.")
            subject = subject_obj
        elif isinstance(subject, str):
            if subject == "stage" or subject == "scene":
                subject_bb = (self.sim.get_active_scene_graph().get_root_node(
                ).cumulative_bb)
                if cam_local_pos is None:
                    cam_local_pos = mn.Vector3(0, 1, 0)
            else:
                from habitat.sims.habitat_simulator.sim_utilities import \
                    get_obj_from_handle

                subject_obj = get_obj_from_handle(self.sim, subject)
                if subject_obj is None:
                    raise AssertionError(f"The string subject, '{subject}', \
is not a valid object handle or an allowed alias from ('stage', 'scene').")
                subject = subject_obj

        if subject_bb is None:
            subject_bb = subject.aabb
            subject_transform = subject.transformation

        return self._peek_bb(
            bb=subject_bb,
            world_transform=subject_transform,
            cam_local_pos=cam_local_pos,
            peek_all_axis=peek_all_axis,
            debug_lines=debug_lines,
            debug_circles=debug_circles,
        )

    def _peek_bb(
        self,
        bb: mn.Range3D,
        world_transform: Optional[mn.Matrix4] = None,
        cam_local_pos: Optional[mn.Vector3] = None,
        peek_all_axis: bool = False,
        debug_lines: Optional[List[Tuple[List[mn.Vector3], mn.Color4]]] = None,
        debug_circles: Optional[List[Tuple[mn.Vector3, float, mn.Vector3,
                                           mn.Color4]]] = None,
    ) -> DebugObservation:
        if self.agent is None:
            self.create_dbv_agent(self.sensor_resolution)

        if world_transform is None:
            world_transform = mn.Matrix4.identity_init()
        look_at = world_transform.transform_point(bb.center())
        bb_size = bb.size()
        fov = 90 if self._equirect else self.sensor._spec.hfov
        aspect = (float(self.sensor._spec.resolution[1]) /
                  self.sensor._spec.resolution[0])
        import math

        distance = (np.amax(np.array(bb_size)) / aspect) / math.tan(
            fov / (360 / math.pi))
        if cam_local_pos is None:
            # default to -Z (forward) of the object
            cam_local_pos = mn.Vector3(0, 0, -1)
        if not peek_all_axis:
            look_from = (
                world_transform.transform_vector(cam_local_pos).normalized() *
                distance + look_at)
            self.render_debug_lines(debug_lines)
            self.render_debug_circles(debug_circles)
            obs = self.get_observation(look_at=look_at, look_from=look_from)
            return obs

        # collect axis observations
        axis_obs: List[DebugObservation] = []
        for axis in range(6):
            axis_vec = mn.Vector3()
            axis_vec[axis % 3] = 1 if axis // 3 == 0 else -1
            look_from = (
                world_transform.transform_vector(axis_vec).normalized() *
                distance + look_at)
            self.render_debug_lines(debug_lines)
            self.render_debug_circles(debug_circles)
            axis_obs.append(self.get_observation(look_at, look_from))
        # stitch images together
        stitched_image = None

        for ix, obs in enumerate(axis_obs):
            obs.create_image()
            if stitched_image is None:
                stitched_image = Image.new(
                    obs.image.mode,
                    (obs.image.size[0] * 3, obs.image.size[1] * 2),
                )
            location = (
                obs.image.size[0] * (ix % 3),
                obs.image.size[1] * (0 if ix // 3 == 0 else 1),
            )
            stitched_image.paste(obs.image, location)
        all_axis_obs = DebugObservation(None)
        all_axis_obs.image = stitched_image
        return all_axis_obs

    def make_debug_video(
        self,
        output_path: Optional[str] = None,
        prefix: str = "",
        fps: int = 4,
        obs_cache: Optional[List[DebugObservation]] = None,
    ) -> None:
        if output_path is None:
            output_path = self.output_path

        check_make_dir(output_path)

        # get a timestamp tag with current date and time for video name
        from datetime import datetime

        date_time = datetime.now().strftime("%m_%d_%Y_%H%M%S")

        if obs_cache is None:
            obs_cache = self.debug_obs

        all_formatted_obs_data = [{
            self.sensor_uuid: obs.obs_data
        } for obs in obs_cache]

        from habitat_sim.utils import viz_utils as vut

        file_path = os.path.join(output_path, prefix + date_time)
        logger.info(f"DebugVisualizer: Saving debug video to {file_path}")
        vut.make_video(
            all_formatted_obs_data,
            self.sensor_uuid,
            "color",
            file_path,
            fps=fps,
        )
