import os
import random

import cv2
import habitat
import numpy as np
from habitat.utils.geometry_utils import quaternion_to_list
from tqdm import tqdm

SAVE_PATH = "sft_data"


def save_image(id, image, folder="image"):
    save_path = f"{SAVE_PATH}/{folder}/{id}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
    return save_path[len(SAVE_PATH) + 1:]


def save_video(id, image_list):
    save_path = f"{SAVE_PATH}/video/{id}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    height, width, _ = image_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, 10.0, (width, height))

    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    video.release()
    return save_path[len(SAVE_PATH) + 1:]


if __name__ == "__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.system(f"rm -rf {SAVE_PATH}/*")
    random.seed(42)
    np.random.seed(42)
    config = habitat.config.get_config_and_task(
        config_path="benchmark/nav/octonav/sft_generate.yaml",
        overrides=[
            "habitat.environment.iterator_options.max_scene_repeat_steps=-1"
        ],
    )

    added_scene = {}
    trajs = 0
    with habitat.Env(config=config) as env:
        num_episodes = len(env.episodes)
        num_sft = 40000
        num_per_scene = 100
        pbar = tqdm(range(num_episodes))
        for _ in pbar:
            if trajs >= num_sft:
                break
            observations = env.reset()
            episode = env.current_episode
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            if added_scene.get(scene_id, 0) >= num_per_scene:
                continue
            cur_id = f"{scene_id}_{episode.episode_id}"
            for key, value in observations["instruction"].items():
                if key != "text":
                    save_image(f"{cur_id}_{key}", value)
            image_list = []
            action_list = []
            for idx, step in enumerate(env.current_episode.steps):
                image_list.append(observations["rgb"])
                action = step["action"]
                action_list.append(action)

                agent_state = env._sim.get_agent_state()
                cur_position = agent_state.position.tolist()
                cur_rotation = quaternion_to_list(agent_state.rotation)
                position = step["position"]
                rotation = step["rotation"]
                if (np.sum(np.abs(np.array(cur_position) - np.array(position)))
                        > 0.2):
                    assert False, "position error"
                if (np.sum(np.abs(np.array(cur_rotation) - np.array(rotation)))
                        > 0.2):
                    assert False, "rotation error"
                observations = env.step(action)
            if (env.task.measurements.measures["octonav_success"].get_metric()
                [-1] != 1.0):  # noqa: E129
                assert False, "success error"
            save_video(cur_id, image_list)
            trajs += 1
            pbar.set_postfix(trajs=trajs)
            added_scene[scene_id] = added_scene.get(scene_id, 0) + 1
