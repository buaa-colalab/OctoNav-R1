#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from typing import List
from collections import defaultdict
from typing import Dict
from PIL import Image
import json
import cv2

import numpy as np

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps

class RandomAgent:
    def __init__(self):
        pass
    
    def act(self, observations):
        actions = ['look_down', 'look_up', 'move_forward', 'turn_left', 'turn_right']
        action = np.random.choice(actions)
        return action

def octonav_dataset_example():
    config = habitat.config.get_config_and_task(
        config_path="benchmark/nav/octonav/octonav_bench_val.yaml",
    )
    # you can change the agent to your own agent
    agent = RandomAgent()
    with habitat.Env(config=config) as env:
        num_episodes = len(env.episodes)
        agg_metrics: Dict[str, Dict] = defaultdict(Dict)
        task_cnt: Dict = defaultdict(int)
        for _ in range(num_episodes):
            # reset the environment and switch to the next episode
            observations = env.reset()
            # get the instruction and images
            instruction = observations['instruction']['text']
            imagenav = observations['instruction'].get('ImageNav', None)
            instanceimagenav = observations['instruction'].get('InstanceImageNav', None)
            for step in range(10):
                # get the observation image
                img = observations['rgb']
                # inference of your agent here
                action = agent.act(observations)
                # perform an action
                observations = env.step(action)
            # after finished, stop the agent
            observations = env.step(HabitatSimActions.stop)

            # get the metrics
            for task, v in env.get_metrics().items():
                if task not in agg_metrics.keys():
                    agg_metrics.setdefault(task, defaultdict(float))
                for metric, value in v.items():
                    if metric == "top_down_map":
                        top_down_map = maps.colorize_draw_agent_and_fit_to_height(value, 480)
                        # output the top-down map if needed
                        # cv2.imwrite(f"output_image.png", top_down_map)
                    else:
                        agg_metrics[task][metric] += value
                task_cnt[task] += 1
            break

    for task_name in agg_metrics.keys():
        for m in agg_metrics[task_name].keys():
            agg_metrics[task_name][m] /= task_cnt[task_name]
    print(agg_metrics)
    return agg_metrics


if __name__ == "__main__":
    octonav_dataset_example()
