#!/usr/bin/env python3

import os
from collections import defaultdict
from typing import Dict, Optional

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from tqdm import tqdm


class Benchmark:

    def __init__(self,
                 config_paths: Optional[str] = None,
                 eval_remote: bool = False) -> None:
        config_env = get_config(config_paths)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(self, agent: "Agent") -> Dict[str, float]:
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity)
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity)
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity)

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
                            action))).SerializedEntity)

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
                        action))).SerializedEntity)

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(self,
                       agent: "Agent",
                       num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Run evaluation of an Agent in the Env locally.

        :param agent: The Agent to evaluate.
        :param num_episodes: The number of episodes to evaluate.
        :return: The results dictionary containing metrics.
        """
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes,
                                             len(self._env.episodes)))

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset()

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1
            pbar.update(1)

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def evaluate(self,
                 agent: "Agent",
                 num_episodes: Optional[int] = None) -> Dict[str, float]:
        if self._eval_remote is True:
            return self.remote_evaluate(agent)
        else:
            return self.local_evaluate(agent, num_episodes)
