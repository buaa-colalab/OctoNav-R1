#!/usr/bin/env python3

from enum import Enum
from typing import Dict

import attr
from habitat.core.utils import Singleton


class _DefaultHabitatSimActions(Enum):
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3
    look_up = 4
    look_down = 5


@attr.s(auto_attribs=True, slots=True)
class HabitatSimActionsSingleton(metaclass=Singleton):
    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultHabitatSimActions:
            self._known_actions[action.name] = action.value

    def extend_action_space(self, name: str) -> int:
        assert name not in self._known_actions, (
            "Cannot register an action name twice")
        self._known_actions[name] = len(self._known_actions)

        return self._known_actions[name]

    def has_action(self, name: str) -> bool:
        """Checks to see if action :p:`name` is already register

        :param name: The name to check
        :return: Whether or not :p:`name` already exists
        """

        return name in self._known_actions

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()
