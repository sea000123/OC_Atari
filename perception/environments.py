"""
Examples of environments wrapper around Perception environments that provide an easy way.
"""

import gymnasium as gym
import numpy as np
from perception.core import Perception


class PositionHistoryEnv(gym.Env):
    """Perception Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected. No hud information.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.

    """

    def __init__(self, env_name: str) -> None:
        super().__init__()
        self.perception_env = Perception(
            env_name, mode="ram", hud=False, obs_mode=None)
        self.reference_list = self._init_ref_vector()
        self.current_vector = np.zeros(
            4 * len(self.reference_list), dtype=np.float32)

    @property
    def observation_space(self):
        vl = len(self.reference_list) * 4
        return gym.spaces.Box(low=-2**63, high=2**63 - 2, shape=(vl, ), dtype=np.float32)

    @property
    def action_space(self):
        return self.perception_env.action_space

    def step(self, *args, **kwargs):
        _, reward, truncated, terminated, info = self.perception_env.step(
            *args, **kwargs)
        self._obj2vec()
        return self.current_vector, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        _, info = self.perception_env.reset(*args, **kwargs)
        self._obj2vec()
        return self.current_vector, info

    def render(self, *args, **kwargs):
        return self.perception_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.perception_env.close(*args, **kwargs)

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}
        for o in self.perception_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

    @property
    def enumerated_reference_list(self):
        nb_entities = {}
        erl = []
        for o in self.reference_list:
            if o not in nb_entities.keys():
                nb_entities[o] = 0
            erl.append(f"{o}_{nb_entities[o]}")
            nb_entities[o] += 1
        return erl

    def _obj2vec(self):
        temp_ref_list = self.reference_list.copy()
        for o in self.perception_env.objects:  # populate out_vector with object instance
            # at position of first category occurance
            idx = temp_ref_list.index(o.category)
            start = idx * 4
            flat = [item for sublist in o.h_coords for item in sublist]
            self.current_vector[start:start + 4] = flat  # write the slice
            temp_ref_list[idx] = ""  # remove reference from reference list
        for i, d in enumerate(temp_ref_list):
            if d != "":  # fill not populated category instances wiht 0.0's
                self.current_vector[i*4:i*4+4] = [0.0, 0.0, 0.0, 0.0]
