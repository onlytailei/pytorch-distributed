import numpy as np
import os
import gym
from gym.spaces.box import Box
from core.envs.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass

def make_env(args, rank):
    def _thunk():
        env = gym.make(args.game)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(args.game)
        env.seed(args.seed + rank)
        if is_atari:
            env = wrap_deepmind(env, frame_stack=True)
        return env
    return _thunk
