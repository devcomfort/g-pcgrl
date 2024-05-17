import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper

# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""


class ToImage(Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, render_mode="rgb_array", **kwargs)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env, **kwargs)
        self.shape = None
        depth = 0
        max_value = 0
        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'
            if self.shape == None:
                self.shape = self.env.observation_space[n].shape
            new_shape = self.env.observation_space[n].shape
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert self.shape[0] == new_shape[0] and self.shape[1] == new_shape[
                1], 'This wrapper only works when all objects have same width and height'
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(low=0, high=max_value, shape=(self.shape[0], self.shape[1], depth),
                                                dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        obs = self.transform(obs)
        return obs, info

    def transform(self, obs):
        final = np.empty([])
        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2)
        return final


"""
Transform any object in the dictionary to one hot encoding

can be stacked
"""


class OneHotEncoding(Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(
            name)
        self.name = name

        self.observation_space = gym.spaces.Dict({})
        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space[self.name].shape
        self.dim = self.observation_space[self.name].high.max() - self.observation_space[self.name].low.min() + 1
        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)

    def step(self, action):
        # action = get_action(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.transform(obs)
        return obs, info

    def transform(self, obs):
        old = obs[self.name]
        try:
            obs[self.name] = np.eye(self.dim)[old].astype(np.uint8)
        except IndexError:
            obs[self.name] = np.eye(self.dim + 1)[old].astype(np.uint8)
        return obs


"""
Transform the input space to a 3D map of values where the argmax value will be applied

can be stacked
"""


class ActionMap(Wrapper):
    def __init__(self, game, render_mode="rgb_array", **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, render_mode, **kwargs)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'
        self.old_obs = None
        self.one_hot = len(self.env.observation_space['map'].shape) > 2
        w, h, dim = 0, 0, 0
        if self.one_hot:
            h, w, dim = self.env.observation_space['map'].shape
        else:
            h, w = self.env.observation_space['map'].shape
            dim = self.env.observation_space['map'].high.max()
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
        # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h * w * self.dim)

    def reset(self):
        self.old_obs, info = self.env.reset()
        return self.old_obs, info

    def step(self, action):
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))
        # truncated = False
        # if 'pos' in self.old_obs:
        #    o_x, o_y = self.old_obs['pos']
        #    if o_x == x and o_y == y:
        #        obs, reward, done, truncated, info = self.env.step(v)
        #    else:
        #        o_v = self.old_obs['map'][o_y][o_x]
        #        if self.one_hot:
        #            o_v = o_v.argmax()
        #        obs, reward, done, truncated, info = self.env.step(v)
        #        info = self.env.step(o_v)
        # else:

        self.env._max_episode_steps = 288
        obs, reward, done, truncated, info = self.env.step([x, y, v])
        self.old_obs = obs
        return obs, reward, done, truncated, info


class SwapActionMap(ActionMap):
    # swapping full with first row
    def __int__(self, game, render_mode, **kwargs):
        super().__init__(game, render_mode, **kwargs)
        self.action_space = gym.spaces.Discrete((self.h * self.w + self.h * self.w) * 1, dtype=np.uint)

    def step(self, action):
        # y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, y2, x2, v = np.unravel_index(action, (self.h, self.w, self.h, self.w, 1))
        if 'pos' in self.old_obs:
            o_x, o_y = self.old_obs['pos']
            if o_x == x and o_y == y:
                obs, reward, done, truncated, info = self.env.step(v)
            else:
                o_v = self.old_obs['map'][o_y][o_x]
                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, truncated, info = self.env.step(o_v)
        else:
            obs, reward, done, truncated, info = self.env.step([x, y, x2, y2, 1])
        self.old_obs = obs
        return obs, reward, done, truncated, info


class SwapFullActionMap(gym.Wrapper):
    # swapping full 
    def __init__(self, game, width, height, **kwargs):
        super().__init__(game)
        self.w = width
        self.h = height
        self.action_space = gym.spaces.Discrete(self.h * self.w * self.h * self.w * 2)

    def step(self, action):
        x1, y1, x2, y2, v = np.unravel_index(action, (self.h, self.w, self.h, self.w, 2))
        # print(self.env)
        obs, reward, done, truncated, info = self.env.step([x1, y1, x2, y2, v])
        self.old_obs = obs
        return obs, reward, done, truncated, info


"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""


class Cropped(Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)  # render_mode="rgb_array",
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a position'
        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(
            name)
        assert len(self.env.observation_space.spaces[name].shape) == 2, "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size // 2
        self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})
        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=high_value, shape=(crop_size, crop_size),
                                                                  dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        obs = self.transform(obs)
        return obs, info

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs['pos']

        # View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y + self.size, x:x + self.size]
        obs[self.name] = cropped
        # return obs
        return obs[self.name]


class SwapCropped(Cropped):
    def __init__(self, game, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, render_mode="rgb_array", **kwargs)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        Wrapper.__init__(self, self.env)
        self.name = name
        self.size = 3  # cropsize 3
        self.pad = 1
        self.pad_value = pad_value

        high_value = self.env.observation_space[self.name].high.max() + 1
        self.observation_space = gym.spaces.Box(low=0, high=high_value, shape=(kwargs["width"], 3, high_value),
                                                dtype=np.uint8)  # 5
        # for (k,s) in self.env.observation_space.spaces.items():
        #    self.observation_space.spaces[k] = s
        # self.observation_space.spaces["pos1"] = gym.spaces.Box(low=0, high=high_value, shape=(self.size, self.size), dtype=np.uint8)
        # self.observation_space.spaces["pos2"] = gym.spaces.Box(low=0, high=high_value, shape=(self.size, self.size), dtype=np.uint8)

    def transform(self, obs):
        map = obs[self.name]
        x1, y1, x2, y2 = obs['pos']
        dim = self.env.observation_space[self.name].high.max() - self.env.observation_space[self.name].low.min() + 1

        # View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        pos1 = padded[y1:y1 + 3, x1:x1 + 3]
        pos2 = padded[y2:y2 + 3, x2:x2 + 3]
        pos1 = np.eye(dim)[pos1]
        pos2 = np.eye(dim)[pos2]
        return np.append(pos1, pos2, axis=0).astype(np.uint8)
        # return {"pos1": obs["pos1"], "pos2": obs["pos2"]}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        obs = self.transform(obs)
        return obs, info


################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for narrow and turtle experiments
"""


class CroppedImagePCGRLWrapper(Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game, **kwargs)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        if "swap" in game:
            env = SwapCropped(self.pcgrl_env, self.pcgrl_env.get_border_tile(), **kwargs)
        else:
            env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')

        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indices = ['map']
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


class ImageGrapPCGRLWrapper(Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game, **kwargs)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        if "swap" in game:
            env = SwapCropped(self.pcgrl_env, self.pcgrl_env.get_border_tile(), 'map')
        else:
            env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')

        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indices = ['map']
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""


class ActionMapImagePCGRLWrapper(Wrapper):
    def __init__(self, game, render_mode="rgb_array", **kwargs):
        self.pcgrl_env = gym.make(game, render_mode, **kwargs)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indices = ['map']
        env = self.pcgrl_env

        # Add the action map wrapper

        if "swap" in game:
            # print("Using swap action map wrapper.")
            env = SwapActionMap(env, render_mode, **kwargs)
        else:
            env = ActionMap(env, render_mode, **kwargs)
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset()


class SwapFullWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game, render_mode="rgb_array", **kwargs)
        # self.pcgrl_env.adjust_param(**kwargs)
        flat_indices = ['map']

        # Add the action map wrapper
        env = SwapFullActionMap(self.pcgrl_env, **kwargs)
        env = OneHotEncoding(env, 'map')
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset()
