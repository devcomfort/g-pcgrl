import numpy as np
import gymnasium as gym
from gymnasium.core import Wrapper

from gym_pcgrl.envs.reps.narrow_graph import triangular, triangular_root
from gym_pcgrl.wrappers import OneHotEncoding


class OneHotGraph(OneHotEncoding):
    def __init__(self, game, name, **kwargs):
        super().__init__(game, name, **kwargs)
        self.dim = kwargs["node_types"]

class GraphWrapper(Wrapper):
    # use with graph-wide representation
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)
        else:
            self.env = game
        #get_pcgrl_env(self.env).adjust_param(**kwargs)
        self.env = OneHotGraph(self.env, 'map', **kwargs)
        self.width = kwargs["width"]
        self.size = triangular(self.width)
        self.num_tiles = kwargs["node_types"]
        gym.Wrapper.__init__(self, self.env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[self.width, self.width, self.num_tiles], dtype=np.float32)


    def step(self, action):
        #pos, replace = np.unravel_index(action, (self.size, 2)) # size, num_tiles, replace 0/1       
        #obs, reward, done, truncated, info = self.env.step([pos, replace])
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, stats = self.env.reset()
        obs = self.transform(obs)
        return obs, stats
        
    def transform(self, obs):
        return obs[self.name] #* 255
        #return th.tensor(obs[self.name]).to(dtype=th.float32) * 255 #.to(torch.device("cuda"))
        
        
class GraphNarrowWrapper(GraphWrapper):
    # use with graph-wide representation
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game, **kwargs)
        else:
            self.env = game
        #get_pcgrl_env(self.env).adjust_param(**kwargs)
        self.env = OneHotGraph(self.env, 'map', **kwargs)
        self.width = kwargs["width"]
        self.size = triangular(self.width)
        self.num_tiles = kwargs["node_types"]
        gym.Wrapper.__init__(self, self.env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[2, self.width, self.num_tiles], dtype=np.float32)


    def step(self, action):
        #pos, replace = np.unravel_index(action, (self.size, 2)) # size, num_tiles, replace 0/1       
        #obs, reward, done, truncated, info = self.env.step([pos, replace])
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, stats = self.env.reset()
        obs = self.transform(obs)
        return obs, stats
        
    def transform(self, obs):
        #print(obs)
        m = obs["map"].copy()
        #print(m)
        y, x = obs["pos"][0], obs["pos"][1]
        # select col/row "around the corner" of a-matrix
        row = m[y]
        row[y+1:] = m[:, y][y+1:]
        col = m[x]
        col[x+1:] = m[:, x][x+1:]
        #print(row, col)
        return np.array([row, col])
        #return th.tensor(np.array([row, col])).to(dtype=th.float32) #* 255