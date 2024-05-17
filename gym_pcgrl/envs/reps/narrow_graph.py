from gym_pcgrl.envs.reps.representation import Representation
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from PIL import Image
from gymnasium import spaces
import numpy as np
from collections import OrderedDict
from gymnasium.utils import seeding
from random import random
import math


def triangular_root(R):    
    # Calculate the discriminant
    discriminant = 1 + 8 * R
    
    # Calculate the positive values of n (idnight formula)
    n = (-1 + math.sqrt(discriminant)) / 2
    return int(math.ceil(n))

def triangular(n):
    return int((n * (n + 1)) / 2)


class NarrowGraphRepresentation(NarrowRepresentation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self, init_random_map=True, **kwargs):
        super().__init__()
        self._random_tile = True
        self.init_random_map = init_random_map
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.size = triangular(self.width-1)

    def init_random_pos(self):
        self.pos = np.random.randint(0, self.size)
        
        pos = self.pos + 1
        self._y = triangular_root(pos) - 1
        self._x = pos - triangular(self._y) - 1
        self._y += 1
        
        #print("RESET", "y", self._y, "x", self._x, "pos", self.pos)
        
        #self._x = triangular_root(self.pos) - 1
        #self._y = self.pos - triangular(self._x) - 1
        #print("x", self._x, "y", self._y)
        
        
    def reset(self, width, height, prob):
        self.init_random_pos()
        
        if self._random_start or self._old_map is None:
            self._map = self.init_random_map(self._random, width, height, 2, prob)
            self._old_map = self._map.copy().astype(np.uint8)
        else:
            self._map = self._old_map.copy().astype(np.uint8)

            
    def get_action_space(self, width, height, num_tiles):
        #return spaces.Discrete(num_tiles + 1)
        return spaces.Discrete(2)

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(2, width))
        })

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._y, self._x], dtype=np.uint8),
            "map": self._map.copy()
        })
    
    def get_observation2(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        })
    
    def get_observation_space2(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })
  
    def update(self, action):
        change = False
                
        #print("y", self._y, "x", self._x, "pos", self.pos)    
        
        if action == 1: # just flip
            if self._map[self._y][self._x] == 1:
                self._map[self._y][self._x] = 0
            else:
                self._map[self._y][self._x] = 1
            change = True
        else:
            change = False
            
        self.pos +=1
        if self.pos >= self.size:
            self.pos = 0
            
        pos = self.pos + 1
        self._y = triangular_root(pos) - 1
        self._x = pos - triangular(self._y) - 1
        self._y += 1
        
        return change, self._x, self._y


    def update2(self, action):
        # predict change of random position
        change = 0
        if action > 0:
            print(self._y)
            change += [0,1][self._map[self._y][self._x] != action-1]
            self._map[self._y][self._x] = action-1
            
        self.init_random_pos()
        return change, self._x, self._y

    
    def render(self, lvl_image, tile_size, border_size):
        # TODO
        pass
