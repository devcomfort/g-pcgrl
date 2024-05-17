from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gymnasium import spaces
import numpy as np
from gym_pcgrl.envs.helper import gen_preset_random_map, gen_random_map
from gym_pcgrl.envs.reps.narrow_graph import triangular, triangular_root


"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""


class WideGraphRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_random_map = kwargs["init_random_map"]
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.size = triangular(self.height-1)
        
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([self.size, 2])

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles - 1, dtype=np.uint8, shape=(height, width))
        })


    def get_observation(self):
        return {
            "map": self._map.copy()
        }
    
    def update4(self, action):
        # predict change at predicted position then swap value 0/1
        #print(action)
        pos = action[0] + 1
        tile = action[1] # 0 or 1
        y = triangular_root(pos) - 1
        x = pos - triangular(y) - 1
        y += 1
        
        if self._map[y][x] == tile:
            return False, x, y
        else:
            self._map[y][x] = tile
            return True, x, y

    
    def update(self, action):
        # predict change at predicted position then swap value 0/1
        #print(action)
        pos = action[0] + 1
        replace = action[1] # 0 or 1
        y = triangular_root(pos) - 1
        x = pos - triangular(y) - 1
        y += 1
        
        if replace == 1: # just flip
            if self._map[y][x] == 1:
                self._map[y][x] = 0
            else:
                self._map[y][x] = 1
                #print("Changing to 1 at pos", y, x)
                #print(self._map)
            return True, x, y
        else:
            return False, x, y
    
    
    def update3(self, action):
        # predict change at predicted position with predicted tile
        pos = action[0] + 1
        tile = action[1]
        replace = action[2] # 0 or 1
        y = triangular_root(pos) - 1
        x = pos - triangular(y) - 1
        y += 1
        
        #print("action", action, "x", x, "y", y)
        
        if self._map[y][x] == tile:
            return False, x, y
        elif replace == 1:
            self._map[y][x] = tile
            return True, x, y
        else:
            return False, x, y
    
    
    def update2(self, action):
        pos = action[0] + 1
        tile = action[1]
        replace = action[2] # 0 or 1
        x = triangular_root(pos) - 1
        y = pos - triangular(x) - 1  
        
        if self._map[y][x] == tile:
            return False, x, y
        else:
            self._map[y][x] = tile
            return True, x, y

    def reset(self, width, height, prob):
        if self._random_start or self._old_map is None:
            self._map = self.init_random_map(self._random, width, height, prob)
            self._old_map = self._map.copy().astype(np.uint8)
        else:
            self._map = self._old_map.copy().astype(np.uint8)
