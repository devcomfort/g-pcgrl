import os
from PIL import Image
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_pcgrl.envs.reps.wide_graph import triangular, triangular_root
from random import random
import pandas as pd
from gymnasium.utils import seeding

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from enum import Enum


class NodeType(Enum):
    LOOT = 3
    CRAFT = 4
    ITEM = 5
    EMPTY = 2
    A = 6
    B = 7
    C = 8
    
    def __str__(self):
        return self.name
    

class GraphProb(Problem):
    def __init__(self, width=6, height=6, **kwargs):
        super().__init__()

        self._width = width
        self._height = height
        self._prob = {"noedge": 0.2, "edge": 0.5, "loot": 0.1, "craft": 0.1, "item": 0.1}
        self._border_size = (0, 0)
        self._border_tile = "noedge"
        self.init_random_map = self.init_map
        
        if "rules" in kwargs:
            self.rules = kwargs["rules"]
        else:
            self.rules = {
                NodeType.LOOT: [NodeType.CRAFT],
                NodeType.CRAFT: [NodeType.LOOT, NodeType.ITEM],
                NodeType.ITEM: [NodeType.CRAFT]
            }
        self.positions = self.init_triang_pos()

    def init_triang_pos(self):
        positions = []
        for i in range(triangular(self._width-1)):
            pos = i + 1
            y = triangular_root(pos) - 1
            x = pos - triangular(y) - 1
            y += 1
            positions.append((y, x))
        return positions

    def get_tile_types(self):
        return ["noedge", "edge", "loot", "craft", "item"]
    
    def get_change_reward(self, m1, m2):
        m1 = GraphProb.clear_empty_nodes(m1)
        m2 = GraphProb.clear_empty_nodes(m2)
        
        change_y, change_x = np.where(m1 != m2)
        try:
            action = m2[change_y[0]][change_x[0]]
        except IndexError:
            return 0 # no change
        
        nodes = m2.diagonal()
        node1, node2 = NodeType(nodes[change_y[0]]), NodeType(nodes[change_x[0]])
        check_rule = node2 in self.rules[node1]
        #print(action, check_rule)
        
        if action == 0 and check_rule is False: # removed wrong edge
            return 2
        elif action == 1 and check_rule is False: # created wrong edge
            return -2
        elif action == 0 and check_rule is True: # removed correct edge
            return -2
        elif action == 1 and check_rule is True: # created correct edge
            return 2
        else:
            return 0

    def get_stats(self, m):
        valid = self.validate(m)
        m_ = GraphProb.clear_empty_nodes(m)
        num_edges = len(np.where(m_ == 1)[0])
        return {"valid": valid, "valids": 0, "num_edges": num_edges, "map": m.copy()}
    
    def get_reward(self, new_stats, old_stats):
        try:
            reward = self.get_change_reward(old_stats["map"], new_stats["map"])
        except Exception:
            reward = 0
        if new_stats["valid"] is True:
            reward += 10#3
        return reward
    
    def get_reward2(self, new_stats, old_stats):
        reward = 0
        for (kn, vn), (ko, vo) in zip(new_stats["valids"].items(), old_stats["valids"].items()):
            reward += get_range_reward(vn, vo, 1, 4)
        
        # having at least num_nodes-1 edges
        reward += get_range_reward(new_stats["num_edges"], old_stats["num_edges"], 3, 7) * 3
            
        if new_stats["valid"] is True:
            reward += 10
        return reward

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["valid"] == True

    @staticmethod
    def map_to_graph2(m):
        np.fill_diagonal(m, 0)
        m = m.astype(int)

        rows, cols = np.where(m == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        return gr
    
    @staticmethod
    def map_to_graph(m):
        m = m.astype(int)
        m_ = m.copy()
        np.fill_diagonal(m_, 0)
        return nx.from_numpy_array(m_)
    
    @staticmethod
    def get_labels(m):
        return {i:v for i, v in zip(range(m.shape[0]), m.diagonal())}

    def clear_empty_nodes(m):
        # return copy of m with cleaned EMPTY nodes out
        m_ = m.copy()
        none_nodes = list(np.where(np.diagonal(m_) == NodeType.EMPTY.value)[0])
        m_ = np.delete(m_, none_nodes, axis=0)
        m_ = np.delete(m_, none_nodes, axis=1)
        return m_
        
    def render(self, m):
        # delete all empty nodes
        m = GraphProb.clear_empty_nodes(m)

        # plot
        nodes_labels = m.diagonal()
        labels = {idx: NodeType(n).name for idx, n in zip(range(len(nodes_labels)), list(nodes_labels))}

        m = m.astype(int)
        m_ = m.copy()
        np.fill_diagonal(m_, 0)
        gr = nx.from_numpy_array(m_)
        #gr = nx.from_numpy_array(m_, create_using=nx.DiGraph)
        nx.draw(gr, node_size=400, labels=labels, with_labels=True, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', arrows=True,     connectionstyle="arc3,rad=0.2")
        plt.show()
        
        
    def get_debug_info(self, new_stats, old_stats):
        return new_stats
    
    def reset(self, start_stats):
        super().reset(start_stats)
        
    def init_map(self):
        size = sum(range(self.width+1))
        random, seed = seeding.np_random(seed)
        map = random.choice(list(self._prob.keys()), size=(size), p=list(prob.values())).astype(int)
        m = np.zeros([self.height, self.width]).astype(int).astype(str)
        for i in range(0, self.width):
            m[i][i:] = map[:self.width-i]
            del map[:self.width-i]
        #return np.rot90(np.fliplr(m))
    
    def edges_missing(actual, should):
        l = {a for a in actual if a in should}
        return len(l) - len(should)
    
    def validate(self, m):
        m_ = m.copy()
        none_nodes = list(np.where(np.diagonal(m_) == NodeType.EMPTY.value)[0])
        m_ = np.delete(m_, none_nodes, axis=0)
        m_ = np.delete(m_, none_nodes, axis=1)
        nodes = m_.diagonal().copy()

        np.fill_diagonal(m_, 0)
        m_[m_ > 1] = 0
        g = nx.from_numpy_array(m_)
        for n_idx in g.nodes():
            try:
                node = NodeType(nodes[n_idx])
                neighbors = [NodeType(nodes[nei_idx]) for nei_idx in g.neighbors(n_idx)]
            except ValueError:
                # for e.g., wide when 0 or 1 is on Node position
                return False
            for node_rule in self.rules[node]:
                #print(node_rule)
                if node_rule not in neighbors:
                    #print("node", node, "must have neighbor", node_rule)
                    return False
            for nei in neighbors:
                if nei not in self.rules[node]:      
                    #print("node", node, "must not have neighbor", nei)
                    return False   
        return True
    
    def validate2(self, m):
        nodes = [NodeType(n) for n in m.diagonal()]
        for y, x in self.positions:
            # ignore edges if empty node
            if nodes[x] == NodeType.EMPTY or nodes[y] == NodeType.EMPTY:
                continue
            if m[y][x] == 1:
                # check if edge is valid
                if nodes[x] not in self.rules[nodes[y]]:
                    return False
            else:
                # check if edge should be there
                if nodes[x] in self.rules[nodes[y]]:
                    # edge is missing, but maybe a valid connection exist to different node of same type
                    return False
        return True