import math

import numpy as np

from gym_pcgrl.envs.probs.graph_prob import NodeType


def triangular_root(R):
    # Calculate the discriminant
    discriminant = 1 + 8 * R
    # Calculate the positive values of n (idnight formula)
    n = (-1 + math.sqrt(discriminant)) / 2
    return int(math.ceil(n))


def triangular(n):
    return int((n * (n + 1)) / 2)


def build_graph_map(conf, random, width):
    empty = width - sum(conf.values())
    nodes = [*[NodeType.A.value] * conf[NodeType.A], *[NodeType.B.value] * conf[NodeType.B],
             *[NodeType.C.value] * conf[NodeType.C], *[NodeType.EMPTY.value] * empty]
    random.shuffle(nodes)

    prob = {0: 0.6, 1: 0.4}
    size = sum(range(width + 1))
    map = list(np.random.choice(list(prob.keys()), size=(size), p=list(prob.values())))
    m = np.full([width, width], 0).astype(int)
    for i in range(0, width):
        m[i][i:] = map[:width - i]
        del map[:width - i]
    np.fill_diagonal(m, nodes)
    return np.rot90(np.fliplr(m))


def init_graph_random(random, width, height, prob, *args, **kwargs):
    max_size = width
    nodes = [1, 1, 1]  # , 1]
    i = np.random.randint(0, len(nodes))
    size = np.random.randint(3, max_size + 1)
    while sum(nodes) <= size:
        r = np.random.randint(0, 2)
        nodes[i] += r
        if i == len(nodes):
            i = np.random.randint(0, len(nodes))

    conf = {
        NodeType.A: nodes[0],
        NodeType.B: nodes[1],
        NodeType.C: 0  # nodes[2] #0
    }
    return build_graph_map(conf, random, width)


def init_graph_controllable(conf, size=6):
    assert sum(conf.values()) <= size and sum(conf.values()) >= 3, f"Sum of nodes must be between 3 and {size}"
    return build_graph_map(conf, np.random, size)


def graph_valid(env):
    return env.unwrapped.get_prob().validate(env.unwrapped.get_map())


def inference(model_, env=None):
    done = False
    reward_total = 0
    for i in range(200):
        if not done:
            action, _ = model_.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
            reward_total += reward
            if done:
                break
    return env, info, reward_total