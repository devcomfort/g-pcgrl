from gym_pcgrl.envs.reps.narrow_graph import NarrowGraphRepresentation
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.wide_graph import WideGraphRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowgraph": NarrowGraphRepresentation,
    "widegraph": WideGraphRepresentation
}
