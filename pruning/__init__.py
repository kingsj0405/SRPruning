from pruning.pruning import Pruning, RandomPruning, MagnitudePruning, MagnitudeFilterPruning

pruning_map = {
    'RandomPruning': RandomPruning,
    'MagnitudePruning': MagnitudePruning,
    'MagnitudeFilterPruning': MagnitudeFilterPruning
}