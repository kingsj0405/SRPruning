from pruning.pruning import Pruning, RandomPruning, MagnitudePruning, MagnitudeFilterPruning, AttentionPruning

pruning_map = {
    'RandomPruning': RandomPruning,
    'MagnitudePruning': MagnitudePruning,
    'MagnitudeFilterPruning': MagnitudeFilterPruning,
    'AttentionPruning': AttentionPruning
}