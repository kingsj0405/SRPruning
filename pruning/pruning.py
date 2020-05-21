import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pruning:
    def __init__(self, params, pruning_rate, exclude_biases=True):
        # Set variables
        self.pruning_rate = pruning_rate
        # Initialize params
        if exclude_biases:
            self.params = [p for p in params if p.dim() > 1]
        else:
            self.params = [p for p in params]
        # Initilaize masks
        self.masks = []
        for i, p in enumerate(self.params):
            self.masks.append(torch.ones_like(p))

    def clone_params(self):
        return [p.clone() for p in self.params]

    def rewind(self, cloned_params):
        for p_old, p_new in zip(self.params, cloned_params):
            p_old.data = p_new.data

    def update(self, channel_mask=None):
        if channel_mask is not None:  # Load channel_mask from other
            self.channel_mask = channel_mask
            for layer_index, mask in enumerate(self.channel_mask):
                m = self.masks[layer_index]
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i][j] == 1:
                            m[i][j] = torch.zeros_like(m[i][j])
                        elif mask[i][j] == 0:
                            m[i][j] = torch.ones_like(m[i][j])
                        else:
                            raise Exception(
                                f"mask should be 0 or 1, cur val is: {mask[i][j]}")
        else:
            self._update()

    def _update(self):
        raise NotImplementedError

    def zero(self):
        for m, p in zip(self.masks, self.params):
            p.data = m * p.data


class RandomPruning(Pruning):
    def __init__(self, params, pruning_rate, exclude_biases=True):
        super(
            RandomPruning,
            self).__init__(
            params,
            pruning_rate,
            exclude_biases)

    def _update(self):
        # Initialize channel_mask
        self.channel_mask = []
        for i, (m, p) in enumerate(zip(self.masks, self.params)):
            # Generate random index and border
            with torch.no_grad():
                p_size = p.size()[0] * p.size()[1]
                random_index = np.arange(p_size, dtype=np.uint32)
                np.random.shuffle(random_index)
                random_index = random_index.reshape(p.size()[:2])
                border = np.round(p_size * self.pruning_rate)
            # Update mask
            for i in range(p.size()[0]):
                for j in range(p.size()[1]):
                    index = i * p.size()[1] + j
                    if random_index[i][j] < border:
                        m[i][j] = torch.zeros_like(m[i][j])
            # Append channel_mask
            self.channel_mask.append(random_index < border)


class MagnitudePruning(Pruning):
    def __init__(self, params, pruning_rate, exclude_biases=True):
        super(
            MagnitudePruning,
            self).__init__(
            params,
            pruning_rate,
            exclude_biases)

    def _update(self):
        # Initialize channel_mask
        self.channel_mask = []
        for i, (m, p) in enumerate(zip(self.masks, self.params)):
            # Get norm of each kernel
            with torch.no_grad():
                norm_value = p.pow(2).sum(-1).sum(-1).detach().cpu().numpy()
                sorted_index = sorted(norm_value.flatten())
                border = sorted_index[int(len(sorted_index) * self.pruning_rate)]
            # Set new mask
            for i in range(p.size()[0]):
                for j in range(p.size()[1]):
                    if norm_value[i][j] < border:
                        m[i][j] = torch.zeros_like(m[i][j])
            # Append channel_mask
            self.channel_mask.append(norm_value < border)