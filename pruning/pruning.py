import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomPruning():
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

    def update(self):
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
        # Save random_index
        self.mask_index = random_index < border

    def zero(self):
        for m, p in zip(self.masks, self.params):
            p.data = m * p.data


class MagnitudePruning():
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
        for p in self.params:
            self.masks.append(torch.ones_like(p))

    def clone_params(self):
        return [p.clone() for p in self.params]

    def rewind(self, cloned_params):
        for p_old, p_new in zip(self.params, cloned_params):
            p_old.data = p_new.data

    def update(self):
        for i, (m, p) in enumerate(zip(self.masks, self.params)):
            # Get norm of each kernel
            with torch.no_grad():
                norm_value = p.pow(2).sum(-1).sum(-1).detach().cpu().numpy()
                sorted_index = norm_value.flatten()
                sorted_index.sort()
                border = sorted_index[int(sorted_index.shape[0] * self.pruning_rate)]
            # Set new mask
            for i in range(p.size()[0]):
                for j in range(p.size()[1]):
                    if norm_value[i][j] < border:
                        m[i][j] = torch.zeros_like(m[i][j])
        self.mask_index = norm_value < border

    def zero(self):
        for m, p in zip(self.masks, self.params):
            p.data = m * p.data
