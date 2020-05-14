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

    def step(self):
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
        self.random_index = random_index < border

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

    def _border(self, flat_params):
        assert flat_params.dim() == 1
        # Compute border value
        with torch.no_grad():
            border_index = round(pruning_rate * flat_params.size()[0])
            values, __indices = torch.sort(torch.abs(flat_params))
            border = values[border_index]
        return border

    def clone_params(self):
        return [p.clone() for p in self.params]

    def rewind(self, cloned_params):
        for p_old, p_new in zip(self.params, cloned_params):
            p_old.data = p_new.data

    def step(self):
        # Gather all masked parameters
        flat_params = torch.cat([p[m == 1].view(-1)
                                 for m, p in zip(self.masks, self.params)])
        # Compute border value
        border = self._border(flat_params)
        # Calculate updated masks
        for i, (m, p) in enumerate(zip(self.masks, self.params)):
            new_mask = torch.where(torch.abs(p) < border,
                                   torch.zeros_like(p), m)
            self.masks[i] = new_mask

    def zero(self):
        for m, p in zip(self.masks, self.params):
            p.data = m * p.data
