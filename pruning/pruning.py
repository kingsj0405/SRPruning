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
        for p in self.params:
            self.masks.append(torch.ones_like(p))

    def clone_params(self):
        return [p.clone() for p in self.params]

    def rewind(self, cloned_params):
        for p_old, p_new in zip(self.params, cloned_params):
            p_old.data = p_new.data

    def step(self):
        for i, (m, p) in enumerate(zip(self.masks, self.params)):
            # Generate random index
            # FIXME: Choose from 1 masks only
            random_index = torch.arange(p.size()[:1])
            print(f"random_index.shape before view: {random_index.shape}")
            np.random.shuffle(random_index)
            random_index = random_index.view(-1)
            print(f"random_index.shape after view: {random_index.shape}")
            # Update mask
            border = np.round(random_index.shape[0] / pruning_rate)
            new_mask = torch.where(random_index < border,
                                   torch.zeros_like(p),
                                   m)
            self.masks[i] = new_mask

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
