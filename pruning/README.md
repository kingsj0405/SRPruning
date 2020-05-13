# Pruning

Inspired by [lgalke/torch-pruning](https://github.com/lgalke/torch-pruning)

## Usage

```python
import torch

from pruning import MagnitudePruning


net = # an arbitrary pytorch nn.Module instance
dataloader = # some pytorch dataloader instance
optimizer = torch.optim.SGD(net.parameters(), 0.01, weight_decay=1e-5)
pruning = MagnitudePruning(net.parameters(), 0.1)
w_0 = pruning.clone_params()  # Save initial parameters for rewinding


# train function
def train(net, dataloader, n_epochs=1):
    # Some standard training loop ...
    for epoch in range(n_epochs):
        for x, y in dataloader:
            pruning.zero()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Train, Prune, Rewinding, and Re-train
train(net, dataloader, n_epochs=100)
pruning.step()  # Update masks!
pruning.zero()  # Real prune!
pruning.rewind(w_0)  # Rewind parameters to their values at init
train(net, dataloader, n_epochs=100)
```