"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak

Updated on Tue May 12 22:05:25 2020
@author: Sejong Yang - github.com/kingsj0405
"""
import os
import numpy as np

import torch
from torch.optim import Adam

from visualization.misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, conv_index, layer_index,
                 filter_index, save_dir='../generated'):
        self.model = model
        self.model.eval()
        self.conv_index = conv_index  # Semantic conv index
        self.layer_index = layer_index  # Real conv index
        self.filter_index = filter_index
        self.conv_output = 0
        self.save_dir = save_dir
        # Create the folder to export images if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def visualise_layer_with_hooks(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.filter_index]
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model.modules()):
                # ModuleList can't forwarded
                if layer.__class__.__name__.lower().find('modulelist') != -1:
                    continue
                # Add hook
                if index == self.layer_index:
                    layer.register_forward_hook(hook_function)
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.layer_index:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print(f'[INFO] Iteration:{i}, Loss:{loss.data.numpy()}')
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                name = f'Conv2d.{self.conv_index}.{self.filter_index}_iter{i}.png'
                save_image(self.created_image, f'{self.save_dir}/{name}')
                print(f"[INFO] Visualization saved on {self.save_dir}/{name}")
