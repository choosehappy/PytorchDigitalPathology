"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable


from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, device,selected_layer, selected_filter ,niter=31):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.niter = niter
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer

        #for module_pos, module in enumerate(self.model.features._modules.items()):
        for module_pos, module in enumerate(self.model.features):
            if(module_pos==self.selected_layer):
                module[1].register_forward_hook(hook_function)
                break

        #self.model[self.selected_layer].register_forward_hook(hook_function)

    # def visualise_layer_with_hooks(self):
        # # Hook the selected layer
        # self.hook_layer()
        # # Process image and return variable
        # self.processed_image = preprocess_image(self.created_image, False)

        # # Define optimizer for the image
        # optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        # for i in range(1, self.niter):
            # optimizer.zero_grad()
            # # Assign create image to a variable to move forward in the model
            # x = self.processed_image
            # for index, layer in enumerate(self.model.features._modules.items()):
                # # Forward pass layer by layer
                # # x is not used after this point because it is only needed to trigger
                # # the forward hook function
                # x = layer[1](x)
                # # Only need to forward until the selected layer is reached
                # if index == self.selected_layer:
                    # # (forward hook function triggered)
                    # break
            # # Loss function is the mean of the output of the selected layer/filter
            # # We try to minimize the mean of the output of that specific filter
            # loss = -torch.mean(self.conv_output)
            # print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # # Backward
            # loss.backward()
            # # Update image
            # optimizer.step()
            # # Recreate image
            # self.created_image = recreate_image(self.processed_image)
            # # Save image
            # if i % 5 == 0:
                # im_path = 'generated/layer_vis_l' + str(self.selected_layer) + \
                    # '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                # save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, resize_im=False)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            #for index, layer in enumerate(self.model.features._modules.items()):
            for index, layer in enumerate(self.model.features):
                # Forward pass layer by layer
                #x = layer[1](x)
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 5 == 0:
                im_path = 'generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 17
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
