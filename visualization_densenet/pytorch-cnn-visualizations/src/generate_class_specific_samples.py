"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np

import torch
from torch.optim import SGD, Adam
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable

from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, device, target_class , niter=100):
#        self.mean = [-0.485, -0.456, -0.406]
#        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.device=device
        self.model = model.to(device)
        self.model.eval()
        self.target_class = target_class
        self.niter = niter
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(100, 200, (224, 224, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def generate(self):
        initial_learning_rate = 6
                #---# self.processed_image = preprocess_image(self.created_image, False)
        self.processed_image = transforms.functional.to_tensor(self.created_image).to(self.device)
        self.processed_image.unsqueeze_(0)
        # Convert to Pytorch variable
        


        # Define optimizer for the image
        
        

        for i in range(1, self.niter+1):
            self.processed_image = Variable(self.processed_image, requires_grad=True)
            optimizer = Adam([self.processed_image], lr=initial_learning_rate, weight_decay=1e2)
            #optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Process image and return variable
            
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class] #+self.processed_image.sum()/50
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.cpu().numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.processed_image=torch.clamp(self.processed_image.detach(),min=0,max=1)
            self.created_image = recreate_image(self.processed_image.cpu())
            if i % 10 == 0:
                # Save image
                im_path = 'generated/c_specific_iteration_'+str(i)+'.png'
                save_image(self.created_image, im_path)
        #return self.processed_image


if __name__ == '__main__':
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()
