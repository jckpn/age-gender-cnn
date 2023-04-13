from torch import nn
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms


# Attention inspired by ...
# Analyses the whole image as well as chosen patches from grid
class GridAttentionNet(nn.Module):
    def __init__(self, num_outputs, analysis_model, attention_model,
                 grid_size=4, num_patches=2):
        super().__init__()

        self.num_outputs = num_outputs
        self.grid_size = grid_size
        self.num_patches = num_patches

        patch_output = 32

        self.attention_net = attention_model(num_outputs=grid_size**2)
        self.fullimg_analysis_net = analysis_model(num_outputs=patch_output)
        self.patch_analysis_net = analysis_model(num_outputs=patch_output)
        self.final_classifier = nn.Linear(in_features=(self.num_patches + 1)*patch_output, out_features=num_outputs)

        # Final classification network:
        total_CNNs = num_patches + 1
        self.final_classifier = nn.Linear(total_CNNs*patch_output, self.num_outputs)
    
    def forward_single(self, input):
            input = input.unsqueeze(0) # 1x batch

            # Run attention network to get top N patches
            patch_scores = self.attention_net(input)
            patch_scores = patch_scores.squeeze(0)
            top_patches = torch.argsort(patch_scores, descending=True)[:self.num_patches]
            top_patches = top_patches.numpy()

            # Get original input image to extract patches
            input_img = input.squeeze(0).numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img_w, input_img_h = input_img.shape[1], input_img.shape[0]

            # Iteratively extract and run patches through CNN
            patch_img_w, patch_img_h = input_img_w//self.grid_size, input_img_h//self.grid_size
            patch_outputs = []

            for patch in top_patches:
                # Get 2D coords of patch
                patch_grid_x = patch % self.grid_size
                patch_grid_y = patch // self.grid_size
            
                # Extract patch from image
                patch_img_x = patch_grid_x*patch_img_w
                patch_img_y = patch_grid_y*patch_img_h
                patch_img = input_img[
                    patch_img_y:patch_img_y+patch_img_h,
                    patch_img_x:patch_img_x+patch_img_w]
            
                # Transform patch back to tensor
                patch_input = transforms.ToTensor()(patch_img)
                patch_input = patch_input.unsqueeze(0)

                # Feed patch through patch CNN
                patch_output = self.patch_analysis_net(patch_input)
                patch_outputs.append(patch_output)
            
            # Analyse full image as well
            img_output = self.fullimg_analysis_net(input)
            patch_outputs.append(img_output)
            
            # Final classification
            final_output = self.final_classifier(torch.cat(patch_outputs, dim=1))
            return final_output

    def forward(self, input_batch):
        output_batch = []
        for input in input_batch:
            output = self.forward_single(input)
            output_batch.append(output.squeeze(0))
        return torch.stack(output_batch, dim=0)

    
    def predict(self, input):
        output = self.forward(input)

        if self.num_outputs > 1:     # Regression
            pred = torch.argmax(output)
        else:                        # Classification
            pred = output[0][0]

        pred = pred.cpu().detach().numpy()
        return pred

