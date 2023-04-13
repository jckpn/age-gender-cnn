from torch import nn
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms

# Analyses chosen patches of any size from grid
class VariableAttentionNet(nn.Module):
    def __init__(self, num_outputs, attention_model, analysis_model, num_patches=2):
        super(VariableAttentionNet, self).__init__()

        self.num_outputs = num_outputs
        self.num_patches = num_patches

        patch_output = 50

        self.attention_net = nn.Sequential(attention_model(num_outputs=num_patches*4),
                                           nn.Sigmoid()) # Need output to be 0-1
        self.fullimg_analysis_net = analysis_model(num_outputs=patch_output)
        self.patch_analysis_net = analysis_model(num_outputs=patch_output)
        self.final_classifier = nn.Linear(num_patches*patch_output, self.num_outputs)
    
    def forward_single(self, input):
            input = input.unsqueeze(0) # 1x batch

            # Run attention network to get top N patches
            attention_output = self.attention_net(input)
            attention_output = attention_output.squeeze(0)
            attention_output = attention_output.detach().numpy()

            # Get original input image to extract patches
            input_img = input.squeeze(0).numpy()
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img_w, input_img_h = input_img.shape[1], input_img.shape[0]

            # Iteratively extract and run patches through CNN
            patch_outputs = []

            for patch_num in range(0, self.num_patches):
                # Get 2D coords of patch
                n = patch_num*4
                patch_x = int(attention_output[n] * input_img_w)
                patch_y = int(attention_output[n+1] * input_img_h)
                patch_w = int(attention_output[n+2] * input_img_w)
                patch_h = int(attention_output[n+2] * input_img_h)

                patch_x = max(1, patch_x)
                patch_y = max(1, patch_y)
                patch_w = max(1, patch_w)
                patch_h = max(1, patch_h)

                # Extract patch from image
                patch_img = input_img[patch_y:patch_y+patch_h,
                            patch_x:patch_x+patch_w]
                patch_img = cv.resize(patch_img, (100, 100))    # Patch cnn needs
                                                                # consistent size
                # Transform patch back to tensor
                patch_input = transforms.ToTensor()(patch_img)
                patch_input = patch_input.unsqueeze(0)

                # Feed patch through patch CNN
                patch_output = self.patch_analysis_net(patch_input)
                patch_outputs.append(patch_output)
            
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