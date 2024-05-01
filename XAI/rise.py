"""
This code implements RISE method to explain AI models

Paper link to "RISE : Randomized Input Sampling for Explanation of Black-box Models"
https://arxiv.org/pdf/1806.07421


authors : Y. Jmal, I. Kallel, T. Taieb, S. Vouters
"""


import numpy as np
import random
import torch
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from PIL import Image
import matplotlib.pyplot as plt
import metrics

# Mask generation
def generate_masks(N, p, H, W, h=7, w=7):
    # Step 1: Sample N binary masks of size h × w to 1 with probability p
    binary_masks = torch.rand(N, h, w) < p

    # Step 2: Upsample all masks to size (h + 1)CH × (w + 1)CW using bilinear interpolation
    CH = H // h
    CW = W // w

    upsampled_masks = torch.nn.functional.interpolate(binary_masks.unsqueeze(1).float(),
                                    size=((h + 1) * CH, (w + 1) * CW),
                                    mode='bilinear', align_corners=False).squeeze()

    # Apply Gaussian blur to smooth out the edges
    blur = GaussianBlur(kernel_size=25, sigma=1.0)
    blurred_masks = blur(upsampled_masks.float())


    # Step 3: Crop areas H × W with uniformly random indents from (0, 0) up to (CH, CW)
    cropped_masks = []
    for mask in blurred_masks:
        indent_h = random.randint(0, CH)
        indent_w = random.randint(0, CW)
        cropped_mask = mask[indent_h:indent_h+H, indent_w:indent_w+W]
        cropped_masks.append(cropped_mask)

    return torch.stack(cropped_masks)



# Saliency map calculation
def rise_map(model, label, image, p, masks):
    saliency_map = torch.zeros_like(image)
    batch_size = len(masks)

    # Predict with masked images batch
    for mask in tqdm(masks):
        # Create masked image
        masked_image = image * mask

        # Predict with masked image
        with torch.no_grad():
            prediction = model(masked_image.unsqueeze(0))

        # If resnet is used
        if not torch.is_tensor(prediction):
            prediction = prediction.logits
        
        # Update saliency map
        saliency_map += prediction.squeeze(0)[int(label)] * mask
    
    # Normalize saliency map
    W, H = masks[0].size()
    saliency_map /= (batch_size * p * W * H)

    return saliency_map



# Ploting image and saliency map together
def plot_saliency_map(ax, image, saliency_map):
    saliency_map_np = saliency_map.cpu().detach().permute(1, 2, 0).numpy()

    # Normalize the saliency map to range [0, 1]
    saliency_map_np = (saliency_map_np - saliency_map_np.min()) / (saliency_map_np.max() - saliency_map_np.min())

    # Convert the image tensor to a numpy array
    image_np = image.cpu().permute(1, 2, 0).numpy()  # Assuming image is in CHW format

    # Apply a colormap (coolwarm) to the saliency map
    cmap = plt.get_cmap('jet')
    colored_saliency = cmap(saliency_map_np[:,:,0])[:, :, :3]

    # Superpose saliency map on image
    alpha = 0.5
    superposed_image = alpha * colored_saliency + (1 - alpha) * image_np

    # Plot superposed image with saliency map
    ax.imshow(superposed_image)
    ax.set_title('RISE Saliency Map')
    ax.axis('off')



# Implementing the whole RISE analysis with the metrics of insertion and deletion
def analysis(ax, image, label, model, masks, N=4000, p=0.5) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image.to(device)
    masks.to(device)

    # Saliency map calculations and display
    saliency_map = rise_map(model=model, label=label, image=image, p=p, masks=masks)
    plot_saliency_map(ax, image, saliency_map)

    # Metrics calculations
    deletion_score, x_axis_del, scores_del = metrics.deletion(model, image, label, saliency_map[0,:,:], N=100)
    insertion_score, x_axis_ins, scores_ins = metrics.insertion(model, image, label, saliency_map[0,:,:], N=100)

    metric = {}
    metric['deletion'] = {'auc' : deletion_score, 
                          'x_axis' : x_axis_del,
                          'scores' : scores_del}
    metric['insertion'] = {'auc' : insertion_score, 
                          'x_axis' : x_axis_ins,
                          'scores' : scores_ins}
    
    return metric

