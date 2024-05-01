"""
This code implements GradCAM method to explain AI models

Paper link to "RISE : Randomized Input Sampling for Explanation of Black-box Models"
https://arxiv.org/pdf/1806.07421


authors : Y. Jmal, I. Kallel, T. Taieb, S. Vouters
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
import metrics
from matplotlib import cm
from scipy.ndimage import zoom


# Class allowing to observe activations of a given layer
class HookFeatures():
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)
    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)
    def gradient_hook_fn(self, grad):
        self.gradients = grad
    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


def analysis (ax, image, label, model, hook) :

    output = model(image.unsqueeze(0))
    if not torch.is_tensor(output):
        output = output.logits
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    gradients = hook.gradients
    activations = hook.features
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    image_np = image.cpu().permute(1, 2, 0).numpy()
    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap)
    
    # Resize heatmap to match image size
    scale_y = image.shape[1] / heatmap.shape[0]
    scale_x = image.shape[2] / heatmap.shape[1]
    heatmap_resized = zoom(heatmap, (scale_y, scale_x), order=1) 

    # Apply a colormap (RAINBOW) to heatmap
    heatmap_colored = cm.rainbow(heatmap_resized)[:, :, :3] 

    # Superpose heatmap on image
    superposed_img = heatmap_colored * 0.5 + image_np * 0.5 


    ax.imshow(np.clip(superposed_img,0,1))
    ax.set_title('GradCAM Saliency Map')
    ax.axis('off')

    mask = heatmap_colored[:,:,0]

    hook.close()

    deletion_score, x_axis_del, scores_del = metrics.deletion(model, image, label, torch.tensor(mask), N=100)
    insertion_score, x_axis_ins, scores_ins = metrics.insertion(model, image, label, torch.tensor(mask), N=100)

    metric = {}
    metric['deletion'] = {'auc' : deletion_score, 
                          'x_axis' : x_axis_del,
                          'scores' : scores_del}
    metric['insertion'] = {'auc' : insertion_score, 
                          'x_axis' : x_axis_ins,
                          'scores' : scores_ins}
    
    return metric
