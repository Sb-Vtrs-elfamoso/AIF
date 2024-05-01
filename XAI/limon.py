"""
This code implements LIME method to explain AI models

Paper link to "RISE : Randomized Input Sampling for Explanation of Black-box Models"
https://arxiv.org/pdf/1806.07421


authors : Y. Jmal, I. Kallel, T. Taieb, S. Vouters
"""

from lime import lime_image
import matplotlib.pyplot as plt
import torch
import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_saliency_map(ax, image, saliency_map):

    # Convert the image tensor to a numpy array
    #image_np = image.cpu().permute(1, 2, 0).numpy()  # Assuming image is in CHW format

    ax.imshow(image, cmap='gray', interpolation='nearest')

    # Create a color map for the mask
    cmap = mcolors.ListedColormap(['none', 'red'])  # 'none' is transparent, 'red' for the mask
    bounds = [0, 0.5, 1]  # Boundaries for the colors
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Display the mask
    ax.imshow(saliency_map, cmap=cmap, norm=norm, interpolation='none', alpha=0.2)  # alpha controls transparency
    ax.set_title('LIME Saliency Map')
    ax.axis('off')


def analysis (ax, image, label, model, batch_predict) :
    image_np = image.cpu().permute(1, 2, 0).numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np,
                                            batch_predict, # classification function
                                            top_labels=5,
                                            hide_color=0,
                                            num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    
    plot_saliency_map(ax, temp, mask)

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


