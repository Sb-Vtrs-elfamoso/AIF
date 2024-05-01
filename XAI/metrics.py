"""
This code implements evaluation metrics used in this paper

Paper link to "RISE : Randomized Input Sampling for Explanation of Black-box Models"
https://arxiv.org/pdf/1806.07421


authors : Y. Jmal, I. Kallel, T. Taieb, S. Vouters
"""


import torch
import numpy as np
from sklearn.metrics import auc
from torchvision.transforms import GaussianBlur
from tqdm import tqdm


def deletion(model, image, label, saliency_map, N):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Flatten the saliency map to sort pixel indices by importance
    flat_saliency_map = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency_map.cpu().numpy())[::-1]  # Descending order of importance

    scores = []
    modified_image = image.clone().to(device)

    # Get initial score
    with torch.no_grad():
        output = model(modified_image.unsqueeze(0))

        if not (torch.is_tensor(output)):
            output = output.logits

        initial_score = output.squeeze(0)[int(label)].item()
    scores.append(initial_score)

    # Begin pixel deletion process
    for step in tqdm(range(0, len(sorted_indices), N)):
        # Set the next N most important pixels to zero
        indices_to_zero = sorted_indices[step:step + N]
        modified_image = modified_image.cpu().numpy()
        np.put(modified_image, indices_to_zero, 0)

        # Evaluate the model on the modified image
        with torch.no_grad():
            modified_image = torch.tensor(modified_image).cpu()
            output = model(modified_image.unsqueeze(0))
            if not (torch.is_tensor(output)):
                output = output.logits
            score = output.squeeze(0)[int(label)].item()
        scores.append(score)

    # Compute deletion score: Area Under the Curve of normalized scores
    # Normalize the steps to go from 0 to 1
    x_axis = np.linspace(0, 1, len(scores))
    d = auc(x_axis, scores)  # Compute the area under the curve

    return d, x_axis, scores

# image: A single-channel or three-channel image tensor (CxHxW)
# saliency_map: A saliency map of the same HxW dimensions as the image
# N: Number of pixels to remove in each step
# deletion_score, x_axis, scores = deletion(model, image, saliency_map, N)




def insertion(model, image, label, saliency_map, N):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Apply Gaussian blur to smooth out the edges
    blur = GaussianBlur(kernel_size=25, sigma=1.0)
    blurred_image = blur(image.float())

    # Flatten the saliency map to sort pixel indices by importance
    flat_saliency_map = saliency_map.flatten()
    sorted_indices = np.argsort(np.array(flat_saliency_map))[::-1]  # Descending order of importance

    scores = []
    modified_image = blurred_image.clone()

    # Get initial score with the blurred image
    with torch.no_grad():
        output = model(modified_image.unsqueeze(0))
        if not (torch.is_tensor(output)):
            output = output.logits
        initial_score = output.squeeze(0)[int(label)].item()
    scores.append(initial_score)

    # Begin pixel insertion process
    for step in tqdm(range(0, len(sorted_indices), N)):
        # Restore the next N most important pixels from the original image
        indices_to_restore = sorted_indices[step:step + N]
        modified_image = modified_image.cpu().numpy()
        np.put(modified_image, indices_to_restore, np.take(image, indices_to_restore))

        # Evaluate the model on the modified image
        with torch.no_grad():
            modified_image = torch.tensor(modified_image).cpu()
            output = model(modified_image.unsqueeze(0))
            if not (torch.is_tensor(output)):
                output = output.logits
            score = output.squeeze(0)[int(label)].item()
        scores.append(score)

    # Compute insertion score: Area Under the Curve of normalized scores
    x_axis = np.linspace(0, 1, len(scores))
    d = auc(x_axis, scores)  # Compute the area under the curve

    return d, x_axis, scores

# image: A single-channel or three-channel image tensor (CxHxW)
# saliency_map: A saliency map of the same HxW dimensions as the image
# N: Number of pixels to restore in each step
# insertion_score, x_axis, scores = insertion(model, image, saliency_map, N)