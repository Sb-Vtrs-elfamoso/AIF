import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

df = pd.read_pickle('df.pkl')

def process_image(image):
    # Here you would extract the vector from the image for example using a mobile net
    # For the example I just generate a random vector
    image = transforms.ToTensor()(image)
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    image = transforms.Normalize(mean, std)(image)
    image = transforms.Resize((224, 224))(image)
    image = image.unsqueeze(0) # image shape torch.Size([1, 3, 224, 224])

    mobilenet = models.mobilenet_v3_small(pretrained=True)
    model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())
    model = model.eval()
    vector = model(image).detach().numpy().tolist()

    # Now we send the vector to the API
    # Replace 'annoy-db:5000' with your Flask server address if different (see docker-compose.yml)
    response = requests.post('http://0.0.0.0:8888/reco', json={'vector': vector})
    if response.status_code == 200:
        indices = response.json()

        # Retrieve paths for the indices
        paths = df[df.index.isin(indices)]['path'].tolist()
        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0", debug=True) # the server will be accessible externally under this address

