import gradio as gr
from PIL import Image
import requests
import numpy as np
import os
import torch
from torchvision import transforms
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from annoy import AnnoyIndex
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Adjust these paths as necessary
paths = np.load('paths.npy', allow_pickle=True)  # Load movie poster paths

# Transformations applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Pre-trained model and transformations setup
# Load the pretrained MobileNetV3 model
mobilenet = models.mobilenet_v3_small(pretrained=True)

# Create a subset of the model to extract features
model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()).to(device)

model.eval()

def extract_features(image):    
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0).to(device)
    print(image)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()  # Ensure features are correctly shaped
    return features




            
        
        

def recommend_similar_movies(uploaded_image):
    try:
        feature = extract_features(uploaded_image)
        # Inside your recommend_similar_movies function, replace the hard-coded URL with:
        api_url = os.getenv("FLASK_API_URL", "http://annoy-db:5000/query")
        response = requests.post(api_url, json={'feature': feature.tolist()})
        response.raise_for_status()  # Check for HTTP request errors
        nearest_ids = response.json()  # Get the list of nearest neighbor indices from the response


        # Initialize a list to store the PIL Images of the recommended movies
        images = []
        for i in nearest_ids:
            img_path = paths[i]  # Retrieve the path for each recommended movie
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
            else:
                print(f"Image path does not exist: {img_path}")
        
        return images  # Return the list of PIL Images to display in Gradio
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


iface = gr.Interface(
    fn=recommend_similar_movies,
    inputs=gr.Image(),
    outputs=gr.Gallery(),
    title="Movie Recommender",
    description="Upload a movie poster to see similar movie posters."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
