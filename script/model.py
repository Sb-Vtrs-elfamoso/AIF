import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from annoy import AnnoyIndex


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path


mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])
dataset = ImageAndPathsDataset('MLP-20M', transform)

dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)

mobilenet = models.mobilenet_v3_small(pretrained=True)
model = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())
model = model.eval()

features_list = []
paths_list = []

for x, paths in tqdm(dataloader):
    with torch.no_grad():
        embeddings = model(x)
        features_list.extend(embeddings.cpu().numpy())
        paths_list.extend(paths)

df = pd.DataFrame({
    'features': features_list,
    'path': paths_list
})


df.to_pickle('df.pkl')
print("df saved !")

features = np.vstack(features_list)
cosine_sim = cosine_distances(features, features)


dim = 576
annoy_index = AnnoyIndex(dim, 'angular')

for i, embedding in enumerate(features_list):
    annoy_index.add_item(i, embedding)

annoy_index.build(10) # 10 trees
annoy_index.save('annoy_db.ann')
print("annoy index saved !")