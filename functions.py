import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import numpy as np

def preprocess(filename):
    input_image = Image.open(filename)
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    input_tensor = preprocess(input_image)
    print(f"input tensor shape: {input_tensor.shape}")
    input_batch = input_tensor.unsqueeze(0) 

    return input_batch

def extract_features(model, input, num_layers=6):

    # Select the 6th layer
    model = nn.Sequential(*list(model.children())[:num_layers])
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        features = model(input)

    return features.squeeze()

# Perform k-means clustering on the extracted features
def k_means_clustering(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    labels = kmeans.fit_predict(features)
    return labels




