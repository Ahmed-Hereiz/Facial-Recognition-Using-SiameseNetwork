import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
    

def load_models(siamese_model):
    
    person_detect_path = "models/detect-person.pt"
    model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=person_detect_path)
    class_names = ['person', 'person']
    
    checkpoint = torch.load('models/siamese_model_resnet18_triplet_retrained_more.pth', map_location='cpu')
    siamese_model.load_state_dict(checkpoint)
    siamese_model.eval()

    return siamese_model, model_detect, class_names
    
    
def load_preprocessing():
    transform = torchvision.transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor()
    ])
    
    return transform


def column_to_tensor(series):
    if series.shape != (128,):
        raise ValueError("Series shape must be (128,)")

    tensor_data = torch.from_numpy(series.values).view(1, -1)

    return tensor_data


def tensor_to_column(tensor):
    if tensor.shape != torch.Size([1, 128]):
        raise ValueError("Tensor shape must be torch.Size([1, 128])")

    numpy_array = tensor.detach().view(-1).numpy()
    
    return numpy_array