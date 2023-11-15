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

from ImportScript import load_models, load_preprocessing, column_to_tensor, tensor_to_column  
from get_embeddings import get_embeddings_dataset
from model import SiameseNetwork
    

def activate_siamese_net(model_detect, model_embedding, class_names, embedding_df, threshold):
    try:
        capture = cv2.VideoCapture(0)
        window_opened = False
        detected_images = []

        while True:
            ret, frame = capture.read()

            results = model_detect(frame)
            objects = results.pred[0]

            for obj in objects:
                x1, y1, x2, y2, confidence, class_id = obj.tolist()
                pt1 = (int(x1) - 25, int(y1) - 60)
                pt2 = (int(x2) + 25, int(y2) + 10)
                label = class_names[int(class_id)]
                
                if label == 'person':
                    detected_person = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    detected_person = Image.fromarray(np.uint8(detected_person))
                    detected_person_transform = transform(detected_person).view(1,3,200,200)
                    img_embed = model_embedding.get_embedding(detected_person_transform)
                    dist_list = []
                    for i in embedding_df.columns:
                        saved_embed = column_to_tensor(embedding_df[i])
                        euclidean_distance = F.pairwise_distance(saved_embed, img_embed)
                        dist_list.append(euclidean_distance.item())
                        
                    most_near_dist = min(dist_list)
                    most_near_person = dist_list.index(most_near_dist)

                    if most_near_dist < threshold:
                        name = embedding_df.columns[most_near_person]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                if 0 <= pt1[0] < pt2[0] < frame.shape[1] and 0 <= pt1[1] < pt2[1] < frame.shape[0]:
                    cv2.putText(frame, name, (pt1[0], pt1[1] - 5), font, font_scale, color, thickness)
                    cv2.rectangle(frame, pt1, pt2, color, thickness)
                    
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break


        capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print("An error occurred:", str(e))

    
if __name__ == "__main__":
    
    siamesenet = SiameseNetwork()  
    siamese_model, model_detect, class_names = load_models(siamesenet)
    
    transform = load_preprocessing()
    df = pd.read_csv("people_embedding/data.csv", index_col="Unnamed: 0")
    
    activate_siamese_net(model_detect=model_detect, model_embedding=siamese_model, class_names=class_names, embedding_df=df, threshold=2.0)
    