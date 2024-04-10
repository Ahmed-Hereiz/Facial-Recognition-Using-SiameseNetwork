import cv2
import numpy as np
from PIL import Image
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn.functional as F

from ImportScript import load_preprocessing, column_to_tensor
from model import SiameseNetwork

def non_max_suppression(detections, threshold=0.1):
    boxes = []
    scores = []
    for detection in detections:
        x1, y1, x2, y2, _, score, _ = detection 
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
    
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.0)
        filtered_detections = detections[indices.flatten()]
    else:
        filtered_detections = []

    return filtered_detections


def activate_siamese_net(model_detect, model_embedding, class_names, embedding_df, threshold, transform):
    try:
        capture = cv2.VideoCapture(0)
        window_opened = False
        detected_images = []

        while True:
            ret, frame = capture.read()

            input_name = model_detect.get_inputs()[0].name
            d_frame = np.transpose(frame.astype(np.float32),(2,0,1))
            d_frame = np.expand_dims(d_frame,axis=0)
            results = model_detect.run(None, {input_name: d_frame})

            detections = results[0][0]
            filtered_detections = non_max_suppression(detections, threshold=0.95)
            print(len(filtered_detections))

            for obj in filtered_detections:
                x1, y1, x2, y2, confidence, class_id, _ = obj.tolist()
                pt1 = (int(x1) - 25, int(y1) - 60)
                pt2 = (int(x2) + 25, int(y2) + 10)
                label = class_names[int(class_id)]

                if label == 'person':
                    
                    detected_person = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    detected_person = Image.fromarray(np.uint8(detected_person))
                    detected_person_transform = transform(detected_person).view(1, 3, 200, 200)
                    

                    img_embed = model_embedding.run(None, {"input.1": detected_person_transform.numpy()})
                    img_embed = torch.tensor(img_embed)

                    dist_list = []
                    for i in embedding_df.columns:
                        saved_embed = column_to_tensor(embedding_df[i])
                        euclidean_distance = F.pairwise_distance(saved_embed, img_embed[0])
                        dist_list.append(euclidean_distance.item())

                    most_near_dist = min(dist_list)
                    most_near_person = dist_list.index(most_near_dist)

                    if most_near_dist < threshold:
                        name = embedding_df.columns[most_near_person]
                        print(name)
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        print(name)
                        color = (0, 0, 255)


                    thickness = 2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    if 0 <= pt1[0] < pt2[0] < frame.shape[1] and 0 <= pt1[1] < pt2[1] < frame.shape[0]:
                        cv2.putText(frame, name, (pt1[0], pt1[1] - 5), font, font_scale, color, thickness)
                        cv2.rectangle(frame, pt1, pt2, color, thickness)
                        print("iam here")
                    print("on the edge")
                    break

            print("donnnnnnnnnes")
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":

    # Load the SiameseNetwork ONNX model for embedding
    siamese_model = ort.InferenceSession('models/siamese_model_resnet18_triblet.onnx')
    
    # Load the object detection model (assuming it's also an ONNX model)
    model_detect = ort.InferenceSession('models/detect-person.onnx')

    # Load class names and preprocessing transform
    class_names = ['person']  # Assuming 'person' is the only class for simplicity
    transform = load_preprocessing()

    # Load the embedding DataFrame
    df = pd.read_csv("people_embedding/data.csv", index_col="Unnamed: 0")

    activate_siamese_net(model_detect=model_detect, model_embedding=siamese_model, class_names=class_names,
                          embedding_df=df, threshold=2.0, transform=transform)
