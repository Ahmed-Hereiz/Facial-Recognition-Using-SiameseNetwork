import os
from PIL import Image
import pandas as pd

from ImportScript import tensor_to_column, load_models, load_preprocessing
from model import SiameseNetwork


def get_embeddings_dataset(folder_path,model,transform):
    
    folder_path = "Images"
    df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg", ".gif", ".bmp")):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)

            transformed_img = transform(img).view(1, 3, 200, 200)
            em_img = model.get_embedding(transformed_img)

            person_name = os.path.splitext(filename)[0]
            df[person_name] = tensor_to_column(em_img)


    df.to_csv("people_embedding/data.csv")
    
    
    
if __name__ == "__main__":
    siamesenet = SiameseNetwork()  
    siamese_model, model_detect, class_names = load_models(siamesenet)
    
    transform = load_preprocessing()
    
    get_embeddings_dataset(folder_path="Images",model=siamese_model,transform=transform)
    df = pd.read_csv("people_embedding/data.csv", index_col="Unnamed: 0")
    print("new dataframe saved !!")
    print(df)