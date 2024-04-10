import torch

from ImportScript import load_models
from model import SiameseNetwork

def convert_to_onnx(model, input_shape, output_file):
    model.eval()
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(model,dummy_input,output_file,verbose=True)


if __name__ == "__main__":
    siamesenet = SiameseNetwork()
    siamese_model, model_detect, class_names = load_models(siamesenet)

    convert_to_onnx(siamese_model, (1,3,200,200), "models/siamese_model_resnet18_triblet.onnx")
    convert_to_onnx(model_detect, (1,3,480,640), "models/detect-person.onnx")
