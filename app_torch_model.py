import torch
import torch.nn as nn
import cv2
import numpy as np
from io import BytesIO

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.leaky_relu3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure model is loaded on the correct device
    model.eval()
    return model

def preprocessing(img):
    if isinstance(img, BytesIO):
        img.seek(0)
        img_array = np.frombuffer(img.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise ValueError("Invalid input image format.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)  # Move tensor to device

    return img

def predicted_img(model, img):
    img = preprocessing(img)
    with torch.no_grad():
        predicted = model(img)
    predicted = predicted.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Ensure output is moved to CPU
    predicted = np.clip(predicted * 255.0, 0, 255).astype(np.uint8)
    return predicted
