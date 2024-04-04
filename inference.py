# inference.py

import json
import torch
import torch.nn as nn
import numpy as np
from six import BytesIO

# Define your MalConv model (ensure this matches your training definition)
class MalConv(nn.Module):
    def __init__(self, input_length=2000000, embedding_dim=8, window_size=128, output_dim=1):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(256, embedding_dim)
        self.filter = nn.Conv1d(embedding_dim, window_size, kernel_size=512, stride=512, bias=True)
        self.attention = nn.Conv1d(embedding_dim, window_size, kernel_size=512, stride=512, bias=True)
        self.fc1 = nn.Linear(window_size, window_size)
        self.fc2 = nn.Linear(window_size, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x.long())
        x = x.transpose(1, 2)
        filt = self.filter(x)
        attn = self.attention(x)
        attn = self.sigmoid(attn)
        gated = filt * attn
        x = self.maxpool(gated)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load the PyTorch model
def model_fn(model_dir):
    model = MalConv()
    with open(f"{model_dir}/model.pt", "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model

# Deserialize the Invoke request body into an appropriate Python type
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/octet-stream':
        data = np.frombuffer(request_body, np.uint8)
        return torch.tensor(data, dtype=torch.long)
    else:
        # Handle other content-types here or raise an exception
        raise ValueError("Unsupported content type: {}".format(request_content_type))

# Perform prediction on the deserialized data, return value is a Tensor or list of Tensors
def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        return model(input_data.unsqueeze(0))

# Serialize the prediction result into the desired response content type
def output_fn(prediction_output, accept='application/json'):
    if accept == 'application/json':
        output = prediction_output.numpy().tolist()
        return json.dumps(output)
    raise ValueError("Unsupported accept type: {}".format(accept))
