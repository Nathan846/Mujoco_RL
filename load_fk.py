"""
Forward kinematics solver, currently redundant
"""
import torch
import torch.nn as nn

class FKNN(nn.Module):
    def __init__(self):
        super(FKNN, self).__init__()
        self.fc1 = nn.Linear(12, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x1 = self.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        x2 = self.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)
        x3 = self.relu(self.bn3(self.fc3(x2))) + x1
        out = self.fc_out(x3)
        return out

def load_trained_fk_model(model_path="fk_nn_model.pth", device="cpu"):
    model = FKNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
