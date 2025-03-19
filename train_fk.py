import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
fk_data = pd.read_csv("cropped_fk.csv")

# Extract input (joint angles) and output (EEF position + quaternion)
X = fk_data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values
X = np.hstack([np.sin(X), np.cos(X)])  # Encode angles as sin/cos

y = fk_data[['eef_x', 'eef_y', 'eef_z', 'eef_qx', 'eef_qy', 'eef_qz', 'eef_qw']].values

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define Neural Network Model with Residual Connections
class FKNN(nn.Module):
    def __init__(self):
        super(FKNN, self).__init__()
        self.fc1 = nn.Linear(12, 256)  # Adjusted for sin/cos encoding
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
        x3 = self.relu(self.bn3(self.fc3(x2))) + x1  # Residual Connection
        return self.fc_out(x3)

# Initialize model, loss, and optimizer
model = FKNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)  # Lower LR + L2 Regularization

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate model on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    mae_loss = torch.mean(torch.abs(test_outputs - y_test_tensor))
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    print(f"Test MAE: {mae_loss.item():.4f}")

    # Print a few predictions vs actual
    for i in range(5):
        print(f"Actual: {y_test_tensor[i].numpy()}, Predicted: {test_outputs[i].numpy()}")

# Save the trained model
torch.save(model.state_dict(), "fk_nn_model.pth")

print("Model training complete and saved as 'fk_nn_model.pth'")
