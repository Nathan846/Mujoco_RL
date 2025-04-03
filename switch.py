import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_switch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for state, label in dataloader:
            state = state.to(device)
            logits = model(state)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Q1 (phase 0)", "Q2 (phase 1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Switch Head Confusion Matrix")
    plt.show()

    print("Confusion matrix:\n", cm)

class SwitchHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.switch_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        features = self.shared_fc(x)
        logits = self.switch_head(features)
        return logits
class SwitchDataset(Dataset):
    def __init__(self, q1_data_file, q2_data_file):
        self.samples = []
        for s, _, _, _, _ in torch.load(q1_data_file):
            self.samples.append((torch.FloatTensor(s), 0))
        for s, _, _, _, _ in torch.load(q2_data_file):
            self.samples.append((torch.FloatTensor(s), 1))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, label = self.samples[idx]
        return state, label
def train_switch(model, dataloader, optimizer, device, epochs=5):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for state, label in dataloader:
            state = state.to(device)
            label = label.to(device)

            logits = model(state)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        acc = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"[Switch] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
def train_switch_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SwitchDataset("expert_p1.pt", "expert_p2.pt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    model = SwitchHead(in_dim=53).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_switch(model, dataloader, optimizer, device, epochs=10)

    torch.save(model.state_dict(), "switch_head.pth")
    print("✅ Switch head saved as 'switch_head.pth'")
    return model, dataloader
if __name__ == "__main__":
    model, dataloader = train_switch_main()
    device = 'cpu'
    evaluate_switch(model, dataloader, device)