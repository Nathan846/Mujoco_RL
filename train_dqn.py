import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# === Simple DQN Model ===
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)

# === Dataset ===
class ExpertTensorDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.data[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        )

# === Training ===
def train_dqn(model, target_model, dataloader, optimizer, device, gamma=0.99, epochs=10, label="Q"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for state, action, reward, next_state, done in dataloader:
            state = state.to(device)
            action = action.squeeze(1).to(device)
            reward = reward.squeeze(1).to(device)
            next_state = next_state.to(device)
            done = done.squeeze(1).to(device)

            # Q(s, a)
            q_values = model(state)
            current_q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Double DQN Target: r + γ * Q_target(s', argmax_a Q_online(s'))
            with torch.no_grad():
                next_q_online = model(next_state)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                next_q_target = target_model(next_state)
                max_next_q = next_q_target.gather(1, next_actions).squeeze(1)

                target_q = reward + gamma * (1 - done) * max_next_q

33333333333333333333333333333333333333333333333            loss = F.smooth_l1_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        # Logging
        avg_loss = total_loss / max(1, len(dataloader))
        with torch.no_grad():
            avg_q = current_q.mean().item()
            max_q = current_q.max().item()
            min_q = current_q.min().item()
        print(f"[{label}] Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Q(mean/max/min): {avg_q:.2f}/{max_q:.2f}/{min_q:.2f}")

        # Optional: update target model periodically
        target_model.load_state_dict(model.state_dict())  # full sync each epoch

# === Main Training Script ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 53 
    num_actions = 14
    batch_size = 64
    learning_rate = 5e-4
    num_epochs = 10

    q1_dataset = ExpertTensorDataset("expert_p1.pt")
    q2_dataset = ExpertTensorDataset("expert_p2.pt")
    q1_loader = DataLoader(q1_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    q2_loader = DataLoader(q2_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"🔧 Loaded {len(q1_dataset)} transitions for Q1 (done=0)")
    print(f"🔧 Loaded {len(q2_dataset)} transitions for Q2 (done=1)")

    # === Build Models ===
    q1_model = DQN(state_dim, num_actions).to(device)
    q2_model = DQN(state_dim, num_actions).to(device)
    q1_target = DQN(state_dim, num_actions).to(device)
    q2_target = DQN(state_dim, num_actions).to(device)

    q1_target.load_state_dict(q1_model.state_dict())
    q2_target.load_state_dict(q2_model.state_dict())

    q1_optimizer = optim.Adam(q1_model.parameters(), lr=learning_rate)
    q2_optimizer = optim.Adam(q2_model.parameters(), lr=learning_rate)

    print("\n🚀 Training Q1...")
    train_dqn(q1_model, q1_target, q1_loader, q1_optimizer, device, gamma=0.99, epochs=num_epochs, label="Q1")

    print("\n🚀 Training Q2...")
    train_dqn(q2_model, q2_target, q2_loader, q2_optimizer, device, gamma=0.99, epochs=num_epochs, label="Q2")

    torch.save(q1_model.state_dict(), "q1_model.pth")
    torch.save(q2_model.state_dict(), "q2_model.pth")
    print("\n✅ Models saved: q1_model.pth, q2_model.pth")

if __name__ == "__main__":
    main()
