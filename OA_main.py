import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from OA_env import OA_env

class HierarchicalDQN(nn.Module):
    def __init__(self, in_features, num_actions, hidden_dim=128):
        super(HierarchicalDQN, self).__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU()
        )
        self.device = 'cpu'
        self.switch_fc = nn.Linear(hidden_dim, 2)
        self.q_head1 = nn.Linear(hidden_dim, num_actions)
        self.q_head2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x, use_hard_switch=False):
        features = self.shared_fc(x)
        switch_logits = self.switch_fc(features)
        switch_probs = F.softmax(switch_logits, dim=-1)
        q1 = self.q_head1(features)
        q2 = self.q_head2(features)
        if use_hard_switch:
            switch_choice = torch.argmax(switch_probs, dim=-1, keepdim=True)
            final_q = torch.where(switch_choice == 0, q1, q2)
        else:
            final_q = switch_probs[:, 0].unsqueeze(-1) * q1 + switch_probs[:, 1].unsqueeze(-1) * q2
        return final_q, switch_probs, q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(agent, target_agent, replay_buffer, batch_size, optimizer, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return None
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    device = agent.device
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    current_q, _, _, _ = agent(states, use_hard_switch=False)
    current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q, _, _, _ = target_agent(next_states, use_hard_switch=False)
        max_next_q = next_q.max(1)[0]
        target_q = rewards + gamma * (1 - dones) * max_next_q
    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = OA_env(device=str(device))
    state_dim = env.obs_dim
    num_actions = env.action_space.n
    hidden_dim = 128
    replay_capacity = 10000
    batch_size = 32
    num_episodes = 1000
    gamma = 0.99
    update_target_freq = 50
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 500
    agent = HierarchicalDQN(state_dim, num_actions, hidden_dim).to(device)
    target_agent = HierarchicalDQN(state_dim, num_actions, hidden_dim).to(device)
    target_agent.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(replay_capacity)
    epsilon = epsilon_start
    total_steps = 0
    for episode in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                q_values, switch_probs, _, _ = agent(state_tensor, use_hard_switch=True)
                action = q_values.argmax(dim=-1).item()
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_steps += 1
            epsilon = max(epsilon_final, epsilon_start - total_steps / epsilon_decay)
            loss = train_dqn(agent, target_agent, replay_buffer, batch_size, optimizer, gamma)
        if episode % update_target_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        print(f"Episode: {episode}, Reward: {ep_reward}, Epsilon: {epsilon:.3f}, Loss: {loss if loss is not None else 'N/A'}")
        
if __name__ == "__main__":
    main()
