"""
Redudant option critic architecture. Attached only for reference.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import numpy as np
from math import exp
from copy import deepcopy

class OptionCriticFeatures(nn.Module):
    def __init__(self,
                 in_features,
                 num_actions,
                 num_options,
                 temperature=1.0,
                 eps_start=1.0,
                 eps_min=0.1,
                 eps_decay=int(1e6),
                 eps_test=0.05,
                 device='cpu',
                 testing=False):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)
        self.terminations = nn.Linear(64, num_options)
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 2:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    device = model.device
    
    batch_idx = torch.arange(len(options)).long().to(device)
    options   = torch.LongTensor(options).to(device)
    rewards   = torch.FloatTensor(rewards).to(device)
    masks     = 1 - torch.FloatTensor(dones).to(device)

    states = model.get_state(obs).squeeze(0)
    Q      = model.get_Q(states)

    with torch.no_grad():
        next_states_prime = model_prime.get_state(next_obs).squeeze(0)
        next_Q_prime      = model_prime.get_Q(next_states_prime)

        next_states = model.get_state(next_obs).squeeze(0)
        next_termination_probs = model.get_terminations(next_states)
        next_options_term_prob = next_termination_probs[batch_idx, options]

    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] 
         + next_options_term_prob * next_Q_prime.max(dim=-1)[0])

    td_err = 0.5 * (Q[batch_idx, options] - gt).pow(2).mean()
    return td_err

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(obs)
    next_state = model.get_state(next_obs)
    next_state_prime = model_prime.get_state(next_obs)

    option_term_prob = model.get_terminations(state)[:, option]
    with torch.no_grad():
        next_option_term_prob = model.get_terminations(next_state)[:, option]
    Q = model.get_Q(state).squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).squeeze()

    done_flag = float(done)
    gt = reward + (1 - done_flag) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] 
         + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() 
                                           + args.termination_reg) * (1 - done_flag)
    
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    return termination_loss + policy_loss
