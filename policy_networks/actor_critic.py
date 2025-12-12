import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_layer(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):


    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden_gain = 2. ** 0.5

        # shared embedding
        self.embedding = nn.Sequential(init_layer(nn.Linear(input_dim, 64), gain=hidden_gain),
                                       nn.Tanh(),
                                       init_layer(nn.Linear(64, 64), gain=hidden_gain),
                                       nn.Tanh())
        
        # actor head
        self.actor = nn.Sequential(init_layer(nn.Linear(64, 64), gain=hidden_gain),
                                   nn.Tanh(),
                                   init_layer(nn.Linear(64, 64), gain=hidden_gain),
                                   nn.Tanh(),
                                   init_layer(nn.Linear(64, output_dim), gain=0.01))
        
        # critic head
        self.critic = nn.Sequential(init_layer(nn.Linear(64, 64), gain=hidden_gain),
                                    nn.Tanh(),
                                    init_layer(nn.Linear(64, 64), gain=hidden_gain),
                                    nn.Tanh(),
                                    init_layer(nn.Linear(64, 1), gain=1.0))
        

    def forward(self, obs_batch, deterministic=False):
        # we assume a batch dim ofc
        embed_batch = self.embedding(obs_batch)
        action_logits = self.actor(embed_batch)
        value_tensor = self.critic(embed_batch)
        action_dists = Categorical(logits=action_logits)
        if deterministic:
            action_tensor = action_dists.mode
        else:
            action_tensor = action_dists.sample()
        return action_tensor, value_tensor, action_dists.log_prob(action_tensor)
    

    def evaluate_actions(self, obs_batch, actions_batch):
        embed_batch = self.embedding(obs_batch)
        action_logit_batch = self.actor(embed_batch)
        action_dist_batch = Categorical(logits=action_logit_batch)
        log_prob_batch = action_dist_batch.log_prob(actions_batch)
        entropy_batch = action_dist_batch.entropy()
        value_batch = self.critic(embed_batch)
        return log_prob_batch, entropy_batch, value_batch
    

    def value(self, obs_batch):
        embed_batch = self.embedding(obs_batch)
        value_batch = self.critic(embed_batch)
        return value_batch

