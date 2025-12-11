from gymnasium import Env
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from policy_networks.mlp_policy import MlpPolicyNetwork
from torch.distributions import Categorical


class Reinforce:

    def __init__(self, env:Env):
        self.env = env

        # get the observation space which we assume to be continuous and of shape (d,)
        assert(len(env.observation_space.shape) == 1)
        input_dim = env.observation_space.shape[0]

        # we assume a discrete action space
        output_dim = env.action_space.n

        # create the policy
        self.policy = MlpPolicyNetwork(input_dim=input_dim, output_dim=output_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99


    def predict(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        action_logits = self.policy(obs_tensor)
        distribution = Categorical(logits=action_logits)
        if deterministic:
            action_tensor = distribution.mode  # argmax or mean
        else:
            action_tensor = distribution.sample()
        
        action = action_tensor.item()
        return action, None 
    

    def learn(self, total_timesteps):
        steps_taken = 0
        last_log = 0
        while steps_taken <= total_timesteps:
            obs, info = self.env.reset()
            rewards = []
            log_probs = []
            episode_done = False
            while not episode_done:

                # get our next action
                obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                action_logits = self.policy(obs_tensor)
                distribution = Categorical(logits=action_logits)
                action_tensor = distribution.sample()

                # get the log probability
                log_prob = distribution.log_prob(action_tensor)
                log_prob = log_prob.sum(dim=-1)
                log_probs.append(log_prob)

                # get the reward and next state
                obs, reward, terminated, truncated, info = self.env.step(action_tensor.item())
                rewards.append(reward)

                # increment total steps
                steps_taken += 1

                # check whether the episode is over
                if terminated or truncated:
                    episode_done = True

            # update model if we are under the total budget
            if steps_taken <= total_timesteps:
                current_return = 0.
                # back prop time
                self.optimizer.zero_grad()

                # loop through rewards backwards
                for reward, log_prob in zip(reversed(rewards), reversed(log_probs)):
                    current_return *= self.gamma
                    current_return += reward
                    loss = -1. * current_return * log_prob

                    # add to the gradient
                    loss.backward()

                # step the optimizer
                self.optimizer.step()

            # print every once in a while what we got going
            if steps_taken > last_log + 10000:
                print('steps learned:', steps_taken)
                print('latest reward:', sum(rewards))
                last_log = steps_taken