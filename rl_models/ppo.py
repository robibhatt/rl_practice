from gymnasium import Env
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from policy_networks.actor_critic import ActorCritic
from typing import Any
import torch.nn.functional as F



class RolloutBuffer:

    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.obs = None
        self.actions = None
        self.log_probs_target = None
        self.advantages = None
        self.returns = None
        self.idx = 0

    def add(self, obs, action, log_probs, adv, ret):
        # we assume obs, action, log_probs, adv, ret are all batched tensors

        # create the tensors if they werent already
        if self.obs is None:
            self.obs = torch.zeros((self.n_steps,) + obs.shape[1:])
            self.actions = torch.zeros((self.n_steps,) + action.shape[1:], dtype=torch.long)
            self.log_probs_target = torch.zeros(self.n_steps)
            self.advantages = torch.zeros(self.n_steps)
            self.returns = torch.zeros(self.n_steps)

        # make sure we aren't adding too much
        assert(self.idx < self.n_steps)

        # add the values
        self.obs[self.idx] = obs.squeeze(0)
        self.actions[self.idx] = action.squeeze(0)
        self.log_probs_target[self.idx] = log_probs.item()
        self.advantages[self.idx] = adv.item()
        self.returns[self.idx] = ret.item()

        # increment the index
        self.idx += 1

        # normalize if we need to
        if self.idx == self.n_steps:
            self.normalize_advantages()

    def normalize_advantages(self):
        assert(self.idx == self.n_steps)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_minibatches(self, batch_size):
        assert(self.idx == self.n_steps)
        indices = torch.randperm(self.n_steps)

        for start in range(0, self.n_steps, batch_size):
            end = min(start+batch_size, self.n_steps)
            mb_idx = indices[start:end]
            yield(
                self.obs[mb_idx],
                self.actions[mb_idx],
                self.log_probs_target[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx]
            )



class PPO:

    def __init__(self, env:Env):
        self.env = env

        # get the observation space which we assume to be continuous and of shape (d,)
        assert(len(env.observation_space.shape) == 1)
        input_dim = env.observation_space.shape[0]

        # we assume a discrete action space
        output_dim = env.action_space.n

        # create the policy
        self.policy = ActorCritic(input_dim=input_dim, output_dim=output_dim)
        self.target_policy = ActorCritic(input_dim=input_dim, output_dim=output_dim)

        # copy the weights and prevent gradients
        self.target_policy.load_state_dict(self.policy.state_dict())
        for p in self.target_policy.parameters():
            p.requires_grad_(False)

        # create optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5)

        # decay params
        self.gamma = 0.99
        self.gae_lambda = 0.95

        # rollout buffer params
        self.n_steps = 2048 # this is how many steps we roll out from the target model to train on
        self.batch_size = 64 # the minibatch size we use for training
        self.n_epochs = 10 # how many epochs we train the loader for

        # variance reduction params
        self.clip_range = 0.2 # key clipping parameter hack for 'trust region' based updates
        self.max_grad_norm = 0.5 # just prevent huge updates

        # loss weights
        self.vf_coef = 0.5
        self.ent_coef = 0.0


    def predict(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        action_tensor, value_tensor, log_prob_tensor = self.policy(obs_tensor, deterministic)        
        action = action_tensor.item()
        return action, None
    

    @torch.no_grad()
    def generate_rollout(self, obs, reward, terminated, truncated)->tuple[RolloutBuffer, Any, float, bool, bool]:

        # create the lists of things we will use
        observations = [] # tensors
        actions = [] # tensors
        log_probs = [] # tensors
        rewards = [] # floats
        advantages = [] # tensors
        returns = [] # tensors
        values = [] # tensors
        truncates = [] # floats 1/0
        terminates = [] # float 1/0

        # here we go counting time step the 0
        steps_taken = 0
        while steps_taken < self.n_steps:

            # convert observation into tensor
            obs_tensor = torch.as_tensor(obs).unsqueeze(0)
            observations.append(obs_tensor)


            # use our target policy to generate our action
            action_tensor, value_tensor, log_probs_tensor = self.target_policy(obs_tensor)

            # insert everything else into our lists
            actions.append(action_tensor)
            values.append(value_tensor)
            log_probs.append(log_probs_tensor)
            rewards.append(reward)
            truncates.append(1. if truncated else 0.)
            terminates.append(1. if terminated else 0.)

            # add placeholder for the advantages and returns
            advantages.append(0.)
            returns.append(0.)


            # step in the environment
            if terminated or truncated:
                obs, info = self.env.reset()
                terminated = False
                truncated = False
                reward = 0.
            else:
                obs, reward, terminated, truncated, info = self.env.step(action_tensor.item())
                steps_taken += 1


        # confirm sizes
        assert(len(observations) == len(actions))
        assert(len(observations) == len(values))
        assert(len(observations) == len(log_probs))
        assert(len(observations) == len(rewards))
        assert(len(observations) == len(truncates))
        assert(len(observations) == len(terminates))
        assert(len(observations) == len(advantages))
        assert(len(observations) == len(returns))

        # set up stuff to begin computing the advantages
        next_reward = reward
        obs_tensor = torch.as_tensor(obs).unsqueeze(0) # this is the final obs
        next_value = self.target_policy.value(obs_tensor)
        next_truncate = 1. if truncated else 0.
        next_terminate = 1. if terminated else 0.
        next_advantage = torch.zeros_like(next_value)


        # get the total size which might be more than n_steps
        for step in range(len(observations)-1, -1, -1):
            
            if terminates[step] + truncates[step] > 0.1:
                advantages[step] = torch.zeros_like(next_value)
                returns[step] = torch.zeros_like(next_value)

            else:
                delta = next_reward + self.gamma * next_value * (1 - next_terminate) - values[step]
                advantages[step] = delta + (1 - next_terminate) * (1 - next_truncate) * next_advantage * self.gamma * self.gae_lambda
                returns[step] = advantages[step] + values[step]

            next_reward = rewards[step]
            next_value = values[step]
            next_truncate = truncates[step]
            next_terminate = terminates[step]
            next_advantage = advantages[step]

        # now we add into rollout buffer
        buffer = RolloutBuffer(n_steps = self.n_steps)
        dudes_added = 0
        for idx in range(len(observations)):
            if terminates[idx] + truncates[idx] > 0.1:
                pass
            else:
                buffer.add(observations[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx])
                dudes_added += 1
        
        assert(dudes_added == self.n_steps)
        return buffer, obs, reward, terminated, truncated
    

    def learn_buffer(self, buffer:RolloutBuffer):
        total_points = 0
        total_loss = 0.
        for obs_batch, actions_batch, log_probs_target_batch, adv_batch, ret_batch in buffer.get_minibatches(batch_size=self.batch_size):
            
            # get our models opinions about what's going on here.
            log_prob_batch, entropy_batch, value_batch = self.policy.evaluate_actions(obs_batch=obs_batch, actions_batch=actions_batch)

            # get the batch size
            b = entropy_batch.shape[0]
            total_points += b

            # handle the entropy
            entropy_loss = -self.ent_coef * 1/b * entropy_batch.sum()

            # handle the value loss
            value_loss = self.vf_coef * F.mse_loss(value_batch, ret_batch)
            
            # and the ppo loss
            ratio = torch.exp(log_prob_batch - log_probs_target_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            loss = policy_loss + value_loss + entropy_loss

            total_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return total_loss / total_points

    
    def learn(self, total_timesteps):
        self.policy.train()
        obs, info = self.env.reset()
        terminated = False
        truncated = False
        reward = 0.
        for i in range(total_timesteps // self.n_steps):

            # get the buffer
            buffer, obs, reward, terminated, truncated = self.generate_rollout(obs, reward, terminated, truncated)

            # train over the buffer
            for epoch in range(self.n_epochs):
                avg_loss = self.learn_buffer(buffer)
                print('Buffer rollout:', i, 'Epoch:', epoch, 'avg_tr_loss:', avg_loss)

            # copy the weights
            self.target_policy.load_state_dict(self.policy.state_dict())
