import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils
from agents.networks.actor import Actor_v1 as Actor
from agents.networks.critic import Critic
from agents.networks.encoder import Encoder

class SACAgent:
    def __init__(self,
                 name,
                 nstep,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 log_std_min,
                 log_std_max,
                 init_temperature,
                 actor_update_frequency,
                 critic_target_update_frequency,
                 meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.init_temperature = init_temperature
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim, log_std_min, log_std_max).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim
        # optimizers

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.train()
        self.critic_target.train()
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        
    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        self.log_alpha = torch.tensor(other.log_alpha).to(self.device)
        self.log_alpha.requires_grad = True
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        dist = self.actor(inpt)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        #action = action.clamp(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, obs_aug, action, reward, done, discount, next_obs,\
                      next_obs_aug, step):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample() ##sample
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        self.log_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_opt.step()
        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['entropy'] = -log_prob.mean().item()
            metrics['target_entropy'] = self.target_entropy

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

        batch = next(replay_iter)
        obs_, action, reward, discount, next_obs_, done = utils.to_torch(
            batch, self.device)
        # augment and encode
        obs = self.aug(obs_)
        obs_aug = self.aug(obs_)
        
        next_obs = self.aug(next_obs_)
        next_obs_aug = self.aug(next_obs_)
        
        obs = self.encoder(obs)
        obs_aug = self.encoder(obs_aug)
        
        next_obs = self.encoder(next_obs).detach()
        next_obs_aug = self.encoder(next_obs_aug).detach()
        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        # update critic
        metrics.update(
            self.update_critic(obs, obs_aug, action, reward, done, \
                               discount, next_obs, next_obs_aug, step))

        # update actor
        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return metrics