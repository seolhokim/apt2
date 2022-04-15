import torch
import torch.nn as nn

import utils.utils as utils
from agents.algorithms.sac import SACAgent


class ICM(nn.Module):
    """
    Same as ICM, with a trunk to save memory for KNN
    """
    def __init__(self, obs_dim, action_dim, icm_rep_hidden_dim, icm_rep_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, icm_rep_dim), 
                                   nn.LayerNorm(icm_rep_dim),nn.Tanh())

        self.forward_net = nn.Sequential(
            nn.Linear(icm_rep_dim + action_dim, icm_rep_hidden_dim), nn.ReLU(),
            nn.Linear(icm_rep_hidden_dim, icm_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * icm_rep_dim, icm_rep_hidden_dim), nn.ReLU(),
            nn.Linear(icm_rep_hidden_dim, action_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))
        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

    def get_rep(self, next_obs):
        rep = self.trunk(next_obs)
        return rep

class ICMAPTAgent(SACAgent):
    def __init__(self, icm_scale, knn_rms, knn_k, knn_avg, knn_clip,
                 update_encoder, icm_hidden_dim, icm_rep_dim, icm_rep_hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self.icm_scale = icm_scale
        self.update_encoder = update_encoder
        self.icm = ICM(self.obs_dim, self.action_dim, 
                       icm_rep_hidden_dim, icm_rep_dim).to(self.device)
        # optimizers
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=self.lr)

        self.icm.train()

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() + backward_error.mean()

        self.icm_opt.zero_grad()
        self.encoder_opt.zero_grad()
        loss.backward()
        self.icm_opt.step()
        self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics

    def aug_and_encoding(self, obs):
        obs1 = self.aug(obs)
        obs2 = self.aug(obs)
        
        obs1 = self.encoder(obs1)
        obs2 = self.encoder(obs2)
        return obs1, obs2
    def compute_intr_reward(self, next_obs):
        rep = self.icm.get_rep(next_obs)
        reward = self.pbe(rep)
        reward = reward.reshape(-1, 1)
        return reward
    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, done = utils.to_torch(
            batch, self.device)
        # augment and encode
        obs, obs_aug = self.aug_and_encoding(obs)
        next_obs, next_obs_aug = self.aug_and_encoding(next_obs)
        next_obs, next_obs_aug = next_obs.detach(), next_obs_aug.detach()
        if self.reward_free:
            metrics.update(self.update_icm(obs, action, next_obs, step))
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(next_obs).detach()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            if self.reward_free :
                metrics['intr_reward'] = intr_reward.mean().item()
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if self.reward_free:
            obs = obs.detach()
            obs_aug = obs_aug.detach()        

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
