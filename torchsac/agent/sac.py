import numpy as np
import torch
import torch.nn.functional as F

from torchsac.agent import Agent
from torchsac import utils
from torchsac.agent.critic import DoubleQCritic
from torchsac.agent.actor import DiagGaussianActor


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self, cfg, device):
        super().__init__()

        obs_dim = cfg.obs_dim
        action_dim = cfg.action_dim

        self.action_range = cfg.action_range
        self.device = device
        self.discount = cfg.discount
        self.critic_tau = cfg.critic_tau
        self.actor_update_frequency = cfg.actor_update_frequency
        self.critic_target_update_frequency = cfg.critic_target_update_frequency
        self.batch_size = cfg.batch_size
        self.learnable_temperature = cfg.learnable_temperature

        self.critic = DoubleQCritic(
            obs_dim, action_dim, cfg.critic.hidden_dim, cfg.critic.hidden_depth
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim, action_dim, cfg.critic.hidden_dim, cfg.critic.hidden_depth
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(
            obs_dim,
            action_dim,
            cfg.actor.hidden_dim,
            cfg.actor.hidden_depth,
            cfg.actor.log_std_bounds,
        ).to(self.device)

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(
            self.device
        )
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, betas=cfg.critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=cfg.alpha_lr, betas=cfg.alpha_betas
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(
        self, obs, action, reward, next_obs, not_done, logger, step
    ):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = (
            torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        )
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            logger.log("train_alpha/loss", alpha_loss, step)
            logger.log("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        (
            obs,
            action,
            reward,
            next_obs,
            not_done,
            not_done_no_max,
        ) = replay_buffer.sample(self.batch_size)

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(
            obs, action, reward, next_obs, not_done_no_max, logger, step
        )

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_tau
            )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load("%s/actor_%s.pt" % (model_dir, step))
        )
        print(f"Successfully loaded weights from {model_dir}.")
