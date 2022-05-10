import os
import time

import yaml

import torch
from ml_collections import ConfigDict
from torchsac import utils
from torchsac.agent.sac import SACAgent
from torchsac.logger import Logger
from torchsac.replay_buffer import ReplayBuffer
from torchsac.video import VideoRecorder
import wandb 

def evaluate(agent, env, step, cfg, logger, video_recorder, wandb_logging):
    average_episode_reward = 0
    average_episode_success = 0
    for episode in range(cfg.num_eval_episodes):
        obs = env.reset()
        agent.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, info = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        average_episode_success += info["eval_score"]
        average_episode_reward += episode_reward
        video_recorder.save(f"{step}.mp4")
    average_episode_reward /= cfg.num_eval_episodes
    average_episode_success /= cfg.num_eval_episodes
    logger.log("eval/episode_reward", average_episode_reward, step)
    logger.log("eval/episode_success", average_episode_success, step)

    if wandb_logging:
         wandb.log({'eval/episode_reward': average_episode_reward}, step=step+1)
         wandb.log({'eval/episode_success': average_episode_success}, step=step+1)
        
    logger.dump(step)


def train(env, logger, video_recorder, cfg, agent, replay_buffer, wandb_logging):
    step = 0
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    while step < cfg.num_train_steps:
        if done:
            if step > 0:
                logger.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step, save=(step > cfg.num_seed_steps))
                if wandb_logging:
                    wandb.log({'train/duration0':time.time() - start_time}, step=step+1)

            # Evaluate agent periodically.
            if step > cfg.num_seed_steps and step % cfg.eval_frequency == 0:
                if wandb_logging:
                    wandb.log({'eval/episode':episode}, step=step+1)
                
                logger.log("eval/episode", episode, step)
                evaluate(agent, env, step, cfg, logger, video_recorder, wandb_logging)

            logger.log("train/episode_reward", episode_reward, step)

            if wandb_logging:
                wandb.log({'train/episode_reward':episode_reward}, step=step+1)

            obs = env.reset()
            agent.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            logger.log("train/episode", episode, step)

            if wandb_logging:
                wandb.log({'train/episode': episode}, step=step+1)

        # Sample action for data collection.
        if step < cfg.num_seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # Run training update.
        if step >= cfg.num_seed_steps:
            agent.update(replay_buffer, logger, step)

        next_obs, reward, done, info = env.step(action)

        # Allow infinite bootstrap.
        done = float(done)
        done_no_max = 0 if episode_step + 1 == env.max_episode_steps else done
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

        obs = next_obs
        episode_step += 1
        step += 1


def main(cfg, make_env, experiment_name, wandb_logging=False):


    exp_dir = os.path.join(cfg.save_dir, experiment_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
            yaml.dump(ConfigDict.to_dict(cfg), fp)
    else:
        raise ValueError("Experiment already exists.")
    logger = Logger(
        exp_dir, save_tb=True, log_frequency=cfg.log_frequency, agent="sac",
    )
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env()
    cfg.sac.obs_dim = env.observation_space.shape[0]
    cfg.sac.action_dim = env.action_space.shape[0]
    cfg.sac.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    agent = SACAgent(cfg.sac, device)
    replay_buffer = ReplayBuffer(
        env.observation_space.shape,
        env.action_space.shape,
        int(cfg.replay_buffer_capacity),
        device,
    )
    video_recorder = VideoRecorder(exp_dir if cfg.save_video else None)
    train(env, logger, video_recorder, cfg, agent, replay_buffer, wandb_logging)
