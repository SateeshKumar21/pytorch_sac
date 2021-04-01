import torch
import os
import time
import yaml
from video import VideoRecorder
from logger import Logger
from agent.sac import SACAgent
from ml_collections import ConfigDict
from replay_buffer import ReplayBuffer
import utils


def evaluate(agent, env, step, cfg, logger, video_recorder):
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
            if cfg.sparse_reward:
                reward = env.score_on_end_of_traj()
            video_recorder.record(env)
            episode_reward += reward
        average_episode_success += info['eval_score']
        average_episode_reward += episode_reward
        video_recorder.save(f'{step}.mp4')
    average_episode_reward /= cfg.num_eval_episodes
    average_episode_success /= cfg.num_eval_episodes
    logger.log('eval/episode_reward', average_episode_reward, step)
    logger.log('eval/episode_success', average_episode_success, step)
    logger.dump(step)


def train(env, logger, video_recorder, cfg, agent, replay_buffer):
    step = 0
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    while step < cfg.num_train_steps:
        if done:
            if step > 0:
                logger.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step, save=(step > cfg.num_seed_steps))

            # Evaluate agent periodically.
            if step > cfg.num_seed_steps and step % cfg.eval_frequency == 0:
                logger.log('eval/episode', episode, step)
                evaluate(agent, env, step, cfg, logger, video_recorder)

            logger.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            agent.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            logger.log('train/episode', episode, step)

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

def main(cfg, make_env, experiment_name):
    exp_dir = os.path.join(cfg.save_dir, experiment_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
            yaml.dump(ConfigDict.to_dict(cfg), fp)
    else:
        raise ValueError("Experiment already exists.")
    logger = Logger(
        exp_dir,
        save_tb=True,
        log_frequency=cfg.log_frequency,
        agent='sac',
    )
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    env = make_env(cfg)
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
    train(env, logger, video_recorder, cfg, agent, replay_buffer)
