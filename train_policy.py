"""Train an RL policy on the x-magical benchmark."""

import gym
from absl import app, flags
from ml_collections.config_flags import config_flags

from train import main as launch_rl

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")

config_flags.DEFINE_config_file(
    "config",
    "config/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("experiment_name")


def make_env(cfg) -> gym.Env:
    from xmagical import register_envs
    register_envs()
    env = gym.make(f"SweepToTop-Longstick-State-Allo-TestLayout-v0")
    return env


def main(_):
    launch_rl(FLAGS.config, make_env, FLAGS.experiment_name)


if __name__ == "__main__":
    app.run(main)
