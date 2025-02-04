import ml_collections


def get_config():
  """Default config values, taken from [1].

  References:
    [1]: https://github.com/denisyarats/pytorch_sac/blob/master/config/
  """
  config = ml_collections.ConfigDict()

  # ================================================= #
  # Placeholders.
  # ================================================= #
  # These values will be filled at runtime once the gym.Env is loaded.
  obs_dim = ml_collections.FieldReference(None, field_type=int)
  action_dim = ml_collections.FieldReference(None, field_type=int)
  action_range = ml_collections.FieldReference(None, field_type=tuple)

  # ================================================= #
  # Main parameters.
  # ================================================= #
  config.save_dir = "runs/"
  config.seed = 1
  config.device = "cpu"
  config.action_repeat = 1
  config.frame_stack = 3

  # Set this to True to allow CUDA to find the best convolutional algorithm to
  # use for the given parameters. When False, cuDNN will deterministically
  # select the same algorithm at a possible cost in performance.
  config.cudnn_benchmark = True
  # Enforce CUDA convolution determinism. The algorithm itself might not be
  # deterministic so setting this to True ensures we make it repeatable.
  config.cudnn_deterministic = False

  # ================================================= #
  # Training parameters.
  # ================================================= #
  config.num_train_steps = 75_000
  config.replay_buffer_capacity = 1_000_000
  config.num_seed_steps = 5_000
  config.num_eval_episodes = 50
  config.eval_frequency = 10_000
  config.checkpoint_frequency = 50_000
  config.log_frequency = 10_000
  config.save_video = True

  # ================================================= #
  # SAC parameters.
  # ================================================= #
  config.sac = ml_collections.ConfigDict()

  config.sac.obs_dim = obs_dim
  config.sac.action_dim = action_dim
  config.sac.action_range = action_range
  config.sac.discount = 0.99
  config.sac.init_temperature = 0.1
  config.sac.alpha_lr = 1e-4
  config.sac.alpha_betas = [0.9, 0.999]
  config.sac.actor_lr = 1e-4
  config.sac.actor_betas = [0.9, 0.999]
  config.sac.actor_update_frequency = 1
  config.sac.critic_lr = 1e-4
  config.sac.critic_betas = [0.9, 0.999]
  config.sac.critic_tau = 0.005
  config.sac.critic_target_update_frequency = 2
  config.sac.batch_size = 1024
  config.sac.learnable_temperature = True

  # ================================================= #
  # Critic parameters.
  # ================================================= #
  config.sac.critic = ml_collections.ConfigDict()

  config.sac.critic.obs_dim = obs_dim
  config.sac.critic.action_dim = action_dim
  config.sac.critic.hidden_dim = 1024
  config.sac.critic.hidden_depth = 2

  # ================================================= #
  # Actor parameters.
  # ================================================= #
  config.sac.actor = ml_collections.ConfigDict()

  config.sac.actor.obs_dim = obs_dim
  config.sac.actor.action_dim = action_dim
  config.sac.actor.hidden_dim = 1024
  config.sac.actor.hidden_depth = 2
  config.sac.actor.log_std_bounds = [-5, 2]

  # ================================================= #

  return config
