# Begin file: pmbrl/configs.py
import pprint

MOUNTAIN_CAR_CONFIG = "mountain_car"
CUP_CATCH_CONFIG = "cup_catch"
HALF_CHEETAH_RUN_CONFIG = "half_cheetah_run"
HALF_CHEETAH_FLIP_CONFIG = "half_cheetah_flip"
REACHER_CONFIG = "reacher"
ANT_MAZE = "ant_maze"
DEBUG_CONFIG = "debug"

def print_configs():
    print(f"[{MOUNTAIN_CAR_CONFIG}, {CUP_CATCH_CONFIG}, {HALF_CHEETAH_RUN_CONFIG}, {HALF_CHEETAH_FLIP_CONFIG}, {REACHER_CONFIG}, {ANT_MAZE}, {DEBUG_CONFIG}]")

def get_config(args):
    if args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == CUP_CATCH_CONFIG:
        config = CupCatchConfig()
    elif args.config_name == HALF_CHEETAH_RUN_CONFIG:
        config = HalfCheetahRunConfig()
    elif args.config_name == HALF_CHEETAH_FLIP_CONFIG:
        config = HalfCheetahFlipConfig()
    elif args.config_name == REACHER_CONFIG:
        config = ReacherConfig()
    elif args.config_name == ANT_MAZE:
        config = AntMazeConfig()
    elif args.config_name == DEBUG_CONFIG:
        config = DebugConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    config.set_logdir(args.logdir)
    config.set_seed(args.seed)
    config.set_strategy(args.strategy)
    
    config.n_trials = args.n_trials
    config.n_episodes = args.n_episodes
        
    config.use_exploration = args.use_exploration
    config.use_reward = args.use_reward
    config.expl_scale = args.expl_scale
    config.reward_scale = args.reward_scale
    
    config.context_length = args.context_length
    config.plan_horizon = args.plan_horizon
    
    # Set new parameters from args
    # config.global_goal_weight = args.global_goal_weight
    # config.max_subgoal_distance = args.max_subgoal_distance
    # config.initial_goal_std = args.initial_goal_std
    # config.goal_std_decay = args.goal_std_decay
    # config.min_goal_std = args.min_goal_std
    config.goal_achievement_scale = args.goal_achievement_scale
    # config.goal_mean_weight = args.goal_mean_weight
    config.global_goal_scale = args.global_goal_scale
    # config.subgoal_scale = args.subgoal_scale

    return config

# Base Configuration Class
class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 0
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.record_every = None
        self.coverage = False

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 3
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.strategy = "information"  # Renamed from expl_strategy to strategy for consistency
        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False

        self.expl_scale = 1.0
        self.reward_scale = 1.0
        self.action_noise_scale = 0.1

        self.context_length = 7
        self.n_trials = 25

        # New parameters with default values
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1
        self.subgoal_scale = 1.0
        self.global_goal_scale = 1.0
        self.max_steps = 10
        self.step_size = 0.1 #For goal stepping stone mechanism
        

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def set_strategy(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return pprint.pformat(vars(self))


# Derived Configuration Classes

class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v0"
        self.n_episodes = 25
        # self.expl_scale = 1.0
        # self.reward_scale = 10.0
        self.n_train_epochs = 400
        self.max_episode_len = 200
        self.hidden_size = 64
        self.plan_horizon = 5
        self.context_length = 7
        self.record_every = 0  # Record every episode for debugging

        # self.expl_scale = 1.0
        # self.reward_scale = 1.0
        
        # Overriding new parameters for debugging
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.env_name = "SparseMountainCar"

        # self.n_train_epochs = 100
        self.n_train_epochs = 1000
        self.plan_horizon = 15
        self.context_length = 18
        self.max_episode_len = 200
        # self.max_episode_len = 200
        # self.n_train_epochs = 100
        self.n_seed_episodes = 5
        self.expl_scale = 1.0
        self.n_episodes = 5
        # self.ensemble_size = 5
        self.n_candidates = 500
        self.top_candidates = 50
        self.record_every = 0

        # Overriding new parameters for Mountain Car
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class CupCatchConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "catch"
        self.env_name = "DeepMindCatch"
        self.max_episode_len = 1000
        self.action_repeat = 4
        self.plan_horizon = 12
        self.expl_scale = 0.1
        self.action_noise_scale = 0.2
        self.record_every = None
        self.n_episodes = 50

        # Overriding new parameters for Cup Catch
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class HalfCheetahRunConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_run"
        self.env_name = "HalfCheetahRun"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1
        self.action_noise_scale = 0.3

        # Overriding new parameters for Half Cheetah Run
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class HalfCheetahFlipConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_flip"
        self.env_name = "HalfCheetahFlip"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1
        self.action_noise_scale = 0.4

        # Overriding new parameters for Half Cheetah Flip
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class AntMazeConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "ant_maze"
        self.env_name = "AntMaze"
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.max_episode_len = 300
        self.action_repeat = 4
        self.coverage = True

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 200
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_reward = False
        self.use_mean = True
        self.expl_scale = 1.0
        self.action_noise_scale = 0.5

        # Overriding new parameters for Ant Maze
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1


class ReacherConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "reacher"
        self.env_name = "DeepMindReacher"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 1000
        self.action_repeat = 4

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_exploration = True
        self.use_reward = True
        self.use_mean = True
        self.expl_scale = 0.1
        self.action_noise_scale = 0.1

        # Overriding new parameters for Reacher
        self.global_goal_weight = 1.0
        self.max_subgoal_distance = 1.0
        self.initial_goal_std = 1.0
        self.goal_std_decay = 0.99
        self.min_goal_std = 0.1
        self.goal_mean_weight = 0.5
        self.goal_achievement_scale = 0.1
