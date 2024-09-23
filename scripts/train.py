# # pylint: disable=not-callable
# # pylint: disable=no-member

import sys
import time
import pathlib
import argparse
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl.utils import Logger
from pmbrl import get_config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    logger = Logger(args.logdir, args.seed)
    logger.log("\n=== Starting Training ===\n")
    logger.log(args)

    # Define configurations to run
    configurations = [
        {'use_high_level': True, 'plan_horizon': args.plan_horizon, 'label': 'Hierarchical'},
        {'use_high_level': False, 'plan_horizon': args.plan_horizon, 'label': 'Non-Hierarchical'}
    ]

    # Initialize metrics storage
    all_metrics = {}
    for config in configurations:
        config_label = config['label']
        all_metrics[config_label] = {
            'reward': []
        }
        if config['use_high_level']:
            all_metrics[config_label]['goal_reward'] = []
            all_metrics[config_label]['goal_achievement'] = []

    for config in configurations:
        config_label = config['label']
        logger.log(f"\n=== Running Configuration: {config_label} ===")
        
        logger.log(f'n trials: {args.n_trials} | n episodes: {args.n_episodes}')

        # For each configuration, run trials
        for trial in range(1, args.n_trials + 1):
            logger.log(f"\n=== Starting Trial {trial} for Configuration: {config_label} ===")

            # Set a unique seed for each trial
            trial_seed = args.seed + trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(trial_seed)

            # Reinitialize environment and models
            env = GymEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=trial_seed)
            action_size = env.action_space.shape[0]
            state_size = env.observation_space.shape[0]

            normalizer = Normalizer()
            buffer = Buffer(state_size, action_size, args.ensemble_size, normalizer, device=DEVICE)

            ensemble = EnsembleModel(
                state_size + action_size,
                state_size,
                args.hidden_size,
                args.ensemble_size,
                normalizer,
                device=DEVICE,
            )
            reward_model = RewardModel(state_size + action_size, args.hidden_size, device=DEVICE)

            trainer = Trainer(
                ensemble,
                reward_model,
                buffer,
                n_train_epochs=args.n_train_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epsilon=args.epsilon,
                grad_clip_norm=args.grad_clip_norm,
                logger=logger,
            )

            # Define the global goal state
            global_goal_state = env.max_reward_state
            if args.env_name in ['HalfCheetahRun', 'HalfCheetahFlip']:
                global_goal_state = None
            elif args.env_name == 'AntMaze':
                args.global_goal_scale = 0.0

            planner = Planner(
                env,
                ensemble,
                reward_model,
                action_size,
                args.ensemble_size,
                plan_horizon=config['plan_horizon'],
                optimisation_iters=args.optimisation_iters,
                n_candidates=args.n_candidates,
                top_candidates=args.top_candidates,
                use_reward=args.use_reward,
                use_exploration=args.use_exploration,
                use_mean=args.use_mean,
                expl_scale=args.expl_scale,
                reward_scale=args.reward_scale,
                strategy=args.strategy,
                use_high_level=config['use_high_level'],
                context_length=args.context_length,
                goal_achievement_scale=args.goal_achievement_scale,
                global_goal_state=torch.tensor(global_goal_state, dtype=torch.float32).to(DEVICE) if global_goal_state is not None else None,
                device=DEVICE,
                global_goal_scale=args.global_goal_scale,
                logger=logger,
            )

            agent = Agent(env, planner, logger=logger)

            agent.get_seed_episodes(buffer, args.n_seed_episodes)
            logger.log(f"\nCollected seeds: [{args.n_seed_episodes} episodes | {buffer.total_steps} frames]")

            # Initialize metrics for this trial
            trial_metrics = {
                'reward': []
            }
            if config['use_high_level']:
                trial_metrics['goal_reward'] = []
                trial_metrics['goal_achievement'] = []

            for episode in range(1, args.n_episodes + 1):
                logger.log(f"\n=== Trial {trial} | Episode {episode} ===")

                start_time = time.time()
                logger.log(f"Training on [{buffer.total_steps}/{buffer.total_steps * args.action_repeat}] data points")
                trainer.reset_models()
                ensemble_loss, reward_loss = trainer.train()
                logger.log_losses(ensemble_loss, reward_loss)

                recorder = None
                print(f'Recording every {args.record_every} episodes, episode: {episode}')
                if args.record_every is not None and args.record_every % episode == 0:
                    filename = logger.get_video_path(episode)
                    recorder = VideoRecorder(env.unwrapped, path=filename)
                    logger.log("Setup recorder @ {}".format(filename))

                logger.log("\n=== Collecting data [{}] ===".format(episode))
                reward, steps, stats = agent.run_episode(
                        buffer, action_noise=args.action_noise, recorder=recorder
                    )
                
                logger.log_episode(reward, steps)
                logger.log_stats(stats)

                # Save metrics
                trial_metrics['reward'].append(stats.get('reward', {}).get('mean', 0))

                if config['use_high_level']:
                    trial_metrics['goal_reward'].append(stats.get('goal_reward', {}).get('mean', 0))
                    trial_metrics['goal_achievement'].append(stats.get('goal_achievement', {}).get('mean', 0))

                logger.log_time(time.time() - start_time)

            # Append trial metrics to all_metrics
            all_metrics[config_label]['reward'].append(trial_metrics['reward'])

            if config['use_high_level']:
                all_metrics[config_label]['goal_reward'].append(trial_metrics['goal_reward'])
                all_metrics[config_label]['goal_achievement'].append(trial_metrics['goal_achievement'])

            # Save metrics after each trial
            logger.save_metrics(all_metrics, trial)

        # Generate plots after running all trials for this configuration
        generate_plots(all_metrics, args)

    logger.log(f"Completed Training for all configurations.")

def generate_plots(all_metrics, args):
    episodes = np.arange(1, args.n_episodes + 1)
    metrics_names = ['reward', 'goal_reward', 'goal_achievement']

    for metric_name in metrics_names:
        plt.figure()
        data_plotted = False
        for config_label, metrics in all_metrics.items():
            if metric_name not in metrics:
                continue  # Skip if this metric is not collected for this configuration
            data = np.array(metrics[metric_name])  # Shape: (n_trials, n_episodes)
            if data.size == 0:
                continue  # Skip if data is empty
            mean = np.mean(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)

            plt.plot(episodes, mean, label=f'{config_label}')
            plt.fill_between(episodes, q25, q75, alpha=0.3)
            data_plotted = True

        if not data_plotted:
            print(f"No data to plot for metric '{metric_name}'. Skipping plot.")
            plt.close()
            continue  # Skip saving the plot if no data was plotted

        plt.xlabel('Episode')
        plt.ylabel(f'Average {metric_name}')
        plt.title(f'{metric_name.replace("_", " ").capitalize()} vs Episode')
        plt.legend()
        plt.grid(True)

        # Set x-axis to integer ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save plot
        plot_path = os.path.join(args.logdir, f'{metric_name}_comparison_plot.png')
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="mountain_car")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=25)

    parser.add_argument("--expl_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)

    parser.add_argument("--use_reward", action="store_true", default=True)
    parser.add_argument("--use_exploration", action="store_true", default=True)

    parser.add_argument("--goal_achievement_scale", type=float, default=1.0)
    parser.add_argument("--global_goal_scale", type=float, default=1.0)
    parser.add_argument("--use_high_level", action="store_true", default=True)
    parser.add_argument("--context_length", type=int, default=7, help="Context length for high-level planner.")
    parser.add_argument("--plan_horizon", type=int, default=5)

    args = parser.parse_args()
    config = get_config(args)
    main(config)



# import sys
# import time
# import pathlib
# import argparse

# import numpy as np
# import torch
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# sys.path.append(str(pathlib.Path(__file__).parent.parent))

# from pmbrl.envs import GymEnv
# from pmbrl.training import Normalizer, Buffer, Trainer
# from pmbrl.models import EnsembleModel, RewardModel
# from pmbrl.control import Planner, Agent
# from pmbrl.utils import Logger
# from pmbrl import get_config

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def main(args):
#     logger = Logger(args.logdir, args.seed)
#     logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
#     logger.log(args)

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)

#     env = GymEnv(
#         args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed
#     )
    
#     action_size = env.action_space.shape[0]
#     state_size = env.observation_space.shape[0]

#     normalizer = Normalizer()
#     buffer = Buffer(state_size, action_size, args.ensemble_size, normalizer, device=DEVICE)

#     ensemble = EnsembleModel(
#         state_size + action_size,
#         state_size,
#         args.hidden_size,
#         args.ensemble_size,
#         normalizer,
#         device=DEVICE,
#     )
#     reward_model = RewardModel(state_size + action_size, args.hidden_size, device=DEVICE)

#     trainer = Trainer(
#         ensemble,
#         reward_model,
#         buffer,
#         n_train_epochs=args.n_train_epochs,
#         batch_size=args.batch_size,
#         learning_rate=args.learning_rate,
#         epsilon=args.epsilon,
#         grad_clip_norm=args.grad_clip_norm,
#         logger=logger,
#     )

#     # Define the global goal state (maximum reward state)
#     global_goal_state = env.max_reward_state  # Ensure your environment has this method
    
#     if args.env_name in ['HalfCheetahRun', 'HalfCheetahFlip']:
#         global_goal_state = None  # Will be set dynamically in the planner
#     elif args.env_name == 'AntMaze':
#         args.global_goal_scale = 0.0
    
#     planner = Planner(
#         env,
#         ensemble,
#         reward_model,
#         action_size,
#         args.ensemble_size,
#         plan_horizon=args.plan_horizon,
#         optimisation_iters=args.optimisation_iters,
#         n_candidates=args.n_candidates,
#         top_candidates=args.top_candidates,
#         use_reward=args.use_reward,
#         use_exploration=args.use_exploration,
#         use_mean = args.use_mean,
#         expl_scale=args.expl_scale,
#         reward_scale=args.reward_scale,
#         strategy=args.strategy,
#         use_high_level=True,
#         context_length=args.context_length,
#         goal_achievement_scale=args.goal_achievement_scale,
#         global_goal_state=torch.tensor(global_goal_state, dtype=torch.float32).to(DEVICE),
#         device=DEVICE,
#         global_goal_scale=args.global_goal_scale,
#         logger=logger,  # Pass the logger here
#     )


#     agent = Agent(env, planner, logger=logger)

#     agent.get_seed_episodes(buffer, args.n_seed_episodes)
#     msg = "\nCollected seeds: [{} episodes | {} frames]"
#     logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

#     for episode in range(1, args.n_episodes + 1):
#         logger.log("\n=== Episode {} ===".format(episode))
#         start_time = time.time()

#         msg = "Training on [{}/{}] data points"
#         logger.log(msg.format(buffer.total_steps, buffer.total_steps * args.action_repeat))
#         trainer.reset_models()
#         ensemble_loss, reward_loss = trainer.train()
#         logger.log_losses(ensemble_loss, reward_loss)

#         recorder = None
#         print(f'Recording every {args.record_every} episodes, episode: {episode}')
#         if args.record_every is not None and args.record_every % episode == 0:
#             filename = logger.get_video_path(episode)
#             recorder = VideoRecorder(env.unwrapped, path=filename)
#             logger.log("Setup recorder @ {}".format(filename))

#         logger.log("\n=== Collecting data [{}] ===".format(episode))
#         reward, steps, stats = agent.run_episode(
#                 buffer, action_noise=args.action_noise, recorder=recorder
#             )
#         logger.log_episode(reward, steps)
#         logger.log_stats(stats)

#         logger.log_time(time.time() - start_time)
#         logger.log("Saving metrics...")
#         logger.save()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--logdir", type=str, default="log")
#     parser.add_argument("--config_name", type=str, default="mountain_car")
#     parser.add_argument("--strategy", type=str, default="information")
#     parser.add_argument("--seed", type=int, default=0)

#     parser.add_argument("--expl_scale", type=float, default=1.0)
#     parser.add_argument("--reward_scale", type=float, default=1.0)
#     parser.add_argument("--use_reward", action="store_true", default=True)
#     parser.add_argument("--use_exploration", action="store_true", default=True)
    
#     parser.add_argument("--goal_achievement_scale", type=float, default=1.0)
#     parser.add_argument("--global_goal_scale", type=float, default=1.0)
#     # New stepping stone parameters
#     # parser.add_argument("--step_size", type=float, default=0.1, help="Step size towards the global goal for subgoals.")
#     # parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of intermediate subgoals.")
    
#     args = parser.parse_args()
#     config = get_config(args)
#     main(config)
