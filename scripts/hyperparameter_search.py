import argparse
import itertools
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl.utils import Logger
from pmbrl import get_config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_trial(
    config,
    hyperparams,
    agent_config,
    trial_num,
    logger,
    env_name,
    max_episode_len,
    action_repeat,
    n_episodes,
):
    """
    Runs a single trial with the specified hyperparameters and agent configuration.
    """
    np.random.seed(config.seed + trial_num)
    torch.manual_seed(config.seed + trial_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + trial_num)

    env = GymEnv(env_name, max_episode_len, action_repeat=action_repeat, seed=config.seed)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    normalizer = Normalizer()
    buffer = Buffer(
        state_size,
        action_size,
        config.ensemble_size,
        normalizer,
        device=DEVICE,
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        config.hidden_size,
        config.ensemble_size,
        normalizer,
        device=DEVICE,
    )
    reward_model = RewardModel(state_size + action_size, config.hidden_size, device=DEVICE)

    trainer = Trainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=config.n_train_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        grad_clip_norm=config.grad_clip_norm,
        logger=logger,
    )

    # Get the global goal state
    global_goal_state = env.max_reward_state
    if env_name in ['HalfCheetah-v2', 'HalfCheetahRun', 'HalfCheetahFlip']:
        # For HalfCheetah environments, set the global goal state dynamically
        global_goal_state = None  # Will be set during runtime in the planner
    elif env_name == 'AntMaze':
        # For AntMaze, set global_goal_scale to 0.0
        hyperparams['global_goal_scale'] = 0.0

    # Update hyperparameters in the planner configuration
    planner_config = {
        'env': env,
        'ensemble': ensemble,
        'reward_model': reward_model,
        'action_size': action_size,
        'ensemble_size': config.ensemble_size,
        'plan_horizon': hyperparams.get('plan_horizon', config.plan_horizon),
        'optimisation_iters': hyperparams.get('optimisation_iters', config.optimisation_iters),
        'n_candidates': config.n_candidates,
        'top_candidates': config.top_candidates,
        'use_reward': agent_config['use_reward'],
        'use_exploration': agent_config['use_exploration'],
        'use_mean': config.use_mean,
        'expl_scale': hyperparams.get('expl_scale', config.expl_scale),
        'reward_scale': hyperparams.get('reward_scale', config.reward_scale),
        'strategy': config.strategy,
        'use_high_level': agent_config['use_high_level'],
        'context_length': hyperparams.get('context_length', config.context_length),
        'goal_achievement_scale': hyperparams.get(
            'goal_achievement_scale', config.goal_achievement_scale
        ),
        'global_goal_state': (
            torch.tensor(global_goal_state, dtype=torch.float32).to(DEVICE)
            if global_goal_state is not None
            else None
        ),
        'device': DEVICE,
        'global_goal_scale': hyperparams.get('global_goal_scale', config.global_goal_scale),
        'logger': logger,
    }

    planner = Planner(**planner_config)

    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, config.n_seed_episodes)

    episode_rewards = []
    for episode in range(1, n_episodes + 1):
        trainer.reset_models()
        trainer.train()
        # TODO: Optionally pass the recorder
        reward, steps, stats = agent.run_episode(buffer)
        episode_rewards.append(reward)
        logger.log_trial_episode(
            trial_num, episode, reward, steps, stats, agent_config['name']
        )

    return episode_rewards


def hyperparameter_search(config):
    # Define hyperparameters to search
    hyperparams_ranges = {
        'plan_horizon': list(range(10, 101, 10)),
        'context_length': list(range(10, 101, 10)),
        'optimisation_iters': list(range(10, 101, 10)),
        'expl_scale': [round(x * 0.1, 1) for x in range(1, 11)],
        'reward_scale': [round(x * 0.1, 1) for x in range(1, 11)],
        'goal_achievement_scale': [round(x * 0.1, 1) for x in range(1, 11)],
        'global_goal_scale': [round(x * 0.1, 1) for x in range(1, 11)],
    }

    # Agent configurations
    agent_configs = [
        {
            'name': 'Hierarchical',
            'use_high_level': True,
            'use_reward': True,
            'use_exploration': True,
        },
        {
            'name': 'LowPlannerOnly',
            'use_high_level': False,
            'use_reward': True,
            'use_exploration': True,
        },
        {
            'name': 'HighPlannerOnly',
            'use_high_level': True,
            'use_reward': False,
            'use_exploration': False,
        },
    ]

    default_hyperparams = {
        'plan_horizon': config.plan_horizon,
        'context_length': config.context_length,
        'optimisation_iters': config.optimisation_iters,
        'expl_scale': config.expl_scale,
        'reward_scale': config.reward_scale,
        'goal_achievement_scale': config.goal_achievement_scale,
        'global_goal_scale': config.global_goal_scale,
    }

    # Prepare results dictionary
    results = defaultdict(lambda: defaultdict(list))

    for hyperparam_name, hyperparam_values in hyperparams_ranges.items():
        for value in hyperparam_values:
            # Create a copy of default hyperparameters
            hyperparams = default_hyperparams.copy()
            # Update the current hyperparameter
            hyperparams[hyperparam_name] = value

            # Skip goal hyperparameters when not using goal planner
            if hyperparam_name in ['goal_achievement_scale', 'global_goal_scale']:
                continue  # These will be tested separately

            print(f"Testing hyperparameter {hyperparam_name} with value {value}")

            for agent_config in agent_configs:
                # Skip goal hyperparameters when agent is not using goal planner
                if not agent_config['use_high_level'] and hyperparam_name in [
                    'goal_achievement_scale',
                    'global_goal_scale',
                ]:
                    continue

                # Initialize logger
                logdir = os.path.join(
                    config.logdir,
                    f"{agent_config['name']}_{hyperparam_name}_{value}",
                )
                logger = Logger(logdir, config.seed)
                logger.log(f"Running {agent_config['name']} with {hyperparam_name}={value}")

                all_rewards = []
                for trial in range(1, config.n_trials + 1):
                    episode_rewards = run_trial(
                        config,
                        hyperparams,
                        agent_config,
                        trial,
                        logger,
                        config.env_name,
                        config.max_episode_len,
                        config.action_repeat,
                        config.n_episodes,
                    )
                    all_rewards.append(episode_rewards)
                    logger.log('-' * 50)

                # Aggregate results
                mean_rewards = np.mean(all_rewards, axis=0)
                results[agent_config['name']][hyperparam_name].append(
                    {'value': value, 'mean_rewards': mean_rewards}
                )
                logger.log('=' * 50)

    # Save results and generate plots
    save_results_and_plots(results, config)

    # Generate summary
    generate_summary(results, config)


def save_results_and_plots(results, config):
    import matplotlib.pyplot as plt

    for agent_name, hyperparam_results in results.items():
        for hyperparam_name, values_list in hyperparam_results.items():
            plt.figure()
            for entry in values_list:
                value = entry['value']
                mean_rewards = entry['mean_rewards']
                episodes = np.arange(1, len(mean_rewards) + 1)
                plt.plot(episodes, mean_rewards, label=f"{hyperparam_name}={value}")

            plt.xlabel('Episode')
            plt.ylabel('Average Return')
            plt.title(f"{agent_name} - {hyperparam_name}")
            plt.legend()
            plot_dir = os.path.join(config.logdir, 'plots', agent_name)
            os.makedirs(plot_dir, exist_ok=True)
            plt_path = os.path.join(plot_dir, f"{hyperparam_name}.png")
            plt.savefig(plt_path)
            plt.close()


def generate_summary(results, config):
    summary = {}
    for agent_name, hyperparam_results in results.items():
        summary[agent_name] = {}
        for hyperparam_name, values_list in hyperparam_results.items():
            best_value = None
            best_return = -np.inf
            for entry in values_list:
                value = entry['value']
                mean_rewards = entry['mean_rewards']
                avg_return = np.mean(mean_rewards)
                if avg_return > best_return:
                    best_return = avg_return
                    best_value = value
            summary[agent_name][hyperparam_name] = {
                'best_value': best_value,
                'best_return': best_return,
            }

    # Save summary to JSON file
    summary_path = os.path.join(config.logdir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="mountain_car")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--use_reward", action="store_true", default=False)
    parser.add_argument("--use_exploration", action="store_true", default=False)
    parser.add_argument("--expl_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--goal_achievement_scale", type=float, default=1.0)
    parser.add_argument("--global_goal_scale", type=float, default=1.0)

    args = parser.parse_args()
    config = get_config(args)
    hyperparameter_search(config)
