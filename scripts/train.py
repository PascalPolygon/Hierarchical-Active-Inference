# pylint: disable=not-callable
# pylint: disable=no-member

# Begin file: scripts/train.py
import sys
import time
import pathlib
import argparse

import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

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
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = GymEnv(
        args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed
    )
    
    # action_space = gym.spaces.Box(
    #     low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
    # )
    # observation_space = gym.spaces.Box(
    #     low=self.low_state, high=self.high_state, dtype=np.float32
    # )
        
    # logger.log(env.action_space)
    # logger.log(env.unwrapped.action_space)
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

    # Define the global goal state (maximum reward state)
    global_goal_state = env.max_reward_state  # Ensure your environment has this method
    
    # Get the global goal state
    global_goal_state = env.max_reward_state
    if args.env_name in ['HalfCheetahRun', 'HalfCheetahFlip']:
        global_goal_state = None  # Will be set dynamically in the planner
    elif args.env_name == 'AntMaze':
        args.global_goal_scale = 0.0
    
    planner = Planner(
        env,
        ensemble,
        reward_model,
        action_size,
        args.ensemble_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        # use_reward=False,
        # use_exploration=False,
        use_reward=args.use_reward,
        use_exploration=args.use_exploration,
        use_mean = args.use_mean,
        # use_mean=args.use_mean,
        expl_scale=args.expl_scale,
        reward_scale=args.reward_scale,
        strategy=args.strategy,
        use_high_level=True,
        context_length=args.context_length,
        goal_achievement_scale=args.goal_achievement_scale,
        global_goal_state=torch.tensor(global_goal_state, dtype=torch.float32).to(DEVICE),
        device=DEVICE,
        # New parameters
        # global_goal_weight=args.global_goal_weight,
        # max_subgoal_distance=args.max_subgoal_distance,
        # initial_goal_std=args.initial_goal_std,
        # goal_std_decay=args.goal_std_decay,
        # min_goal_std=args.min_goal_std,
        # goal_mean_weight=args.goal_mean_weight,
        # Additional parameters
        # subgoal_scale=args.subgoal_scale,
        global_goal_scale=args.global_goal_scale,
        logger=logger,  # Pass the logger here
    )
    
    # planner = Planner(
    #     env,
    #     ensemble,
    #     reward_model,
    #     action_size,
    #     args.ensemble_size,
    #     plan_horizon=args.plan_horizon,
    #     optimisation_iters=args.optimisation_iters,
    #     n_candidates=args.n_candidates,
    #     top_candidates=args.top_candidates,
    #     use_reward=args.use_reward,
    #     use_exploration=args.use_exploration,
    #     use_mean=args.use_mean,
    #     expl_scale=args.expl_scale,
    #     reward_scale=args.reward_scale,
    #     strategy=args.strategy,
    #     use_high_level=True,
    #     context_length=args.context_length,
    #     goal_achievement_scale=args.goal_achievement_scale,
    #     global_goal_state=torch.tensor(global_goal_state, dtype=torch.float32).to(DEVICE),
    #     device=DEVICE,
    #     global_goal_scale=args.global_goal_scale,
    #     logger=logger,  # Pass the logger here
    #     step_size=args.step_size,      # New parameter
    #     max_steps=args.max_steps,      # New parameter
    # )


    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    for episode in range(1, args.n_episodes + 1):
        logger.log("\n=== Episode {} ===".format(episode))
        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(msg.format(buffer.total_steps, buffer.total_steps * args.action_repeat))
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

        logger.log_time(time.time() - start_time)
        logger.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="mountain_car")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--expl_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--use_reward", action="store_true", default=False)
    parser.add_argument("--use_exploration", action="store_true", default=False)
    
    parser.add_argument("--goal_achievement_scale", type=float, default=1.0)
    parser.add_argument("--global_goal_scale", type=float, default=1.0)
    # New stepping stone parameters
    # parser.add_argument("--step_size", type=float, default=0.1, help="Step size towards the global goal for subgoals.")
    # parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of intermediate subgoals.")
    
    args = parser.parse_args()
    config = get_config(args)
    main(config)
