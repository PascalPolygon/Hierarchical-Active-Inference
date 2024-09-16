from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

class Agent(object):
    def __init__(self, env, planner, logger=None):
        self.env = env
        self.planner = planner
        self.logger = logger

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                goal = np.zeros(buffer.goal_size)  # Use zero goals for seed episodes
                buffer.add(state, action, reward, next_state, goal)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=None, recorder=None, goal_state=None):
        total_reward = 0
        total_steps = 0
        done = False

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                action = self.planner(state)
                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log(
                        "> Step {} [reward {:.2f}]".format(total_steps, total_reward)
                    )

                if buffer is not None:
                    # Retrieve the current goal from the planner
                    if hasattr(self.planner, 'current_goal'):
                        goal = self.planner.current_goal.cpu().numpy()
                    else:
                        goal = np.zeros(buffer.goal_size)
                    buffer.add(state, action, reward, next_state, goal)
                if recorder is not None:
                    recorder.capture_frame()

                state = deepcopy(next_state)
                if done:
                    break

        if recorder is not None:
            recorder.close()
            del recorder

        self.env.close()
        stats = self.planner.return_stats()
        return total_reward, total_steps, stats

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
