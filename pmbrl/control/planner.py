# Begin file: pmbrl/control/planner.py
import torch
import torch.nn as nn

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random

class Planner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        action_size,
        ensemble_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        use_reward=True,
        use_exploration=True,
        use_mean=False,
        expl_scale=1.0,
        reward_scale=1.0,
        strategy="information",
        device="cpu",
        use_high_level=False,  # High-level planner flag
        context_length=1,  # Context length
        goal_achievement_scale=1.0,  # Scale for goal-achievement reward
        global_goal_state=None,  # The known global goal state
        # New parameters
        global_goal_weight=1.0,
        max_subgoal_distance=1.0,
        initial_goal_std=1.0,
        goal_std_decay=0.99,
        min_goal_std=0.1,
        goal_mean_weight=0.5,
        # Additional parameters
        subgoal_scale=1.0,
        global_goal_scale=1.0,
        logger=None,  # Logger for debugging
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates

        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.use_mean = use_mean
        self.expl_scale = expl_scale
        self.reward_scale = reward_scale
        self.device = device

        self.use_high_level = use_high_level  # High-level planner flag
        self.context_length = context_length
        self.goal_achievement_scale = goal_achievement_scale
        self.global_goal_state = global_goal_state.to(device)  # Ensure on correct device

        # New parameters
        self.global_goal_weight = global_goal_weight
        self.max_subgoal_distance = max_subgoal_distance
        self.initial_goal_std = initial_goal_std
        self.goal_std_decay = goal_std_decay
        self.min_goal_std = min_goal_std
        self.goal_mean_weight = goal_mean_weight

        # Additional parameters
        self.subgoal_scale = subgoal_scale
        self.global_goal_scale = global_goal_scale
        self.logger = logger  # Pass logger for debugging

        if strategy == "information":
            self.measure = InformationGain(self.ensemble, scale=expl_scale)
        elif strategy == "variance":
            self.measure = Variance(self.ensemble, scale=expl_scale)
        elif strategy == "random":
            self.measure = Random(self.ensemble, scale=expl_scale)
        elif strategy == "none":
            self.use_exploration = False

        self.trial_rewards = []
        self.trial_bonuses = []
        self.to(device)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        # Initialize action and subgoal distributions
        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )

        # For subgoals, determine how many subgoals we have based on the context length
        num_subgoals = (self.plan_horizon + self.context_length - 1) // self.context_length
        goal_size = state_size  # Assuming goal is in the same space as state

        goal_mean = self.global_goal_state.unsqueeze(0).unsqueeze(1).repeat(num_subgoals, 1, 1)  # Shape: (num_subgoals, 1, goal_size)
        goal_std_dev = torch.ones(num_subgoals, 1, goal_size).to(self.device) * self.initial_goal_std

        for _ in range(self.optimisation_iters):
            # Sample action candidates
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            # Sample subgoal candidates
            goals = goal_mean + goal_std_dev * torch.randn(
                num_subgoals,
                self.n_candidates,
                goal_size,
                device=self.device,
            )

            # Perform rollout with sampled actions
            states, delta_vars, delta_means = self.perform_rollout(state, actions)

            # Compute returns
            returns = torch.zeros(self.n_candidates).float().to(self.device)

            if self.use_reward:
                _states = states.view(-1, state_size)
                _actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _actions = _actions.view(-1, self.action_size)
                rewards = self.reward_model(_states, _actions)
                rewards = rewards * self.reward_scale
                rewards = rewards.view(
                    self.plan_horizon, self.ensemble_size, self.n_candidates
                )
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards
                self.trial_rewards.append(rewards)

            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                returns += expl_bonus
                self.trial_bonuses.append(expl_bonus)

            # Add goal-achievement reward
            if self.use_high_level:
                goal_achievements = self.compute_goal_achievement(states, goals, state)
                returns += goal_achievements * self.goal_achievement_scale

            # Update distributions using top candidates
            action_mean, action_std_dev, goal_mean, goal_std_dev = self._update_distributions(
                actions, goals, returns
            )

            # Decay the goal standard deviation
            goal_std_dev = torch.clamp(goal_std_dev * self.goal_std_decay, min=self.min_goal_std)

        # Extract the first action and the first subgoal to return
        selected_action = action_mean[0].squeeze(dim=0)  # Shape: (action_size,)
        selected_goal = goal_mean[0].squeeze(dim=1)      # Shape: (goal_size,)

        return selected_action, selected_goal

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * (T - 1)
        delta_vars = [torch.empty(0)] * (T - 1)

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.use_mean:
                next_state = states[t] + delta_mean
            else:
                next_state = states[t] + self.ensemble.sample(delta_mean, delta_var)
            states[t + 1] = next_state
            delta_means[t] = delta_mean
            delta_vars[t] = delta_var

        # Stack the lists into tensors
        states = torch.stack(states[1:], dim=0)  # Exclude initial state
        delta_vars = torch.stack(delta_vars, dim=0)
        delta_means = torch.stack(delta_means, dim=0)

        return states, delta_vars, delta_means

    def compute_goal_achievement(self, states, goals, current_state):
        goal_achievements = torch.zeros(self.n_candidates).float().to(self.device)
        c = self.context_length
        num_subgoals = goals.size(0)

        for i in range(num_subgoals):
            t = i * c + c - 1  # Last timestep in the context
            if t >= self.plan_horizon:
                t = self.plan_horizon - 1  # Ensure t is within bounds
            predicted_state = states[t]  # (ensemble_size, n_candidates, state_size)
            predicted_state = predicted_state.mean(dim=0)  # (n_candidates, state_size)
            desired_state = goals[i, :, :]  # (n_candidates, goal_size)

            # Distance between predicted state and subgoal
            diff_subgoal = desired_state - predicted_state  # (n_candidates, state_size)
            distance_subgoal = torch.norm(diff_subgoal, dim=1)  # (n_candidates,)

            # Distance between subgoal and global goal state
            diff_global_goal = self.global_goal_state.unsqueeze(0) - desired_state  # (n_candidates, state_size)
            distance_global_goal = torch.norm(diff_global_goal, dim=1)  # (n_candidates,)

            # Distance between current state and subgoal to ensure feasibility
            current_state_repeated = current_state.unsqueeze(0).repeat(self.n_candidates, 1)
            diff_feasibility = desired_state - current_state_repeated  # (n_candidates, state_size)
            distance_feasibility = torch.norm(diff_feasibility, dim=1)  # (n_candidates,)

            # Penalize subgoals that are too far from the current state
            feasibility_penalty = torch.where(
                distance_feasibility > self.max_subgoal_distance,
                -1e6 * torch.ones_like(distance_feasibility),  # Use large negative finite value
                torch.zeros_like(distance_feasibility)
            )

            # Compute goal achievement reward with scaling factors
            goal_reward = -self.subgoal_scale * distance_subgoal ** 2 \
                          - self.global_goal_scale * distance_global_goal ** 2 \
                          + feasibility_penalty
            goal_achievements += goal_reward

            # Logging for debugging
            if self.logger is not None:
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} distances mean: {distance_subgoal.mean().item():.4f}")
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} global goal distances mean: {distance_global_goal.mean().item():.4f}")
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} feasibility penalties mean: {feasibility_penalty.mean().item():.4f}")
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} goal rewards mean: {goal_reward.mean().item():.4f}")

        return goal_achievements

    def _update_distributions(self, actions, goals, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)

        # Update action distribution
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean = best_actions.mean(dim=1, keepdim=True)
        action_std_dev = best_actions.std(dim=1, unbiased=False, keepdim=True)

        # Update goal distribution
        best_goals = goals[:, topk.view(-1)].reshape(
            goals.size(0), self.top_candidates, -1
        )
        goal_mean_new = best_goals.mean(dim=1, keepdim=True)
        goal_std_dev_new = best_goals.std(dim=1, unbiased=False, keepdim=True)

        # Blend the new goal mean with the global goal state
        goal_mean = self.goal_mean_weight * goal_mean_new + \
                    (1 - self.goal_mean_weight) * self.global_goal_state.unsqueeze(0).unsqueeze(1)

        goal_std_dev = goal_std_dev_new

        return action_mean, action_std_dev, goal_mean, goal_std_dev

    def return_stats(self):
        if self.use_reward:
            reward_stats = self._create_stats(self.trial_rewards)
        else:
            reward_stats = {}
        if self.use_exploration:
            info_stats = self._create_stats(self.trial_bonuses)
        else:
            info_stats = {}
        self.trial_rewards = []
        self.trial_bonuses = []
        return reward_stats, info_stats

    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)
        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }
