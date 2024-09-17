# Begin file: pmbrl/control/planner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random

class Planner(nn.Module):
    def __init__(
        self,
        env,
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
        # Additional parameters
        subgoal_scale=1.0,
        global_goal_scale=1.0,
        logger=None,  # Logger for debugging
        action_low=None,  # Lower bounds of action space
        action_high=None,  # Upper bounds of action space
    ):
        super().__init__()
        self.env = env
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

        # Additional parameters
        self.subgoal_scale = subgoal_scale
        self.global_goal_scale = global_goal_scale
        self.logger = logger  # Pass logger for debugging

        # Action bounds
        if self.env.action_space is None:
            raise ValueError("Action bounds (action_low and action_high) must be provided.")
        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32).to(device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32).to(device)

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

        # Initialize action distributions
        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(self.device)
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(self.device)

        # Generate feasible subgoals using the dynamics model
        num_subgoals = (self.plan_horizon + self.context_length - 1) // self.context_length
        goal_size = state_size

        # Generate subgoals by simulating future states
        goal_mean = self.generate_feasible_subgoals(state, num_subgoals)

        for iter in range(self.optimisation_iters):
            # Sample action candidates
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            # Clamp actions to respect environment bounds
            actions = self.clamp_actions(actions)

            # Perform rollouts with sampled actions
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
                goal_achievements = self.compute_goal_achievement(states, goal_mean)
                returns += goal_achievements * self.goal_achievement_scale

            # Update action distributions using top candidates
            action_mean, action_std_dev = self._update_action_distribution(
                actions, returns
            )

        # Extract the first action to return
        selected_action = action_mean[0].squeeze(dim=0)  # Shape: (action_size,)

        # For visualization, return the first subgoal
        selected_goal = goal_mean[0]  # Shape: (goal_size,)

        return selected_action, selected_goal

    def clamp_actions(self, actions):
        """
        Clamp the action sequences to respect the environment's action bounds.
        """
        # Expand action_low and action_high to match the actions' shape
        # actions: (plan_horizon, n_candidates, action_size)
        # action_low and action_high: (action_size,)
        # Need to reshape to (1, 1, action_size) for broadcasting
        action_low = self.action_low.view(1, 1, -1)
        action_high = self.action_high.view(1, 1, -1)
        return torch.clamp(actions, min=action_low, max=action_high)

    def generate_feasible_subgoals(self, current_state, num_subgoals):
        """
        Generate feasible subgoals by simulating future states using the dynamics model.
        """
        # Initialize
        n_subgoal_candidates = 100  # Number of action sequences to sample for subgoal generation
        current_state = current_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, state_size)
        current_state = current_state.repeat(self.ensemble_size, n_subgoal_candidates, 1)  # Shape: (ensemble_size, n_subgoal_candidates, state_size)
        subgoals = []

        # Sample random action sequences
        total_steps = num_subgoals * self.context_length

        action_sequences = torch.randn(
            total_steps,
            n_subgoal_candidates,
            self.action_size,
            device=self.device,
        )  # Shape: (total_steps, n_subgoal_candidates, action_size)
        # Clamp actions using the environment's action bounds
        action_sequences = self.clamp_actions(action_sequences)

        # Perform rollout to get predicted states
        states = [current_state]  # List to hold states at each timestep
        for t in range(total_steps):
            actions_t = action_sequences[t].unsqueeze(0).repeat(self.ensemble_size, 1, 1)  # Shape: (ensemble_size, n_subgoal_candidates, action_size)
            delta_mean, delta_var = self.ensemble(states[-1], actions_t)
            next_state = states[-1] + delta_mean  # Use mean prediction
            states.append(next_state)

        # Collect possible subgoals
        for i in range(1, num_subgoals + 1):
            idx = i * self.context_length
            if idx >= len(states):
                idx = len(states) - 1  # Ensure idx is within bounds
            predicted_state = states[idx].mean(dim=0)  # Mean over ensemble members: (n_subgoal_candidates, state_size)
            # Select the subgoal that is closest to the global goal state
            distances_to_goal = torch.norm(predicted_state - self.global_goal_state.unsqueeze(0), dim=1)  # (n_subgoal_candidates,)
            min_distance_idx = torch.argmin(distances_to_goal)
            subgoal = predicted_state[min_distance_idx]
            subgoals.append(subgoal)

        goal_mean = torch.stack(subgoals)  # Shape: (num_subgoals, state_size)
        return goal_mean

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * (T - 1)
        delta_vars = [torch.empty(0)] * (T - 1)

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)  # Shape: (1,1,state_size)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)  # Shape: (ensemble_size, n_candidates, state_size)
        states[0] = current_state

        actions = actions.unsqueeze(0)  # Shape: (1, plan_horizon, n_candidates, action_size)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)  # Shape: (plan_horizon, ensemble_size, n_candidates, action_size)

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
        states = torch.stack(states[1:], dim=0)  # Exclude initial state: (plan_horizon, ensemble_size, n_candidates, state_size)
        delta_vars = torch.stack(delta_vars, dim=0)  # (plan_horizon, ensemble_size, n_candidates, state_size)
        delta_means = torch.stack(delta_means, dim=0)  # (plan_horizon, ensemble_size, n_candidates, state_size)

        return states, delta_vars, delta_means

    def compute_goal_achievement(self, states, goals):
        goal_achievements = torch.zeros(self.n_candidates).float().to(self.device)
        c = self.context_length
        num_subgoals = goals.size(0)

        for i in range(num_subgoals):
            t = i * c + c - 1  # Last timestep in the context
            if t >= self.plan_horizon:
                t = self.plan_horizon - 1  # Ensure t is within bounds
            predicted_state = states[t]  # (ensemble_size, n_candidates, state_size)
            predicted_state = predicted_state.mean(dim=0)  # (n_candidates, state_size)
            desired_state = goals[i].unsqueeze(0).repeat(self.n_candidates, 1)  # (n_candidates, state_size)

            # Distance between predicted state and subgoal
            diff_subgoal = desired_state - predicted_state  # (n_candidates, state_size)
            distance_subgoal = torch.norm(diff_subgoal, dim=1)  # (n_candidates,)

            # Compute goal achievement reward with scaling factor
            goal_reward = -self.subgoal_scale * distance_subgoal  # Higher reward for closer distances

            goal_achievements += goal_reward

            # Logging for debugging
            if self.logger is not None:
                subgoal_values = desired_state.mean(dim=0).cpu().numpy()
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} values: {subgoal_values}")
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} distances mean: {distance_subgoal.mean().item():.4f}")
                self.logger.log(f"Subgoal {i+1}/{num_subgoals} goal rewards mean: {goal_reward.mean().item():.4f}")
                self.logger.log("=========================================================================")

        return goal_achievements

    def _update_action_distribution(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)

        # Update action distribution
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean = best_actions.mean(dim=1, keepdim=True)
        action_std_dev = best_actions.std(dim=1, unbiased=False, keepdim=True)

        return action_mean, action_std_dev

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
# End file: pmbrl/control/planner.py
