# Begin file: pmbrl/control/planner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
        # subgoal_scale=1.0,
        global_goal_scale=1.0,
        logger=None,  # Logger for debugging
        action_low=None,  # Lower bounds of action space
        action_high=None,  # Upper bounds of action space
        # New flags
        use_high_level_reward=True,
        use_high_level_exploration=True,
    ):
        super().__init__()
        self.env = env
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        
        # self.use_goal_achievement = use_goal_achievement 

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
        # self.subgoal_scale = subgoal_scale
        self.global_goal_scale = global_goal_scale
        self.logger = logger  # Pass logger for debugging

        # Action bounds
        if self.env.action_space is None:
            raise ValueError("Action bounds (action_low and action_high) must be provided.")
        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32).to(device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32).to(device)

        # High-level planner flags
        self.use_high_level_reward = use_high_level_reward
        self.use_high_level_exploration = use_high_level_exploration

        if strategy == "information":
            self.measure = InformationGain(self.ensemble, scale=expl_scale)
        elif strategy == "variance":
            self.measure = Variance(self.ensemble, scale=expl_scale)
        elif strategy == "random":
            self.measure = Random(self.ensemble, scale=expl_scale)
        elif strategy == "none":
            self.use_exploration = False
        
        # Action sequence bonuses (low-level)
        self.trial_rewards = []
        self.trial_bonuses = []
        self.trial_goal_achievements = []
        # Goal bonuses
        self.trial_goal_rewards = []
        self.trial_goal_bonuses = []
        
        self.logger.log(f'exploration scale: {self.expl_scale}')
        self.logger.log(f'reward scale: {self.reward_scale}')
        # self.logger.log(f'subgoal scale: {self.subgoal_scale}')
        self.logger.log(f'global goal scale: {self.global_goal_scale}')
        self.logger.log(f'Goal achievement scale: {self.goal_achievement_scale}')
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
        goal_mean = state.unsqueeze(0).repeat(self.ensemble_size, 1, 1)  # Shape: (ensemble_size, 1, state_size)
        if self.use_high_level:
            # Generate subgoals by simulating future states
            goal_mean = self.generate_feasible_subgoals(state, num_subgoals)

        # Select the current subgoal (e.g., the first one)
        current_subgoal = goal_mean[0]  # Shape: (state_size,)

        for iter in range(self.optimisation_iters):
            # Sample action candidates
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            # Clamp actions to respect environment bounds
            # actions = self.clamp_actions(actions)

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

            if self.use_high_level:
                goal_achievement = self.compute_goal_achievement(states, current_subgoal)* self.goal_achievement_scale
                returns += goal_achievement
                self.trial_goal_achievements.append(goal_achievement)
            # else:
            #     # Add subgoal distance penalty
            #     subgoal_penalty = self.compute_subgoal_distance_penalty(states, current_subgoal)
            #     returns += subgoal_penalty

            # Update action distributions using top candidates
            action_mean, action_std_dev = self._update_action_distribution(
                actions, returns
            )

        # Extract the first action to return
        selected_action = action_mean[0].squeeze(dim=0)  # Shape: (action_size,)

        # For visualization, return the current subgoal
        selected_goal = current_subgoal  # Shape: (goal_size,)

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
        n_subgoal_candidates = 1000  # Number of action sequences to sample for subgoal generation
        current_state = current_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, state_size)
        current_state = current_state.repeat(self.ensemble_size, n_subgoal_candidates, 1)  # Shape: (ensemble_size, n_subgoal_candidates, state_size)

        # Initialize action distributions for subgoal optimization
        goal_action_mean = torch.zeros(self.context_length, n_subgoal_candidates, self.action_size).to(self.device)
        goal_action_std_dev = torch.ones(self.context_length, n_subgoal_candidates, self.action_size).to(self.device)

        for iter in range(self.optimisation_iters):
            # Sample action sequences for subgoal candidates
            goal_actions = goal_action_mean + goal_action_std_dev * torch.randn(
                self.context_length,
                n_subgoal_candidates,
                self.action_size,
                device=self.device,
            )
            # Clamp actions
            # goal_actions = self.clamp_actions(goal_actions)

            # Perform rollouts to get predicted subgoal states
            goal_states, goal_delta_vars, goal_delta_means = self.perform_subgoal_rollout(
                current_state, goal_actions
            )

            # Compute returns for subgoal candidates
            goal_returns = torch.zeros(n_subgoal_candidates).float().to(self.device)

            # if self.use_high_level_reward:
            #     # Compute rewards for subgoal candidates
            #     _states = goal_states.view(-1, current_state.size(-1))
            #     _actions = goal_actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
            #     _actions = _actions.view(-1, self.action_size)
            #     rewards = self.reward_model(_states, _actions)
            #     rewards = rewards * self.reward_scale
            #     rewards = rewards.view(
            #         self.context_length, self.ensemble_size, n_subgoal_candidates
            #     )
            #     rewards = rewards.mean(dim=1).sum(dim=0)
            #     goal_returns += rewards
            
            # In generate_feasible_subgoals method
            if self.use_high_level_reward:
                # Compute rewards for subgoal candidates
                _states = goal_states.view(-1, current_state.size(-1))
                _actions = goal_actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _actions = _actions.view(-1, self.action_size)
                rewards = self.reward_model(_states, _actions)
                rewards = rewards * self.reward_scale  # Use self.reward_scale
                rewards = rewards.view(
                    self.context_length, self.ensemble_size, n_subgoal_candidates
                )
                rewards = rewards.mean(dim=1).sum(dim=0)
                goal_returns += rewards
                self.trial_goal_rewards.append(rewards)  # Collect stats


            if self.use_high_level_exploration:
                # Compute exploration bonuses for subgoal candidates
                expl_bonus = self.measure(goal_delta_means, goal_delta_vars) * self.expl_scale
                goal_returns += expl_bonus
                self.trial_goal_bonuses.append(expl_bonus)  # Collect stats
                

            # Include distance to global goal (encourage subgoals closer to global goal)
            final_goal_states = goal_states[-1].mean(dim=0)  # Shape: (n_subgoal_candidates, state_size)
            distances_to_global_goal = torch.norm(final_goal_states - self.global_goal_state.unsqueeze(0), dim=1)
            goal_returns -= self.global_goal_scale * distances_to_global_goal

            # Update goal action distributions using top candidates
            goal_action_mean, goal_action_std_dev = self._update_goal_action_distribution(
                goal_actions, goal_returns
            )

        # After optimization, select the best action sequence based on the highest return
        best_return, best_index = goal_returns.max(dim=0)
        best_goal_actions = goal_actions[:, best_index, :].unsqueeze(1)  # Shape: (context_length, 1, action_size)
        current_state_best = current_state[:, best_index, :].unsqueeze(1)  # Shape: (ensemble_size, 1, state_size)

        # Perform rollout with the best actions to get subgoal state
        goal_states, _, _ = self.perform_subgoal_rollout(current_state_best, best_goal_actions)
        subgoal_state = goal_states[-1].mean(dim=0).squeeze(0)  # Shape: (state_size,)
        goal_mean = subgoal_state.unsqueeze(0)  # Shape: (1, state_size)

        # Repeat the subgoal for the number of subgoals needed
        goal_mean = goal_mean.repeat(num_subgoals, 1)  # Shape: (num_subgoals, state_size)

        return goal_mean

    def perform_subgoal_rollout(self, current_state, actions):
        """
        Perform rollout to predict future states for subgoal candidates.

        Args:
            current_state (torch.Tensor): The current state tensor, shape (ensemble_size, n_candidates, state_size)
            actions (torch.Tensor): Actions for subgoal candidates, shape (context_length, n_candidates, action_size)

        Returns:
            Tuple of predicted states, delta_vars, delta_means.
        """
        T = actions.size(0) + 1
        n_candidates = actions.size(1)
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * (T - 1)
        delta_vars = [torch.empty(0)] * (T - 1)

        states[0] = current_state  # Shape: (ensemble_size, n_candidates, state_size)

        # Repeat actions for each ensemble member
        actions = actions.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1)  # Shape: (context_length, ensemble_size, n_candidates, action_size)
        actions = actions.permute(1, 0, 2, 3)  # Shape: (ensemble_size, context_length, n_candidates, action_size)

        for t in range(T - 1):
            action_t = actions[:, t, :, :]  # Shape: (ensemble_size, n_candidates, action_size)
            delta_mean, delta_var = self.ensemble(states[t], action_t)
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
    
    # def compute_subgoal_distance_penalty(self, states, current_subgoal):
    #     """
    #     Compute a penalty based on the expected distance to the current subgoal at the end of the planning horizon.

    #     Args:
    #         states (torch.Tensor): Predicted states from rollouts, shape (plan_horizon, ensemble_size, n_candidates, state_size)
    #         current_subgoal (torch.Tensor): The current subgoal, shape (state_size,)

    #     Returns:
    #         torch.Tensor: Subgoal distance penalties for each candidate, shape (n_candidates,)
    #     """
    #     # Use the last predicted state
    #     predicted_state = states[-1]  # Shape: (ensemble_size, n_candidates, state_size)
    #     predicted_state = predicted_state.mean(dim=0)  # Mean over ensemble: (n_candidates, state_size)
    #     desired_state = current_subgoal.unsqueeze(0).repeat(self.n_candidates, 1)  # (n_candidates, state_size)

    #     # Compute distance between predicted state and current subgoal
    #     diff_subgoal = desired_state - predicted_state  # (n_candidates, state_size)
    #     distance_subgoal = torch.norm(diff_subgoal, dim=1)  # (n_candidates,)

    #     # Compute subgoal distance penalty (negative reward)
    #     subgoal_penalty = -self.subgoal_scale * distance_subgoal

    #     return subgoal_penalty


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
    
    def compute_goal_achievement(self, states, current_subgoal):
        """
        compute_goal_achievement function to consider only the final state rather than summing over the entire planning horizon. 
        This is more appropriate for environments where reaching and maintaining a goal state is important.

        Args:
            states (_type_): _description_
            current_subgoal (_type_): _description_

        Returns:
            _type_: _description_
        """
        predicted_state = states[-1]  # Last predicted state
        predicted_state = predicted_state.mean(dim=0)  # Mean over ensemble
        desired_state = current_subgoal.unsqueeze(0).repeat(self.n_candidates, 1)
        diff_subgoal = desired_state - predicted_state
        distance_subgoal = torch.norm(diff_subgoal, dim=1)
        goal_achievement = -distance_subgoal
        return goal_achievement


    # def compute_goal_achievement(self, states, current_subgoal):
    #     """
    #     Compute the goal achievement reward based on the distance to the subgoal at each timestep.

    #     Args:
    #         states (torch.Tensor): Predicted states from rollouts, shape (plan_horizon, ensemble_size, n_candidates, state_size)
    #         current_subgoal (torch.Tensor): The current subgoal, shape (state_size,)

    #     Returns:
    #         torch.Tensor: Goal achievement rewards for each candidate, shape (n_candidates,)
    #     """
    #     goal_achievements = torch.zeros(self.n_candidates).float().to(self.device)
    #     for t in range(self.plan_horizon):
    #         predicted_state = states[t]  # (ensemble_size, n_candidates, state_size)
    #         predicted_state = predicted_state.mean(dim=0)  # (n_candidates, state_size)
    #         desired_state = current_subgoal.unsqueeze(0).repeat(self.n_candidates, 1)  # (n_candidates, state_size)
    #         diff_subgoal = desired_state - predicted_state  # (n_candidates, state_size)
    #         distance_subgoal = torch.norm(diff_subgoal, dim=1)  # (n_candidates,)

    #         # Compute goal achievement reward
    #         goal_reward = -distance_subgoal
    #         goal_achievements += goal_reward  # Accumulate over time steps

    #     return goal_achievements


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

    def _update_goal_action_distribution(self, actions, returns):
        """
        Update the action distribution for subgoal selection using top candidates.

        Args:
            actions (torch.Tensor): Actions sampled for subgoal candidates, shape (context_length, n_subgoal_candidates, action_size)
            returns (torch.Tensor): Returns computed for subgoal candidates, shape (n_subgoal_candidates,)

        Returns:
            Tuple of updated action mean and std dev.
        """
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)

        # Update action distribution
        best_actions = actions[:, topk.view(-1)].reshape(
            self.context_length, self.top_candidates, self.action_size
        )
        action_mean = best_actions.mean(dim=1, keepdim=True)
        action_std_dev = best_actions.std(dim=1, unbiased=False, keepdim=True)

        return action_mean, action_std_dev

    # def return_stats(self):
    #     if self.use_reward:
    #         reward_stats = self._create_stats(self.trial_rewards)
    #     else:
    #         reward_stats = {}
    #     if self.use_exploration:
    #         info_stats = self._create_stats(self.trial_bonuses)
    #     else:
    #         info_stats = {}
    #     self.trial_rewards = []
    #     self.trial_bonuses = []
    #     return reward_stats, info_stats
    
    def return_stats(self):
        stats = {}
        if self.use_reward:
            reward_stats = self._create_stats(self.trial_rewards)
            stats['reward'] = reward_stats
        if self.use_exploration:
            info_stats = self._create_stats(self.trial_bonuses)
            stats['exploration'] = info_stats
        if self.trial_goal_achievements:
            goal_achievement_stats = self._create_stats(self.trial_goal_achievements)
            stats['goal_achievement'] = goal_achievement_stats
        if self.trial_goal_rewards:
            goal_reward_stats = self._create_stats(self.trial_goal_rewards)
            stats['goal_reward'] = goal_reward_stats
        if self.trial_goal_bonuses:
            goal_exploration_stats = self._create_stats(self.trial_goal_bonuses)
            stats['goal_exploration'] = goal_exploration_stats
        # Reset stats
        self.trial_rewards = []
        self.trial_bonuses = []
        self.trial_goal_achievements = []
        self.trial_goal_rewards = []
        self.trial_goal_bonuses = []
        return stats



    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)
        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }
