import torch
import torch.nn as nn

from pmbrl.control.measures import InformationGain, Disagreement, Variance, Random
# from pmbrl.models import GoalModel  # Import the GoalModel

class Planner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        goal_model,  # Added goal_model
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
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.goal_model = goal_model  # Store the goal model
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
        
        # Predict the goal using the goal model
        if self.use_high_level:
            goal = self.goal_model(state)
            self.current_goal = goal  # Store the current goal
        else:
            goal = torch.zeros(state_size).to(self.device)
            self.current_goal = goal

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            # Perform rollout with goal transition logic
            states, delta_vars, delta_means, goals = self.perform_rollout(state, actions)

            returns = torch.zeros(self.n_candidates).float().to(self.device)
            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale
                returns += expl_bonus
                self.trial_bonuses.append(expl_bonus)

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

            # Add goal-achievement reward if using high-level planner
            if self.use_high_level:
                goal_achievements = self.compute_goal_achievement(states, goals)
                returns += goal_achievements * self.goal_achievement_scale

            action_mean, action_std_dev = self._fit_gaussian(actions, returns)

        # Store the current goal for use in the agent
        if self.use_high_level:
            self.current_goal = goals[0]  # Store the first goal for the current state
        else:
            self.current_goal = torch.zeros(state_size).to(self.device)

        return action_mean[0].squeeze(dim=0)

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T
        goals = [torch.empty(0)] * (T - 1)  # List to store goals for each time step

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0)
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1)
        states[0] = current_state

        actions = actions.unsqueeze(0)
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3)

        c = self.context_length

        # Initialize the goal at the beginning
        if self.use_high_level:
            with torch.no_grad():
                goal_mu, goal_logvar = self.goal_model(
                    current_state.mean(dim=0).mean(dim=0)
                )
                goal_std = torch.exp(0.5 * goal_logvar)
                goal_eps = torch.randn_like(goal_std)
                goal = goal_mu + goal_eps * goal_std
        else:
            goal = torch.zeros_like(current_state[0, 0])

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t])
            if self.use_mean:
                next_state = states[t] + delta_mean
            else:
                next_state = states[t] + self.ensemble.sample(delta_mean, delta_var)
            states[t + 1] = next_state
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

            if self.use_high_level:
                if t % c == 0 and t != 0:
                    # At context boundary, sample a new goal from the goal model
                    with torch.no_grad():
                        goal_mu, goal_logvar = self.goal_model(
                            states[t].mean(dim=0).mean(dim=0)
                        )
                        goal_std = torch.exp(0.5 * goal_logvar)
                        goal_eps = torch.randn_like(goal_std)
                        goal = goal_mu + goal_eps * goal_std
                else:
                    # Use the goal transition function for intra-context goal generation
                    goal = self.goal_transition(states[t], goal, states[t + 1])

            goals[t] = goal  # Store the goal at time t

        # Stack the lists into tensors
        states = torch.stack(states[1:], dim=0)  # Exclude initial state
        delta_vars = torch.stack(delta_vars[1:], dim=0)
        delta_means = torch.stack(delta_means[1:], dim=0)
        goals = torch.stack(goals, dim=0)  # Shape: (plan_horizon, state_size)

        return states, delta_vars, delta_means, goals

    def goal_transition(self, state, current_goal, next_state):
        """
        Goal transition function:
        next_goal = state + current_goal - next_state
        """
        next_goal = state + current_goal - next_state
        return next_goal

    def compute_goal_achievement(self, states, goals):
        goal_achievements = torch.zeros(self.n_candidates).float().to(self.device)

        c = self.context_length

        for t in range(self.plan_horizon):
            if t % c == c - 1:
                # Compute goal achievement at the end of each context
                predicted_state = states[t]  # (ensemble_size, n_candidates, state_size)
                predicted_state = predicted_state.mean(dim=0)  # (n_candidates, state_size)
                desired_state = states[0][0] + goals[t]  # Initial state + predicted goal
                diff = desired_state - predicted_state  # (n_candidates, state_size)
                distances = torch.norm(diff, dim=1)  # (n_candidates,)
                # Negative squared distance as reward
                goal_rewards = -distances ** 2
                goal_achievements += goal_rewards

        return goal_achievements


    def _fit_gaussian(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )
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
