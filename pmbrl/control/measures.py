# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np
from scipy.special import psi, gamma


class Random(object):
    def __init__(self, model, scale=1.0):
        self.model = model
        self.scale = scale

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """

        n_candidates = delta_means.size(2)
        randoms = torch.randn(n_candidates).float().to(delta_means.device)

        return randoms


class Variance(object):
    def __init__(self, model, scale=1.0):
        self.model = model
        self.scale = scale

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """
        plan_horizon = delta_means.size(0)
        n_candidates = delta_means.size(2)
        delta_means = self.model.normalizer.renormalize_state_delta_means(delta_means)
        variance = (
            torch.zeros(plan_horizon, n_candidates).float().to(delta_means.device)
        )
        for t in range(plan_horizon):
            variance[t, :] = self.get_variance(delta_vars[t])

        variance = variance * self.scale
        return variance.sum(dim=0)

    def get_variance(self, delta_vars):
        """ensemble_size, candidates, n_dim"""
        variance = delta_vars.sum(dim=0).sum(dim=-1)
        return variance


class Disagreement(object):
    def __init__(self, model, scale=1.0):
        self.model = model
        self.scale = scale

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """
        plan_horizon = delta_means.size(0)
        n_candidates = delta_means.size(2)
        delta_means = self.model.normalizer.renormalize_state_delta_means(delta_means)
        disagreements = (
            torch.zeros(plan_horizon, n_candidates).float().to(delta_means.device)
        )
        for t in range(plan_horizon):
            disagreements[t, :] = self.get_disagreement(delta_means[t])

        disagreements = disagreements * self.scale
        return disagreements.sum(dim=0)

    def get_disagreement(self, delta_means):
        """ensemble_size, candidates, n_dim"""
        disagreement = delta_means.std(dim=0).sum(dim=-1)
        return disagreement


class InformationGain(object):
    def __init__(self, model, scale=1.0):
        self.model = model
        self.scale = scale

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """

        plan_horizon = delta_means.size(0)
        n_candidates = delta_means.size(2)

        delta_means = self.model.normalizer.renormalize_state_delta_means(delta_means)
        delta_vars = self.model.normalizer.renormalize_state_delta_vars(delta_vars)
        delta_states = self.model.sample(delta_means, delta_vars)
        info_gains = (
            torch.zeros(plan_horizon, n_candidates).float().to(delta_means.device)
        )

        for t in range(plan_horizon):
            """
                Detailed explaination of Information Gain and why our environment entropy is modeled as an ensemble model.
                https://chatgpt.com/share/66e8825f-7480-8001-b910-50fa17aa73da
                
                # TODO: For next paper, what'a a better world model to use then an ensemble model?
                How to ensure that the model we use is as true as possible to true dynamics of the environment?
                Is there a better way to represent the true dynamics of the environment?
                
                
                In practice, especially in model-based reinforcement learning and active inference frameworks, 
                the ensemble model serves as an approximation of the true dynamics of the environment. 
                
                Model Approximation:
                The ensemble represents multiple hypotheses about the environment's dynamics. 
                By aggregating these predictions, entropy_of_average captures the uncertainty modeled by the ensemble, serving as a stand-in for the true entropy H(Y).

                Ensemble as Proxy for True Distribution:
                In scenarios where the true distribution is unknown or intractable, an ensemble model provides a practical way to estimate 
                H(Y). The diversity within the ensemble captures model uncertainty, which is crucial for effective exploration.

                Y: Future state of the environment.
                X: Ensemble model's predictions based on a candidate action sequence.
                H(Y): Entropy of Y (uncertainty about without any knowledge of X).
                H(Y∣X): Conditional entropy of Y given X (remaining uncertainty about Y after knowing X).
                I(Y; X): Mutual information between Y and X (reduction in uncertainty about Y after knowing X).

            """
            ent_avg = self.entropy_of_average(delta_states[t])  # Estimates H(Y) - Entropy of the -- supposed true, approximated using model itself-- dynamics of the environment
            avg_ent = self.average_of_entropy(delta_vars[t])  # Estimates H(Y | X)
            info_gains[t, :] = ent_avg - avg_ent  # Computes I(Y; X)

        info_gains = info_gains * self.scale
        return info_gains.sum(dim=0)

    def entropy_of_average(self, samples):
        """
        samples (ensemble_size, n_candidates, n_dim)
        """
        samples = samples.permute(1, 0, 2)
        n_samples = samples.size(1)
        dims = samples.size(2)
        k = 3

        distances_yy = self.batched_cdist_l2(samples, samples)
        y, _ = torch.sort(distances_yy, dim=1)
        v = self.volume_of_the_unit_ball(dims)
        h = (
            np.log(n_samples - 1)
            - psi(k)
            + np.log(v)
            + dims * torch.sum(torch.log(y[:, k - 1]), dim=1) / n_samples
            + 0.5
        )
        return h

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = (
            torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2)
            .add_(x1_norm)
            .clamp_min_(1e-30)
            .sqrt_()
        )
        return res

    def volume_of_the_unit_ball(self, dim):
        return np.pi ** (dim / 2) / gamma(dim / 2 + 1)

    def average_of_entropy(self, delta_vars):
        return torch.mean(self.gaussian_diagonal_entropy(delta_vars), dim=0)

    def gaussian_diagonal_entropy(self, delta_vars):
        min_variance = 1e-8
        return 0.5 * torch.sum(
            torch.log(2 * np.pi * np.e * torch.clamp(delta_vars, min=min_variance)),
            dim=len(delta_vars.size()) - 1,
        )

