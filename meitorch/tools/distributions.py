import torch
from torch import nn


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_components, input_dim, fixed_variance=False):
        super(GaussianMixtureModel, self).__init__()

        self.num_components = num_components
        self.input_dim = input_dim

        # Parameters for the Gaussian mixture components
        self.weights = nn.Parameter(torch.randn(num_components))
        self.means = nn.Parameter(torch.randn(num_components, input_dim))
        if fixed_variance:
            self.variances = nn.Parameter(torch.ones(num_components, input_dim)*fixed_variance)
        else:
            self.variances = nn.Parameter(torch.rand(num_components, input_dim))

    def forward(self, x):
        # Compute the Gaussian component probabilities
        log_probs = []
        for i in range(self.num_components):
            weight = self.weights[i]
            mean = self.means[i]
            var = self.variances[i]
            log_prob = torch.log(weight) + self._log_gaussian(x, mean, var)
            log_probs.append(log_prob)

        # Calculate the log-sum-exp of log probabilities
        log_probs = torch.stack(log_probs, dim=1)
        log_sum_exp = torch.logsumexp(log_probs, dim=1)

        return log_sum_exp

    @staticmethod
    def _log_gaussian(x, mean, var):
        # Calculate the log likelihood of a Gaussian distribution
        return -0.5 * torch.sum(((x - mean) / var)**2, dim=1) - 0.5 * torch.sum(torch.log(var))
