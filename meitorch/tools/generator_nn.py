from torch import nn
import numpy as np
import torch


class GenerativeMLPNet(nn.Module):
    """
    Parametric multilayer perceptron network

    :param input_size: int, the size of the input
    :param hidden_sizes: list of int, the sizes of the hidden layers
    :param output_size: int, the size of the output
    :param residual: bool, whether to use residual connections
    :param activation: torch.nn.Module, the activation function to use
    """

    def __init__(self, hidden_sizes, output_shape, trainable_input=False,
                 activation=nn.ReLU(), activate_output=True, distribution_base='normal', fixed_stddev=0.4):
        super(GenerativeMLPNet, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.output_shape = output_shape
        self.input = nn.Parameter(torch.randn(hidden_sizes[0]),
                                  requires_grad=trainable_input)

        layers = []
        output_shape = 2*np.prod(output_shape) if fixed_stddev else 2*hidden_sizes[0]
        sizes = hidden_sizes + [output_shape]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or activate_output:
                layers.append(self.activation)
        self.mlp_layers = nn.Sequential(*layers)

        if distribution_base == 'normal':
            self.dist = torch.distributions.Normal
        elif distribution_base == 'laplace':
            self.dist = torch.distributions.Laplace
        else:
            raise ValueError('Unknown distribution base: {}'.format(distribution_base))

    def forward(self, batch_size, use_mean=False):
        batch = torch.tile(self.input, (batch_size, 1))
        output = self.mlp_layers(batch)
        if self.fixed_stddev:
            y_mu = output
            y_sigma = torch.ones_like(output) * self.fixed_stddev
        else:
            y_mu, y_sigma = output.reshape(batch_size, 2, -1).chunk(2, dim=1)
        output_distribution = self.dist(y_mu, torch.exp(y_sigma))
        samples = output_distribution.sample() if not use_mean else output_distribution.mean
        return samples.reshape(batch_size, *self.output_shape)


class GenerativeConvNet(nn.Module):
    """
    Parametric convolutional network
    based on Efficient-VDVAE paper

    :param n_layers: int, the number of convolutional layers
    :param in_filters: int, the number of input filters
    :param bottleneck_ratio: float, the ratio of bottleneck filters to input filters
    :param kernel_size: int or tuple of int, the size of the convolutional kernel
    :param init_scaler: float, the scaler for the initial weights_imagenet
    :param residual: bool, whether to use residual connections
    :param use_1x1: bool, whether to use 1x1 convolutions
    :param pool_strides: int or tuple of int, the strides for the pooling layers
    :param unpool_strides: int or tuple of int, the strides for the unpooling layers
    :param output_ratio: float, the ratio of output filters to input filters
    :param activation: torch.nn.Module, the activation function to use
    """

    def __init__(self, hidden_sizes, shape, kernel_size, activation=nn.SiLU(), activate_output=False,
                 trainable_input=False, distribution_base='normal', fixed_stddev=0.4):
        super(GenerativeConvNet, self).__init__()
        self.shape = shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.input = nn.Parameter(torch.randn(shape),
                                  requires_grad=trainable_input)
        self.fixed_stddev = fixed_stddev

        layers = []
        output_channels = shape[0] if fixed_stddev else 2*shape[0]
        sizes = hidden_sizes + [output_channels]
        for i in range(len(sizes) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=sizes[i], out_channels=sizes[i + 1],
                    kernel_size=kernel_size, padding='same'))
            if i < len(sizes) - 2 or activate_output:
                layers.append(self.activation)
        self.conv = nn.Sequential(*layers)

        if distribution_base == 'normal':
            self.dist = torch.distributions.Normal
        elif distribution_base == 'laplace':
            self.dist = torch.distributions.Laplace
        else:
            raise ValueError(f'Unknown distribution base: {distribution_base}')

    def forward(self, batch_size, use_mean=False):
        batch = torch.tile(self.input, (batch_size, 1, 1, 1))
        output = self.conv(batch)
        if self.fixed_stddev:
            y_mu = output
            y_sigma = torch.ones_like(output) * self.fixed_stddev
        else:
            y_mu, y_sigma = output.reshape(batch_size, 2, -1).chunk(2, dim=1)
        output_distribution = self.dist(y_mu, torch.exp(y_sigma))
        samples = output_distribution.rsample() if not use_mean else output_distribution.mean
        return samples.reshape(batch_size, *self.shape)


