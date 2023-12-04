from torch import nn
import numpy as np
import torch


class GenerativeMLPNet(nn.Module):
    """
    Parametric multilayer perceptron network

    :param hidden_sizes: list of int, the sizes of the hidden layers
    :param output_shape: int, the size of the output
    :param trainable_input: bool, whether the input of the transformation should be trainable
    :param activation: torch.nn.Module, the activation function to use
    :param activate_output: bool, whether to apply the activation function to the output
    :param distribution_base: str, the base distribution to use (normal or laplace)
    :param fixed_stddev: float, the fixed standard deviation to use for the output, False if it should be learned
    """

    def __init__(self, hidden_sizes, output_shape, trainable_input=False,
                 activation=nn.ReLU(), activate_output=True, distribution_base='normal', fixed_stddev=0.4):
        super(GenerativeMLPNet, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.output_shape = output_shape
        self.fixed_stddev = fixed_stddev
        self.input = nn.Parameter(torch.randn(hidden_sizes[0]),
                                  requires_grad=trainable_input)

        layers = []
        output_shape = np.prod(output_shape) if fixed_stddev else 2*np.prod(output_shape)
        sizes = hidden_sizes + [output_shape]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or activate_output:
                layers.append(self.activation)
        self.mlp_layers = nn.Sequential(*layers)
        
        self.distribution = distribution_base
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
    
    def __str__(self):
        return f" mlp net | hidden_sizes: {self.hidden_sizes} | " \
               f"distribution: {self.distribution} | fixed_stddev: {self.fixed_stddev} "


class GenerativeConvNet(nn.Module):
    """
    Parametric convolutional network

    :param hidden_sizes: list of int, the sizes of the hidden layers
    :param shape: tuple of int, the shape of the output
    :param kernel_size: int, the size of the convolutional kernel
    :param activation: torch.nn.Module, the activation function to use
    :param activate_output: bool, whether to apply the activation function to the output
    :param distribution_base: str, the base distribution to use (normal or laplace)
    :param fixed_stddev: float, the fixed standard deviation to use for the output, False if it should be learned
    :param use_mean: bool, whether to use the mean of the distribution instead of sampling
    """

    def __init__(self, hidden_sizes, shape, kernel_size, activation=nn.SiLU(), activate_output=False,
                 distribution_base='normal', fixed_stddev=0.4, use_mean=False, scale=1):
        super(GenerativeConvNet, self).__init__()
        self.shape = shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.scale = scale
        inputs = torch.tensor(self.generate_random_noise(self.shape), dtype=torch.float32)
        self.input = nn.Parameter(inputs, requires_grad=True)
        self.fixed_stddev = fixed_stddev
        self.kernel_size = kernel_size
        self.distribution_base = distribution_base
        self.use_mean = use_mean

        layers = []
        output_channels = shape[0] if fixed_stddev else 2*shape[0]
        sizes = hidden_sizes + [output_channels]
        if len(sizes) == 1:
            conv2d = nn.Conv2d(
                in_channels=sizes[0], out_channels=sizes[0],
                kernel_size=kernel_size, padding='same')
            layers.append(conv2d)
        else:
            for i in range(len(sizes) - 1):
                conv2d = nn.Conv2d(
                        in_channels=sizes[i], out_channels=sizes[i + 1],
                        kernel_size=kernel_size, padding='same')
                layers.append(conv2d)
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
        output_distribution = self.dist(y_mu, y_sigma)
        samples = output_distribution.mean if use_mean or self.use_mean else output_distribution.rsample()
        return samples.reshape(batch_size, *self.shape)

    def generate_random_noise(self, shape):
        # generate initial random image
        background_color = np.float32([0])
        gen_img = np.random.normal(background_color, self.scale/4, shape)
        gen_img = np.clip(gen_img, -1, 1)
        return gen_img

    def __str__(self):
        return f" cnn net | hidden_sizes: {self.hidden_sizes} | kernel_size: {self.kernel_size} | " \
               f"distribution: {self.distribution} | fixed_stddev: {self.fixed_stddev} "
