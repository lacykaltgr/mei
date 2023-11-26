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
        return f" mlp net | hidden_sizes: {self.hidden_sizes} | distribution: {self.distribution} | fixed_stddev: {self.fixed_stddev} "


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
                 distribution_base='normal', fixed_stddev=0.4, use_mean=False, scale=1):
        super(GenerativeConvNet, self).__init__()
        self.shape = shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.scale = scale
        self.input = nn.Parameter(torch.tensor(self.generate_random_noise(self.shape),
                                              dtype=torch.float32),
                                  requires_grad=True)
        self.fixed_stddev = fixed_stddev
        self.kernel_size = kernel_size
        self.distribution_base = distribution_base
        self.use_mean = use_mean

        layers = []
        output_channels = shape[0] if fixed_stddev else 2*shape[0]
        sizes = hidden_sizes + [output_channels]
        for i in range(len(sizes) - 1):
            conv2d = nn.Conv2d(
                    in_channels=sizes[i], out_channels=sizes[i + 1],
                    kernel_size=kernel_size, padding='same')
            layers.append(conv2d)
            if i < len(sizes) - 2 or activate_output:
                layers.append(self.activation)
        if len(sizes) == 1:
            conv2d = nn.Conv2d(
                    in_channels=sizes[0], out_channels=sizes[0],
                    kernel_size=kernel_size, padding='same')
            layers.append(conv2d)
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
        return f" cnn net | hidden_sizes: {self.hidden_sizes} | kernel_size: {self.kernel_size} distribution: {self.distribution} | fixed_stddev: {self.fixed_stddev} "

    
class AdaptiveGenerativeConvNet(nn.Module):
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

    def __init__(self, hidden_sizes, steps, shape, kernel_size, activation=nn.SiLU(), activate_output=False,
                 distribution_base='normal', fixed_stddev=0.4, device="cpu"):
        super(AdaptiveGenerativeConvNet, self).__init__()
        self.shape = shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_output
        self.input = nn.Parameter(torch.tensor(self.generate_random_noise(self.shape),
                                              dtype=torch.float32),
                                  requires_grad=True)
        self.fixed_stddev = fixed_stddev
        self.kernel_size = kernel_size
        self.distribution_base = distribution_base
        self.steps = steps
        self.step_i = 0
        self.layer_step = 0
        self.device = device
        self.output_size = shape[0]
        self.sizes = hidden_sizes 
        self.new_layer = []
        for i, step in enumerate(steps):
            self.new_layer.append([
                nn.Conv2d(in_channels=self.sizes[self.layer_step], out_channels=self.sizes[self.layer_step+1],
                            kernel_size=self.kernel_size, padding='same', device=self.device),
                self.activation.to(self.device),
                nn.Conv2d(in_channels=self.sizes[self.layer_step]+1, out_channels=self.output_size,
                            kernel_size=self.kernel_size, padding='same', device=self.device)])
        self.conv = nn.Sequential()

        if distribution_base == 'normal':
            self.dist = torch.distributions.Normal
        elif distribution_base == 'laplace':
            self.dist = torch.distributions.Laplace
        else:
            raise ValueError(f'Unknown distribution base: {distribution_base}')

    def forward(self, batch_size, use_mean=False):
        if not use_mean:
            self.add_layer()
        batch = torch.tile(self.input, (batch_size, 1, 1, 1))
        output = self.conv(batch)
        y_mu = output
        y_sigma = torch.ones_like(output) * self.fixed_stddev
        output_distribution = self.dist(y_mu, y_sigma)
        samples = output_distribution.rsample() if not use_mean else output_distribution.mean
        return samples.reshape(batch_size, *self.shape)
    
    def add_layer(self):
        self.step_i += 1
        if len(self.steps) > self.layer_step and self.step_i == self.steps[self.layer_step]:
            if len(self.conv) > 0:
                self.conv = self.conv[:-1]
            self.conv.extend(self.new_layer[self.layer_step])
            self.layer_step += 1
            
    
    
    def generate_random_noise(self, shape):
        # generate initial random image
        background_color = np.float32([0])
        gen_img = np.random.normal(background_color, 1 / 4, shape)
        gen_img = np.clip(gen_img, -1, 1)
        return gen_img
    
    
    def __str__(self):
        return f" cnn net | hidden_sizes: {self.hidden_sizes} | kernel_size: {self.kernel_size} distribution: {self.distribution} | fixed_stddev: {self.fixed_stddev} "

    
class GenerativeConvNetB(nn.Module):
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
                 distribution_base='normal', fixed_stddev=0.4):
        super(GenerativeConvNetB, self).__init__()
        self.shape = shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.activate_output = activate_outputs
        self.distribution_base = distribution_base
              
        self.mean = nn.Parameter(torch.tensor(self.generate_random_noise(self.shape),
                                              dtype=torch.float32),
                                  requires_grad=True)
        self.std = nn.Parameter(torch.ones_like(self.mean) * fixed_stddev,
                               requires_grad=False)
                
        if distribution_base == 'normal':
            self.dist = torch.distributions.Normal(self.mean, self.std)
        elif distribution_base == 'laplace':
            self.dist = torch.distributions.Laplace(self.mean, self.std)
        else:
            raise ValueError(f'Unknown distribution base: {distribution_base}')

        layers = []
        output_channels = shape[0]
        sizes = hidden_sizes + [output_channels]
        for i in range(len(sizes) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=sizes[i], out_channels=sizes[i + 1],
                    kernel_size=kernel_size, padding='same'))
            if i < len(sizes) - 2 or activate_output:
                layers.append(self.activation)
        self.conv = nn.Sequential(*layers)



            
    def forward(self, batch_size, use_mean=False):
        device = next(self.conv.parameters()).device
        batch = self.dist.rsample((batch_size, )).to(device)
        output = self.conv(batch)
        return output.reshape(batch_size, *self.shape)
    
    
    def generate_random_noise(self, shape):
        # generate initial random image
        background_color = np.float32([0])
        gen_img = np.random.normal(background_color, 1 / 16, shape)
        gen_img = np.clip(gen_img, -1, 1)
        return gen_img


