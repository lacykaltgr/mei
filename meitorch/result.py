from meitorch.tools.optimizer import get_optimizer
from abc import ABC, abstractmethod
from torch import nn
import torch
import numpy as np

from meitorch.tools.distributions import GaussianMixtureModel


class MEI_result(nn.Module, ABC):
    def __init__(self, img_shape, **MEIParams):
        super().__init__()
        self.img_shape = img_shape
        self.param_dict = MEIParams
        self.result_dict = dict()

        """
        self.iter_n = 1000 if 'iter_n' not in MEIParams else MEIParams['iter_n']
        self.start_sigma = 1.5 if 'start_sigma' not in MEIParams else MEIParams['start_sigma']
        self.end_sigma = 0.01 if 'end_sigma' not in MEIParams else MEIParams['end_sigma']
        self.precond = 0  # .1          if 'precond' not in MEIParams else MEIParams['precond']
        self.jitter = 0 if 'jitter' not in MEIParams else MEIParams['jitter']
        self.blur = True if 'blur' not in MEIParams else MEIParams['blur']
        self.norm = -1 if 'norm' not in MEIParams else MEIParams['norm']
        self.train_norm = -1 if 'train_norm' not in MEIParams else MEIParams['train_norm']
        self.clip = True if 'clip' not in MEIParams else MEIParams['clip']
        """
        self.loss_history = []
        self.image_history = []

    def show_results(self):
        for key, value in self.result_dict.items():
            print(key, value)

    def init_optimizer(self):
        optimizer_hparams = self.param_dict["optimizer"]
        optimizer_hparams.update(
            {"params": [self.image],
             "iter_n": self.iter_n})
        self.optimizer = get_optimizer(**optimizer_hparams)

    def spatial_frequency(self):
        """
        Compute the spatial frequency of the image and plot it
        :return: frequency of each column, row in arrays, magnitude spectrum
        """
        from .analyze import Analyze
        return Analyze.compute_spatial_frequency(self.image)

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def get_samples(self):
        pass

    def get_activation(self):
        if "activation" not in self.result_dict.keys():
            return None
        return self.result_dict["activation"]

    def generate_random_noise(self, shape):
        # generate initial random image
        background_color = np.float32([self.bias] * self.img_shape[-1])
        gen_img = np.random.normal(background_color, self.scale / 20, shape)
        gen_img = np.clip(gen_img, -1, 1)
        return gen_img


class MEI_image(MEI_result):
    def __init(self, n_images, shape, **MEIParams):
        super().__init__(shape, **MEIParams)
        self.n_images = n_images
        self.batch_shape = (n_images, *shape)
        self.imgage = self.generate_random_noise(self.batch_shape)

    def get_image(self):
        return self.image

    def get_samples(self):
        # add batch dimension
        # jitter, random noise etc.
        return self.image

    def show_image(self):
        """
        Plot the image
        """
        import matplotlib.pyplot as plt
        if self.image is None:
            return None
        if self.image.shape[0] == 1:
            image = self.image.squeeze(0)
        else:
            image = self.image
        plt.imshow(image)
        plt.show()

    def save_image(self, path):
        """
        Save the image

        :param path: The path to save the image
        """
        from meitorch.tools.utils import write_image_to_disk
        if self.image is None:
            return None
        write_image_to_disk(path, self.image.detach().cpu().numpy())


class MEI_distibution(MEI_result):
    def __init__(self, distribution, img_shape, **MEIParams):
        super().__init__(img_shape, **MEIParams)

        self.distribution_type = distribution
        assert "fixed_stddev" in MEIParams, "fixed_stddev must be specified for uniform distribution"
        if distribution == "normal":
            mean, std = self.get_mean_std(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Normal(mean, std)
        elif distribution == "laplace":
            mean, std = self.get_mean_std(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Laplace(mean, std)
        elif distribution == "uniform":
            mean, std = self.get_mean_std(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Uniform(mean, std)
        elif distribution == "mixture_of_gaussians":
            assert "n_components" in MEIParams, "n_components must be specified for mixture of gaussians"
            n_components = MEIParams["n_components"]
            self.distribution = GaussianMixtureModel(n_components, self.img_shape, MEIParams["fixed_stddev"])
        else:
            raise ValueError("Distribution not supported")

    def get_image(self):
        return self.distribution.mean

    def get_samples(self):
        # multiple samples
        return self.distribution.rsample()

    def get_mean_std(self, fixed_stddev=False):
        mean = self.generate_random_noise(self.img_shape)
        if not fixed_stddev:
            std = self.generate_random_noise(self.img_shape)
        else:
            std = torch.ones(self.img_shape) * fixed_stddev
        return mean, std


class MEI_neural_network(MEI_result):
    def __init__(self, model, input_type, input_shape, img_shape,  **MEIParams):
        super().__init__(img_shape, **MEIParams)
        self.model = model
        self.input_shape = input_shape

    def get_image(self):
        return self.model(self.generate_noise())

    def get_samples(self):
        sample_batch = []
        for i in range(self.param_dict["sample_n"]):
            sample_batch.append(self.generate_noise())
        sample_batch = torch.stack(sample_batch)
        return self.model(sample_batch)

    def generate_noise(self):
        """
        Generate noise based on the input shape
        :return: noise
        """
        noise = torch.randn(self.input_shape)
        return noise










