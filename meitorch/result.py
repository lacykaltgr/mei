from meitorch.objective.optimizer import get_optimizer
from abc import ABC, abstractmethod
from torch import nn
import torch
import numpy as np

from meitorch.tools.distributions import GaussianMixtureModel
from meitorch.tools.schedules import ConstantLearningRateSchedule, LRScheduler, CosineAnnealingLR, Scheduler
from meitorch.tools.denoisers import Denoiser
from meitorch.analyze import Analyze


class MEI_result(nn.Module, ABC):
    def __init__(self, shape, n_samples=1, **MEIParams):
        super(MEI_result, self).__init__()
        self.img_shape = shape
        self.n_samples = n_samples
        self.param_dict = MEIParams
        self.result_dict = dict()

        if "scaler" in self.param_dict.keys() and self.param_dict["scaler"] is not None:
            if isinstance(self.param_dict["scaler"], (int, float)):
                self.scaler = lambda step: self.param_dict["scaler"]
            elif isinstance(self.param_dict["scaler"], Scheduler):
                self.scaler = self.param_dict["scaler"]
            else:
                raise ValueError("scaler must be float or Scheduler")
        else:
            self.scaler = None

        if "precond" in self.param_dict.keys() and self.param_dict["precond"] is not None:
            if isinstance(self.param_dict["precond"], (int, float)):
                self.precond = lambda step: self.param_dict["precond"]
            elif isinstance(self.param_dict["precond"], Scheduler):
                self.precond = self.param_dict["precond"]
            else:
                raise ValueError("precond must be float or Scheduler")
        else:
            self.precond = None

        if "jitter" in self.param_dict.keys() and self.param_dict["jitter"] is not None:
            self.jitter = self.param_dict["jitter"]
        else:
            self.jitter = None

        if "blur" in self.param_dict.keys() and self.param_dict["blur"] is not None:
            if isinstance(self.param_dict["blur"], Denoiser):
                self.blur = self.param_dict["blur"]
            elif isinstance(self.param_dict["blur"], str):
                assert "blur_params" in self.param_dict.keys(), "blur_params must be specified"
                self.blur = Denoiser.get_denoiser(self.param_dict["blur"], self.param_dict["blur_params"])
        else:
            self.blur = None

        if "clip" in self.param_dict.keys() and self.param_dict["clip"] is not None:
            self.clip = self.param_dict["clip"]
        else:
            self.clip = None

        if "norm" in self.param_dict.keys() and self.param_dict["norm"] is not None:
            self.norm = self.param_dict["norm"]
        else:
            self.norm = None

        if "train_norm" in self.param_dict.keys() and self.param_dict["train_norm"] is not None:
            self.train_norm = self.param_dict["train_norm"]
        else:
            self.train_norm = None

        assert "bias" in self.param_dict.keys(), "bias must be specified"
        self.bias = self.param_dict["bias"]
        assert "scale" in self.param_dict.keys(), "scale must be specified"
        self.scale = self.param_dict["scale"]

        self.loss_history = []
        self.image_history = []

    def show_results(self):
        for key, value in self.result_dict.items():
            print(key, value)

    def init_optimizer(self):
        assert "optimizer" in self.param_dict.keys(), "optimizer must be specified"
        if "optimizer_params" in self.param_dict.keys():
            optimizer_hparams = self.param_dict["optimizer_params"]
        else:
            optimizer_hparams = dict()
        optimizer_hparams.update(
            {"params": self.parameters(),
             "iter_n": self.iter_n})
        self.optimizer = get_optimizer(self.param_dict["optimizer"], **optimizer_hparams)

        #assert "lr" in self.param_dict["optimizer_params"].keys(), "lr must be specified"
        #step_gain = self.param_dict["optimizer_params"]["lr"]
        #if isinstance(step_gain, (int, float)):
        #    scheduler = ConstantLearningRateSchedule(step_gain)
        #elif isinstance(step_gain, (LRScheduler, CosineAnnealingLR)):
        #    scheduler = step_gain
        #else:
        #    raise ValueError("lr must be float or LRScheduler")
        #self.scheduler = scheduler(self.optimizer)

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

    def show_image(self):
        import matplotlib.pyplot as plt
        image = self.get_image().detach().cpu().numpy()
        if image is None:
            return None
        if image.shape[0] == 1:
            image = image.squeeze(0)
        plt.imshow(image)
        plt.show()

    def save_image(self, path):
        from meitorch.tools.utils import write_image_to_disk
        image = self.get_image()
        if image is None:
            return None
        write_image_to_disk(path, image.detach().cpu().numpy())

    @abstractmethod
    def __getstate__(self):
        state = dict(
            img_shape=self.img_shape,
            n_samples=self.n_samples,
            param_dict=self.param_dict,
            result_dict=self.result_dict,
        )
        return state

    @abstractmethod
    def __setstate__(self, state):
        self.img_shape = state["img_shape"]
        self.n_samples = state["n_samples"]
        self.param_dict = state["param_dict"]
        self.result_dict = state["result_dict"]

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

    def __getattr__(self, item):
        if item in self.param_dict.keys():
            return self.param_dict[item]
        else:
            return super().__getattr__(item)



class MEI_image(MEI_result):
    def __init__(self, shape, n_samples=1, image=None, **MEIParams):
        super(MEI_image, self).__init__(shape, n_samples, **MEIParams)
        self.batch_shape = (n_samples, *shape)

        if image is None:
            self.image = torch.nn.Parameter(
                torch.tensor(self.generate_random_noise(self.batch_shape), dtype=torch.float32),
                requires_grad=True)
        else:
            self.image = image

        self.init_optimizer()

    def get_image(self):
        return self.image

    def get_samples(self):
        return self.image

    def __getstate__(self):
        state = super().__getstate__()
        state.update({"image": self.image})
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.image = state["image"]


class MEI_distibution(MEI_result):
    def __init__(self, distribution, img_shape, **MEIParams):
        assert "n_samples_per_batch" in MEIParams, "n_samples_per_batch must be specified"
        n_samples = MEIParams["n_samples_per_batch"]
        super(MEI_distibution, self).__init__(img_shape, n_samples, **MEIParams)

        self.distribution_type = distribution
        assert "fixed_stddev" in MEIParams, "fixed_stddev must be specified for uniform distribution"
        if distribution == "normal":
            mean, std = self.generate_loc_scale(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Normal(mean, std)
        elif distribution == "laplace":
            mean, std = self.generate_loc_scale(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Laplace(mean, std)
        elif distribution == "uniform":
            mean, std = self.generate_loc_scale(MEIParams["fixed_stddev"])
            self.distribution = torch.distributions.Uniform(mean, std)
        elif distribution == "mixture_of_gaussians":
            assert "n_components" in MEIParams, "n_components must be specified for mixture of gaussians"
            n_components = MEIParams["n_components"]
            self.distribution = GaussianMixtureModel(n_components, img_shape, MEIParams["fixed_stddev"])
        else:
            raise ValueError("Distribution not supported")

        self.init_optimizer()

    def get_image(self):
        return self.distribution.mean

    def get_samples(self):
        return self.distribution.rsample(self.n_samples)

    def generate_loc_scale(self, fixed_stddev=False):
        mean = self.generate_random_noise(self.img_shape)
        mean = torch.nn.Parameter(torch.tensor(mean), requires_grad=True)

        if fixed_stddev:
            std = torch.ones(self.img_shape) * fixed_stddev
        else:
            std = self.generate_random_noise(self.img_shape)
            std = torch.nn.Parameter(torch.tensor(std), requires_grad=True)
        return mean, std

    def __getstate__(self):
        state = super().__getstate__()
        state.update({"distribution": self.distribution})
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.distribution = state["distribution"]


class MEI_neural_network(MEI_result):
    def __init__(self, net, img_shape, shape, **MEIParams):
        n_samples = MEIParams["n_samples"]
        super(MEI_neural_network, self).__init__(img_shape, n_samples, **MEIParams)

        self.net = net
        self.batch_shape = (n_samples, *img_shape)

        self.init_optimizer()

    def get_image(self):
        return self.net(self.generate_random_noise(self.batch_shape))

    def get_samples(self):
        sample_batch = torch.nn.Parameter(
            torch.tensor(self.generate_random_noise(self.batch_shape)),
            requires_grad=True)
        return self.net(sample_batch)

    def __getstate__(self):
        state = super().__getstate__()
        state.update({"net": self.net})
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.net = state["net"]











