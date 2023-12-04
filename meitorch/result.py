
from abc import ABC, abstractmethod
from torch import nn
import torch
import numpy as np

from .tools.denoisers import Denoiser
from .objective.optimizer import get_optimizer
from .tools.schedules import get_lr_scheduler


class MEI_result(nn.Module, ABC):
    def __init__(self, shape, n_samples=1, device='cpu',  **MEIParams):
        super(MEI_result, self).__init__()
        self.img_shape = shape
        self.n_samples = n_samples
        self.param_dict = MEIParams
        self.result_dict = dict()
        self.device = device
        self.scheduler = None
        self.optimizer = None

        assert "iter_n" in self.param_dict.keys(), "iter_n must be specified"
        self.iter_n = self.param_dict["iter_n"]

        self.save_every = self.param_dict["save_every"] if "save_every" in self.param_dict.keys() else self.iter_n

        if "scaler" in self.param_dict.keys() and self.param_dict["scaler"] is not None:
            if isinstance(self.param_dict["scaler"], (int, float)):
                self.scaler = lambda step: self.param_dict["scaler"]
            elif callable(self.param_dict["scaler"]):
                self.scaler = self.param_dict["scaler"](self.iter_n)
            else:
                raise ValueError("scaler must be float or Scheduler")
        else:
            self.scaler = None

        if "precond" in self.param_dict.keys() and self.param_dict["precond"] is not None:
            if isinstance(self.param_dict["precond"], (int, float)):
                self.precond = lambda step: self.param_dict["precond"]
            elif callable(self.param_dict["precond"]):
                self.precond = self.param_dict["precond"](self.iter_n)
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
                self.blur = Denoiser.get_denoiser(self.param_dict["blur"],
                                                  self.iter_n,  **self.param_dict["blur_params"])
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

        self.diverse = "diverse" in self.param_dict.keys() and self.param_dict["diverse"]
        if self.diverse:
            if "diverse_params" in self.param_dict.keys():
                self.diverse_params = self.param_dict["diverse_params"]
            else:
                self.diverse_params = dict()

        self.best_loss = np.inf
        self.best_image = None

        assert "bias" in self.param_dict.keys(), "bias must be specified"
        self.bias = self.param_dict["bias"]
        assert "scale" in self.param_dict.keys(), "scale must be specified"
        self.scale = self.param_dict["scale"]

        self.loss_history = dict()
        self.image_history = []

    def show_results(self):
        for key, value in self.result_dict.items():
            print(key, value)

    def init_optimizer(self):
        assert "optimizer" in self.param_dict.keys(), "optimizer must be specified"
        assert "optimizer_params" in self.param_dict.keys(), "optimizer_hparams must be specified"

        optimizer_params = self.param_dict["optimizer_params"]
        scheduler_params = None
        if "scheduler_params" in optimizer_params.keys():
            scheduler_params = optimizer_params["scheduler_params"]
            del optimizer_params["scheduler_params"]

        self.optimizer = get_optimizer(self.parameters(),
                                       self.param_dict["optimizer"],
                                       self.iter_n, **optimizer_params)

        if scheduler_params is not None:
            self.scheduler = get_lr_scheduler(self.optimizer, scheduler_params)

    def spatial_frequency(self):
        """
        Compute the spatial frequency of the image and plot it
        :return: frequency of each column, row in arrays, magnitude spectrum
        """
        from .analyze import Analyze
        return Analyze.compute_spatial_frequency(self.get_image()[0])

    def update_loss_history(self, loss_dict):
        for key, value in loss_dict.items():
            if key not in self.loss_history.keys():
                self.loss_history[key] = []
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy().mean()
            self.loss_history[key].append(value)

    def plot_losses(self, show=False, save_path=None, ranges=None):
        # plot multiple losses in one plot in different colors
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots()
        for key, value in self.loss_history.items():
            if ranges is not None:
                assert len(ranges) == 2
                clipped_value = np.clip(value, *ranges)
            else:
                clipped_value = value
            ax.plot(clipped_value, label=key)
        ax.legend()
        ax.set_ylabel("iteration")
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    def plot_image_and_losses(self, save_path=None, ranges=None):
        # plot multiple losses in one plot in different colors
        import matplotlib.pyplot as plt 
        fig, (ax_im, ax_loss) = plt.subplots(1, 2, figsize=(6, 3),
                                             gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1]})
        image = self.get_image()[0].detach().cpu().numpy()
        ax_im.imshow(image.reshape(40, 40))
        ax_im.set_title(f"Activation: {self.get_activation():.2f}")
        ax_im.axis("off")

        for key, value in self.loss_history.items():
            if ranges is not None:
                assert len(ranges) == 2
                clipped_value = np.clip(value, *ranges)
            else:
                clipped_value = value
            ax_loss.plot(clipped_value, label=key)
        ax_loss.legend()
        
        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        return fig

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
        gen_img = np.random.normal(background_color, self.scale / 16, shape)
        gen_img = np.clip(gen_img, -1, 1)
        return gen_img


class MEI_image(MEI_result):
    def __init__(self, shape, n_samples=1, init=None, device='cpu', **MEIParams):
        super(MEI_image, self).__init__(shape, n_samples, device=device, **MEIParams)
        self.batch_shape = (n_samples, *shape)

        if init is None:
            self.image = torch.nn.Parameter(
                torch.tensor(self.generate_random_noise(self.batch_shape),
                             dtype=torch.float32, device=self.device),
                requires_grad=True)
        else:
            if isinstance(init, np.ndarray):
                image = torch.tensor(init, device=device)
            elif isinstance(init, torch.Tensor):
                image = init.to(device)
            else:
                raise NotImplementedError(f"init must be np.ndarray or torch.Tensor, not {type(init)}")
            self.image = torch.nn.Parameter(image, requires_grad=True)
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


class MEI_variational(MEI_result):
    def __init__(self, distribution, img_shape, init=None, device='cpu', **MEIParams):
        assert "n_samples_per_batch" in MEIParams, "n_samples_per_batch must be specified"
        n_samples = MEIParams["n_samples_per_batch"]
        super(MEI_variational, self).__init__(img_shape, n_samples, device, **MEIParams)

        if distribution == "normal":
            dist = torch.distributions.Normal
        elif distribution == "laplace":
            dist = torch.distributions.Laplace
        elif distribution == "uniform":
            dist = torch.distributions.Uniform
        else:
            raise NotImplementedError(f"Distribution {distribution} not supported")

        assert "fixed_stddev" in MEIParams, "fixed_stddev must be specified"
        self.mean, self.std = self.generate_loc_scale(init, MEIParams["fixed_stddev"])
        self.register_parameter("mu", self.mean)
        if not MEIParams["fixed_stddev"]:
            self.register_parameter("sigma", self.std)
        self.distribution = dist(self.mean, self.std)

        self.init_optimizer()

    def get_image(self):
        return self.distribution.mean

    def get_samples(self):
        return self.distribution.rsample(torch.Size(self.n_samples))

    def generate_loc_scale(self, mean=None, fixed_stddev=False) -> (nn.Parameter, nn.Parameter):
        mean = self.generate_random_noise(self.img_shape) if mean is None else mean
        mean = torch.nn.Parameter(torch.tensor(mean,
                                               dtype=torch.float32,
                                               device=self.device),  requires_grad=True)
        
        stddev = fixed_stddev if fixed_stddev else 0.4
        std = torch.ones(self.img_shape, dtype=torch.float32, device=self.device) * stddev
        if not fixed_stddev:
            std = torch.nn.Parameter(std, requires_grad=True)
        return mean, std

    def __getstate__(self):
        state = super().__getstate__()
        state.update({"distribution": self.distribution})
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.distribution = state["distribution"]


class MEI_transformation(MEI_result):
    def __init__(self, transform, img_shape, device='cpu', **MEIParams):
        n_samples = MEIParams["n_samples"]
        del MEIParams["n_samples"]
        super(MEI_transformation, self).__init__(img_shape, n_samples, device, **MEIParams)

        self.transform = transform.to(self.device)
        self.batch_shape = (n_samples, *img_shape)

        self.init_optimizer()

    def get_image(self):
        return self.transform(self.n_samples, use_mean=True)

    def get_samples(self):
        return self.transform(self.n_samples)

    def __getstate__(self):
        state = super().__getstate__()
        state.update({"transform": self.transform})
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.transform = state["transform"]
