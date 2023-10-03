from torch import optim
import torch
from meitorch.tools.schedules import _ConstantSchedule, Scheduler


def get_optimizer(optimizer_type: str = "adam", **kwargs):
    """
    Get an optimizer based on the optimizer type and the learning rate

    :param optimizer_type: The optimizer type
    :param lr: The learning rate
    :param kwargs: Additional arguments for the optimizer
    :return: The optimizer
    """
    if optimizer_type == "adam":
        del kwargs["iter_n"]
        return optim.Adam(**kwargs)
    elif optimizer_type == "sgd":
        return optim.SGD(**kwargs)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(**kwargs)
    elif optimizer_type == "mei":
        return MEIoptimizer(**kwargs)
    elif optimizer_type == "meibatch":
        return MEIBatchoptimizer(**kwargs)
    else:
        raise ValueError("Invalid optimizer type (must be 'adam', 'sgd' or 'rmsprop')")


class MEIoptimizer(optim.Optimizer):

    def __init__(self, params, defaults):
        super(MEIoptimizer, self).__init__(params, defaults)

        assert "iter_n" in defaults, "iter_n must be specified"
        self.iter_n = defaults["iter_n"]

        step_size = defaults["start_step_size"] if "start_step_size" in defaults else 0.125
        if isinstance(step_size, (int, float)):
            self.step_size = _ConstantSchedule(step_size)
        elif isinstance(step_size, Scheduler):
            self.step_size = step_size

        self.eps = defaults["eps"] if "eps" in defaults else 1e-8


    def step(self, step_i):
        step_size = self.step_size(step_i)
        step = None
        for param_group in self.param_groups:
            for param in param_group["params"]:
                grad = param.grad.data
                a = step_size / (torch.abs(grad).mean() + self.eps)
                b = param_group["lr"] * grad.data  # itt (step gain -255) volt az egyik szorzó
                step = a * b
                param.data += step
        return step


class MEIBatchoptimizer(MEIoptimizer):
    def __init__(self, params, defaults):
        super(MEIBatchoptimizer, self).__init__(params, defaults)

    def step(self, step_i):
        step_size = self.step_size(step_i)
        step = None
        for param_group in self.param_groups:
            for param in param_group["params"]:
                grad = param.grad.data
                a = step_size / (torch.mean(torch.abs(grad.data), dim=0, keepdim=True) + self.eps)
                b = param_group["lr"] * grad.data
                step = a * b
                param.data += step
        return step

# * both versions are equivalent for a single-image batch, for batches with more than
# one image the first one is better but it drawns out the gradients that are spatially
# wide; for instance a gradient of size 5 x 5 pixels all at amplitude 1 will produce a
# higher change in an image of the batch than a gradient of size 20 x 20 all at
# amplitude 1 in another. This is alright in most cases, but when generating diverse
# images with min linkage (i.e, all images receive gradient from the signal and two
# get the gradient from the diversity term) it drawns out the gradient generated from
# the diversity term (because it is usually bigger spatially than the signal gradient)
# and becomes hard to find very diverse images (i.e., increasing the diversity term
# has no effect because the diversity gradient gets rescaled down to smaller values
# than the signal gradient)
# In any way, gradient mean is only used as normalization here and using the mean is
# alright (also image generation works normally).

