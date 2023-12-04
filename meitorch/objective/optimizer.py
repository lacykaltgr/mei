from torch import optim
import torch
from ..tools.schedules import ConstantSchedule


def get_optimizer(
        params,
        optimizer_type: str = "adam",
        iter_n = None,
        **kwargs):
    """
    Get an optimizer based on the optimizer type and the learning rate

    :param params:
    :param optimizer_type: The optimizer type
    :param iter_n: The number of iterations
    :param kwargs: Additional arguments for the optimizer
    :return: The optimizer nn.Optimizer instance
    """
    if optimizer_type == "adam":
        return optim.Adam(params, **kwargs)
    elif optimizer_type == "sgd":
        return optim.SGD(params, **kwargs)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(params, **kwargs)
    elif optimizer_type == "mei":
        assert iter_n is not None, "iter_n must be specified"
        kwargs.update({"iter_n": iter_n})
        return MEIoptimizer(params, kwargs)
    elif optimizer_type == "meibatch":
        assert iter_n is not None, "iter_n must be specified"
        kwargs.update({"iter_n": iter_n})
        return MEIBatchoptimizer(params, kwargs)
    else:
        raise ValueError("Invalid optimizer type (must be 'adam', 'sgd' or 'rmsprop')")


class MEIoptimizer(optim.Optimizer):

    def __init__(self, params, defaults):
        assert "lr" in defaults, "lr must be specified"
        self.lr = defaults["lr"]
        super(MEIoptimizer, self).__init__(params, defaults)

        assert "iter_n" in defaults, "iter_n must be specified"
        self.iter_n = defaults["iter_n"]

        step_gain = defaults["step_gain"] if "step_gain" in defaults else 1
        if isinstance(step_gain, (int, float)):
            self.step_gain = ConstantSchedule(step_gain)(self.iter_n)
        elif callable(step_gain):
            self.step_gain = step_gain(self.iter_n)

        self.eps = defaults["eps"] if "eps" in defaults else 1e-8
        self.step_i = 0

    def step(self):
        step_gain = self.step_gain(self.step_i)
        step = None
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if not param.requires_grad or param.grad is None:
                    continue
                grad = param.grad.data
                standard_deviation = 1  # hardcoded nem jó
                a = param_group["lr"] / (torch.abs(grad).mean() + self.eps)
                b = (step_gain / (2 * standard_deviation)) * grad.data
                step = a * b
                param.data -= step
        self.step_i += 1
        return step


class MEIBatchoptimizer(MEIoptimizer):
    def __init__(self, params, defaults):
        super(MEIBatchoptimizer, self).__init__(params, defaults)

    def step(self):
        step_gain = self.step_size(self.step_i)
        step = None
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if not param.requires_grad or param.grad is None:
                    continue
                grad = param.grad.data
                standard_deviation = 1  # hardcoded nem jó
                a = param_group["lr"] / (torch.mean(torch.abs(grad.data), dim=0, keepdim=True) + self.eps)
                b = (step_gain / (2 * standard_deviation)) * grad.data
                step = a * b
                param.data -= step
        self.step_i += 1
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
