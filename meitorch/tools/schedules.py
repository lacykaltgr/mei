from abc import ABC, abstractmethod
from torch import optim
import numpy as np


def LinearSchedule(start, end):
    return lambda iter_n: _LinearSchedule(start, end, iter_n)


def OctaveSchedule(values: list):
    return lambda iter_n: _OctaveSchedule(values, iter_n)


def ConstantSchedule(value):
    return lambda iter_n: _ConstantSchedule(value)


def RandomSchedule(minimum, maximum):
    return lambda iter_n: _RandomSchedule(minimum, maximum)


class Scheduler(ABC):
    @abstractmethod
    def __call__(self, step):
        pass
    
    def __str__(self):
        pass


class _LinearSchedule(Scheduler):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps

    def __call__(self, step):
        return self.start + (self.end - self.start) * step / self.steps
    
    def __str__(self):
        return f"linear {self.start}->{self.end}"
    
    
class _RandomSchedule(Scheduler):
    def __init__(self, minimum, maximum):
        self.maximum = maximum
        self.minimum = minimum
        
    def __call__(self, step):
        return np.random.uniform(self.minimum, self.maximum)
    
    def __str__(self):
        return f"random {self.minimum}->{self.maximum}"


class _OctaveSchedule(Scheduler):
    def __init__(self, values: list, steps):
        self.values = values
        self.steps = steps

    def __call__(self, step):
        octaves = len(self.values)
        steps_per_octave = self.steps // octaves + 1
        step = step // steps_per_octave
        return self.values[step]
    
    def __str__(self):
        return f"octave {self.values}"


class _ConstantSchedule(Scheduler):
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value
    
    def __str__(self):
        return f"constant {self.value}"


def get_lr_scheduler(optimizer, hparams):

    """
    StepLR:     This scheduler decreases the learning rate at specified epochs by a multiplicative factor.
                For example, you could use this scheduler to decrease the learning rate by 0.5 every 10 epochs.

                - step_size (int): The number of epochs between each learning rate decrease.
                - gamma (float): The multiplicative factor by which the learning rate is decreased at each step.

    MultiStepLR: This scheduler is similar to StepLR, but it allows you to specify multiple epochs
                    at which to decrease the learning rate. For example,
                    you could use this scheduler to decrease the learning rate by 0.5 at epochs 10, 20, and 30.

                - milestones (list[int]): A list of epochs at which to decrease the learning rate.
                - gamma (float): The multiplicative factor by which the learning rate is decreased at each step.

    ExponentialLR: This scheduler decreases the learning rate exponentially by a multiplicative factor at each epoch.

                - gamma (float): The multiplicative factor by which the learning rate is decreased at each epoch.

    CosineAnnealingLR: This scheduler decreases the learning rate following a cosine curve,
                        starting from a high value and gradually decreasing to a low value.

                - T_max (int): The maximum number of epochs.
                - eta_min (float): The minimum learning rate.

    CosineAnnealingWarmRestartsLR: This scheduler is similar to CosineAnnealingLR,
                                    but it restarts the cosine annealing process at specified epochs.
                                    This can be useful for preventing the model from getting stuck in local minima.

                - T_max (int): The maximum number of epochs per restart.
                - eta_min (float): The minimum learning rate.
                - T_mult (float): A factor by which the T_max is multiplied after each restart.

    :param optimizer: the optimizer to be used
    :param hparams: the hyperparameters of the schdeler
    :return: the scheduler
    """

    s_type = hparams["type"]
    del hparams["type"]

    if s_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, **hparams)
    elif s_type == "multi_step":
        return optim.lr_scheduler.MultiStepLR(optimizer, **hparams)
    elif s_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **hparams)
    elif s_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **hparams)
    elif s_type == "cosine_warm":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **hparams)

