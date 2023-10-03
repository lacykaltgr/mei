import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from abc import ABC, abstractmethod



def LinearSchedule(start, end):
    return lambda step: _LinearSchedule(start, end, step)

def OctaveSchedule(values: list):
    return lambda step: _OctaveSchedule(values, step)

def ConstantSchedule(value):
    return lambda step: _ConstantSchedule(value)


class Scheduler(ABC):
    @abstractmethod
    def __call__(self, step):
        pass


class _LinearSchedule(Scheduler):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps

    def __call__(self, step):
        return self.start + (self.end - self.start) * step / self.steps


class _OctaveSchedule(Scheduler):
    def __init__(self, values: list, steps):
        self.values = values
        self.steps = steps

    def __call__(self, step):
        return self.values[step // self.steps]


class _ConstantSchedule(Scheduler):
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value


def ConstantLearningRateSchedule(warmup_steps, last_epoch=-1):
    return lambda optimizer: _ConstantLearningRate(optimizer, warmup_steps, last_epoch)

def NarrowExponentialDecaySchedule(decay_steps, decay_rate, decay_start, minimum_learning_rate, last_epoch=-1):
    return lambda optimizer: _NarrowExponentialDecay(optimizer, decay_steps, decay_rate, decay_start,
                                                     minimum_learning_rate, last_epoch)

def NarrowCosineDecaySchedule(decay_steps, warmup_steps, decay_start=0, minimum_learning_rate=None, last_epoch=-1):
    return lambda optimizer: _NarrowCosineDecay(optimizer, decay_steps, warmup_steps, decay_start,
                                                minimum_learning_rate, last_epoch)

def NoamScheduleSchedule(warmup_steps=4000, last_epoch=-1):
    return lambda optimizer: _NoamSchedule(optimizer, warmup_steps, last_epoch)


class _ConstantLearningRate(LRScheduler):
    """
    Constant learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        if warmup_steps != 0:
            self.warmup_steps = torch.tensor(warmup_steps)
        else:
            self.warmup_steps = torch.tensor(1)
        super(_ConstantLearningRate, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)
        # self.last_epoch = last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps))
                for v in self.base_lrs]

    def _get_closed_form_lr(self):
        return [v * (torch.minimum(torch.tensor(1.), torch.tensor(self.last_epoch / self.warmup_steps)))
                for v in self.base_lrs]



class _NarrowExponentialDecay(LRScheduler):
    """
    Narrow exponential learning rate decay scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, decay_steps, decay_rate, decay_start,
                 minimum_learning_rate, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate

        super(_NarrowExponentialDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs

    def _get_closed_form_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs


class _NarrowCosineDecay(CosineAnnealingLR):
    """
    Narrow cosine learning rate decay scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, decay_steps, warmup_steps, decay_start=0, minimum_learning_rate=None, last_epoch=-1,
                 verbose=False):
        self.decay_steps = decay_steps
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate
        self.warmup_steps = warmup_steps

        assert self.warmup_steps <= self.decay_start

        super(_NarrowCosineDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, T_max=decay_steps,
                                                 eta_min=self.minimum_learning_rate)

    def get_lr(self):
        if self.last_epoch < self.decay_start:

            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(_NarrowCosineDecay, self).get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.decay_start:
            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(_NarrowCosineDecay, self)._get_closed_form_lr()


class _NoamSchedule(LRScheduler):
    """
    Noam learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps=4000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(_NoamSchedule, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]
