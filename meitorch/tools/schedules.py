import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR


def get_gamma_schedule(params):
    """
    Get gamma schedule for VAE
    Used to weight the KL loss of each group
    :return: nn.Module or None
    """
    return GammaSchedule(max_steps=params.loss_params.gamma_max_steps,
                         num_groups=params.optimizer_params.gamma_n_groups,
                         scaled_gamma=params.loss_params.scaled_gamma) \
        if params.loss_params.use_gamma_schedule \
        else None


def get_beta_schedule(params):
    """
    Get beta schedule for VAE
    Used to weight the KL loss of each group
    :return: nn.Module or uniform tensor function
    """
    return LogisticBetaSchedule(
        activation_step=params.loss_params.vae_beta_activation_steps,
        growth_rate=params.loss_params.vae_beta_growth_rate) \
        if params.loss_params.variation_schedule == 'Logistic' \
        else LinearBetaSchedule(
        anneal_start=params.loss_params.vae_beta_anneal_start,
        anneal_steps=params.loss_params.vae_beta_anneal_steps,
        beta_min=params.loss_params.vae_beta_min) \
        if params.loss_params.variation_schedule == 'Linear' \
        else lambda x: torch.as_tensor(1.)


def get_schedule(optimizer, decay_scheme, warmup_steps, decay_steps, decay_rate, decay_start,
                 min_lr, last_epoch, checkpoint):
    """
    Get learning rate schedule

    :param optimizer: torch.optim.Optimizer, the optimizer to schedule
    :param decay_scheme: str, the decay scheme to use
    :param warmup_steps: int, the number of warmup steps
    :param decay_steps: int, the number of decay steps
    :param decay_rate: float, the decay rate
    :param decay_start: int, the number of steps before starting decay
    :param min_lr: float, the minimum learning rate
    :param last_epoch: int, the last epoch
    :param checkpoint: Checkpoint, the checkpoint to load the scheduler from

    :return: torch.optim.lr_scheduler.LRScheduler, the scheduler
    """
    if decay_scheme == 'noam':
        schedule = NoamSchedule(optimizer=optimizer, warmup_steps=warmup_steps, last_epoch=last_epoch)

    elif decay_scheme == 'exponential':
        schedule = NarrowExponentialDecay(optimizer=optimizer,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          decay_start=decay_start,
                                          minimum_learning_rate=min_lr,
                                          last_epoch=last_epoch)

    elif decay_scheme == 'cosine':
        schedule = NarrowCosineDecay(optimizer=optimizer,
                                     decay_steps=decay_steps,
                                     decay_start=decay_start,
                                     minimum_learning_rate=min_lr,
                                     last_epoch=last_epoch,
                                     warmup_steps=warmup_steps)

    elif decay_scheme == 'constant':
        schedule = ConstantLearningRate(optimizer=optimizer, last_epoch=last_epoch, warmup_steps=warmup_steps)

    else:
        raise NotImplementedError(f'{decay_scheme} is not implemented yet!')

    if checkpoint is not None:
        schedule.load_state_dict(checkpoint.scheduler_state_dict)
        print('Loaded Scheduler Checkpoint')

    return schedule


class LogisticBetaSchedule:
    """
    Logistic beta schedule for VAE
    from Efficient-VDVAE paper
    """
    def __init__(self, activation_step, growth_rate):
        self.beta_max = 1.
        self.activation_step = activation_step
        self.growth_rate = growth_rate

    def __call__(self, step):
        return self.beta_max / (1. + torch.exp(-self.growth_rate * (step - self.activation_step)))


class LinearBetaSchedule:
    """
    Linear beta schedule for VAE
    from Efficient-VDVAE paper
    """
    def __init__(self, anneal_start, anneal_steps, beta_min):
        self.beta_max = 1.
        self.anneal_start = anneal_start
        self.anneal_steps = anneal_steps
        self.beta_min = beta_min

    def __call__(self, step):
        return torch.clamp(torch.tensor((step - self.anneal_start) / (self.anneal_start + self.anneal_steps)),
                           min=self.beta_min, max=self.beta_max)


class GammaSchedule:
    """
    Gamma schedule for VAE
    from Efficient-VDVAE paper
    """
    def __init__(self, max_steps, num_groups, scaled_gamma=False):
        self.max_steps = max_steps
        self.num_groups = num_groups
        self.scaled_gamma = scaled_gamma

    def __call__(self, kl_losses, avg_kl_losses, step_n, epsilon=0.):
        avg_kl_losses = torch.stack(avg_kl_losses, dim=0) * np.log(2)  # [n]
        assert kl_losses.size() == avg_kl_losses.size() == (self.num_groups,)

        if step_n <= self.max_steps:
            if self.scaled_gamma:
                alpha_hat = (avg_kl_losses + epsilon)
            else:
                alpha_hat = kl_losses + epsilon
            alpha = self.num_groups * alpha_hat / torch.sum(alpha_hat)

            kl_loss = torch.tensordot(alpha.detach(), kl_losses, dims=1)

        else:
            kl_loss = torch.sum(kl_losses)

        return kl_loss


class ConstantLearningRate(LRScheduler):
    """
    Constant learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        if warmup_steps != 0:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = 1
        super(ConstantLearningRate, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)
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


class NarrowExponentialDecay(LRScheduler):
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

        super(NarrowExponentialDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs

    def _get_closed_form_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs


class NarrowCosineDecay(CosineAnnealingLR):
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

        super(NarrowCosineDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, T_max=decay_steps,
                                                eta_min=self.minimum_learning_rate)

    def get_lr(self):
        if self.last_epoch < self.decay_start:

            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self).get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.decay_start:
            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self)._get_closed_form_lr()


class NoamSchedule(LRScheduler):
    """
    Noam learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps=4000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(NoamSchedule, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]
