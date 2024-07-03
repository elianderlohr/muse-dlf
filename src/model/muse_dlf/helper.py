import torch
from torch.nn.functional import log_softmax, softmax, sigmoid, logsigmoid


def sample_gumbel(shape, eps=1e-20, device="cpu"):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


# Softmax


def gumbel_softmax_sample(logits, t):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return softmax(y / t, dim=-1)


def gumbel_logsoftmax_sample(logits, t):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return log_softmax(y / t, dim=-1)


def custom_gumbel_softmax(logits, tau, hard=False, log=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    tau: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    if log:
        y = gumbel_logsoftmax_sample(logits, tau)
    else:
        y = gumbel_softmax_sample(logits, tau)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard
    return y


# Sigmoid


def gumbel_sigmoid_sample(logits, t):
    """Draw a sample from the Gumbel-Sigmoid distribution"""
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return sigmoid(y / t)


def gumbel_logsigmoid_sample(logits, t):
    """Draw a sample from the Gumbel-Sigmoid distribution"""
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return logsigmoid(sigmoid(y / t))


def custom_gumbel_sigmoid(logits, tau, hard=False, log=False):
    """Sample from the Gumbel-Sigmoid distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    tau: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Sigmoid distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    if log:
        y = gumbel_logsigmoid_sample(logits, tau)
    else:
        y = gumbel_sigmoid_sample(logits, tau)

    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard
    return y
