"""
distributions.py — Probability distributions module for AMEVA-Forge
Provides PyTorch-compatible torch.distributions APIs with reparameterization trick (rsample) and analytical KL divergence.
"""
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import math
from .tensor import Tensor
from .ops import tensor, _require_cpu_data, where, sum_axis, mean_op


class Distribution:
    """Base class for probability distributions in AMEVA-Forge."""
    has_rsample = False
    
    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        raise NotImplementedError

    def rsample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self) -> Tensor:
        raise NotImplementedError

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        return self.variance.sqrt()


class Normal(Distribution):
    """
    Creates a normal (also called Gaussian) distribution parameterized by loc and scale.
    Supports reparameterized sampling (rsample) for VAEs, Diffusion, and RL.
    """
    has_rsample = True

    def __init__(self, loc: Union[Tensor, float], scale: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.loc = loc if isinstance(loc, Tensor) else tensor(loc)
        self.scale = scale if isinstance(scale, Tensor) else tensor(scale)

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2.0)

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        shape = tuple(sample_shape) + self.loc.shape
        eps = np.random.randn(*shape).astype(np.float32)
        res = self.loc.numpy() + self.scale.numpy() * eps
        return tensor(res, device=self.loc.device, dtype=self.loc.dtype)

    def rsample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        shape = tuple(sample_shape) + self.loc.shape
        eps = np.random.randn(*shape).astype(np.float32)
        eps_tensor = tensor(eps, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * eps_tensor

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.scale.pow(2.0)
        log_scale = self.scale.log()
        return -((value - self.loc).pow(2.0)) / (2.0 * var) - log_scale - math.log(math.sqrt(2.0 * math.pi))

    def entropy(self) -> Tensor:
        return 0.5 + 0.5 * math.log(2.0 * math.pi) + self.scale.log()


class Uniform(Distribution):
    """
    Generates uniformly distributed random variables over the half-open interval [low, high).
    """
    has_rsample = True

    def __init__(self, low: Union[Tensor, float], high: Union[Tensor, float]):
        self.low = low if isinstance(low, Tensor) else tensor(low)
        self.high = high if isinstance(high, Tensor) else tensor(high)

    @property
    def mean(self) -> Tensor:
        return (self.low + self.high) / 2.0

    @property
    def variance(self) -> Tensor:
        return (self.high - self.low).pow(2.0) / 12.0

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        shape = tuple(sample_shape) + self.low.shape
        u = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
        res = self.low.numpy() + (self.high.numpy() - self.low.numpy()) * u
        return tensor(res, device=self.low.device, dtype=self.low.dtype)

    def rsample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        shape = tuple(sample_shape) + self.low.shape
        u = tensor(np.random.uniform(0.0, 1.0, size=shape).astype(np.float32), device=self.low.device, dtype=self.low.dtype)
        return self.low + (self.high - self.low) * u

    def log_prob(self, value: Tensor) -> Tensor:
        return -(self.high - self.low).log()

    def entropy(self) -> Tensor:
        return (self.high - self.low).log()


class Bernoulli(Distribution):
    """
    Creates a Bernoulli distribution parameterized by probs or logits.
    """
    has_rsample = False

    def __init__(self, probs: Optional[Union[Tensor, float]] = None, logits: Optional[Union[Tensor, float]] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified, but not both.")
        if probs is not None:
            self.probs = probs if isinstance(probs, Tensor) else tensor(probs)
            self.logits = (self.probs / (1.0 - self.probs)).log()
        else:
            self.logits = logits if isinstance(logits, Tensor) else tensor(logits)
            self.probs = 1.0 / (1.0 + (-self.logits).exp())

    @property
    def mean(self) -> Tensor:
        return self.probs

    @property
    def variance(self) -> Tensor:
        return self.probs * (1.0 - self.probs)

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        shape = tuple(sample_shape) + self.probs.shape
        data_p = _require_cpu_data(self.probs, "probs")
        res = np.random.binomial(1, data_p, size=shape).astype(np.float32)
        return tensor(res, device=self.probs.device, dtype=self.probs.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        return value * self.probs.log() + (1.0 - value) * (1.0 - self.probs).log()

    def entropy(self) -> Tensor:
        p = self.probs
        return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


class Categorical(Distribution):
    """
    Creates a categorical distribution parameterized by either probs or logits.
    Used widely in Reinforcement Learning (PPO, A2C, Policy Gradients).
    """
    has_rsample = False

    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Either probs or logits must be specified, but not both.")
        if probs is not None:
            p_sum = probs.sum(axis=-1, keepdim=True)
            self.probs = probs / p_sum
            self.logits = self.probs.log()
        else:
            from .functional import log_softmax, softmax
            self.logits = log_softmax(logits, axis=-1)
            self.probs = softmax(logits, axis=-1)

    @property
    def mean(self) -> Tensor:
        return tensor(float('nan'))

    @property
    def variance(self) -> Tensor:
        return tensor(float('nan'))

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        data_p = _require_cpu_data(self.probs, "probs")
        orig_shape = data_p.shape[:-1]
        num_classes = data_p.shape[-1]
        flat_p = data_p.reshape(-1, num_classes)
        
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = []
        for _ in range(n_samples):
            sub_samples = [np.random.choice(num_classes, p=p_row) for p_row in flat_p]
            samples.append(np.array(sub_samples, dtype=np.int32).reshape(orig_shape))
            
        res = np.stack(samples, axis=0) if sample_shape else samples[0]
        return tensor(res, dtype="int32", device=self.probs.device)

    def log_prob(self, value: Tensor) -> Tensor:
        data_val = _require_cpu_data(value, "value").astype(np.int64)
        data_log_p = _require_cpu_data(self.logits, "logits")
        selected = np.take_along_axis(data_log_p, np.expand_dims(data_val, axis=-1), axis=-1).squeeze(axis=-1)
        return tensor(selected, dtype="float32", device=self.logits.device)

    def entropy(self) -> Tensor:
        return -(self.probs * self.logits).sum(axis=-1)


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """
    Computes analytical KL divergence KL(p || q) between two distributions.
    """
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_p = p.variance
        var_q = q.variance
        return q.scale.log() - p.scale.log() + (var_p + (p.loc - q.loc).pow(2.0)) / (2.0 * var_q) - 0.5
    elif isinstance(p, Categorical) and isinstance(q, Categorical):
        return (p.probs * (p.logits - q.logits)).sum(axis=-1)
    elif isinstance(p, Bernoulli) and isinstance(q, Bernoulli):
        return (p.probs * (p.probs.log() - q.probs.log()) + (1.0 - p.probs) * ((1.0 - p.probs).log() - (1.0 - q.probs).log()))
    raise NotImplementedError(f"Analytical KL divergence between {type(p).__name__} and {type(q).__name__} not implemented.")