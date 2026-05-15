# flake8: noqa
from __future__ import annotations
import torch


torch.set_rng_state(
    [  # E: Argument 1 to "set_rng_state" has incompatible type "list[int]"; expected "Tensor"  [arg-type]
        1,
        2,
        3,
    ]
)
