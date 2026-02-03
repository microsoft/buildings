# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def weighted_train_val_split(all_data, weights, val_size, random_state=None):
    n_samples = len(all_data[0]) if isinstance(all_data, tuple) else len(all_data)
    val_count = int(n_samples * val_size)

    weights = np.array(weights)
    weights = weights / weights.sum()

    rng = np.random.RandomState(random_state)
    val_indices = rng.choice(n_samples, size=val_count, replace=False, p=weights)

    mask = np.ones(n_samples, dtype=bool)
    mask[val_indices] = False
    train_indices = np.nonzero(mask)[0]

    if isinstance(all_data, tuple):
        train_split = tuple([data[i] for i in train_indices] for data in all_data)
        val_split = tuple([data[i] for i in val_indices] for data in all_data)
        return train_split, val_split

    return [all_data[i] for i in train_indices], [all_data[i] for i in val_indices]


def apply_temperature(raw_weights, temperature, eps=1e-10):
    if temperature == 1.0:
        return raw_weights

    # Work in log space for numerical stability
    log_weights = np.log(raw_weights + eps)
    temp_log_weights = log_weights / temperature

    # Subtract max for numerical stability
    temp_log_weights = temp_log_weights - temp_log_weights.max()

    weights = np.exp(temp_log_weights)

    # Scale to match original magnitude at temperature=1
    # This makes the function continuous at temperature=1
    scale_factor = raw_weights.sum() / weights.sum()
    return weights * scale_factor
