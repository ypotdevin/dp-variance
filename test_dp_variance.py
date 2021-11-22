import itertools
import math
from typing import Optional, Sequence

import numpy as np

import dp_variance


def naive_k_min_variance_subset(
        k: int,
        sample: Sequence[float]
    ) -> np.ndarray:
    canditate, var = None, math.inf
    for k_subset in itertools.combinations(sample, k):
        current_var = np.std(k_subset)
        if current_var < var:
            var = current_var
            canditate = np.array(list(k_subset))
    return canditate # type: ignore

def naive_k_max_variance_subset(
        k: int,
        sample: Sequence[float]
    ) -> np.ndarray:
    canditate, var = None, -math.inf
    for k_subset in itertools.combinations(sample, k):
        current_var = np.std(k_subset)
        if current_var > var:
            var = current_var
            canditate = np.array(list(k_subset))
    return canditate # type: ignore

def test_k_min_variance_subset_indices(
        n: int = 100,
        seed: Optional[int] = None
    ) -> None:
    rng = np.random.default_rng(seed = seed)
    for _ in range(n):
        k = rng.integers(low = 1, high = 15)
        sample = rng.random(size = 15) * 100 - 50
        sample.sort()
        subset2 = naive_k_min_variance_subset(k, sample)
        subset3 = sample[dp_variance.k_min_variance_subset_indices(k, sample)]
        assert np.array_equal(subset2, subset3)
        assert np.std(subset2) == np.std(subset3)


def test_k_max_variance_subset_indices(
        n: int = 100,
        seed: Optional[int] = None
    ) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = rng.integers(low = 1, high = 15)
        if k == 1:
            continue
        sample = rng.random(size = 15) * 100 - 50
        sample.sort()
        subset2 = naive_k_max_variance_subset(k, sample)
        subset3 = sample[dp_variance.k_max_variance_subset_indices(k, sample)]
        assert np.array_equal(subset2, subset3)
        assert np.std(subset2) == np.std(subset3)

def test__recurrent_std(n: int = 100, seed: Optional[int] = None) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        sample = rng.random(size = n) - 0.5
        var_without_e = np.var(sample[:-1])
        mean_without_e = np.mean(sample[:-1])
        e = sample[-1]
        std = dp_variance._recurrent_std(e, mean_without_e, var_without_e, n)
        assert np.isclose(std, np.std(sample))

def test__mean_var_without(n: int = 100, seed: Optional[int] = None) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = rng.integers(low = 0, high = n)
        sample = rng.random(size = n) - 0.5
        sample_var = np.var(sample)
        sample_mean = np.mean(sample)
        e = sample[k]
        (mean, var) = dp_variance._mean_var_without(
            e = e,
            var_with_e = sample_var,
            mean_with_e = sample_mean,
            n_with_e = n
        )
        sample_without_e = np.delete(sample, k)
        assert np.isclose(mean, np.mean(sample_without_e))
        assert np.isclose(var, np.var(sample_without_e))

def _naive_local_sensitivity(
        sample: np.ndarray,
        L: float,
        U: float,
        mean: float
    ) -> float:
    std = np.std(sample)
    dist_from_std = 0.0
    for i in range(sample.size):
        old_value = sample[i]
        sample[i] = L
        v1 = abs(std - np.std(sample))
        sample[i] = U
        v2 = abs(std - np.std(sample))
        sample[i] = mean
        v3 = abs(std - np.std(sample))
        dist_from_std = max([dist_from_std, v1, v2, v3])
        sample[i] = old_value
    local_sens = dist_from_std
    return local_sens

def test_tunings(n: int = 200, seed: Optional[int] = None) -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    L = -0.5
    U = 0.5
    mean = 0.0
    for _ in range(n):
        k = rng.integers(low = 0, high = n)
        sample = rng.random(size = n) - 0.5
        sample.sort()

        wcn = dp_variance._worst_case_k_neighbor(
            k, sample, 'max_var', L, U, mean, np.std
        )
        ls1 = _naive_local_sensitivity(wcn, L, U, mean)
        ls2 = dp_variance._local_sensitivity(wcn, k, L, U, mean, np.std)
        assert(np.isclose(ls1, ls2))

        wcn = dp_variance._worst_case_k_neighbor(
            k, sample, 'min_var', L, U, mean, np.std
        )
        ls1 = _naive_local_sensitivity(wcn, L, U, mean)
        ls2 = dp_variance._local_sensitivity(wcn, k, L, U, mean, np.std)
        assert(np.isclose(ls1, ls2))
