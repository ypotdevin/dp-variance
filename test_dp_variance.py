import itertools
import math
from typing import Callable, Optional, Sequence

import numpy as np

import dp_variance


def naive_k_min_variance_subset(
        k: int,
        sample: Sequence[float]
    ) -> np.ndarray:
    canditate, var = None, math.inf
    for k_subset in itertools.combinations(sample, k):
        _k_subset = np.array(k_subset)
        current_var = np.var(_k_subset)
        if current_var < var:
            var = current_var
            canditate = _k_subset
    return canditate # type: ignore

def naive_k_max_variance_subset(
        k: int,
        sample: Sequence[float]
    ) -> np.ndarray:
    canditate, var = None, -math.inf
    for k_subset in itertools.combinations(sample, k):
        current_var = np.var(k_subset)
        if current_var > var:
            var = current_var
            canditate = np.array(list(k_subset))
    return canditate # type: ignore

def test_k_min_variance_subset_indices(
        n: int = 200,
        seed: Optional[int] = None
    ) -> None:
    rng = np.random.default_rng(seed = seed)
    print(rng)
    for _ in range(n):
        k = rng.integers(low = 1, high = 15)
        sample = rng.random(size = 15) * 100 - 50
        sample.sort()
        subset2 = naive_k_min_variance_subset(k, sample)
        subset3 = sample[dp_variance.k_min_variance_subset_indices(k, sample)]
        assert np.array_equal(subset2, subset3)


def test_k_max_variance_subset_indices(
        n: int = 200,
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

def test__mean_var_with(n: int = 100, seed: Optional[int] = None) -> None:
    _test__mean_var_with(2, 1.0, seed)
    _test__mean_var_with(3, 10.0, seed)
    _test__mean_var_with(n, 1e-100, seed)
    _test__mean_var_with(n, 1.0, seed)
    _test__mean_var_with(n, 1e100, seed)

def _test__mean_var_with(n: int, scale: float, seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        sample = (rng.random(size = n) - 0.5) * scale
        var_without_e = np.var(sample[:-1])
        mean_without_e = np.mean(sample[:-1])
        e = sample[-1]
        (mean, var) = dp_variance._mean_var_with(
            e, mean_without_e, var_without_e, n
        )
        assert np.isclose(mean, np.mean(sample), rtol = 1e-9, atol = 1e-18)
        assert np.isclose(var, np.var(sample), rtol = 1e-9, atol = 1e-18)

def test_iteration__mean_var_with(
        n: int = 100, seed: Optional[int] = None
    ) -> None:
    _test_iteration__mean_var_with(1, seed)
    _test_iteration__mean_var_with(2, seed)
    _test_iteration__mean_var_with(3, seed)
    _test_iteration__mean_var_with(n, seed)


def _test_iteration__mean_var_with(n: int, seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    sample = rng.random(size = n) - 0.5
    online_mean = sample[0]
    online_var = 0.0
    for i in range(1, len(sample)):
        (online_mean, online_var) = dp_variance._mean_var_with(
            sample[i], online_mean, online_var, i + 1
        )
    assert np.isclose(online_mean, np.mean(sample), rtol = 1e-9, atol = 1e-18)
    assert np.isclose(online_var, np.var(sample), rtol = 1e-9, atol = 1e-18)

def test__mean_var_without(n: int = 100, seed: Optional[int] = None) -> None:
    _test__mean_var_without(2, 1.0, seed)
    _test__mean_var_without(3, 10.0, seed)
    _test__mean_var_without(n, 1e-100, seed)
    _test__mean_var_without(n, 1.0, seed)
    _test__mean_var_without(n, 1e100, seed)

def _test__mean_var_without(n: int, scale: float, seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = rng.integers(low = 0, high = n)
        sample = (rng.random(size = n) - 0.5) * scale
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
        assert np.isclose(
            mean, np.mean(sample_without_e), rtol = 1e-9, atol = 1e-18
        )
        assert np.isclose(
            var, np.var(sample_without_e), rtol = 1e-9, atol = 1e-18
        )

def test_iteration__mean_var_without(
        n: int = 100, seed: Optional[int] = None
    ) -> None:
    _test_iteration__mean_var_without(1, seed)
    _test_iteration__mean_var_without(2, seed)
    _test_iteration__mean_var_without(3, seed)
    _test_iteration__mean_var_without(n, seed)

def _test_iteration__mean_var_without(n: int, seed: Optional[int]) -> None:
    rng = np.random.default_rng(seed)
    sample = rng.random(size = n) - 0.5
    online_mean = np.mean(sample)
    online_var = np.var(sample)
    for i in range(len(sample) - 2):
        (online_mean, online_var) = dp_variance._mean_var_without(
            sample[i], online_mean, online_var, n - i
        )
    assert np.isclose(online_mean, np.mean(sample[-2:]), rtol = 1e-9, atol = 1e-18)
    assert np.isclose(online_var, np.var(sample[-2:]), rtol = 1e-9, atol = 1e-18)

def test_composition__mean_var_with_without(
        n: int = 100, scale: float = 1.0, seed: Optional[int] = None
    ) -> None:
    _test_composition__mean_var__mean_var_without(2, scale, seed)
    _test_composition__mean_var__mean_var_without(3, scale, seed)
    _test_composition__mean_var__mean_var_without(4, scale, seed)
    _test_composition__mean_var__mean_var_without(n, scale, seed)

def _test_composition__mean_var__mean_var_without(
        n: int, scale: float, seed: Optional[int]
    ) -> None:
    rng = np.random.default_rng(seed)
    sample = (rng.random(size = n) - 0.5) * scale
    mean = np.mean(sample)
    var = np.var(sample)
    for i in range(n - 2):
        (mean, var) = dp_variance._mean_var_without(sample[i], mean, var, n - i)
    for i in range(n - 2):
        (mean, var) = dp_variance._mean_var_with(sample[i], mean, var, i + 3)
    assert np.isclose(mean, np.mean(sample), rtol = 1e-9, atol = 1e-18)
    assert np.isclose(var, np.var(sample), rtol = 1e-9, atol = 1e-18)

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

def test__local_sensitivity(n: int = 200, seed: Optional[int] = None) -> None:
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

def _naive_worst_case_k_neighbor(
        k: int,
        sample: np.ndarray,
        mode: str,
        L: float,
        U: float,
        mean: float,
        dispersion: Callable[[np.ndarray], float]
    ) -> np.ndarray:
    """Compute k-neighbors which are good candidate for having the
    maximal local sensitivity among all k-neighbors of `sample`.
    """
    if k == 0:
        return sample
    if mode == 'max_var':
        max_var_indices = dp_variance.k_max_variance_subset_indices(k, sample)
        max_var_complement = sample[
            dp_variance._complement(len(sample), max_var_indices)
        ]
        # It doesn't matter where I add the new values, as the variance
        # will stay the same.
        worst_case = np.concatenate([np.tile(mean, k), max_var_complement])
    elif mode == 'min_var':
        min_var_indices = dp_variance.k_min_variance_subset_indices(k, sample)
        min_var_complement = sample[
            dp_variance._complement(len(sample), min_var_indices)
        ]
        worst_case_candidates = (
            np.concatenate([extr, min_var_complement])
            for extr in dp_variance._extreme_value_combinations(k, L, U)
        )
        worst_case = max(
            worst_case_candidates, key = lambda seq: dispersion(seq)
        ) # type: ignore
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    return worst_case # type: ignore

def test__worst_case_k_neighbor(
        n: int = 100, seed: Optional[int] = None
    ) -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    L = -0.5
    U = 0.5
    mean = 0.0
    for _ in range(n):
        k = rng.integers(low = 0, high = n)
        sample = rng.random(size = n) - 0.5
        sample.sort()
        wcn1 = _naive_worst_case_k_neighbor(k, sample, 'max_var', L, U, mean, np.var)
        wcn2 = dp_variance._worst_case_k_neighbor(k, sample, 'max_var', L, U, mean, np.var)
        assert np.array_equal(wcn1, wcn2)
        wcn3 = _naive_worst_case_k_neighbor(k, sample, 'min_var', L, U, mean, np.var)
        wcn4 = dp_variance._worst_case_k_neighbor(k, sample, 'min_var', L, U, mean, np.var)
        assert np.array_equal(wcn3, wcn4)
