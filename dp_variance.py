import itertools
import logging
import math
from typing import Callable, Iterator, Optional, Sequence, Tuple, cast

import numpy as np  # type: ignore


def dp_standard_deviation(
    sample: Sequence[float],
    mean: float,
    epsilon: float,
    beta: float,
    m: int,
    L: float,
    U: float,
    seed: Optional[int] = None,
) -> float:
    """
    Parameters
    ----------
    sample : Sequence[float]
        The sample to compute the standard deviation for, in a
        differentially private manner.
    mean : float
    epsilon : float
        The differential privacy parameter.
    beta : float
        The beta-smooth sensitivity parameter.
    m : int, optional
        The trimming parameter (discard the m smallest and m largest
        elements in `sample`.
    seed : int
        A random seed used to draw from the Cauchy distribution.
    """
    return _dp_disp("std", sample, mean, epsilon, beta, m, L, U, seed)


def dp_variance(
    sample: Sequence[float],
    mean: float,
    epsilon: float,
    beta: float,
    m: int,
    L: float,
    U: float,
    seed: Optional[int] = None,
) -> float:
    """
    Parameters
    ----------
    sample : Sequence[float]
        The sample to compute the variance for, in a differentially
        private manner.
    mean : float
    epsilon : float
        The differential privacy parameter.
    beta : float
        The beta-smooth sensitivity parameter.
    m : int, optional
        The trimming parameter (discard the m smallest and m largest
        elements in `sample`.
    seed : int
        A random seed used to draw from the Cauchy distribution.
    """
    return _dp_disp("var", sample, mean, epsilon, beta, m, L, U, seed)


def _dp_disp(
    dispersion: str,
    sample: Sequence[float],
    mean: float,
    epsilon: float,
    beta: float,
    m: int,
    L: float,
    U: float,
    seed: Optional[int] = None,
) -> float:
    if dispersion == "std":
        dispersion_fun = np.std
    elif dispersion == "var":
        dispersion_fun = np.var
    else:
        raise ValueError("Expected one of 'var' or 'std'. Got {}.".format(dispersion))
    assert epsilon > 0 and beta > 0 and m >= 0 and L < U
    logging.debug("calculating dp variance for %s", str(sample))
    _sample = np.array(sample)
    _sample.sort()
    trimmed_sample = _sample[m : len(sample) - m]
    s = smooth_sensitivity(trimmed_sample, mean, beta, L, U, dispersion)
    scale = math.sqrt(2) * s / epsilon
    rng = np.random.default_rng(seed=seed)
    noise = rng.standard_cauchy() * scale
    return dispersion_fun(trimmed_sample) + noise  # type: ignore


def smooth_sensitivity(
    sample: np.ndarray, mean: float, beta: float, L: float, U: float, dispersion: str
) -> float:
    """Compute the beta-smooth sensitivity of a dispersion function on
    the given sample.

    Parameters
    ----------
    sample : Sequence[float]
        The sample to compute the beta-smooth sensitivity of the std
        function on. It is assumed to be sorted.
    mean : float
    beta : float
        The decaying factor in the smooth sensitivity term.
    L : float
        The domain specific lower bound on the values of `sample`.
    U : float
        The domain specific upper bound on the values of `sample`.
    dispersion : str
        One of 'var' or 'std'.
    """
    if dispersion == "std":
        dispersion_fun = np.std
    elif dispersion == "var":
        dispersion_fun = np.var
    else:
        raise ValueError("Expected one of 'var' or 'std'. Got {}.".format(dispersion))
    assert L <= sample.min() and sample.max() <= U, "min={}, max={}".format(
        sample.min(), sample.max
    )
    logging.debug("calculating smooth sensitivity for %s", str(sample))
    local_sensitivities = _local_sensitivities(sample, L, U, mean, dispersion_fun)
    discounted_sensitivities = [
        loc_sens * math.exp(-beta * distance)
        for (loc_sens, distance) in local_sensitivities
    ]
    logging.debug("discounted sensitivities %s", str(discounted_sensitivities))
    return max(discounted_sensitivities)


def _local_sensitivities(
    sample: np.ndarray,
    L: float,
    U: float,
    mean: float,
    dispersion: Callable[[np.ndarray], float],
) -> Sequence[Tuple[float, int]]:
    """Compute the local sensitivities of all the relevant k-neighbors,
    for k = 0, …, n, of `sample."""
    logging.debug("calculating local sensitivities for %s", str(sample))
    sensitivities = []
    for k in range(sample.size + 1):
        wc_neighbor1 = _worst_case_k_neighbor(
            k, sample, "max_var", L, U, mean, dispersion
        )
        ls1 = _local_sensitivity(wc_neighbor1, k, L, U, mean, dispersion)
        wc_neighbor2 = _worst_case_k_neighbor(
            k, sample, "min_var", L, U, mean, dispersion
        )
        ls2 = _local_sensitivity(wc_neighbor2, k, L, U, mean, dispersion)

        sensitivities.append((max(ls1, ls2), k))
    logging.debug("local sensitivities %s", str(sensitivities))
    return sensitivities


def _worst_case_k_neighbor(
    k: int,
    sample: np.ndarray,
    mode: str,
    L: float,
    U: float,
    mean: float,
    dispersion: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Compute k-neighbors which are good candidate for having the
    maximal local sensitivity among all k-neighbors of `sample`.

    Notes
    -----
    Currently only the dispersions 'var' and 'std' are supported.
    """
    logging.debug(
        "calculating worst case k-neighbor for k=%d, sample=%s, mode=%s",
        k,
        str(sample),
        mode,
    )
    if k == 0:
        return sample
    if not dispersion in [np.var, np.std]:
        raise ValueError(
            "Dispersion {} is currently not supported. Use 'var' or "
            "'std' instead".format(dispersion)
        )
    # In this case it does not matter whether I calculate var or std,
    # as both of them yield the same worst case.
    if mode == "max_var":
        max_var_indices = k_max_variance_subset_indices(k, sample)
        max_var_complement = sample[_complement(len(sample), max_var_indices)]
        # It doesn't matter where I add the new values, as the variance
        # will stay the same.
        worst_case = np.concatenate([np.tile(mean, k), max_var_complement])
    elif mode == "min_var":
        min_var_indices = k_min_variance_subset_indices(k, sample)
        min_var_complement = sample[_complement(len(sample), min_var_indices)]

        evcs = _extreme_value_combinations(k, L, U)
        evc = next(evcs)
        worst_case = current_case = np.concatenate([evc, min_var_complement])
        worst_case_var = current_var = np.var(current_case)
        current_mean = np.mean(current_case)
        n = len(sample)
        for evc in evcs:
            # Use the fact that in each iteration, an U is replaced by L
            (previous_mean, previous_var) = _mean_var_without(
                U, current_mean, current_var, n
            )
            (current_mean, current_var) = _mean_var_with(
                L, previous_mean, previous_var, n
            )
            if current_var > worst_case_var:
                worst_case_var = current_var
                worst_case = np.concatenate([evc, min_var_complement])
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    logging.debug("worst case k-neighbor %s", str(worst_case))
    return worst_case  # type: ignore


def _complement(n: int, indices: np.ndarray) -> np.ndarray:
    """Return those indices among [0, 1, ..., n - 1], which do not occur
    in `indices`."""
    return np.setdiff1d(np.arange(n), indices)  # type: ignore


def _extreme_value_combinations(k: int, L: float, U: float) -> Iterator[np.ndarray]:
    """For each value of i in [0, 1, …, `k`], yield the sequence i times
    `L`, followed by (k - i) times `U`.

    Notes
    -----
    The iterator yields always the same object, which is manipulated in
    place, rather than yielding fresh arrays.
    """
    combination = np.tile(U, k)
    yield combination
    for i in range(k):
        combination[i] = L
        yield combination


def _local_sensitivity(
    sample: np.ndarray,
    k: int,
    L: float,
    U: float,
    mean: float,
    dispersion: Callable[[np.ndarray], float],
) -> float:
    logging.debug("calculating local sensitivity for %s", str(sample))
    disp = dispersion(sample)
    sample_mean = np.mean(sample)
    n = len(sample)
    dist_from_disp = 0.0
    for i in itertools.chain((0,), range(k - 1, sample.size)):
        if dispersion is np.std:
            base_mean, base_var = _mean_var_without(
                sample[i], sample_mean, disp ** 2, n
            )
            dists = [
                abs(disp - math.sqrt(_mean_var_with(e, base_mean, base_var, n)[1]))
                for e in [L, U, mean]
            ]
        elif dispersion is np.var:
            base_mean, base_var = _mean_var_without(sample[i], sample_mean, disp, n)
            dists = [
                abs(disp - _mean_var_with(e, base_mean, base_var, n)[1])
                for e in [L, U, mean]
            ]
        else:
            raise ValueError()
        dist_from_disp = max([dist_from_disp] + dists)
    local_sens = dist_from_disp
    logging.debug("local sensitivity %f", local_sens)
    return local_sens


def _mean_var_with(
    e: float, previous_mean: float, previous_var: float, current_n: int
) -> Tuple[float, float]:
    """Calculate the sample mean and variance of a sample where `e` has
    been added to, based on the previous sample mean and variance.

    Returns
    -------
    (mean, var) : (float, float)
        The sample mean and variance of a sample where `e` has
        (implicitly) been added to.

    Notes
    -----
    Implementation based on
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance,
    therefore the re-evaluation of the whole sample is not needed.
    """
    new_mean = previous_mean + (e - previous_mean) / current_n
    new_var = (
        previous_var + ((e - previous_mean) * (e - new_mean) - previous_var) / current_n
    )
    if new_var < 0.0:
        raise ArithmeticError(
            "Numerical instability detected. Variance is negative. "
            "e={}, previous_mean={}, previous_var={}, current_n={}, "
            "mean={}, var={}".format(
                e, previous_mean, previous_var, current_n, new_mean, new_var
            )
        )
    return (new_mean, new_var)


def _mean_var_without(
    e: float, mean_with_e: float, var_with_e: float, n_with_e: int
) -> Tuple[float, float]:
    """Given a sample variance and a corresponding sample mean, both
    including element `e`, calculate the mean and variance of the sample
    without element `e`.

    Returns
    -------
    (mean, var) : (float, float)
        The mean an variance which remain when the effect of `e` is
        removed from `var_with_e` and `mean_with_e`.

    Notes
    -----
    This implementation is ased on
    https://stackoverflow.com/a/30876815/3389669 and therefore does not
    need to reiterate over the whole sample again.
    """
    if n_with_e < 2:
        raise ValueError("Mean and variance undefined for (to be) empty sequence.")
    mean_without_e = mean_with_e - (e - mean_with_e) / (n_with_e - 1)
    if n_with_e == 2:
        return (mean_without_e, 0.0)
    # The first summand is multiplied by `n_with_e`, because the
    # aggregate in the cited formula is just the sum of squared
    # differences, not their average.
    var_without_e = (
        var_with_e * n_with_e - (e - mean_without_e) * (e - mean_with_e)
    ) / (n_with_e - 1)
    if var_without_e < 0.0:
        logging.warning(
            "Numerical instability detected. Variance is negative. "
            "e={}, mean_with_e={}, var_with_e={}, n_with_e={}, "
            "mean={}, var={}. Setting variance to 0.0.".format(
                e, mean_with_e, var_with_e, n_with_e, mean_without_e, var_without_e
            )
        )
        return (mean_without_e, 0.0)
    else:
        return (mean_without_e, var_without_e)


def k_min_variance_subset_indices(k: int, sample: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    sample:
        An ascendingly sorted sequence to extract the indices of a
        k-minimal variance subset from.

    Returns:
    --------
    indices : np.ndarray
        The indices of the variance minimizing subset of `sample`.
        """
    if k == 0:
        return np.array([])
    elif k == 1:
        return np.array([0])

    n = len(sample)
    assert 0 <= k <= n
    indices = np.arange(n)
    min_var_i = 0
    min_var = current_var = np.var(sample[0:k])
    current_mean = np.mean(sample[0:k])
    for i in range(1, n - k + 1):
        (previous_mean, previous_var) = _mean_var_without(
            sample[i - 1], current_mean, current_var, k
        )
        (current_mean, current_var) = _mean_var_with(
            sample[i + k - 1], previous_mean, previous_var, k
        )
        if current_var < min_var:
            min_var = current_var
            min_var_i = i
    return cast(np.ndarray, indices[min_var_i : min_var_i + k])


def k_max_variance_subset_indices(k: int, sample: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    sample:
        An ascendingly sorted sequence to extract the indices of a
        k-maximal variance subset from.

    Returns:
    --------
    indices : np.ndarray
        The indices of the variance maximizing subset of `sample`.
    """
    if k == 0:
        return np.array([])
    elif k == 1:
        return np.array([0])

    n = len(sample)
    assert 0 <= k <= n
    indices = np.arange(n)
    max_var_i = 0
    max_var = current_var = np.var(sample[n - k : n])
    current_mean = np.mean(sample[n - k : n])
    for i in range(1, k + 1):
        (previous_mean, previous_var) = _mean_var_without(
            sample[n - k + i - 1], current_mean, current_var, k
        )
        (current_mean, current_var) = _mean_var_with(
            sample[i - 1], previous_mean, previous_var, k
        )
        if current_var > max_var:
            max_var = current_var
            max_var_i = i
    return cast(
        np.ndarray,
        np.concatenate([indices[0:max_var_i], indices[n - k + max_var_i : n]]),
    )


def speed_test() -> None:
    rng = np.random.default_rng(42)
    big_sample = rng.standard_normal(4500)
    s = smooth_sensitivity(big_sample, 0, 0.2, -5, 5, "std")
    print("Smooth sensitivity of {} element array: {}".format(len(big_sample), s))


if __name__ == "__main__":
    speed_test()
