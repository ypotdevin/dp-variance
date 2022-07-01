from typing import Any, cast, Iterable

import numpy as np
import pandas as pd  # type: ignore
from joblib import Parallel, delayed  # type: ignore

import dp_variance

BenchmarkResults = pd.DataFrame
HyperParameters = dict[str, Any]
SampleParameters = dict[str, Any]
Sample = np.ndarray


def params_generator() -> Iterable[SampleParameters]:
    sample_sizes = [10, 100, 1000, 10000]
    locations = [1e-3, 1e-1, 1e0, 1e1, 1e3, 1e6]

    for variant in ["normal", "laplace"]:
        for loc in locations:
            scales = [1e-1 * loc, 1e0 * loc, 1e1 * loc]
            for scale in scales:
                for n in sample_sizes:
                    yield dict(size=n, loc=loc, scale=scale, variant=variant)
        for n in sample_sizes:
            yield dict(size=n, loc=0.0, scale=1.0, variant=variant)


def sample(parameters: SampleParameters, rng: np.random.Generator,) -> Sample:
    variant_fun_dispatcher = dict(
        normal=rng.normal, laplace=rng.laplace, poisson=rng.poisson,
    )
    params = parameters.copy()
    variant = cast(str, params.pop("variant"))
    fun = variant_fun_dispatcher[variant]
    sample = cast(Sample, fun(**params))
    return sample


def _benchmark(sample: Sample, hyperparams: HyperParameters) -> BenchmarkResults:
    """Perform a single run of the DP variance estimation.

    Args:
        sample (Sample):
            The sample to calculate the DP sample variance for.
        hyperparams (HyperParameters):
            The hyperparameters to use for estimating the sample
            variance. Should contain the keys "epsilon", "beta", "m",
            "L", "U".

    Returns:
       BenchmarkResults: A line of benchmark results, shipped as a data
       frame. It contains the (true) sample variance, the DP estimate
       and key hyperparameters.
    """
    dp_var = dp_variance.dp_variance(sample=sample, mean=sample.mean(), **hyperparams)
    result = dict(variance=sample.var(), dp_variance=dp_var, **hyperparams)
    return pd.DataFrame(result, index=[0])


def benchmark(sample_params: SampleParameters) -> BenchmarkResults:
    n_repetitions = 10
    sampl = sample(sample_params, np.random.default_rng())
    results = []
    for epsilon in [0.01, 0.1, 1.0, 10]:
        for _ in range(n_repetitions):
            hyperparams = dict(
                epsilon=epsilon,
                beta=epsilon / 6,
                m=int(0.1 * len(sampl)),
                L=sampl.min(),
                U=sampl.max(),
            )
            print(f"Current hyperparameters: {hyperparams}")
            result = _benchmark(sampl, hyperparams)
            result = result.assign(**sample_params, **hyperparams)
            results.append(result)
    return pd.concat(results)


if __name__ == "__main__":
    sub_results = Parallel(n_jobs=8)(
        delayed(benchmark)(sample_params) for sample_params in params_generator()
    )
    df = pd.concat(sub_results)
    df.to_csv("benchmark.csv", index=False)
