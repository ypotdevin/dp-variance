# dp-variance
Based on the work of Zafarani and Clifton (2021) in [Differentially Private Naive Bayes Classifier using Smooth Sensitivity](http://arxiv.org/abs/2003.13955), we implement a differentially private estimation of sample variance and standard deviation.

# Benchmark results (unimodal synthetic data)
We analyzed the quality of the differentially private sample variance
approximations of our implementation, based on synthetic samples drawn from Gaussian
and Laplacian distributions. For various 
locations $\mu \in \Set{10^{-3}, 10^{-1}, 0, 1, 10, 10^3, 10^6}$,
scales $\sigma \in \Set{0.1 \mu, \mu, 10 \mu}$,
sample sizes $n \in \Set{10, 100, 1000, 10000}$ and 
privacy budgets $\epsilon \in \Set{0.01, 0.1, 1.0, 10.0}$
we estimated the DP sample variance 10 times and show the statistics in [Table 1](table_1.md).
The lower bounds $L$ and upper bounds $U$ were set to the sample minimum and sample maximum, respectively.
The parameter β, (for the calculation of the β-smooth sensitivity) is set to $\frac{\epsilon}{6}$.

Averaged over the locations and scales, the approximation quality of our implementation is shown in [Table 2](table_2.md).

# Reproducability
The raw data ([benchmark.csv](benchmark.csv)) may be (up to randomness) reproduced via [benchmark.py](benchmark_rmse.py) and the aggregation of the resulting data may be reproduced by [dp-var-bench-aggregation.ipynb](dp-var-bench-aggregation.ipynb).
