| column                    | meaning |
|---------------------------|---------|
| $n$                       | size of the sample |
| $\epsilon$                | privacy budget |
| mean rel. deviation       | the average (over distribution, location, scale and the 10 i.i.d. runs) value of $\frac{\lvert\text{variance} - \text{DP variance}\rvert}{\text{variance}}$ |
| std of the rel. deviation | the standard devition of the above |

|   $n$ |   $\epsilon$ |   mean rel. deviation |   std of the rel. deviation |
|-------|--------------|-----------------------|-----------------------------|
|    10 |         0.01 |            338.865    |                1527.24      |
|    10 |         0.1  |             29.5522   |                 137.454     |
|    10 |         1    |              3.74609  |                  24.4914    |
|    10 |        10    |              0.687286 |                   1.00197   |
|   100 |         0.01 |            137.374    |                 949.683     |
|   100 |         0.1  |            312.458    |                5917.67      |
|   100 |         1    |              0.891666 |                   1.21015   |
|   100 |        10    |              0.658247 |                   0.358514  |
|  1000 |         0.01 |             14.1689   |                  54.3273    |
|  1000 |         0.1  |              1.66007  |                   4.78799   |
|  1000 |         1    |              0.691916 |                   0.44285   |
|  1000 |        10    |              0.668282 |                   0.437989  |
| 10000 |         0.01 |              3.00549  |                  10.1972    |
| 10000 |         0.1  |              0.707248 |                   0.581726  |
| 10000 |         1    |              0.68911  |                   0.916211  |
| 10000 |        10    |              0.644667 |                   0.0827664 |
