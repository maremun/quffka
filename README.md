# quffka

**QUFFKA** &mdash; QUadrature-based Features For Kernel Approximation. It
applies simple quadrature rules to generate random features for kernel
approximation.

Three simple steps to kernel approximation:

To prepare datasets, run:

```bash
    $ cook-datasets
```

To run kernel approximation (beware long computations! First please see
[examples](notebooks/Examples.ipynb)), run:

```bash
    $ approximate-kernels
```

To compare how fast explicit mapping is computed, run:

```bash
    $ measure-time
```
