<h1 align='center'>PINN for Shell Structures - JAX</h1>

This repository provides unofficial, partial JAX-implementation for ['Physics-Informed Neural Networks for shell structures'](https://doi.org/10.1016/j.euromechsol.2022.104849).

By re-writing the [original code](https://github.com/jhbastek/PhysicsInformedShellStructures/tree/67adef6b9afc9996ed1dd82e1056b3ed41e49c87), we could obtain substantial speed-ups.
Please consult with `result_jax_x64.txt` and `result_torch_x64.txt` for a performance comparison.

### How to run
Training a neural network in double-precision: `JAX_ENABLE_X64=True python main.py`.

Measuring relative $L^2$ error:
`python err_measure.py --method=lbfgs`


### Requirements
Please consult with `requirements.txt`.


### Current Implementation
1. Fully clamped hyperbolic paraboloid
    - [data](https://github.com/jhbastek/PhysicsInformedShellStructures/tree/main/FEM_sol)

### Reference
- [Physics-Informed Neural Networks for shell structures](https://doi.org/10.1016/j.euromechsol.2022.104849)
- [JAX](https://github.com/google/jax)
- [Equinox](https://github.com/patrick-kidger/equinox)
