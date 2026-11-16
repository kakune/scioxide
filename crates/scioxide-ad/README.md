# scioxide-ad

Automatic differentiation for `f32` and `f64` by number types.
Write the function once against `Real`, then choose the number: `f64` to evaluate, `Lanes` for SIMD batches, `Dual`, `MultiDual` or `Jet` for forward mode and Taylor coefficients, and `Var` on a caller-owned `Tape` for reverse mode.
Wrappers nest (`Var<Dual<f64>>` gives Hessian-vector products), and `batch_mean_and_grad` and `BatchPlan` average or sum a function over many independent items, such as Monte Carlo paths, with or without derivatives and with identical results for every worker count.

```rust
use scioxide_ad::{Dual, Real, grad};

fn model<R: Real>(x: R) -> R {
    x.sin() + x * x
}
let x = 0.4_f64;
let forward = model(Dual::variable(x)).derivative();
let reverse = grad(|p| Ok(model(p[0])), &[x])?;
let expected = x.cos() + 2.0 * x;
assert!((forward - expected).abs() < 1e-15);
assert!((reverse.as_slice()[0] - expected).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```

## Main entry points

- Number types: `Real`, `Lanes`, `Dual`, `MultiDual`, `Jet`, `Var`.
- One call per derivative: `grad`, `value_and_grad`, `vjp`, `jacobian`, `hvp`, `hessian`, `sparse_jacobian`, `sparse_hessian`, `jvp`, `jet`.
- Reverse mode with reuse: `Tape`, `Linearization`, `JacobianOperator`, `HessianOperator`.
- Batches: `BatchFunction`, `batch_mean`, `batch_mean_and_grad`, `BatchPlan`, `BatchWorkspace`, `BatchDataFn`.
- Structured operations: `solve_into`, `cholesky_into`, `qr_into`, `least_squares_into`, `logsumexp`, `softmax_into`, `implicit_solution_into`, `checkpointed_loop`, `ExternalRule`, `ComplexElementary`.

Lanes vectorize for the target features of the build: compile with `-C target-cpu=native` (or `x86-64-v3`) for AVX2.

## Feature flags

- `std` (default): the `std` features of the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-ad) for the rules, conventions, accuracy and costs.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
