# scioxide-core

Scalar types, checked arithmetic, errors and execution shared by all SciOxide crates.
It also provides correctly rounded sums, log-sum-exp and softmax without overflow, trigonometry in units of π, values with an extended exponent range, and exact combinatorial counts.

## Main entry points

- Scalars: `RealScalar`, `NumericScalar`, `Complex`, `AffineValue`.
- Errors: `SciOxideError`, `SciOxideErrorKind`, `Argument`, `Quantity`.
- Tolerances and limits: `Tolerance`, `StepTolerance`, `ResidualTolerance`, `Budget`, `budget!`, `Interval`, `ExtendedInterval`, `Bounds`.
- Checked arithmetic: `checked_sum`, `checked_sum_of_products`, `checked_midpoint`, `CompensatedSum`, `ScaledNorm`.
- Extended range: `Scaled`, `ScaledAccumulator`, `LogValue`.
- Stable functions: `logsumexp`, `softmax`, `log_softmax`, `sin_pi`, `sincpi`, `cosm1`, `exprel`, `log1mexp`, `xlogy`, `sigmoid`; along an axis of an array, `softmax_axis` and `logsumexp_axis`.
- Combinatorics: `checked_binomial`, `checked_factorial`, `Permutation`, `CombinationEnumerator`.
- Arrays and operators: `StridedLayout`, `NdView`, `NdView::row_major`, `NdViewMut`, `LinearOperator`, `Preconditioner`, `ClosureOperator`, `ClosurePreconditioner`.
- Execution and solver accounting: `ExecutionContext`, `ExecutionPolicy`, `ThreadExecutor`, `SolveState`, `SolveContext`, `SolveBudgets`.

```rust
fn main() -> Result<(), scioxide_core::SciOxideError> {
    // The exact sum is 1; naive left-to-right summation returns 0.
    assert_eq!(scioxide_core::checked_sum(&[1.0e100_f64, 1.0, -1.0e100])?, 1.0);
    Ok(())
}
```

## Features

`std` (default) adds `ThreadExecutor`, `StdClock`, `StdProcessor` and runtime CPU feature detection.
Without it the crate is `no_std` and needs `alloc`.

The [API documentation](https://docs.rs/scioxide-core) states the rules on errors, non-finite values and execution that apply to every SciOxide crate.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
