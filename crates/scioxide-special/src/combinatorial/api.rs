//! Public combinatorial entry points.

use super::*;
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the multifactorial `n!⁽ᵏ⁾ = n (n − k) (n − 2k) ⋯`, the product of the positive terms,
where `k` is `step`.

The empty product `n = 0` is 1. `step = 1` is [`crate::factorial`] and `step = 2` is
[`crate::double_factorial`]. Products of at most 128 factors are formed in double-word
arithmetic. Longer ones use

```text
n!⁽ᵏ⁾ = kᵐ Γ(m + r/k) / Γ(r/k),
```

where `m = ⌈n/k⌉` is the number of factors and `r = n − (m − 1)k` is the smallest factor.

# Examples

```
// 10!⁽³⁾ = 10 · 7 · 4 · 1 = 280
let m = scioxide_special::multifactorial::<f64>(10, 3)?;
assert!((m - 280.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `step = 0`.
- `NumericalInstability` (`Overflow`) if the product exceeds the range of the scalar type.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub multifactorial [S: RealScalar] (n: usize, step: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed multifactorial_with_options -> Report<S>;
    evaluate |options, context| {
        run(
            CombinatorialQuery::Multifactorial(n, step),
            options,
            &mut [],
            None,
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Pochhammer symbol (rising factorial) `(a)ₓ = Γ(a + x) / Γ(a)` (DLMF §5.2(iii)).

For a nonnegative integer `x = n` this is the polynomial `a (a + 1) ⋯ (a + n − 1)`, including
its zeros, and `(a)₀ = 1`. For a negative integer `x = −n` it is
`1 / ((a − 1) (a − 2) ⋯ (a − n))`. Where `Γ(a)` has a pole and `Γ(a + x)` does not, the result
is zero. Complex arguments follow the cut conventions of [`crate::ln_gamma`].

Integer shifts with `|x| ≤ 128` are products. Other shifts use a ratio of Stirling series, or of
gamma functions, with the reflection formula on the negative axis; the shift is kept separate
from the base so that `a + x` loses no digits. [`crate::pochhammer_with_options`] keeps the
logarithm of the magnitude and the sign or phase separately; this is not the analytic complex
logarithm.

# Examples

```
// (2)₃ = 2 · 3 · 4 = 24
let p = scioxide_special::pochhammer(2.0_f64, 3.0)?;
assert!((p - 24.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `a + x = 0, −1, −2, …` and `x` is not a nonnegative integer.
- `NumericalInstability` (`Overflow`) if `|(a)ₓ|` exceeds the range of the scalar type.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub pochhammer [T: NumericScalar] (a: T, x: T) -> T
    where [];
    options options: CombinatorialOptions;
    detailed pochhammer_with_options -> Report<T>;
    evaluate |options, context| {
        run(CombinatorialQuery::Pochhammer(a, x), options, &mut [], None)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `((a)ₓ − 1) / x` without cancellation for small `x`, where `(a)ₓ` is
[`crate::pochhammer`].

At `x = 0` the result is the limit `ψ(a)`, the digamma function. Elsewhere the domain and the
limits of [`crate::pochhammer`] apply, and complex arguments follow its conventions.

# Examples

```
// ((2)₁ − 1) / 1 = 2 − 1 = 1
let r = scioxide_special::pochhammer_relative(2.0_f64, 1.0)?;
assert!((r - 1.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `x = 0` and `a = 0, −1, −2, …`.
- `DomainError` (`Pole`) if `a + x = 0, −1, −2, …` and `x` is not a nonnegative integer.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub pochhammer_relative [T: NumericScalar] (a: T, x: T) -> T
    where [];
    options options: CombinatorialOptions;
    detailed pochhammer_relative_with_options -> Report<T>;
    evaluate |options, context| {
        run(
            CombinatorialQuery::PochhammerRelative(a, x),
            options,
            &mut [],
            None,
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the binomial coefficient `C(z, k) = z (z − 1) ⋯ (z − k + 1) / k!` for real or complex
`z` and integer `k ≥ 0`.

`C(z, k)` is a polynomial of degree `k` in `z`, with no poles or branch cuts, and `C(z, 0) = 1`.
The numerator is a product with a separate binary exponent, so it does not overflow before the
single division by `k!`. The work grows linearly with `k`.

# Examples

```
// C(5, 2) = 10
let c = scioxide_special::generalized_binomial(5.0_f64, 2)?;
assert!((c - 10.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|C(z, k)|` exceeds the range of the scalar type.
- `ConvergenceFailure` if `k` exceeds the iteration budget.

# References

- NIST DLMF, §1.2(i). <https://dlmf.nist.gov/1.2.i>"###)]

    pub generalized_binomial [T: NumericScalar] (z: T, k: usize) -> T
    where [];
    options options: CombinatorialOptions;
    detailed generalized_binomial_with_options -> Report<T>;
    evaluate |options, context| {
        run(
            CombinatorialQuery::GeneralizedBinomial(z, k),
            options,
            &mut [],
            context,
        )
    }
    scoped pub generalized_binomial_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the signed Stirling number of the first kind `s(n, k)`, the coefficient of `xᵏ` in
`x (x − 1) ⋯ (x − n + 1)`.

`s(n, n) = 1`, `s(n, 0) = 0` for `n > 0`, and `s(n, k) = 0` for `k > n`. The sign is `(−1)ⁿ⁻ᵏ`.
The triangular recurrence `s(n + 1, k) = s(n, k − 1) − n s(n, k)` is run on one column of
`k + 1` scaled values, at a cost of `O(k (n − k))` operations. Each call allocates this column.

# Examples

```
// x (x − 1) (x − 2) = x³ − 3x² + 2x
let s = scioxide_special::stirling_first_kind::<f64>(3, 2)?;
assert!((s + 3.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `0 < k < n` and `n` is at least the iteration budget of
  [`crate::CombinatorialOptions`] (65,536 by default).
- `ConvergenceFailure` if the `O(k (n − k))` recurrence exceeds the iteration budget.
- `SizeOverflow` or `AllocationFailure` if the buffer of `k + 1` values cannot be allocated.
- `NumericalInstability` (`Overflow`) if the result exceeds the range of the scalar type.

# References

- NIST DLMF, §26.8. <https://dlmf.nist.gov/26.8>"###)]

    pub stirling_first_kind [S: RealScalar] (n: usize, k: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed stirling_first_kind_with_options -> Report<S>;
    evaluate |options, context| {
        stirling_first_kind_evaluate(n, k, options, context)
    }
    scoped pub stirling_first_kind_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn stirling_first_kind_evaluate<S: RealScalar>(
    n: usize,
    k: usize,
    options: CombinatorialOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<CombinatorialEvaluation<S>, SciOxideError> {
    let query = CombinatorialQuery::StirlingFirst(n, k);
    if n == k || k == 0 || k > n {
        return run(query, options, &mut [], context);
    }
    if k >= options.iteration_budget().get() || n - k >= options.iteration_budget().get() - k {
        let mut w = Work::with_context(options.iteration_budget().get(), context);
        w.check()?;
        w.complete = false;
        return report(
            Err(invalid(
                scioxide_core::argument!("options.iteration_budget"),
                "recurrence exceeds work allowance",
            )),
            &w,
        );
    }
    let mut cells = try_filled(
        k + 1,
        Scaled::ZERO,
        scioxide_core::quantity!("stirling_workspace"),
    )?;
    run(query, options, &mut cells, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Stirling number of the second kind `S(n, k)`, the number of partitions of `n`
objects into `k` nonempty unlabeled subsets.

`S(n, n) = 1`, `S(n, 0) = 0` for `n > 0`, `S(n, 1) = 1` for `n > 0`, and `S(n, k) = 0` for
`k > n`. The triangular recurrence `S(n + 1, k) = k S(n, k) + S(n, k − 1)` is run on one column
of `k + 1` scaled values, at a cost of `O(k (n − k))` operations. Each call allocates this
column.

# Examples

```
// {1, 2, 3} splits into two blocks in 3 ways
let s = scioxide_special::stirling_second_kind::<f64>(3, 2)?;
assert!((s - 3.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `0 < k < n` and `n` is at least the iteration budget of
  [`crate::CombinatorialOptions`] (65,536 by default).
- `ConvergenceFailure` if the `O(k (n − k))` recurrence exceeds the iteration budget.
- `SizeOverflow` or `AllocationFailure` if the buffer of `k + 1` values cannot be allocated.
- `NumericalInstability` (`Overflow`) if the result exceeds the range of the scalar type.

# References

- NIST DLMF, §26.8. <https://dlmf.nist.gov/26.8>"###)]

    pub stirling_second_kind [S: RealScalar] (n: usize, k: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed stirling_second_kind_with_options -> Report<S>;
    evaluate |options, context| {
        stirling_second_kind_evaluate(n, k, options, context)
    }
    scoped pub stirling_second_kind_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn stirling_second_kind_evaluate<S: RealScalar>(
    n: usize,
    k: usize,
    options: CombinatorialOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<CombinatorialEvaluation<S>, SciOxideError> {
    let query = CombinatorialQuery::StirlingSecond(n, k);
    if n == k || k == 0 || k > n {
        return run(query, options, &mut [], context);
    }
    if k >= options.iteration_budget().get() || n - k >= options.iteration_budget().get() - k {
        let mut w = Work::with_context(options.iteration_budget().get(), context);
        w.check()?;
        w.complete = false;
        return report(
            Err(invalid(
                scioxide_core::argument!("options.iteration_budget"),
                "recurrence exceeds work allowance",
            )),
            &w,
        );
    }
    let mut cells = try_filled(
        k + 1,
        Scaled::ZERO,
        scioxide_core::quantity!("stirling_workspace"),
    )?;
    run(query, options, &mut cells, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln(n!!)`, the logarithm of [`crate::double_factorial`], which does not overflow.

`ln(0!!) = ln(1!!) = 0`.

# Examples

```
// 9!! = 945
let v = scioxide_special::ln_double_factorial::<f64>(9)?;
assert!((v - 945.0_f64.ln()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"Every `n` is accepted; no error is specific to this function.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub ln_double_factorial [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed ln_double_factorial_with_options -> Report<S>;
    evaluate |options, context| {
        log_query(CombinatorialQuery::DoubleFactorial(n), options)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln(n!⁽ᵏ⁾)`, the logarithm of [`crate::multifactorial`], which does not overflow.

`k` is `step`. The empty product `n = 0` has logarithm 0.

# Examples

```
// 10!⁽³⁾ = 280
let v = scioxide_special::ln_multifactorial::<f64>(10, 3)?;
assert!((v - 280.0_f64.ln()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `step = 0`.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub ln_multifactorial [S: RealScalar] (n: usize, step: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed ln_multifactorial_with_options -> Report<S>;
    evaluate |options, context| {
        log_query(CombinatorialQuery::Multifactorial(n, step), options)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln C(n, k)`, the logarithm of the binomial coefficient [`crate::binomial`], without
forming the coefficient.

`0 ≤ k ≤ n`, and `ln C(n, 0) = ln C(n, n) = 0`.

# Examples

```
// C(5, 2) = 10
let v = scioxide_special::ln_binomial::<f64>(5, 2)?;
assert!((v - 10.0_f64.ln()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `k > n`.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub ln_binomial [S: RealScalar] (n: usize, k: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed ln_binomial_with_options -> Report<S>;
    evaluate |options, context| {
        log_query(CombinatorialQuery::Binomial(n, k), options)
    }

}

/// Logarithmic report of `query`.
fn log_query<S: RealScalar>(
    query: CombinatorialQuery<S>,
    options: CombinatorialOptions,
) -> Result<CombinatorialEvaluation<S>, SciOxideError> {
    query.validate()?;
    let mut w = Work::new(options.iteration_budget().get());
    let candidate = query.evaluate(&mut [], &mut w).and_then(|c| {
        let l = c
            .logarithmic
            .log_magnitude()
            .ok_or_else(|| ERRORS.domain(scioxide_core::argument!("n"), DomainReason::Undefined))?;
        Candidate::value(
            real(l),
            if l == 0.0 { 0.0 } else { c.log_error / l.abs() },
            algorithm_component!("combinatorial_logarithm"),
        )
    });
    report(candidate, &w)
}
/// Log-factorial report.
pub(crate) fn ln_factorial_report<S: RealScalar>(
    n: usize,
    options: CombinatorialOptions,
) -> Result<CombinatorialEvaluation<S>, SciOxideError> {
    let mut w = Work::new(options.iteration_budget().get());
    let c = if n <= 1 {
        Candidate::value(ZERO, 0.0, algorithm_component!("log_factorial_endpoint"))
    } else {
        factorial_log(n, &mut w).and_then(|l| {
            Candidate::value(
                real(l.value()),
                4.0 * EPS,
                algorithm_component!("log_factorial_gamma"),
            )
        })
    };
    report(c, &w)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the factorial `n! = 1 · 2 ⋯ n = Γ(n + 1)`, with `0! = 1` (DLMF 5.4.1).

Values through `20!` come from an exact table. Larger `n ≤ 128` are products in double-word
arithmetic, and `n > 128` uses `ln Γ(n + 1)`. [`crate::ln_factorial`] gives `ln(n!)` without
overflow.

# Examples

```
// 5! = 120
let f = scioxide_special::factorial::<f64>(5)?;
assert!((f - 120.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range, and `n! ≤ 20!` is
exact in `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `n > 170` (`f64`) or `n > 34` (`f32`).

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub factorial [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed factorial_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        crate::combinatorial::run(CombinatorialQuery::Factorial(n), options, &mut [], None)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln(n!) = ln Γ(n + 1)`, which does not overflow for any `n`.

`ln(0!) = ln(1!) = 0`. Other values use the log-gamma function in double-word arithmetic.

# Examples

```
// ln(5!) = ln 120
let v = scioxide_special::ln_factorial::<f64>(5)?;
assert!((v - 120.0_f64.ln()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"Every `n` is accepted; no error is specific to this function.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub ln_factorial [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed ln_factorial_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        crate::combinatorial::ln_factorial_report(n, options)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the binomial coefficient `C(n, k) = n! / (k! (n − k)!)` for `0 ≤ k ≤ n` (DLMF 26.3.1).

`C(n, 0) = C(n, n) = 1`. If `min(k, n − k) < 128` the coefficient is a product in double-word
arithmetic; otherwise it uses `C(n, k) = 1 / ((n + 1) B(n − k + 1, k + 1))` through the
logarithm of the beta function. The result is rounded to the scalar type, so large coefficients
are not exact integers. [`crate::ln_binomial`] gives `ln C(n, k)` without overflow.

# Examples

```
// C(10, 3) = 120
let c = scioxide_special::binomial::<f64>(10, 3)?;
assert!((c - 120.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `k > n`.
- `NumericalInstability` (`Overflow`) if `C(n, k)` exceeds the range of the scalar type.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub binomial [S: RealScalar] (n: usize, k: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed binomial_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        crate::combinatorial::run(CombinatorialQuery::Binomial(n, k), options, &mut [], None)
    }

}
