//! Levinson–Durbin recursion, circulant division, backward-error checks and the dense LU fallback.
use super::*;
use crate::checked::finite_candidate;
use crate::kernel::standard::{exponent, magnitude, scale};
use scioxide_core::backend::{checked_add, checked_product};

// Ordinary four-lane reductions avoid a dependency chain and permit native
// vectorization. No per-term checked/extended or library FMA operations.
#[inline]
fn dot<T: NumericScalar>(products: impl Iterator<Item = (T, T)>) -> T {
    let mut sums = [T::ZERO; 4];
    for (i, (a, b)) in products.enumerate() {
        sums[i % 4] = sums[i % 4] + a * b;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

#[inline]
fn add_product<T: NumericScalar>(a: T, b: T, c: T) -> T {
    a * b + c
}

pub(super) fn zeroed<T: NumericScalar>(len: usize) -> Result<Vec<T>, SciOxideError> {
    let mut values = scioxide_core::backend::try_with_capacity(
        len,
        scioxide_core::quantity!("structured_storage"),
    )?;
    values.resize(len, T::ZERO);
    Ok(values)
}

pub(super) fn storage_lengths<T: NumericScalar>(
    n: usize,
    compact: Compact,
) -> Result<(usize, usize, usize), SciOxideError> {
    let q = scioxide_core::quantity!("preparation_scalars");
    if n <= 1 {
        return Ok((n, 0, 0));
    }
    let coefficients = checked_product(
        &[
            n,
            if matches!(compact, Compact::Circulant | Compact::HermitianToeplitz) {
                1
            } else {
                2
            },
        ],
        q,
    )?;
    let phases = checked_product(
        &[
            n,
            if compact == Compact::Circulant { 1 } else { 2 },
            if T::IS_REAL { 2 } else { 1 },
        ],
        q,
    )?;
    let spectrum = if compact == Compact::Circulant { n } else { 0 };
    Ok((coefficients, phases, spectrum))
}

pub(super) fn instability(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Cancellation, detail)
}

fn divide<T: NumericScalar>(a: T, b: T) -> Result<T, SciOxideError> {
    if b == T::ZERO {
        return Err(SciOxideError::singular(None));
    }
    finite_candidate(a / b)
}

#[derive(Clone, Copy)]
enum Candidate {
    Computed,
    Rejected,
}

impl<T: NumericScalar> StructuredSolvePlan<'_, T> {
    pub(super) fn stride(&self) -> Result<usize, SciOxideError> {
        if self.order() == 0 {
            return Ok(0);
        }
        // Staged output, two work vectors, and deterministic error/path slots.
        checked_add(
            checked_product(
                &[self.order(), 3],
                scioxide_core::quantity!("solve_workspace"),
            )?,
            2,
            scioxide_core::quantity!("solve_workspace"),
        )
    }

    pub(super) fn prepare(
        &mut self,
        execution: ExecutionContext<'_>,
        scope: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        let mut maximum = T::Real::ZERO;
        for k in 0..n {
            admit(execution, scope)?;
            let c = self.toeplitz_entry(k, 0)?;
            self.coefficients[k] = c;
            maximum = maximum.max(magnitude(c));
            if self.coefficients.len() > n {
                let r = self.toeplitz_entry(0, k)?;
                self.coefficients[n + k] = r;
                maximum = maximum.max(magnitude(r));
            }
        }
        self.matrix_exponent = exponent(maximum)?;
        for value in &mut self.coefficients {
            *value = crate::kernel::standard::restore(*value, -self.matrix_exponent)?;
        }
        if n == 1 {
            return Ok(());
        }
        if self.compact != Compact::Circulant {
            self.prepare_gko_phases(execution)?;
        }
        if self.compact == Compact::Circulant {
            for k in 0..n {
                execution.check()?;
                let angle = crate::checked::rounded_count::<T::Real>(2 * k)?
                    / crate::checked::rounded_count::<T::Real>(n)?;
                let (s, c) = scioxide_core::sin_cos_pi(angle)?;
                if T::IS_REAL {
                    self.phases[k] = T::from_real(c);
                    self.phases[n + k] = T::from_real(s);
                } else {
                    self.phases[k] = T::from_components(c, s);
                }
            }
            if T::IS_REAL {
                // DC, conjugate pairs (real, imaginary), and optional Nyquist.
                for k in 0..=n / 2 {
                    execution.check()?;
                    let re = dot((0..n).map(|j| (self.coefficients[j], self.phases[j * k % n])));
                    let im =
                        -dot((0..n).map(|j| (self.coefficients[j], self.phases[n + j * k % n])));
                    let index = if k == 0 { 0 } else { 2 * k - 1 };
                    self.spectrum[index] = finite_candidate(re)?;
                    if k > 0 && 2 * k < n {
                        self.spectrum[index + 1] = finite_candidate(im)?;
                    }
                }
            } else {
                for k in 0..n {
                    execution.check()?;
                    self.spectrum[k] = finite_candidate(dot(
                        (0..n).map(|j| (self.coefficients[j], self.phases[j * k % n].conj()))
                    ))?;
                }
            }
        }
        Ok(())
    }

    // Coefficients of `op(A)` in the normalized Toeplitz frame. A square Hankel
    // matrix is symmetric, so its operation only conjugates.
    #[inline]
    pub(super) fn entry(&self, op: StructuredOperation, i: usize, j: usize) -> T {
        let (i, j) = if op == StructuredOperation::Normal || self.compact == Compact::Hankel {
            (i, j)
        } else {
            (j, i)
        };
        let value = self.normal_entry(i, j);
        if op == StructuredOperation::Adjoint {
            value.conj()
        } else {
            value
        }
    }

    #[inline]
    fn normal_entry(&self, i: usize, j: usize) -> T {
        let n = self.order();
        if self.compact == Compact::Circulant {
            self.coefficients[(i + n - j) % n]
        } else if i >= j {
            self.coefficients[i - j]
        } else if self.compact == Compact::HermitianToeplitz {
            self.coefficients[j - i].conj()
        } else {
            self.coefficients[n + j - i]
        }
    }

    fn levinson(
        &self,
        rhs: impl Fn(usize) -> T,
        rhs_exponent: i32,
        output: &mut [T],
        prediction: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<Candidate, SciOxideError> {
        let c = &self.coefficients;
        let n = self.order();
        let mut beta = c[0].real_component();
        let threshold = T::Real::EPSILON.sqrt(); // coefficient maximum is in [1/2,1).
        if beta <= threshold {
            return Ok(Candidate::Rejected);
        }
        output[0] = scale(rhs(0), -rhs_exponent) / c[0];
        if n == 1 {
            return Ok(Candidate::Computed);
        }
        prediction[0] = -c[1] / c[0];
        for m in 1..n {
            execution.check()?;
            let reflection = prediction[m - 1].abs();
            beta = beta * (T::Real::ONE - reflection) * (T::Real::ONE + reflection);
            if !beta.is_finite() || beta <= threshold {
                return Ok(Candidate::Rejected);
            }
            let residual =
                scale(rhs(m), -rhs_exponent) - dot((0..m).map(|j| (c[m - j], output[j])));
            let value = residual / T::from_real(beta);
            for j in 0..m {
                output[j] = add_product(value, prediction[m - 1 - j].conj(), output[j]);
            }
            output[m] = value;
            if m + 1 < n {
                let reflection = -(c[m + 1] + dot((0..m).map(|j| (c[m - j], prediction[j]))))
                    / T::from_real(beta);
                for j in 0..m.div_ceil(2) {
                    let k = m - 1 - j;
                    let a = prediction[j];
                    let b = prediction[k];
                    prediction[j] = add_product(reflection, b.conj(), a);
                    prediction[k] = add_product(reflection, a.conj(), b);
                }
                prediction[m] = reflection;
            }
        }
        Ok(if output.iter().all(|v| v.has_finite_components()) {
            Candidate::Computed
        } else {
            Candidate::Rejected
        })
    }

    // Pending and rejected columns can select GKO before final assessment.
    pub(super) fn solve_compact(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize) -> T,
        scratch: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<ColumnOutcome<T::Real>, SciOxideError> {
        let n = self.order();
        execution.check()?;
        if n == 0 {
            return Ok(ColumnOutcome::Assessed(StructuredSolveReport::empty()));
        }
        let rhs_exponent = exponent((0..n).fold(T::Real::ZERO, |v, i| v.max(magnitude(rhs(i)))))?;
        let (output, work) = crate::workspace::split_mut(scratch, n)?;
        if self.compact == Compact::HermitianToeplitz {
            if matches!(
                self.levinson(
                    |i| if op == StructuredOperation::Transpose {
                        rhs(i).conj()
                    } else {
                        rhs(i)
                    },
                    rhs_exponent,
                    output,
                    work,
                    execution
                )?,
                Candidate::Computed
            ) {
                if op == StructuredOperation::Transpose {
                    for value in output.iter_mut() {
                        *value = value.conj();
                    }
                }
                // A finite recurrence can still overflow during restoration.
                // Only numerical rejection selects GKO; execution stops propagate.
                match self
                    .restore(output, rhs_exponent)
                    .and_then(|()| self.assess(op, &rhs, output, &mut work[..n], execution))
                {
                    Ok(mut report) => {
                        report.levinson = usize::from(report.termination.is_success());
                        if report.termination.is_success() || n > 1 {
                            return Ok(ColumnOutcome::Assessed(report));
                        }
                    }
                    Err(error)
                        if error.kind()
                            == scioxide_core::SciOxideErrorKind::NumericalInstability => {}
                    Err(error) => return Err(error),
                }
            }
            if n > 1 {
                return Ok(ColumnOutcome::Pending);
            }
        }
        if n == 1 {
            output[0] = divide(scale(rhs(0), -rhs_exponent), self.entry(op, 0, 0))?;
        } else if self.compact == Compact::Circulant {
            self.circulant(op, &rhs, rhs_exponent, output, work, execution)?;
        } else {
            return Ok(ColumnOutcome::Pending);
        }
        self.restore(output, rhs_exponent)?;
        let mut report = self.assess(op, &rhs, output, &mut work[..n], execution)?;
        if report.termination.is_success() {
            report.gko = usize::from(self.compact != Compact::Circulant);
            report.circulant = usize::from(self.compact == Compact::Circulant);
        }
        Ok(ColumnOutcome::Assessed(report))
    }

    pub(super) fn restore(&self, output: &mut [T], rhs_exponent: i32) -> Result<(), SciOxideError> {
        for value in output {
            *value = crate::kernel::standard::restore(*value, rhs_exponent - self.matrix_exponent)?;
        }
        Ok(())
    }

    pub(super) fn factor_dense<'w>(
        &self,
        op: StructuredOperation,
        storage: &'w mut [T],
        indices: &'w mut [usize],
        gemm: &mut LinalgWorkspace<'_, T>,
        execution: ExecutionContext<'_>,
    ) -> Result<crate::decompose::LuKernelFactor<'w, T>, SciOxideError> {
        let n = self.order();
        let (matrix, factors) = crate::workspace::split_mut(storage, n * n)?;
        for (i, row) in matrix.chunks_exact_mut(n).enumerate() {
            execution.check()?;
            for (j, value) in row.iter_mut().enumerate() {
                *value = self.entry(op, i, j);
            }
        }
        crate::decompose::factor_kernel(
            MatrixView::from_slice(n, n, matrix)?,
            factors,
            indices,
            gemm,
            ExecutionPolicy::Sequential,
            crate::KernelContext::new(execution),
        )
    }

    pub(super) fn solve_dense(
        &self,
        op: StructuredOperation,
        factor: &crate::decompose::LuKernelFactor<'_, T>,
        rhs: impl Fn(usize) -> T,
        scratch: &mut [T],
        gemm: &mut LinalgWorkspace<'_, T>,
        execution: ExecutionContext<'_>,
    ) -> Result<StructuredSolveReport<T::Real>, SciOxideError> {
        let n = self.order();
        let rhs_exponent = exponent((0..n).fold(T::Real::ZERO, |v, i| v.max(magnitude(rhs(i)))))?;
        let (output, work) = crate::workspace::split_mut(scratch, n)?;
        for (i, value) in work[..n].iter_mut().enumerate() {
            *value = scale(rhs(i), -rhs_exponent);
        }
        factor.solve_matrix(
            MatrixView::from_slice(n, 1, &work[..n])?,
            output,
            gemm,
            crate::KernelContext::new(execution),
        )?;
        if self.compact == Compact::Hankel {
            output.reverse();
        }
        self.restore(output, rhs_exponent)?;
        let mut report = self.assess(op, &rhs, output, &mut work[..n], execution)?;
        report.dense_lu = usize::from(report.termination.is_success());
        Ok(report)
    }

    pub(super) fn assess(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize) -> T,
        output: &[T],
        work: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<StructuredSolveReport<T::Real>, SciOxideError> {
        let backward_error = self.backward_error(op, rhs, output, work, execution)?;
        let termination = if self
            .options
            .tolerance
            .is_residual_ok(backward_error, T::Real::ONE)?
        {
            DirectSolveTermination::Accepted
        } else {
            DirectSolveTermination::ResidualRejected
        };
        Ok(StructuredSolveReport {
            termination,
            backward_error,
            ..StructuredSolveReport::empty()
        })
    }

    fn circulant(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize) -> T,
        rhs_exponent: i32,
        output: &mut [T],
        work: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        if T::IS_REAL {
            let (re, im) = crate::workspace::split_mut(work, n)?;
            for k in 0..=n / 2 {
                execution.check()?;
                let br =
                    dot((0..n).map(|j| (scale(rhs(j), -rhs_exponent), self.phases[j * k % n])))
                        .real_component();
                let bi = -dot(
                    (0..n).map(|j| (scale(rhs(j), -rhs_exponent), self.phases[n + j * k % n]))
                )
                .real_component();
                let index = if k == 0 { 0 } else { 2 * k - 1 };
                let lr = self.spectrum[index].real_component();
                // A real transposed or adjoint circulant has the conjugate spectrum.
                let li = if k > 0 && 2 * k < n {
                    let li = self.spectrum[index + 1].real_component();
                    if op == StructuredOperation::Normal {
                        li
                    } else {
                        -li
                    }
                } else {
                    T::Real::ZERO
                };
                let power = exponent(lr.abs().max(li.abs()))?;
                let ar = lr.scalbn(-power);
                let ai = li.scalbn(-power);
                let denominator = ar * ar + ai * ai;
                if denominator == T::Real::ZERO {
                    return Err(SciOxideError::singular(None));
                }
                re[k] = finite_candidate(T::from_real(
                    ((br * ar + bi * ai) / denominator).scalbn(-power),
                ))?;
                im[k] = finite_candidate(T::from_real(
                    ((bi * ar - br * ai) / denominator).scalbn(-power),
                ))?;
            }
            let inverse_n = T::Real::ONE / crate::checked::rounded_count::<T::Real>(n)?;
            for (i, value) in output.iter_mut().enumerate() {
                execution.check()?;
                let mut sum = re[0];
                for k in 1..=n / 2 {
                    let term = re[k] * self.phases[i * k % n] - im[k] * self.phases[n + i * k % n];
                    sum = sum + if 2 * k == n { term } else { term + term };
                }
                *value = finite_candidate(sum * T::from_real(inverse_n))?;
            }
        } else {
            // The transpose has eigenvalues `lambda_{-k}` and the adjoint `conj(lambda_k)`.
            let eigenvalue = |k: usize| match op {
                StructuredOperation::Normal => self.spectrum[k],
                StructuredOperation::Transpose => self.spectrum[(n - k) % n],
                StructuredOperation::Adjoint => self.spectrum[k].conj(),
            };
            let b = &mut work[..n];
            for (k, value) in b.iter_mut().enumerate() {
                execution.check()?;
                *value = divide(
                    dot((0..n)
                        .map(|j| (scale(rhs(j), -rhs_exponent), self.phases[j * k % n].conj()))),
                    eigenvalue(k),
                )?;
            }
            let inverse_n = T::Real::ONE / crate::checked::rounded_count::<T::Real>(n)?;
            for (i, value) in output.iter_mut().enumerate() {
                execution.check()?;
                *value = finite_candidate(
                    dot((0..n).map(|k| (b[k], self.phases[i * k % n]))) * T::from_real(inverse_n),
                )?;
            }
        }
        Ok(())
    }

    // Preserve the original row-scaled error, evaluated after final projection.
    // A common binary frame bounds both Ax and b and leaves their ratio invariant.
    fn backward_error(
        &self,
        op: StructuredOperation,
        rhs: impl Fn(usize) -> T,
        output: &[T],
        scaled_output: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<T::Real, SciOxideError> {
        let n = self.order();
        let xe = exponent(
            output
                .iter()
                .fold(T::Real::ZERO, |m, &x| m.max(magnitude(x))),
        )?;
        let be = exponent((0..n).fold(T::Real::ZERO, |m, i| m.max(magnitude(rhs(i)))))?;
        let frame = (self.matrix_exponent + xe).max(be);
        let product_weight = T::Real::ONE.scalbn(self.matrix_exponent + xe - frame);
        let rhs_shift = -frame;
        let xnorm = output
            .iter()
            .fold(T::Real::ZERO, |m, &x| m.max(scale(x, -xe).abs()));
        for (scaled, &x) in scaled_output.iter_mut().zip(output) {
            *scaled = scale(x, -xe) * T::from_real(product_weight);
        }
        let mut error = T::Real::ZERO;
        for i in 0..n {
            execution.check()?;
            let mut residual = scale(rhs(i), rhs_shift);
            let mut row = T::Real::ZERO;
            if T::IS_REAL && self.compact == Compact::Toeplitz && op == StructuredOperation::Normal
            {
                // Preserve the original left-to-right reduction exactly, while
                // decoding Toeplitz storage only once for each row segment.
                for (&a, &x) in self.coefficients[..=i]
                    .iter()
                    .rev()
                    .zip(&scaled_output[..=i])
                {
                    residual = add_product(-a, x, residual);
                    row += a.abs();
                }
                for (&a, &x) in self.coefficients[n + 1..2 * n - i]
                    .iter()
                    .zip(&scaled_output[i + 1..])
                {
                    residual = add_product(-a, x, residual);
                    row += a.abs();
                }
            } else {
                for (j, &x) in scaled_output.iter().enumerate() {
                    let a = self.entry(
                        op,
                        i,
                        if self.compact == Compact::Hankel {
                            n - 1 - j
                        } else {
                            j
                        },
                    );
                    residual = add_product(-a, x, residual);
                    row += a.abs();
                }
            }
            let denominator = scale(rhs(i), rhs_shift).abs() + row * (xnorm * product_weight);
            let ratio = if denominator == T::Real::ZERO {
                T::Real::ZERO
            } else {
                residual.abs() / denominator
            };
            if !ratio.is_finite() {
                return Err(instability("non-finite structured backward error"));
            }
            error = error.max(ratio);
        }
        Ok(error)
    }

    /// Returns `min |λₖ| / max |λₖ|` over the eigenvalues of a circulant `A`.
    ///
    /// A circulant is normal, so this is its reciprocal 2-norm condition
    /// number `1 / (‖A‖₂ ‖A⁻¹‖₂)`. The eigenvalues were computed when the plan
    /// was prepared, so the call costs O(n). A zero eigenvalue gives zero.
    ///
    /// # Errors
    /// - `InvalidInput` if the plan is not for a circulant matrix.
    /// - `DomainError` if `n = 0`.
    pub fn spectral_condition_estimate(&self) -> Result<ConditionEstimate<T::Real>, SciOxideError> {
        if self.compact != Compact::Circulant {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("self"),
                "spectral conditioning requires a circulant",
            ));
        }
        let n = self.order();
        if n == 0 {
            return Err(crate::analysis::empty_condition());
        }
        let mut low = T::Real::INFINITY;
        let mut high = T::Real::ZERO;
        let count = if n <= 1 {
            n
        } else if T::IS_REAL {
            n / 2 + 1
        } else {
            n
        };
        for k in 0..count {
            let value = if n == 1 {
                self.coefficients[0].abs()
            } else if T::IS_REAL {
                let index = if k == 0 { 0 } else { 2 * k - 1 };
                let re = self.spectrum[index].real_component();
                let im = if k > 0 && 2 * k < n {
                    self.spectrum[index + 1].real_component()
                } else {
                    T::Real::ZERO
                };
                re.hypot(im)
            } else {
                self.spectrum[k].abs()
            };
            low = low.min(value);
            high = high.max(value);
        }
        let reciprocal = if low == T::Real::ZERO {
            T::Real::ZERO
        } else {
            low / high
        };
        ConditionEstimate::new(reciprocal, EstimateReliability::ModelBased)
    }
}
