//! Generalized Schur (QZ) decomposition of a square pencil.
use super::SchurOptions;
use super::qz_types::{
    GeneralizedEigenvalue, GeneralizedSchurDecomposition, GeneralizedSchurReport,
};
use crate::workspace::{take_mut, take_square};
//
// Arena-level QZ kernels shared with plan families (Riccati), reachable as
// `crate::eigen::<name>`; the staged pencil type is
// `crate::eigen::qz_kernel::Pencil { s, t, q: Option<_>, z: Option<_> }` over
// `ScratchMatrix` slices:
//
//   qz_kernel::scratch_len::<S>(n, policy, operation) -> elements
//       Reduction and sweep scratch of an order-`n` pencil for a policy.
//   qz_pencil(&mut Pencil, scratch, &SchurOptions, solve, operation) -> iterations
//       `s`/`t` hold finite row-major `(A, B)` on entry and the generalized
//       Schur form in input units on exit; `q`/`z`, when present, receive the
//       unitary transformations. `scratch` holds `scratch_len` elements for
//       the options' execution policy. Work units are charged to `solve`.
//   qz_values(s, t, n, scratch, values: &mut [R; 3n], &SchurOptions, solve, operation)
//       -> iterations. Values-only QZ without Q/Z; homogeneous eigenvalue `i`
//       is `(alpha.re, alpha.im, beta)` at `values[3i..3i + 3]` in the real
//       arena; rebuild it with `GeneralizedEigenvalue::from_parts`.
//   qz_reorder(&mut Pencil, selection, indices: &mut [usize; 4n], rhs: &mut [Extended<S>],
//       &SchurReorderOptions, solve, operation) -> GeneralizedSchurReorderReport
//       Selected-first reordering of a staged pencil with Q and Z, with
//       estimates when requested (`rhs` holds `2 floor(n/2) ceil(n/2)` values).
//   for_each_generalized_value(&s, &t, operation, visit)
//       Visit diagonal blocks and homogeneous eigenvalues of a pencil.
use super::SchurBlock;
use super::qz_kernel::Pencil;
use super::support::{
    DirectLengths, copy_view, direct_requirements, identity, set_identity, square, square_finite,
    times, total,
};
use super::types::reserve;
use crate::workspace::storage::MatrixStorage;
use crate::{LinalgWorkspace, MatrixView};
use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;
use scioxide_core::{
    Complex, Dimensions, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError, SolveContext,
};
use scioxide_core::{Scaled, ScaledValue};

pub(super) fn decompose_lengths<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<DirectLengths, SciOxideError> {
    Ok(DirectLengths {
        scalars: total(&[
            times(4, square(n)?)?,
            super::qz_kernel::scratch_len::<S>(n, policy)?,
        ])?,
        ..DirectLengths::default()
    })
}

/// Arena-level QZ: `Q^H A Z = S`, `Q^H B Z = T` on a staged pencil.
///
/// `A` and `B` are normalized separately (the eigenvalues `alpha / beta`
/// only change by the known ratio), reduced to Hessenberg-triangular form and
/// iterated to generalized Schur form; S and T are then scaled back while Q
/// and Z are scale-free. `scratch` holds `qz_kernel::scratch_len` elements
/// for the options' execution policy.
pub(crate) fn qz_pencil<S: NumericScalar>(
    pencil: &mut Pencil<'_, S>,
    scratch: &mut [S],
    options: &SchurOptions<S::Real>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let n = pencil.s.rows();
    let sa = normalize(pencil.s.as_mut_slice())?;
    let sb = normalize(pencil.t.as_mut_slice())?;
    if let Some(z) = pencil.z.as_mut() {
        set_identity(z.as_mut_slice(), n);
    }
    let iterations = super::qz_kernel::generalized_schur(
        pencil,
        scratch,
        (
            options.deflation_tolerance(),
            super::qr_iteration_budget(options.iteration_budget(), n)?,
            options.execution_options().execution_policy(),
        ),
        solve,
    )?;
    restore(pencil.s.as_mut_slice(), sa)?;
    restore(pencil.t.as_mut_slice(), sb)?;
    Ok(iterations)
}

/// Values-only arena QZ storing `(alpha.re, alpha.im, beta)` per eigenvalue.
pub(crate) fn qz_values<S: NumericScalar>(
    s: &mut [S],
    t: &mut [S],
    n: usize,
    scratch: &mut [S],
    values: &mut [S::Real],
    options: &SchurOptions<S::Real>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let mut pencil = Pencil {
        s: crate::workspace::storage::ScratchMatrix::new(s, n, n)?,
        t: crate::workspace::storage::ScratchMatrix::new(t, n, n)?,
        q: None,
        z: None,
    };
    let iterations = qz_pencil(&mut pencil, scratch, options, solve)?;
    let mut rest = values;
    let values = take_mut(&mut rest, times(3, n)?)?;
    let mut index = 0;
    for_each_generalized_value(&pencil.s, &pencil.t, |_, block| {
        for value in block {
            values[3 * index] = value.alpha.re;
            values[3 * index + 1] = value.alpha.im;
            values[3 * index + 2] = value.beta;
            index += 1;
        }
    })?;
    Ok(iterations)
}

/// Visit diagonal blocks and homogeneous eigenvalues of a generalized Schur pencil.
///
/// A 1x1 block with both `alpha` and `beta` at roundoff level (relative to
/// their matrices) signals a numerically singular pencil, where every lambda
/// is an eigenvalue; that is reported rather than returning an arbitrary ratio.
pub(crate) fn for_each_generalized_value<S: NumericScalar, M: MatrixStorage<S>>(
    s: &M,
    t: &M,
    mut visit: impl FnMut(SchurBlock, &[GeneralizedEigenvalue<S::Real>]),
) -> Result<(), SciOxideError> {
    let n = s.rows();
    let uncertainty = Scaled::from_value(S::Real::try_from_usize_exact(32)? * S::Real::EPSILON)?;
    let small_a = Scaled::from_value(crate::kernel::standard::slice_maximum(s.as_slice()))?
        .mul(uncertainty)?;
    let small_b = Scaled::from_value(crate::kernel::standard::slice_maximum(t.as_slice()))?
        .mul(uncertainty)?;
    let mut i = 0;
    while i < n {
        let pair = S::IS_REAL && i + 1 < n && s.get_element(i + 1, i) != S::ZERO;
        let block = SchurBlock::sized(i, if pair { 2 } else { 1 });
        let mut values = [GeneralizedEigenvalue::zero(); 2];
        if pair {
            let roots = super::qz_kernel::roots(s, t, i)?;
            values[0] = value(roots[0].0, roots[0].1)?;
            values[1] = value(roots[1].0, roots[1].1)?;
            if values[0].alpha.im < S::Real::ZERO {
                values.swap(0, 1);
            }
        } else {
            let alpha = ScaledValue::from_value(crate::eigen::into_complex(s.get_element(i, i)))?;
            let beta = ScaledValue::from_value(crate::eigen::into_complex(t.get_element(i, i)))?;
            if !(alpha.is_zero() && beta.is_zero())
                && alpha.magnitude()?.abs_le(small_a)
                && beta.magnitude()?.abs_le(small_b)
            {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::IllConditioned,
                    "pencil has an unresolved near-zero homogeneous pair; regularity is numerically ambiguous",
                ));
            }
            values[0] = value(alpha, beta)?;
        }
        visit(block, &values[..block.size()]);
        i += block.size();
    }
    Ok(())
}

pub(super) fn normalize<S: NumericScalar>(a: &mut [S]) -> Result<S::Real, SciOxideError> {
    let scale = crate::kernel::standard::slice_maximum(a);
    if scale == S::Real::ZERO {
        return Ok(S::Real::ONE);
    }
    for v in a.iter_mut() {
        let z = round_complex(
            ScaledValue::from_value(crate::eigen::into_complex(*v))?.div(
                ScaledValue::from_value(scioxide_core::Complex::new(
                    scale,
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                ))?,
            )?,
        )?;
        *v = S::from_components(z.re, z.im);
    }
    Ok(scale)
}
pub(super) fn restore<S: NumericScalar>(a: &mut [S], scale: S::Real) -> Result<(), SciOxideError> {
    for v in a.iter_mut() {
        *v = crate::checked::mul(*v, S::from_real(scale))?;
    }
    Ok(())
}
fn value<R: RealScalar>(
    a: ScaledValue<scioxide_core::Complex<R>>,
    b: ScaledValue<scioxide_core::Complex<R>>,
) -> Result<GeneralizedEigenvalue<R>, SciOxideError> {
    if a.is_zero() && b.is_zero() {
        return Err(scioxide_core::SciOxideError::singular(None));
    }
    if b.is_zero() {
        return Ok(GeneralizedEigenvalue {
            alpha: Complex::new(R::ONE, R::ZERO),
            beta: R::ZERO,
        });
    }
    // Canonical homogeneous form: rotate the pair by `conj(phase(beta))` so
    // beta is real and positive (the ratio is unchanged). A zero beta is an
    // infinite eigenvalue, reported as `(1, 0)`.
    let magnitude = b.magnitude()?;
    let phase = b.div(ScaledValue::from_components(magnitude, Scaled::ZERO))?;
    let a = a.div(phase)?;
    // Preserve independent original units when representable, including ratios
    // spanning twice the scalar exponent range. Otherwise remove a common scale.
    if let (Ok(alpha), Ok(beta)) = (round_complex(a), round_real(magnitude)) {
        return Ok(GeneralizedEigenvalue { alpha, beta });
    }
    // Center the nonzero exponent range rather than forcing the largest
    // component to one: a finite pair may span twice the scalar exponent range.
    let mut parts = [a.real(), a.imaginary(), magnitude];
    let mut low = i64::MAX;
    let mut high = i64::MIN;
    for part in parts {
        if !part.is_zero() {
            let (_, exponent) = scioxide_core::backend::scaled::factor_common_scale([part])?;
            low = low.min(exponent);
            high = high.max(exponent);
        }
    }
    let middle = low / 2 + high / 2 + (low % 2 + high % 2) / 2;
    for part in &mut parts {
        *part = part.shifted(-middle)?;
    }
    Ok(GeneralizedEigenvalue {
        alpha: Complex::new(round_real(parts[0])?, round_real(parts[1])?),
        beta: round_real(parts[2])?,
    })
}

pub(super) fn round_real<R: RealScalar>(value: Scaled<R>) -> Result<R, SciOxideError> {
    let result = value.round()?;
    if result == R::ZERO && !value.is_zero() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "nonzero homogeneous component is not representable",
        ));
    }
    Ok(result)
}
pub(super) fn round_complex<R: RealScalar>(
    value: ScaledValue<scioxide_core::Complex<R>>,
) -> Result<Complex<R>, SciOxideError> {
    Ok(Complex::new(
        round_real(value.real())?,
        round_real(value.imaginary())?,
    ))
}
impl<S: NumericScalar> GeneralizedSchurDecomposition<S> {
    /// Computes the generalized Schur decomposition of the pencil `A − λ B`.
    ///
    /// The conventions are stated on [`GeneralizedSchurDecomposition`]. `B`
    /// is reduced to upper triangular form by Householder QR, with `Qᴴ`
    /// applied to `A`. Givens rotations then reduce `A` to upper Hessenberg
    /// form while keeping `B` triangular. Finally the QZ iteration reduces `A`
    /// to (quasi-)triangular form, on copies of `A` and `B` each scaled by its
    /// largest entry magnitude. Real input uses double shifts for complex
    /// conjugate pairs and single shifts otherwise; complex input uses single
    /// shifts. The cost is O(n³) time and O(n²) memory.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` or `b` is not square, or if their orders differ.
    /// - `NonFinite` if `a` or `b` contains NaN or infinity.
    /// - `ConvergenceFailure` if QZ iteration exhausts its iteration budget.
    /// - `Singular` if a diagonal pair `(Sⱼⱼ, Tⱼⱼ)` is exactly `(0, 0)`: the
    ///   pencil is singular and every `λ` is an eigenvalue.
    /// - `NumericalInstability` with
    ///   [`IllConditioned`](scioxide_core::NumericalIssue::IllConditioned) if a
    ///   diagonal pair is `(0, 0)` up to rounding, so that the regularity of
    ///   the pencil cannot be decided.
    /// - `NumericalInstability` if a result overflows when scaled back, or if
    ///   a nonzero component of an eigenvalue underflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralizedSchurDecomposition, Matrix};
    /// // B is singular, so the pencil has one infinite eigenvalue.
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 0.0, 3.0])?;
    /// let b = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 0.0])?;
    /// let qz = GeneralizedSchurDecomposition::decompose(&a, &b)?;
    /// assert_eq!(qz.report().unwrap().infinite_eigenvalues(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - C. B. Moler and G. W. Stewart, "An algorithm for generalized matrix
    ///   eigenvalue problems", *SIAM J. Numer. Anal.* 10, 1973.
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §7.7.
    pub fn decompose<'a, 'b>(
        a: impl Into<MatrixView<'a, S>>,
        b: impl Into<MatrixView<'b, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::decompose_with_options(a, b, &crate::SchurOptions::default())
    }

    /// Runs [`decompose`](Self::decompose) with `options`, which set the deflation tolerance and
    /// the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`decompose`](Self::decompose), and limit errors from `options` as described
    /// in [Execution](crate#execution).
    pub fn decompose_with_options<'a, 'b>(
        a: impl Into<MatrixView<'a, S>>,
        b: impl Into<MatrixView<'b, S>>,
        options: &SchurOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::decompose_into(
            a,
            b,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the generalized Schur decomposition of `A − λ B` into `output`.
    ///
    /// This is the reusable form of [`Self::decompose`], which describes the
    /// method. Size the workspace with
    /// [`GeneralizedSchurWorkspaceRequest::new`](crate::GeneralizedSchurWorkspaceRequest::new).
    /// The QZ iteration runs on one thread. Each sweep counts toward the
    /// budget set by [`crate::LinalgExecutionOptions::with_iteration_budget`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::decompose`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn decompose_into<'scope, 'a, 'b, 'c>(
        a: impl Into<MatrixView<'a, S>>,
        b: impl Into<MatrixView<'b, S>>,
        options: &SchurOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'c>>,
    ) -> Result<GeneralizedSchurReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        kernels.execution().check()?;
        let a = square_finite(a.into(), kernels.dispatch())?;
        let b = square_finite(b.into(), kernels.dispatch())?;
        if a.rows() != b.rows() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(a.rows(), a.cols()),
                Dimensions::two(b.rows(), b.cols()),
            ));
        }
        let n = a.rows();
        let requirements = direct_requirements::<S>(
            decompose_lengths::<S>(n, options.execution_options().execution_policy())?,
            options.execution.byte_limit(),
        )?;
        workspace.try_reserve(&requirements)?;
        output.reserve(n)?;
        let options = *options;
        workspace.run_scope(
            scope,
            identity(scioxide_core::algorithm_id!("linalg", "qz", 1), n),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, _, solve| {
                workspace.with_direct(|mut scalars, _, _, _| {
                    let mut pencil = Pencil {
                        s: take_square(&mut scalars, n)?,
                        t: take_square(&mut scalars, n)?,
                        q: Some(take_square(&mut scalars, n)?),
                        z: Some(take_square(&mut scalars, n)?),
                    };
                    copy_view(a.as_view(), pencil.s.as_mut_slice());
                    copy_view(b.as_view(), pencil.t.as_mut_slice());
                    let _ = qz_pencil(&mut pencil, scalars, &options, solve)?;
                    for_each_generalized_value(&pencil.s, &pencil.t, |_, _| {})?;
                    solve.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.publish(&pencil, solve.counts())
                })
            },
        )
    }

    /// Reserve output storage for order `n` without changing the current factors.
    pub(crate) fn reserve(&mut self, n: usize) -> Result<(), SciOxideError> {
        for m in [&mut self.s, &mut self.t, &mut self.q, &mut self.z] {
            m.try_reserve_shape(n, n)?;
        }
        reserve(
            &mut self.values,
            n,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        reserve(
            &mut self.blocks,
            n,
            scioxide_core::quantity!("output.blocks"),
        )
    }

    // Copy a validated candidate into reserved storage and rebuild derived
    // homogeneous eigenvalues, blocks and infinite count from (S, T).
    pub(super) fn publish(
        &mut self,
        pencil: &Pencil<'_, S>,
        counts: scioxide_core::SolveCounts,
    ) -> Result<GeneralizedSchurReport, SciOxideError> {
        let n = pencil.s.rows();
        for (target, source) in [
            (&mut self.s, Some(&pencil.s)),
            (&mut self.t, Some(&pencil.t)),
            (&mut self.q, pencil.q.as_ref()),
            (&mut self.z, pencil.z.as_ref()),
        ] {
            if let Some(source) = source {
                target.try_resize_flat(n, n)?;
                target.as_mut_slice().copy_from_slice(source.as_slice());
            }
        }
        self.values.clear();
        self.blocks.clear();
        let (values, blocks) = (&mut self.values, &mut self.blocks);
        for_each_generalized_value(&pencil.s, &pencil.t, |block, block_values| {
            blocks.push(block);
            values.extend_from_slice(block_values);
        })?;
        let report = GeneralizedSchurReport {
            counts,
            infinite: self.values.iter().filter(|v| v.is_infinite()).count(),
        };
        self.report = Some(report);
        Ok(report)
    }

    // Fallible copy used by owned transformations.
    pub(super) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        output.reserve(self.s.rows())?;
        output.s.try_copy_from(&self.s)?;
        output.t.try_copy_from(&self.t)?;
        output.q.try_copy_from(&self.q)?;
        output.z.try_copy_from(&self.z)?;
        output.values.extend_from_slice(&self.values);
        output.blocks.extend_from_slice(&self.blocks);
        output.report = self.report;
        Ok(output)
    }
}
