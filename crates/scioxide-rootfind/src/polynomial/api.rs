//! All roots of a polynomial and their coefficient input.
use super::pipeline::*;
use crate::PolynomialRootOptions;
use alloc::vec::Vec;
use scioxide_core::{
    AttemptKind, Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideError, SolveScope,
};
use scioxide_poly::{ApproxPolynomial, PolynomialRootBasis, PolynomialRootBasisRef};

/// Holds borrowed polynomial coefficients and the basis they refer to.
///
/// Coefficients are in ascending order: `[a₀, a₁, …, aₙ]` represents
/// `p(x) = a₀ φ₀(x) + a₁ φ₁(x) + … + aₙ φₙ(x)`, where `φₖ` is the `k`-th basis
/// polynomial. A slice, array or `Vec` of real or complex coefficients converts
/// to the monomial basis `φₖ(x) = xᵏ`. A [`scioxide_poly::ApproxPolynomial`]
/// converts with its own basis, for example a Chebyshev series, and its roots
/// are computed and checked in that basis without conversion to monomials.
#[derive(Clone, Copy, Debug)]
pub struct PolynomialRootInput<'a, C: NumericScalar> {
    coefficients: &'a [C],
    basis: PolynomialRootBasisRef<'a, C::Real>,
}
impl<'a, C: NumericScalar> From<&'a [C]> for PolynomialRootInput<'a, C> {
    fn from(coefficients: &'a [C]) -> Self {
        Self {
            coefficients,
            basis: PolynomialRootBasisRef::Monomial,
        }
    }
}
impl<'a, C: NumericScalar, const N: usize> From<&'a [C; N]> for PolynomialRootInput<'a, C> {
    fn from(coefficients: &'a [C; N]) -> Self {
        Self::from(coefficients.as_slice())
    }
}
impl<'a, C: NumericScalar> From<&'a Vec<C>> for PolynomialRootInput<'a, C> {
    fn from(coefficients: &'a Vec<C>) -> Self {
        Self::from(coefficients.as_slice())
    }
}
impl<'a, C: NumericScalar, B: PolynomialRootBasis<C>> From<&'a ApproxPolynomial<C, B>>
    for PolynomialRootInput<'a, C>
{
    fn from(polynomial: &'a ApproxPolynomial<C, B>) -> Self {
        Self {
            coefficients: polynomial.basis().coefficients(),
            basis: polynomial.basis().root_basis(),
        }
    }
}

/// Computes all `n` complex roots of a polynomial of degree `n` with the
/// default options and returns them.
///
/// The input is a [`PolynomialRootInput`]: ascending real or complex
/// coefficients `[a₀, …, aₙ]` with `aₙ ≠ 0`, in the monomial basis or in the
/// native basis of a [`scioxide_poly::ApproxPolynomial`]. The roots are returned
/// with multiplicity, sorted by real part and then by imaginary part. They
/// equal the roots of [`polynomial_roots_with_options`] with the default
/// [`PolynomialRootOptions`] bit for bit.
///
/// The method is [`PolynomialRootMethod::Auto`]: closed-form formulas for a
/// monomial polynomial of degree one or two; companion or colleague matrix
/// eigenvalues refined by Aberth–Ehrlich iteration through degree 64, at
/// O(n³) cost and O(n²) memory; Aberth–Ehrlich iteration from Newton-polygon
/// starting points above degree 64, at O(n²) cost per sweep and O(n) memory.
/// Each start has at most 200 iterations, and at most 4 starts are made. After
/// Newton polishing, the solve succeeds when every root `z` has a normalized
/// backward error `η(z) ≤ 2√ε`, where `ε` is the machine epsilon of `S` and
/// `η` is defined on [`polynomial_roots_with_options`]. The call allocates one
/// workspace and the output.
///
/// Use [`polynomial_roots_with_options`] for another method, other tolerances
/// or budgets, or to read the per-root diagnostics, and
/// [`polynomial_roots_into`] for repeated solves without allocation.
///
/// # Errors
///
/// - `InvalidInput` (`input`, `OutOfRange`) if there are fewer than two
///   coefficients or the leading coefficient is zero.
/// - `NonFinite` (origin `Input`) if a coefficient is NaN or infinite.
/// - `SizeOverflow` or `AllocationFailure` if the workspace or the output
///   cannot be allocated.
/// - `ConvergenceFailure` (`IterationLimit` if no start converges,
///   `Stagnation` if a root fails its backward-error check) or
///   `NumericalInstability` if the solve breaks down (see the
///   [crate errors](crate#errors)).
///
/// # Examples
///
/// ```
/// // x³ − 6x² + 11x − 6 = (x − 1)(x − 2)(x − 3).
/// let roots = scioxide_rootfind::polynomial_roots(&[-6.0_f64, 11.0, -6.0, 1.0])?;
/// for (root, expected) in roots.iter().zip([1.0, 2.0, 3.0]) {
///     assert!((root.re - expected).abs() < 1e-12 && root.im.abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. A. Bini, "Numerical computation of polynomial zeros by means of
///   Aberth's method", *Numer. Algorithms* 13 (1996), 179–200.
/// - A. Edelman and H. Murakami, "Polynomial roots from companion matrix
///   eigenvalues", *Math. Comp.* 64 (1995), 763–776.
pub fn polynomial_roots<'a, S, C>(
    input: impl Into<PolynomialRootInput<'a, C>>,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    C: NumericScalar<Real = S> + 'a,
{
    let roots = polynomial_roots_with_options(input, &PolynomialRootOptions::default())?;
    crate::termination::require_success(roots.report().termination(), roots.report().accounting())?;
    Ok(roots.into_parts().0)
}

/// Computes all `n` complex roots of a polynomial of degree `n` and returns
/// them with their diagnostics and report.
///
/// The input is a [`PolynomialRootInput`]: ascending real or complex
/// coefficients `[a₀, …, aₙ]` with `aₙ ≠ 0`, in the monomial basis or in the
/// native basis of a [`scioxide_poly::ApproxPolynomial`]. Roots are returned with
/// multiplicity, sorted by real part and then by imaginary part.
///
/// The solve has three stages:
///
/// 1. For monomial input, a zero root of multiplicity `m` (`a₀ = … = aₘ₋₁ = 0`)
///    is removed exactly, and the coefficients and variable are rescaled by
///    powers of two when no coefficient loses bits.
/// 2. The [`PolynomialRootMethod`] of [`PolynomialRootOptions::method`]
///    generates candidates. Monomial polynomials of degree one or two use
///    closed-form formulas. Companion candidates are the eigenvalues of the
///    companion matrix (monomial basis) or colleague matrix (Chebyshev and
///    other orthogonal bases), at O(n³) cost and O(n²) memory; real
///    coefficients give a real matrix. Aberth–Ehrlich and Durand–Kerner
///    iterations update all candidates simultaneously at O(n²) cost per sweep
///    and O(n) memory, starting from Newton-polygon radii. Companion and
///    Laguerre candidates are refined by Aberth–Ehrlich iteration, and each
///    failed start restarts from new initial points up to
///    [`PolynomialRootOptions::restart_budget`] starts. If rescaling is not
///    possible or the eigenvalue stage fails, the solve continues with
///    Aberth–Ehrlich iteration and reports this in
///    [`PolynomialRootReport::fallback_used`].
/// 3. Optional Newton polishing on the original coefficients (at most 10
///    steps per root, each accepted only if it does not increase the backward
///    error), followed by a check of every root against the original input.
///    For real coefficients, nearly real roots are moved onto the real axis
///    and nearly conjugate pairs are made exact conjugates when the check
///    still passes.
///
/// The call allocates one workspace and the output and runs sequentially. A
/// solve that does not converge, or whose roots fail the backward-error check,
/// still returns `Ok`: check [`PolynomialRootReport::termination`] and the
/// per-root [`PolynomialRootDiagnostic`] before using the roots. Use
/// [`polynomial_roots`] to get an error instead, and
/// [`polynomial_roots_into`] for repeated solves without allocation.
///
/// # Accuracy
///
/// Each root `z` is accepted only if its normalized backward error passes
/// [`PolynomialRootOptions::backward_error_tolerance`]. For monomial
/// coefficients the backward error is componentwise:
///
/// ```text
/// η(z) = |p(z)| / Σₖ |aₖ| |z|ᵏ,
/// ```
///
/// the smallest relative perturbation of the coefficients that makes `z` an
/// exact root. For other bases each coefficient may vary by `maxₖ |aₖ|`
/// (normwise). The default tolerance accepts `η(z) ≤ 2√ε`. The forward error
/// of a root is about its condition number, reported in
/// [`PolynomialRootDiagnostic::condition`], times `η(z)`. Multiple and
/// clustered roots are ill-conditioned: for a root of multiplicity `m` the
/// forward error is of order `η(z)^(1/m)`.
///
/// # Errors
///
/// - `InvalidInput` (`input`, `OutOfRange`) if there are fewer than two
///   coefficients or the leading coefficient is zero.
/// - `NonFinite` (origin `Input`) if a coefficient is NaN or infinite.
/// - `InvalidInput` (`options.method`, `OutOfRange`) if
///   [`PolynomialRootMethod::Laguerre`] or
///   [`PolynomialRootMethod::DurandKerner`] is used with a non-monomial basis.
/// - `ResourceLimit` if the workspace exceeds
///   [`PolynomialRootOptions::workspace_byte_limit`]; `SizeOverflow` or
///   `AllocationFailure` if the workspace or the output cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions, polynomial_roots_with_options};
///
/// // x² − 1 = 0 has the roots −1 and 1, returned in ascending order.
/// let options = PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth);
/// let roots = polynomial_roots_with_options(&[-1.0_f64, 0.0, 1.0], &options)?;
/// assert!(roots.report().termination().is_success());
/// assert!((roots[0].re + 1.0).abs() < 1e-12 && (roots[1].re - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - O. Aberth, "Iteration methods for finding all zeros of a polynomial
///   simultaneously", *Math. Comp.* 27 (1973), 339–344.
/// - D. A. Bini, "Numerical computation of polynomial zeros by means of
///   Aberth's method", *Numer. Algorithms* 13 (1996), 179–200.
/// - A. Edelman and H. Murakami, "Polynomial roots from companion matrix
///   eigenvalues", *Math. Comp.* 64 (1995), 763–776.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, chapter 5 (polynomials).
pub fn polynomial_roots_with_options<'a, S, C>(
    input: impl Into<PolynomialRootInput<'a, C>>,
    options: &PolynomialRootOptions<S>,
) -> Result<PolynomialRoots<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    C: NumericScalar<Real = S> + 'a,
{
    let input = input.into();
    let degree = validate_coefficients(input.coefficients, scioxide_core::argument!("input"))?;
    validate_method(input.basis, options.method)?;
    let mut workspace = PolynomialRootWorkspace::workspace(degree, options)?;
    let mut roots = scioxide_core::backend::try_filled(
        degree,
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("root_output"),
    )?;
    let report = polynomial_roots_into(
        input,
        options,
        &mut roots,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok(PolynomialRoots {
        roots,
        diagnostics: workspace.diagnostics,
        report,
    })
}

/// Computes all roots into `output` with a prepared workspace and returns the
/// report.
///
/// The method, the stopping rule and the accuracy are those of
/// [`polynomial_roots_with_options`]. `output` must have length `n`, and
/// `workspace` must be reserved for this degree and
/// [`PolynomialRootOptions::method`]: create it with
/// [`PolynomialRootWorkspace::workspace`], or size it with
/// [`PolynomialRootWorkspace::workspace_requirements`] and reserve it with
/// [`PolynomialRootWorkspace::try_reserve`]. With such a workspace the call
/// does not allocate. Work counts toward the budgets of `scope` (see
/// [Solve accounting](scioxide_core#solve-accounting)). Per-root diagnostics are
/// in [`PolynomialRootWorkspace::diagnostics`].
///
/// When the solve stops at a limit, by cancellation or by numerical
/// breakdown, `output` holds the current finite candidates and the report
/// has an unsuccessful termination. A candidate is usable only if its
/// diagnostic reports [`PolynomialRootDiagnostic::is_validated`].
///
/// # Errors
///
/// - `InvalidInput` (`input`, `OutOfRange`), `NonFinite` (origin `Input`)
///   and `InvalidInput` (`options.method`, `OutOfRange`) as for
///   [`polynomial_roots_with_options`].
/// - `SizeOverflow` or `ResourceLimit` if the workspace size for this degree
///   overflows or exceeds [`PolynomialRootOptions::workspace_byte_limit`].
/// - `DimensionMismatch` if `output.len()` is not the degree.
/// - `InvalidInput` (`workspace`, `Inconsistent`) if `workspace` is not
///   reserved for this degree and method.
///
/// On error, `output` is unchanged (see
/// [Output on error](scioxide_core#output-on-error)).
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_rootfind::{PolynomialRootOptions, PolynomialRootWorkspace, polynomial_roots_into};
///
/// // The roots ±√c of x² − c for several c reuse one workspace of degree 2.
/// let options = PolynomialRootOptions::<f64>::default();
/// let mut workspace = PolynomialRootWorkspace::workspace(2, &options)?;
/// let mut roots = [Complex::new(0.0, 0.0); 2];
/// for c in [2.0_f64, 3.0, 5.0] {
///     let cx = ExecutionContext::sequential_native();
///     let report = polynomial_roots_into(&[-c, 0.0, 1.0], &options, &mut roots, &mut workspace, cx)?;
///     assert!(report.termination().is_success());
///     assert!((roots[1].re - c.sqrt()).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_roots_into<'a, 'scope, 'capability, S, C>(
    input: impl Into<PolynomialRootInput<'a, C>>,
    options: &PolynomialRootOptions<S>,
    output: &mut [Complex<S>],
    workspace: &mut PolynomialRootWorkspace<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<PolynomialRootReport<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    C: NumericScalar<Real = S> + 'a,
{
    let input = input.into();
    crate::attempt::attempt(
        workspace,
        output,
        scope.into(),
        AttemptKind::Call,
        |workspace, output| {
            let degree =
                validate_coefficients(input.coefficients, scioxide_core::argument!("input"))?;
            validate_method(input.basis, options.method)?;
            if output.len() != degree {
                return Err(SciOxideError::dimension_mismatch([degree], [output.len()]));
            }
            if workspace.check(&PolynomialRootWorkspace::workspace_requirements(
                degree, options,
            )?)? != scioxide_core::Capacity::Sufficient
            {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "reserve polynomial workspace before solving",
                ));
            }
            Ok(crate::attempt::Entry::call(
                scioxide_core::SolveIdentity::new(
                    scioxide_core::ProblemToken::new(0),
                    scioxide_core::algorithm_id!("rootfind", "PolynomialRoots", 1),
                    degree as u128,
                    0,
                ),
                options.budgets,
            ))
        },
        |workspace, output, context| {
            solve_into(
                input.coefficients,
                input.basis,
                output,
                workspace,
                options,
                context,
                scioxide_core::argument!("input"),
            )
        },
        |_, _, result| crate::attempt::Conclusion {
            succeeded: result
                .as_ref()
                .is_ok_and(|report| report.termination().is_success()),
            progress: result.as_ref().ok().cloned(),
        },
    )
}
