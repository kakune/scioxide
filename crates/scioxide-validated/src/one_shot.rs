//! Owned proof entry points with default settings.
use crate::{outcome::*, *};
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    AlgebraicScalar, Complex, DomainReason, ErrorTolerance, ExecutionContext, Interval,
    NumericScalar, RealScalar, SciOxideError, StepTolerance,
};
use scioxide_linalg::{LuDecomposition, MatrixView};

/// Encloses the unique solution of the dense point system `a x = b`.
///
/// Uses LU factorization and an approximate inverse, then proves regularity and
/// an outward componentwise error bound. The certificate encloses the exact
/// solution of the supplied floating-point system; no fixed forward accuracy is
/// promised. For point data the bound exceeds the true error by at most
/// `(1 + q)/(1 − q)`, plus rounding, where `q < 1` bounds `‖I − RA‖`
/// for the approximate inverse `R`.
/// For order `n`, costs O(n³) arithmetic and O(n²) storage, including `2n + 3`
/// operator applications.
/// Returns a certificate only when verification succeeds. Interval or sparse
/// data and several right-hand sides use [`LinearVerifier`].
///
/// ```
/// use scioxide_linalg::Matrix;
/// let a = Matrix::from_vec(2, 2, vec![3.0_f64, 1.0, 1.0, 2.0])?;
/// let x = scioxide_validated::solve(&a, &[1.0, 1.0])?;
/// assert!((x.view().center[0] - 0.2).abs() < 1e-15);
/// assert!(x.enclosure()?[1].re.contains(0.4));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// Returns `DimensionMismatch` for nonsquare `a` or the wrong `b` length,
/// `NonFinite` for nonfinite entries, and `Singular` for a zero LU pivot.
/// `Singular` describes the floating-point factorization, not a singularity proof.
/// Returns `NumericalInstability` when regularity cannot be proved or a bound
/// overflows, and storage or floating-point environment errors as applicable.
/// The floating-point environment is checked before the ordinary LU factorization.
///
/// # References
///
/// S. M. Rump, *Verification methods: rigorous results using floating-point
/// arithmetic*, Acta Numerica 19 (2010), sections 4–5.
pub fn solve<'m, S: NumericScalar>(
    a: impl Into<MatrixView<'m, S>>,
    b: &[S],
) -> Result<LinearCertificate<S>, SciOxideError> {
    crate::environment::check()?;
    let a = a.into();
    crate::shell::exact(a.rows(), b.len())?;
    let factor = LuDecomposition::factor(a)?;
    let system = DenseIntervalOperator::point(a)?;
    let inverse = LuIntervalInverse::from_dense(&factor, ValidatedOptions::new())?;
    let verifier = LinearVerifier::new(&system, &inverse, ValidatedOptions::new())?;
    let mut rhs = try_with_capacity(b.len(), scioxide_core::quantity!("rhs"))?;
    for &value in b {
        rhs.push(point_enclosure(value)?);
    }
    let mut certificate = LinearCertificate::new(a.rows())?;
    let mut workspace = verifier.workspace(1)?;
    let report = verifier.solve_into(
        &rhs,
        certificate.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    linear_result(
        report.status(),
        diagnostics(crate::linear::proof::ALGORITHM, workspace.last_attempt()),
    )?;
    Ok(certificate)
}

/// Isolates every real root of `f` in `[lower, upper]`, using an enclosure `df` of `f′`.
///
/// Interval Newton and monotonicity tests with bisection prove one root per
/// returned interval and no others in the domain. An empty vector proves absence.
/// The result is sorted and pairwise disjoint. A shared bisection endpoint `m`
/// is returned as `[m, m]`; other intervals are the first leaves proving uniqueness,
/// with no width bound smaller than `upper − lower`. A narrow interval around an
/// approximation gives a tight enclosure by a second call.
///
/// Uses at most 1,023 tree nodes and the default split threshold
/// `√ε (1 + max(|a|, |b|))` for a leaf `[a, b]`. Costs O(N²) comparisons,
/// O(N) enclosure evaluations and storage for `N` nodes; callbacks allocate nothing
/// unless the supplied functions do.
///
/// ```
/// use scioxide_validated::{roots, DecoratedInterval as D};
/// let (one, two) = (D::point(1.0_f64)?, D::point(2.0)?);
/// let found = roots(|x| x.square()?.sub(&one), |x| x.mul(&two), -2.0, 2.0)?;
/// assert_eq!(found.len(), 2);
/// assert!(found[0].contains(-1.0) && found[1].contains(1.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// Returns `NonFinite` or `InvalidInput` for invalid bounds, `ConvergenceFailure`
/// for exhausted node capacity or unresolved leaves at the split threshold
/// (including multiple roots), allocation and floating-point environment errors.
/// Callback errors propagate unchanged; an inconclusive result never asserts absence.
///
/// # References
///
/// R. E. Moore, R. B. Kearfott and M. J. Cloud, *Introduction to Interval
/// Analysis*, SIAM (2009), chapter 8.
pub fn roots<S, F, G>(f: F, df: G, lower: S, upper: S) -> Result<Vec<Interval<S>>, SciOxideError>
where
    S: RealScalar,
    F: Fn(DecoratedInterval<S>) -> Result<DecoratedInterval<S>, SciOxideError> + Sync,
    G: Fn(DecoratedInterval<S>) -> Result<DecoratedInterval<S>, SciOxideError> + Sync,
{
    let domain = Interval::new(lower, upper)?;
    let verifier = ScalarRootVerifier::new(&f, &df, ValidatedOptions::new());
    let mut tree = RootIsolation::new(1, 1023)?;
    let mut workspace = RootWorkspace::new();
    let report = verifier.isolate_into(
        domain.into(),
        StepTolerance::default(),
        &mut tree,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    root_result(
        report.termination(),
        diagnostics(
            crate::roots::scalar::INTERVAL_NEWTON,
            workspace.last_attempt(),
        ),
    )?;
    let mut intervals = try_with_capacity(report.isolated(), scioxide_core::quantity!("roots"))?;
    for region in tree.isolated_regions() {
        intervals.push(Interval::try_from(region[0].re)?);
    }
    intervals.sort_unstable_by(|a, b| {
        a.lower()
            .partial_cmp(&b.lower())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut written = 0;
    for i in 0..intervals.len() {
        if written > 0 && intervals[written - 1].upper() == intervals[i].lower() {
            let shared = intervals[i].lower();
            intervals[written - 1] = Interval::new(shared, shared)?;
        } else {
            intervals[written] = intervals[i];
            written += 1;
        }
    }
    intervals.truncate(written);
    Ok(intervals)
}

/// Certifies disks covering every root of a polynomial with ascending coefficients.
///
/// Approximations seed disks with radius `√ε max(|Re z|, |Im z|, 1)`; Rouché
/// tests double undecided radii and merge overlapping disks until proved counts
/// sum to the degree. Returns only a complete cover. Each disk contains its
/// stated number of roots, including multiplicities; the seed radius is not
/// an accuracy promise. Preparation takes O(d) storage and work for degree `d`;
/// a disk test costs O(d²), with O(d + m) storage for `m` approximations.
///
/// ```
/// use scioxide_core::Complex;
/// let seeds = [Complex::new(-1.4142_f64, 0.0), Complex::new(1.4142, 0.0)];
/// let cover = scioxide_validated::polynomial_roots(&[-2.0, 0.0, 1.0], &seeds)?;
/// assert_eq!(cover.view().iter().map(|c| c.count).sum::<usize>(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// Returns `InvalidInput` for the zero polynomial, `NonFinite` for nonfinite
/// coefficients or approximations, `ConvergenceFailure` when counts cannot
/// reach the degree or radii cannot grow, and allocation or environment errors.
///
/// # References
///
/// P. Henrici, *Applied and Computational Complex Analysis*, volume 1 (1974),
/// chapter 6; S. M. Rump, Acta Numerica 19 (2010).
pub fn polynomial_roots<S: NumericScalar>(
    coefficients: &[S],
    approximations: &[Complex<S::Real>],
) -> Result<PolynomialRootCover<S::Real>, SciOxideError> {
    let plan = PolynomialRootPlan::new(coefficients, ValidatedOptions::new())?;
    let mut seeds = try_with_capacity(approximations.len(), scioxide_core::quantity!("seeds"))?;
    for &center in approximations {
        seeds.push(RootDisk {
            center,
            radius: S::Real::SQRT_EPSILON * center.re.abs().max(center.im.abs()).max(S::Real::ONE),
        });
    }
    let mut cover = PolynomialRootCover::new(seeds.len())?;
    let mut workspace = RootWorkspace::new();
    let report = plan.isolate_into(
        &seeds,
        &mut cover,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    root_result(
        report.termination(),
        diagnostics(
            crate::roots::polynomial::POLYNOMIAL_COUNT,
            workspace.last_attempt(),
        ),
    )?;
    Ok(cover)
}

/// Encloses the integral of a twice continuously differentiable real function.
///
/// `f` encloses the function and `d2f` encloses its second derivative on every
/// requested interval. Uses adaptive trapezoids with the second-derivative
/// remainder, then independently verifies the generated certificate. Returns
/// only a verified interval whose radius is at most `√ε (1 + |midpoint|)`.
/// Equal bounds return `[0, 0]`. Uses up to 4,096 panels, O(N log N) scheduling
/// work and O(N) storage for `N` panels. Storage is reserved once; the adapter
/// allocates nothing per callback.
///
/// ```
/// use scioxide_validated::{integrate, DecoratedInterval as D};
/// let integral = integrate(|x| x.square(), |_| D::point(2.0_f64), 0.0, 1.0)?;
/// assert!(integral.contains(1.0 / 3.0));
/// assert!(integral.upper() - integral.lower() < 1e-7);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// Returns `NonFinite` or `InvalidInput` for invalid bounds, `DomainError` for
/// empty or insufficiently continuous enclosures, `ConvergenceFailure` for
/// capacity, resolution or fixed bounds, and `NumericalInstability` for an
/// unbounded proof. Callback, allocation and environment errors propagate.
///
/// # References
///
/// R. E. Moore, R. B. Kearfott and M. J. Cloud, *Introduction to Interval
/// Analysis*, SIAM (2009), chapter 6.
pub fn integrate<S, F, G>(f: F, d2f: G, lower: S, upper: S) -> Result<Interval<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(DecoratedInterval<S>) -> Result<DecoratedInterval<S>, SciOxideError> + Sync,
    G: Fn(DecoratedInterval<S>) -> Result<DecoratedInterval<S>, SciOxideError> + Sync,
{
    let domain = Interval::new(lower, upper)?;
    let pieces = [QuadraturePiece {
        domain: domain.into(),
        assumption: QuadratureAssumption::TwiceContinuouslyDifferentiable,
    }];
    let oracle = |request: QuadratureRequest<S>, output: &mut [IntervalBox<S>]| {
        let input = DecoratedInterval::from(request.domain);
        let (value, argument) = match request.query {
            QuadratureQuery::Value => (f(input)?, scioxide_core::argument!("f")),
            QuadratureQuery::SecondDerivative => (d2f(input)?, scioxide_core::argument!("d2f")),
            _ => return Ok(QuadratureObligation::Unavailable),
        };
        if value.set().is_empty() || value.decoration() < Decoration::Dac {
            return Err(SciOxideError::domain_error(
                argument,
                DomainReason::Undefined,
                "the enclosure is empty or not known to be continuous on a panel",
            ));
        }
        output[0] = Complex::new(value.set(), DecoratedInterval::point(S::ZERO)?.set());
        Ok(QuadratureObligation::Established)
    };
    let verifier = QuadratureVerifier::<S>::new(&pieces, 1, &oracle, ValidatedOptions::new())?;
    let mut certificate = QuadratureCertificate::new(4096, 1)?;
    let mut workspace = QuadratureWorkspace::new();
    let report = verifier.generate_into(
        &ErrorTolerance::default(),
        &mut certificate,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    quadrature_generation_result(
        report.termination(),
        diagnostics(crate::quadrature::GENERATE, workspace.last_attempt()),
    )?;
    let status = verifier.verify_scoped(
        certificate.view(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    quadrature_verification_result(
        status,
        diagnostics(crate::quadrature::VERIFY, workspace.last_attempt()),
    )?;
    Interval::try_from(certificate.view().total[0].re)
}
