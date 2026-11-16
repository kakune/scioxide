//! Numerical resultants, subresultants and discriminants from equilibrated
//! Sylvester submatrices.
use crate::PolynomialApproximationPolicy;
use crate::util::{canonical_coefficients, finite_coefficient_exponent, invalid, scaled_complex};
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::AlgebraicScalar;
use scioxide_core::ScaledValue;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AlgorithmId, Argument, AttemptKind, AttemptOutcome, AttemptRecord, ExecutionContext,
    ExecutionPolicy, InvalidInputReason, NumericScalar, ProblemToken, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix, RankReport, SvdValues,
};

fn scaled_value<C: NumericScalar>(
    value: ScaledValue<scioxide_core::Complex<C::Real>>,
) -> Result<ScaledValue<C>, SciOxideError> {
    ScaledValue::from_component_parts(
        C::from_components(value.real().significand(), value.imaginary().significand()),
        [value.real().exponent(), value.imaginary().exponent()],
    )
}
fn shift<R: RealScalar>(
    value: ScaledValue<scioxide_core::Complex<R>>,
    exponent: i64,
) -> Result<ScaledValue<scioxide_core::Complex<R>>, SciOxideError> {
    Ok(ScaledValue::from_components(
        value.real().shifted(exponent)?,
        value.imaginary().shifted(exponent)?,
    ))
}

// Equilibration shifts by powers of two are exact unless a component underflows
// to zero, which would silently change the Sylvester structure (and therefore
// rank and determinant); that case is reported instead of rounded away.
fn round_preserving<C: NumericScalar>(
    value: ScaledValue<scioxide_core::Complex<C::Real>>,
) -> Result<C, SciOxideError> {
    Ok(C::from_components(
        value.real().try_value()?,
        value.imaginary().try_value()?,
    ))
}

/// Reports the degrees, scaling, numerical rank and conditioning of the
/// matrix whose determinant defines an [`ApproximateElimination`] value.
///
/// The matrix is the equilibrated Sylvester matrix or submatrix described on
/// [`approximate_resultant`]. Rank and condition come from its singular values.
/// They describe the matrix as a dense matrix, not the polynomials as
/// structured data, so they are neither coefficientwise error bounds nor a
/// proof of a common factor. A numerical rank below the matrix order does not
/// change the returned value.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
/// let report = result.report();
/// assert_eq!(report.degrees(), [1, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct EliminationReport<R: RealScalar> {
    degrees: [usize; 2],
    exponents: [i64; 2],
    rank: Option<RankReport<R>>,
    reciprocal_condition: R,
    policy: PolynomialApproximationPolicy<R>,
}
impl<R: RealScalar> EliminationReport<R> {
    /// Returns the degrees `[m, n]` of the inputs after trailing zero
    /// coefficients are removed.
    ///
    /// For a discriminant of degree `n < 2` this is `[n, 0]`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// let report = result.report();
    /// assert_eq!(report.degrees(), [1, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degrees(self) -> [usize; 2] {
        self.degrees
    }
    /// Returns the exponents `[e_f, e_g]` by which the inputs were scaled.
    ///
    /// Each input is multiplied by `2^(−e)`, where `e = ⌊log₂ max |cᵢ|⌋` over its
    /// coefficients (the larger of the real and imaginary parts for complex
    /// coefficients), so its largest component magnitude lies in `[1, 2)`.
    /// The returned value undoes this scaling exactly.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// let report = result.report();
    /// assert_eq!(report.equilibration_exponents(), [0, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn equilibration_exponents(self) -> [i64; 2] {
        self.exponents
    }
    /// Returns the numerical rank of the scaled matrix under the policy's
    /// [`rank_policy`](PolynomialApproximationPolicy::rank_policy).
    ///
    /// `None` when no matrix was formed: an input is the zero polynomial, or the
    /// matrix is empty because both inputs are constants.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// let report = result.report();
    /// assert_eq!(report.rank().map(|rank| rank.rank()), Some(2));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn rank(self) -> Option<RankReport<R>> {
        self.rank
    }
    /// Returns the reciprocal condition number `σ_min / σ_max` of the scaled
    /// matrix, in `[0, 1]`.
    ///
    /// It is 0 when an input is the zero polynomial and 1 when no matrix was
    /// formed. For a discriminant it describes `Res(p, p′)` before the division
    /// by the leading coefficient.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// let report = result.report();
    /// assert!(report.reciprocal_condition() > 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn reciprocal_condition(self) -> R {
        self.reciprocal_condition
    }
    /// Returns the policy whose rank policy determined [`rank`](Self::rank).
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// let report = result.report();
    /// assert_eq!(report.policy(), PolynomialApproximationPolicy::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn policy(self) -> PolynomialApproximationPolicy<R> {
        self.policy
    }
}
/// Holds a resultant, a discriminant or one coefficient of a subresultant,
/// with diagnostics.
///
/// The value is a [`ScaledValue`], whose real and imaginary parts each carry a
/// separate `i64` binary exponent, so it can exceed the range of `C`. Convert it
/// with [`ScaledValue::round`] or [`ScaledValue::try_value`].
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
/// assert!((result.value().round()? - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ApproximateElimination<C: NumericScalar> {
    value: ScaledValue<C>,
    report: EliminationReport<C::Real>,
}
impl<C: NumericScalar> ApproximateElimination<C> {
    /// Returns the value for the unscaled input coefficients.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// assert!((result.value().round()? - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn value(self) -> ScaledValue<C> {
        self.value
    }
    /// Returns the rank and conditioning diagnostics of the defining matrix.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
    /// assert_eq!(result.report().degrees(), [1, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn report(self) -> EliminationReport<C::Real> {
        self.report
    }
}

/// Validated input without trailing zeros, and its scaling exponent.
#[derive(Clone, Copy)]
struct Canonical<'a, C> {
    values: &'a [C],
    exponent: i64,
}
fn canonical<C: NumericScalar>(
    values: &[C],
    argument: Argument,
) -> Result<Canonical<'_, C>, SciOxideError> {
    let values = canonical_coefficients(values, argument)?;
    Ok(Canonical {
        values,
        exponent: finite_coefficient_exponent(values)?,
    })
}

// Standard descending-coefficient Sylvester rows. For subresultant j, retain
// the first L-1 columns and the column of x^k, where L=m+n-2j. Equilibration
// underflow is reported as a numerical failure.
pub(crate) fn fill_minor<C: NumericScalar>(
    left: &[C],
    right: &[C],
    j: usize,
    k: usize,
    exponents: [i64; 2],
    matrix: &mut Matrix<C>,
) -> Result<(), SciOxideError> {
    let (m, n) = (left.len() - 1, right.len() - 1);
    let order = matrix.rows();
    for (row, values) in matrix.as_mut_slice().chunks_mut(order.max(1)).enumerate() {
        let (p, offset, exponent) = if row < n - j {
            (left, row, exponents[0])
        } else {
            (right, row - (n - j), exponents[1])
        };
        for (column, value) in values.iter_mut().enumerate() {
            let column = if column + 1 == order {
                order + j - 1 - k
            } else {
                column
            };
            *value = if column >= offset && column - offset < p.len() {
                round_preserving(shift(
                    scaled_complex(p[p.len() - 1 - (column - offset)])?,
                    -exponent,
                )?)?
            } else {
                C::ZERO
            };
        }
    }
    debug_assert_eq!(order, m + n - 2 * j);
    Ok(())
}

/// Computes resultants or subresultant coefficients for polynomials of fixed
/// degrees.
///
/// This method value holds the validated degrees `m` and `n`, the subresultant
/// index `j` and the policy whose rank rule the reports use; it precomputes
/// nothing.
/// Index 0 selects the resultant. A positive index must satisfy
/// `j < min(m, n)`. The subresultant
///
/// ```text
/// S_j(x) = Σₖ s_{j,k} xᵏ,   k = 0, …, j,
/// ```
///
/// has coefficients `s_{j,k}` equal to the determinant of the
/// `L × L` matrix, `L = m + n − 2j`, formed from `n − j` shifted rows of `f`
/// and `m − j` shifted rows of `g` of the Sylvester matrix, keeping its first
/// `L − 1` columns and the column of `xᵏ`. The degree of `gcd(f, g)` is the
/// smallest `j` with `s_{j,j} ≠ 0`, and `S_j` is then a GCD of `f` and `g`.
///
/// Each coefficient costs O(L³) operations: a singular value decomposition for
/// the diagnostics and an LU factorization with partial pivoting for the
/// determinant. The workspace holds O(L²) values. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::RankPolicy;
/// use scioxide_poly::{PolynomialApproximationPolicy, SylvesterElimination};
///
/// // Res(1 + x, 3 + x) = 3 + x at x = −1, which is 2.
/// let policy = PolynomialApproximationPolicy::default().with_rank_policy(RankPolicy::Exact);
/// let resultant = SylvesterElimination::<f64>::new(1, 1, 0, policy)?;
/// let result = resultant.coefficient(&[1.0_f64, 1.0], &[3.0_f64, 1.0], 0)?;
/// assert!((result.value().round()? - 2.0).abs() < 1e-12);
/// assert_eq!(result.report().policy(), policy);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
/// Cambridge University Press, 2013, §6.3 and §6.10.
#[derive(Clone, Debug)]
pub struct SylvesterElimination<C: NumericScalar> {
    degrees: [usize; 2],
    index: usize,
    policy: PolynomialApproximationPolicy<C::Real>,
    order: usize,
}

/// Holds the matrix, LU factor and singular-value storage of a
/// [`SylvesterElimination`], and the record of its last attempt.
///
/// Created by [`SylvesterElimination::workspace`] with the size the method value
/// fixes.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::*;
/// let plan = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
/// let mut workspace = plan.workspace()?;
/// let result = plan.coefficient_scoped(
///     &[1.0_f64, 1.0],
///     &[2.0_f64, 1.0],
///     0,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!((result.value().round()? - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct EliminationWorkspace<C: NumericScalar> {
    buffers: EliminationBuffers<C>,
    attempt: Attempt,
}
impl<C: NumericScalar> EliminationWorkspace<C> {
    /// Returns the record of the last call that used this workspace.
    ///
    /// A call rejected before its epoch began, for example for an empty input,
    /// is recorded as `PreflightRejected`; otherwise the record holds the epoch,
    /// the outcome and the work counts.
    ///
    /// ```
    /// use scioxide_core::{AttemptOutcome, ExecutionContext};
    /// use scioxide_poly::*;
    /// let resultant = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// let mut workspace = resultant.workspace()?;
    /// let cx = ExecutionContext::sequential();
    /// resultant.coefficient_scoped(&[1.0, 1.0], &[2.0, 1.0], 0, &mut workspace, cx)?;
    /// let attempt = workspace.last_attempt().map(|attempt| attempt.outcome());
    /// assert_eq!(attempt, Some(AttemptOutcome::Succeeded));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.attempt.record.as_ref()
    }
}
#[derive(Debug)]
struct EliminationBuffers<C: NumericScalar> {
    matrix: Matrix<C>,
    factor: LuDecomposition<C>,
    lu: LinalgWorkspace<'static, C>,
    spectrum: SvdValues<C::Real>,
    svd: LinalgWorkspace<'static, C>,
}
/// Solve state and attempt record shared by the elimination workspaces.
#[derive(Debug, Default)]
struct Attempt {
    state: SolveState,
    record: Option<AttemptRecord<()>>,
}
impl Attempt {
    fn reject(&mut self) {
        self.record = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
    }
    /// Run `work` as one epoch of `scope` and record the attempt.
    fn run<'capability, T>(
        &mut self,
        scope: SolveScope<'_, 'capability>,
        method: AlgorithmId,
        order: usize,
        work: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let identity = SolveIdentity::new(ProblemToken::new(0), method, order as u128, 0);
        let mut entered = false;
        let result = scope.run(&mut self.state, identity, SolveBudgets::unbounded(), |cx| {
            entered = true;
            work(cx)
        });
        if !entered {
            self.reject();
            return result;
        }
        self.record = Some(AttemptRecord::new(
            self.state.epoch(),
            AttemptKind::Call,
            if result.is_ok() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Failed
            },
            self.state.counts(),
            result.as_ref().ok().map(|_| ()),
        ));
        result
    }
}
const ELIMINATION: AlgorithmId =
    scioxide_core::algorithm_id!("scioxide-poly", "sylvester-elimination", 1);
const DISCRIMINANT: AlgorithmId =
    scioxide_core::algorithm_id!("scioxide-poly", "sylvester-discriminant", 1);
impl<C: NumericScalar> SylvesterElimination<C> {
    /// Creates the method value for inputs of degrees `left_degree` and
    /// `right_degree`, subresultant index `index` and the rank rule of `policy`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `index` is positive and not below
    ///   both degrees.
    /// - `SizeOverflow` if the matrix order overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let plan = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// assert_eq!(plan.matrix_order(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        left_degree: usize,
        right_degree: usize,
        index: usize,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        if index != 0 && index >= left_degree.min(right_degree) {
            return Err(invalid(
                scioxide_core::argument!("index"),
                InvalidInputReason::Inconsistent,
                "subresultant index must be below both degrees",
            ));
        }
        let order = left_degree
            .checked_add(right_degree)
            .and_then(|n| n.checked_sub(index.checked_mul(2)?))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sylvester_matrix_order"))
            })?;
        Ok(Self {
            degrees: [left_degree, right_degree],
            index,
            policy,
            order,
        })
    }
    /// Returns the order `L = m + n − 2j` of the determinant matrix.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let plan = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// assert_eq!(plan.matrix_order(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn matrix_order(&self) -> usize {
        self.order
    }
    /// Allocates a workspace for this method value.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the matrix and its
    ///   factorization storage cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let plan = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// let mut workspace = plan.workspace()?;
    /// let result = plan.coefficient_scoped(
    ///     &[1.0, 1.0],
    ///     &[2.0, 1.0],
    ///     0,
    ///     &mut workspace,
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert!((result.value().round()? - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(&self) -> Result<EliminationWorkspace<C>, SciOxideError> {
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        let mut factor = LuDecomposition::empty();
        factor.try_reserve(self.order)?;
        let mut spectrum = SvdValues::empty();
        spectrum.try_reserve(self.order)?;
        Ok(EliminationWorkspace {
            buffers: EliminationBuffers {
                matrix: Matrix::zeros(self.order, self.order)?,
                factor,
                spectrum,
                lu: LuDecomposition::workspace(&scioxide_linalg::LuWorkspaceRequest::factor(
                    self.order, &options,
                ))?,
                svd: SvdValues::workspace::<C>(&scioxide_linalg::SvdWorkspaceRequest::new(
                    self.order, self.order, &options,
                ))?,
            },
            attempt: Attempt::default(),
        })
    }
    /// Computes the coefficient `s_{j,k}` of `xᵏ` in the subresultant `S_j` of
    /// `left` and `right`, with `k = coefficient`.
    ///
    /// Coefficients are in ascending order of degree. For the resultant, `j` and
    /// `k` are 0. If either input is the zero polynomial, the value is 0. An
    /// exactly singular LU pivot also gives the value 0, and the report then
    /// shows how close to singular the matrix is. The call allocates a workspace
    /// and runs [`coefficient_scoped`](Self::coefficient_scoped) with
    /// `ExecutionContext::sequential_native()`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `left` or `right` is empty.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if the degrees after removing trailing
    ///   zeros differ from the method value's, or `coefficient` exceeds the
    ///   index.
    /// - `NumericalInstability` (`Underflow`) if a coefficient underflows to
    ///   zero when its input is scaled.
    /// - `SizeOverflow` or `NumericalInstability` (`ExponentOutOfRange`) if the
    ///   binary exponent of the value leaves the `i64` range.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::SylvesterElimination;
    /// // Res(1 + x, 2 + x) = 1.
    /// let resultant = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// let result = resultant.coefficient(&[1.0, 1.0], &[2.0, 1.0], 0)?;
    /// assert!((result.value().round()? - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coefficient(
        &self,
        left: &[C],
        right: &[C],
        coefficient: usize,
    ) -> Result<ApproximateElimination<C>, SciOxideError> {
        self.coefficient_scoped(
            left,
            right,
            coefficient,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )
    }
    /// Computes the coefficient of [`coefficient`](Self::coefficient) inside the
    /// caller's scope, without allocating.
    ///
    /// The call runs as one epoch of `scope`, a child epoch when `scope` comes
    /// from a `&mut SolveContext`, and observes its cancellation and deadline
    /// before, between and after the singular value decomposition and the LU
    /// factorization. On error it changes only the attempt record of
    /// `workspace`, which [`EliminationWorkspace::last_attempt`] reads.
    ///
    /// # Errors
    ///
    /// - The errors of [`coefficient`](Self::coefficient), except
    ///   `AllocationFailure`.
    /// - `DimensionMismatch` if `workspace` was created for another matrix
    ///   order.
    /// - `ExecutionStopped` if the scope is cancelled or its deadline passes.
    /// - `AccountingInvariant` or `SizeOverflow` if the epoch cannot begin.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::SylvesterElimination;
    /// let resultant = SylvesterElimination::<f64>::new(1, 1, 0, Default::default())?;
    /// let mut workspace = resultant.workspace()?;
    /// let cx = ExecutionContext::sequential_native();
    /// let result = resultant.coefficient_scoped(&[1.0, 1.0], &[2.0, 1.0], 0, &mut workspace, cx)?;
    /// assert!((result.value().round()? - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coefficient_scoped<'scope, 'capability>(
        &self,
        left: &[C],
        right: &[C],
        coefficient: usize,
        workspace: &mut EliminationWorkspace<C>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ApproximateElimination<C>, SciOxideError> {
        let operands = canonical(left, scioxide_core::argument!("left"))
            .and_then(|left| Ok((left, canonical(right, scioxide_core::argument!("right"))?)));
        let (left, right) = match operands {
            Ok(operands) => operands,
            Err(error) => {
                workspace.attempt.reject();
                return Err(error);
            }
        };
        let buffers = &mut workspace.buffers;
        workspace
            .attempt
            .run(scope.into(), ELIMINATION, self.order, |cx| {
                self.coefficient_prepared(left, right, coefficient, buffers, cx)
            })
    }
    fn coefficient_prepared(
        &self,
        left: Canonical<'_, C>,
        right: Canonical<'_, C>,
        coefficient: usize,
        workspace: &mut EliminationBuffers<C>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<ApproximateElimination<C>, SciOxideError> {
        cx.execution().check()?;
        if workspace.matrix.rows() != self.order {
            return Err(SciOxideError::dimension_mismatch(
                self.order,
                workspace.matrix.rows(),
            ));
        }
        let exponents = [left.exponent, right.exponent];
        let (left, right) = (left.values, right.values);
        if [left.len() - 1, right.len() - 1] != self.degrees || coefficient > self.index {
            let argument = if coefficient > self.index {
                scioxide_core::argument!("coefficient")
            } else if left.len() - 1 != self.degrees[0] {
                scioxide_core::argument!("left")
            } else {
                scioxide_core::argument!("right")
            };
            return Err(invalid(
                argument,
                InvalidInputReason::Inconsistent,
                "prepared degrees or coefficient index mismatch",
            ));
        }
        let mut report = EliminationReport {
            degrees: self.degrees,
            exponents,
            rank: None,
            reciprocal_condition: C::Real::ONE,
            policy: self.policy,
        };
        cx.execution().check()?;
        let value = if left == [C::ZERO] || right == [C::ZERO] {
            report.reciprocal_condition = C::Real::ZERO;
            scaled_value(ScaledValue::ZERO)?
        } else if workspace.matrix.rows() == 0 {
            scaled_value(scaled_complex(C::ONE)?)?
        } else {
            fill_minor(
                left,
                right,
                self.index,
                coefficient,
                exponents,
                &mut workspace.matrix,
            )?;
            cx.execution().check()?;
            scioxide_linalg::SvdValues::compute_into(
                &workspace.matrix,
                &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut workspace.spectrum,
                &mut workspace.svd,
                &mut *cx,
            )?;
            report.rank = Some(workspace.spectrum.analyze_rank(self.policy.rank_policy())?);
            report.reciprocal_condition = workspace
                .spectrum
                .condition_estimate()?
                .reciprocal()
                .value();
            cx.execution().check()?;
            match LuDecomposition::factor_into(
                &workspace.matrix,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut workspace.factor,
                &mut workspace.lu,
                &mut *cx,
            ) {
                // An exactly singular pivot means the subresultant coefficient is zero; the
                // rank report computed above already describes how close to singular it is.
                Err(ref error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::Singular { .. }
                    ) =>
                {
                    scaled_value(ScaledValue::ZERO)?
                }
                Err(error) => return Err(error),
                Ok(()) => {
                    let determinant = workspace.factor.scaled_determinant()?;
                    // The determinant is homogeneous in each operand's block of rows: scaling the
                    // `n - j` rows of `left` by `2^{-e_l}` and the `m - j` rows of `right` by
                    // `2^{-e_r}` scales it by `2^{-(e_l (n - j) + e_r (m - j))}`. That factor is
                    // undone here, in the exponent, so the result can exceed the scalar range.
                    let exponent = scale_exponent(
                        exponents,
                        [self.degrees[1] - self.index, self.degrees[0] - self.index],
                    )?;
                    // Shift each component here (rather than through the value's own
                    // shift) so exponent exhaustion propagates as a numerical error.
                    scaled_value(shift(
                        ScaledValue::from_components(determinant.real(), determinant.imaginary()),
                        exponent,
                    )?)?
                }
            }
        };
        cx.execution().check()?;
        Ok(ApproximateElimination { value, report })
    }
}
fn scale_exponent(e: [i64; 2], degrees: [usize; 2]) -> Result<i64, SciOxideError> {
    e.into_iter()
        .zip(degrees)
        .try_fold(0i64, |sum, (e, n)| {
            sum.checked_add(e.checked_mul(i64::try_from(n).ok()?)?)
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("determinant_exponent"))
        })
}
/// Computes the resultant of two polynomials in floating-point arithmetic.
///
/// `left` holds the coefficients of `f`, of degree `m` and leading coefficient
/// `a`, and `right` those of `g`, of degree `n`, both in ascending order of
/// degree. Trailing zeros are removed first. The resultant is the determinant
/// of the `(m + n) × (m + n)` Sylvester matrix:
///
/// ```text
/// Res(f, g) = aⁿ ∏ᵢ g(αᵢ),
/// ```
///
/// where `αᵢ` are the roots of `f`. It is 0 if either input is the zero
/// polynomial and 1 if both are constants.
///
/// Each input is first scaled by a power of two so that its largest coefficient
/// lies in `[1, 2)` (see [`EliminationReport::equilibration_exponents`]). The
/// determinant comes from an LU factorization with partial pivoting, and the
/// singular values give the rank and condition diagnostics. The cost is
/// O((m + n)³). The scaled Sylvester matrix is close to singular when `f` and
/// `g` nearly share a root; a small
/// [`reciprocal_condition`](EliminationReport::reciprocal_condition) then warns
/// that the value may have a large relative error. For exact coefficients use
/// [`exact_subresultants`](crate::exact_subresultants); for repeated calls with
/// the same degrees use [`SylvesterElimination`].
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `left` or `right` is empty.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `NumericalInstability` (`Underflow`) if a coefficient underflows to zero
///   when its input is scaled.
/// - `SizeOverflow` or `NumericalInstability` (`ExponentOutOfRange`) if the
///   binary exponent of the value leaves the `i64` range.
/// - `AllocationFailure` if the matrix storage cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// // Res(1 + x, 2 + x) = 2 + x at x = −1, which is 1.
/// let result = approximate_resultant(&[1.0_f64, 1.0], &[2.0_f64, 1.0])?;
/// assert!((result.value().round()? - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn approximate_resultant<C: NumericScalar>(
    left: &[C],
    right: &[C],
) -> Result<ApproximateElimination<C>, SciOxideError> {
    let degrees = [
        canonical(left, scioxide_core::argument!("left"))?
            .values
            .len()
            - 1,
        canonical(right, scioxide_core::argument!("right"))?
            .values
            .len()
            - 1,
    ];
    SylvesterElimination::new(
        degrees[0],
        degrees[1],
        0,
        PolynomialApproximationPolicy::default(),
    )?
    .coefficient(left, right, 0)
}
/// Computes all coefficients of the subresultant `S_j` of index `j ≥ 1`.
///
/// Returns `j + 1` values, the coefficients `s_{j,0}, …, s_{j,j}` of
/// `S_j(x)` in ascending order of degree, each with its own diagnostics. The
/// definition is on [`SylvesterElimination`]. The index must satisfy
/// `1 ≤ j < min(m, n)`; use [`approximate_resultant`] for `j = 0`.
///
/// Costs O((j + 1) L³) operations with `L = m + n − 2j`, and allocates
/// O(L² + j) storage. Use [`SylvesterElimination`] to compute only some
/// coefficients.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `index` is 0.
/// - `InvalidInput` (`Inconsistent`) if `index` is not below both degrees.
/// - `InvalidInput` (`Empty`) if `left` or `right` is empty.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `NumericalInstability` (`Underflow`) if a coefficient underflows to zero
///   when its input is scaled.
/// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated
///   or an exponent overflows.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// let results = approximate_subresultant(&[1.0_f64, 0.0, 1.0], &[2.0_f64, 0.0, 1.0], 1)?;
/// assert_eq!(results.len(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn approximate_subresultant<C: NumericScalar>(
    left: &[C],
    right: &[C],
    index: usize,
) -> Result<Vec<ApproximateElimination<C>>, SciOxideError> {
    if index == 0 {
        return Err(invalid(
            scioxide_core::argument!("index"),
            InvalidInputReason::OutOfRange,
            "subresultant index must be positive; use approximate_resultant for index zero",
        ));
    }
    let degrees = [
        canonical(left, scioxide_core::argument!("left"))?
            .values
            .len()
            - 1,
        canonical(right, scioxide_core::argument!("right"))?
            .values
            .len()
            - 1,
    ];
    let subresultant = SylvesterElimination::new(
        degrees[0],
        degrees[1],
        index,
        PolynomialApproximationPolicy::default(),
    )?;
    let mut workspace = subresultant.workspace()?;
    let count = scioxide_core::backend::checked_add(
        index,
        1,
        scioxide_core::quantity!("subresultant_coefficients"),
    )?;
    let mut output = scioxide_core::backend::try_with_capacity(
        count,
        scioxide_core::quantity!("subresultant_coefficients"),
    )?;
    for k in 0..count {
        output.push(subresultant.coefficient_scoped(
            left,
            right,
            k,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?);
    }
    Ok(output)
}
/// Computes the discriminant of a polynomial in floating-point arithmetic.
///
/// For `p` of degree `n ≥ 2` with leading coefficient `a` and roots `αᵢ`,
///
/// ```text
/// disc(p) = a^(2n−2) ∏_{i<j} (αᵢ − αⱼ)² = (−1)^(n(n−1)/2) Res(p, p′) / a.
/// ```
///
/// For example `disc(a x² + b x + c) = b² − 4ac`. Polynomials of degree 0 or 1
/// have discriminant 1. Coefficients are in ascending order of degree, and
/// trailing zeros are removed first.
///
/// Scales `p` by a power of two, forms `p′ / n` from the scaled coefficients
/// so that `n aₙ` cannot overflow, and computes `Res(p, p′)` as
/// [`approximate_resultant`] does, in O(n³) operations. The report describes
/// the Sylvester matrix of `p` and `p′`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `coefficients` is empty.
/// - `InvalidInput` (`Inconsistent`) if `p` is the zero polynomial.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `NumericalInstability` if a scaled coefficient underflows to zero or an
///   exponent leaves the `i64` range.
/// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// // disc(x² − 1) = 0² − 4 · 1 · (−1) = 4.
/// let result = approximate_discriminant(&[-1.0_f64, 0.0, 1.0])?;
/// assert!((result.value().round()? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn approximate_discriminant<C: NumericScalar>(
    coefficients: &[C],
) -> Result<ApproximateElimination<C>, SciOxideError> {
    let degree = canonical(coefficients, scioxide_core::argument!("coefficients"))?
        .values
        .len()
        - 1;
    SylvesterDiscriminant::new(degree, PolynomialApproximationPolicy::default())?
        .compute(coefficients)
}

/// Computes discriminants of polynomials of a fixed degree.
///
/// The definition and method are on [`approximate_discriminant`]. This method
/// value holds the validated degree `n` and a [`SylvesterElimination`] for `p`
/// and `p′`; it precomputes nothing. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// // disc(x² − 1) = 4.
/// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
/// let result = discriminant.compute(&[-1.0_f64, 0.0, 1.0])?;
/// assert!((result.value().round()? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SylvesterDiscriminant<C: NumericScalar> {
    degree: usize,
    elimination: SylvesterElimination<C>,
}
/// Holds the derivative coefficients and elimination storage of a
/// [`SylvesterDiscriminant`], and the record of its last attempt.
///
/// Created by [`SylvesterDiscriminant::workspace`].
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::*;
/// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
/// let mut workspace = discriminant.workspace()?;
/// let cx = ExecutionContext::sequential();
/// let result = discriminant.compute_scoped(&[-1.0_f64, 0.0, 1.0], &mut workspace, cx)?;
/// assert!((result.value().round()? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct DiscriminantWorkspace<C: NumericScalar> {
    derivative: Vec<C>,
    elimination: EliminationWorkspace<C>,
}
impl<C: NumericScalar> DiscriminantWorkspace<C> {
    /// Returns the record of the last call that used this workspace, as
    /// [`EliminationWorkspace::last_attempt`] describes.
    ///
    /// ```
    /// use scioxide_core::{AttemptOutcome, ExecutionContext};
    /// use scioxide_poly::*;
    /// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
    /// let mut workspace = discriminant.workspace()?;
    /// let cx = ExecutionContext::sequential();
    /// assert!(
    ///     discriminant
    ///         .compute_scoped(&[], &mut workspace, cx)
    ///         .is_err()
    /// );
    /// let attempt = workspace.last_attempt().map(|attempt| attempt.outcome());
    /// assert_eq!(attempt, Some(AttemptOutcome::PreflightRejected));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.elimination.last_attempt()
    }
}
impl<C: NumericScalar> SylvesterDiscriminant<C> {
    /// Creates the method value for polynomials of degree `degree` and the rank
    /// rule of `policy`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the Sylvester matrix order `2n − 1` overflows
    ///   `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
    /// let result = discriminant.compute(&[-1.0_f64, 0.0, 1.0])?;
    /// assert!((result.value().round()? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        degree: usize,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let degrees = if degree < 2 {
            [0, 0]
        } else {
            [degree, degree - 1]
        };
        Ok(Self {
            degree,
            elimination: SylvesterElimination::new(degrees[0], degrees[1], 0, policy)?,
        })
    }
    /// Allocates a workspace for this method value.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::*;
    /// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
    /// let mut workspace = discriminant.workspace()?;
    /// let cx = ExecutionContext::sequential_native();
    /// let result = discriminant.compute_scoped(&[-1.0_f64, 0.0, 1.0], &mut workspace, cx)?;
    /// assert!((result.value().round()? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(&self) -> Result<DiscriminantWorkspace<C>, SciOxideError> {
        Ok(DiscriminantWorkspace {
            derivative: try_filled(
                self.degree,
                C::ZERO,
                scioxide_core::quantity!("derivative_coefficients"),
            )?,
            elimination: self.elimination.workspace()?,
        })
    }
    /// Computes the discriminant of the polynomial with the given coefficients
    /// in ascending order of degree.
    ///
    /// The call allocates a workspace and runs
    /// [`compute_scoped`](Self::compute_scoped) with
    /// `ExecutionContext::sequential_native()`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `coefficients` is empty.
    /// - `InvalidInput` (`Inconsistent`) if the polynomial is zero or its degree
    ///   after removing trailing zeros differs from the method value's.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a scaled coefficient underflows to zero or an
    ///   exponent leaves the `i64` range.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::SylvesterDiscriminant;
    /// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
    /// let result = discriminant.compute(&[-1.0, 0.0, 1.0])?;
    /// assert!((result.value().round()? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compute(&self, coefficients: &[C]) -> Result<ApproximateElimination<C>, SciOxideError> {
        self.compute_scoped(
            coefficients,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )
    }
    /// Computes the discriminant of [`compute`](Self::compute) inside the
    /// caller's scope, without allocating.
    ///
    /// The call runs as one epoch of `scope`, a child epoch when `scope` comes
    /// from a `&mut SolveContext`, including the resultant of `p` and `p′`. It
    /// observes the scope's cancellation and deadline, and on error changes only
    /// the attempt record of `workspace`.
    ///
    /// # Errors
    ///
    /// - The errors of [`compute`](Self::compute), except `AllocationFailure`.
    /// - `DimensionMismatch` if `workspace` was created for another degree.
    /// - `ExecutionStopped` if the scope is cancelled or its deadline passes.
    /// - `AccountingInvariant` or `SizeOverflow` if the epoch cannot begin.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::SylvesterDiscriminant;
    /// let discriminant = SylvesterDiscriminant::<f64>::new(2, Default::default())?;
    /// let mut workspace = discriminant.workspace()?;
    /// let cx = ExecutionContext::sequential_native();
    /// let result = discriminant.compute_scoped(&[-1.0, 0.0, 1.0], &mut workspace, cx)?;
    /// assert!((result.value().round()? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compute_scoped<'scope, 'capability>(
        &self,
        coefficients: &[C],
        workspace: &mut DiscriminantWorkspace<C>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ApproximateElimination<C>, SciOxideError> {
        let DiscriminantWorkspace {
            derivative,
            elimination,
        } = workspace;
        let p = match canonical(coefficients, scioxide_core::argument!("coefficients")) {
            Ok(p) => p,
            Err(error) => {
                elimination.attempt.reject();
                return Err(error);
            }
        };
        let buffers = &mut elimination.buffers;
        elimination
            .attempt
            .run(scope.into(), DISCRIMINANT, self.elimination.order, |cx| {
                self.compute_prepared(p, derivative, buffers, cx)
            })
    }
    fn compute_prepared(
        &self,
        p: Canonical<'_, C>,
        derivative: &mut [C],
        buffers: &mut EliminationBuffers<C>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<ApproximateElimination<C>, SciOxideError> {
        cx.execution().check()?;
        if derivative.len() != self.degree {
            return Err(SciOxideError::dimension_mismatch(
                self.degree,
                derivative.len(),
            ));
        }
        let exponent = p.exponent;
        let p = p.values;
        if p == [C::ZERO] || p.len() - 1 != self.degree {
            return Err(invalid(
                scioxide_core::argument!("coefficients"),
                InvalidInputReason::Inconsistent,
                "nonzero polynomial with prepared degree required",
            ));
        }
        let n = self.degree;
        cx.execution().check()?;
        if n < 2 {
            let one = [C::ONE];
            let one = canonical(&one, scioxide_core::argument!("coefficients"))?;
            let mut result = self
                .elimination
                .coefficient_prepared(one, one, 0, buffers, cx)?;
            cx.execution().check()?;
            result.report.degrees = [n, 0];
            return Ok(result);
        }
        // Normalize coefficients before differentiating so n*a_n cannot overflow.
        // The derivative is stored as `p' / (n 2^e)`: coefficients `(i + 1) p_{i+1}` are
        // divided by `n` so they stay on the scale of `p`. Since `Res(p, c q) = c^{deg p}
        // Res(p, q)`, the factor `n 2^e` is restored `n` times below.
        let width = C::Real::try_from_usize_exact(n)?;
        for (i, value) in derivative.iter_mut().enumerate() {
            cx.execution().check()?;
            *value = round_preserving(shift(scaled_complex(p[i + 1])?, -exponent)?.mul(
                scaled_complex(C::from_real(C::Real::try_from_usize_exact(i + 1)? / width))?,
            )?)?;
        }
        let mut result = self.elimination.coefficient_prepared(
            Canonical {
                values: p,
                exponent,
            },
            Canonical {
                values: derivative,
                exponent: finite_coefficient_exponent(derivative)?,
            },
            0,
            buffers,
            cx,
        )?;
        cx.execution().check()?;
        let mut value = ScaledValue::from_components(result.value.real(), result.value.imaginary());
        let derivative_scale = shift(scaled_complex(C::from_real(width))?, exponent)?;
        for _ in 0..n {
            cx.execution().check()?;
            value = value.mul(derivative_scale)?;
        }
        // `disc(p) = (-1)^{n(n-1)/2} Res(p, p') / a_n`; the sign is negative exactly
        // when `n mod 4` is 2 or 3.
        value = value.div(scaled_complex(p[n])?)?;
        if n % 4 >= 2 {
            value = value.neg();
        }
        result.value = scaled_value(value)?;
        cx.execution().check()?;
        Ok(result)
    }
}
