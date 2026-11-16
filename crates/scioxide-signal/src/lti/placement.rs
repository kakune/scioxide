use super::*;

/// Options for [`StateSpace::place_poles_with_options`] and
/// [`StateSpace::place_poles_scoped`].
///
/// The defaults are [`RankPolicy::DimensionScaledEpsilon`] and a residual
/// tolerance of `√ε`, where `ε` is the machine epsilon. The placement checks
/// the options.
///
/// ```
/// use scioxide_linalg::RankPolicy;
/// use scioxide_signal::PolePlacementOptions;
/// let options = PolePlacementOptions::<f64>::new()
///     .with_rank_policy(RankPolicy::Exact)
///     .with_residual_tolerance(1e-9);
/// assert_eq!(options.residual_tolerance(), 1e-9);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolePlacementOptions<S: RealScalar> {
    rank_policy: RankPolicy<S>,
    residual_tolerance: S,
}

impl<S: RealScalar> PolePlacementOptions<S> {
    /// Creates the default options.
    pub const fn new() -> Self {
        Self {
            rank_policy: RankPolicy::DimensionScaledEpsilon,
            residual_tolerance: S::SQRT_EPSILON,
        }
    }
    /// Sets the rank policy for every numerical rank decision of the placement:
    /// the controllability matrix, `B`, the eigenvector null spaces and the
    /// eigenvector matrix.
    pub const fn with_rank_policy(mut self, rank_policy: RankPolicy<S>) -> Self {
        self.rank_policy = rank_policy;
        self
    }
    /// Sets the tolerance on [`PolePlacementReport::maximum_residual`]. It must
    /// be finite and positive.
    pub const fn with_residual_tolerance(mut self, residual_tolerance: S) -> Self {
        self.residual_tolerance = residual_tolerance;
        self
    }
    /// Returns the rank policy.
    pub const fn rank_policy(self) -> RankPolicy<S> {
        self.rank_policy
    }
    /// Returns the residual tolerance.
    pub const fn residual_tolerance(self) -> S {
        self.residual_tolerance
    }
}

impl<S: RealScalar> Default for PolePlacementOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Termination status of [`StateSpace::place_poles_with_options`] and
/// [`StateSpace::place_poles_scoped`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolePlacementTermination {
    /// The achieved poles lie within the residual tolerance of the requested poles.
    Converged,
    /// The gain and achieved poles are finite, but the achieved poles miss the
    /// residual tolerance.
    ResidualToleranceNotMet,
}

/// Result of [`StateSpace::place_poles`]: the gain, the achieved poles and diagnostics.
///
/// Check [`Self::is_success`] before using the gain. An unsuccessful report
/// still holds the gain and the achieved poles.
#[derive(Clone, Debug)]
pub struct PolePlacementReport<T: NumericScalar> {
    gain: Matrix<T>,
    requested_poles: Vec<Complex<T::Real>>,
    achieved_poles: Vec<Complex<T::Real>>,
    maximum_residual: T::Real,
    controllability: RankReport<T::Real>,
    controllability_condition: ConditionEstimate<T::Real>,
    transfer_condition: Option<ConditionEstimate<T::Real>>,
    input_condition: ConditionEstimate<T::Real>,
    termination: PolePlacementTermination,
    counts: scioxide_core::SolveCounts,
}

impl<T: NumericScalar> PolePlacementReport<T> {
    /// Returns the termination status.
    pub const fn termination(&self) -> PolePlacementTermination {
        self.termination
    }
    /// Returns `true` if the termination is [`PolePlacementTermination::Converged`].
    pub fn is_success(&self) -> bool {
        self.termination == PolePlacementTermination::Converged
    }
    /// Returns the work counts: one iteration per eigenvector attempt (one for
    /// the full-input case), plus the counts of the nested decompositions.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns `ResidualChecked` if the placement converged and `DiagnosticOnly` otherwise.
    pub fn trust(&self) -> scioxide_core::PartialResultTrust {
        if self.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        }
    }
    /// Returns the condition estimate of `B` from its singular values.
    pub const fn input_condition(&self) -> ConditionEstimate<T::Real> {
        self.input_condition
    }
    /// Returns the gain `K`, of shape `m × n`, of the closed-loop matrix `A − B K`.
    pub const fn gain(&self) -> &Matrix<T> {
        &self.gain
    }
    /// Returns the requested poles as placed, sorted.
    ///
    /// For a real system, each nearly conjugate pair is replaced by the exact
    /// conjugates of its member in the upper half-plane, and real poles come first.
    pub fn requested_poles(&self) -> &[Complex<T::Real>] {
        &self.requested_poles
    }
    /// Returns the eigenvalues of `A − B K`, sorted by real part and then by imaginary part.
    pub fn achieved_poles(&self) -> &[Complex<T::Real>] {
        &self.achieved_poles
    }
    /// Returns the largest relative distance between a requested and an achieved pole.
    ///
    /// Each requested pole `λ` is matched in turn to the nearest unmatched
    /// achieved pole `μ`, and the residual is the maximum of `|μ − λ| / (1 + |λ|)`.
    pub const fn maximum_residual(&self) -> T::Real {
        self.maximum_residual
    }
    /// Returns the rank report of the controllability matrix `[B, AB, …, Aⁿ⁻¹B]`.
    pub const fn controllability(&self) -> RankReport<T::Real> {
        self.controllability
    }
    /// Returns the condition estimate of the controllability matrix from its singular values.
    pub const fn controllability_condition(&self) -> ConditionEstimate<T::Real> {
        self.controllability_condition
    }
    /// Returns the condition estimate of the eigenvector matrix `V`.
    ///
    /// It is `None` when `rank(B) = n`, where the gain comes from the full-input
    /// construction without eigenvectors.
    pub const fn transfer_condition(&self) -> Option<ConditionEstimate<T::Real>> {
        self.transfer_condition
    }
}

impl<T> StateSpace<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Computes a state-feedback gain `K` that places the eigenvalues of `A − B K` at `requested`.
    ///
    /// `requested` holds `n` poles. The pair `(A, B)` must be controllable
    /// under the rank policy, and no pole may occur more often than `rank(B)`.
    /// A real system needs conjugate pairs: each non-real pole must have a
    /// partner within relative distance `32 ε` of its conjugate, and the pair
    /// is made exactly conjugate. A pole that close to its own conjugate and
    /// without a partner is moved to the real axis.
    ///
    /// When `rank(B) = n`, `K` solves `B K = A − Λ` by the SVD of `B`, where `Λ`
    /// is block diagonal with the requested poles as eigenvalues (2 × 2 real
    /// blocks for the conjugate pairs of a real system). Otherwise
    /// each pole `λⱼ` gives a null space of `[A − λⱼ I, −B]`. Choosing a vector
    /// `(vⱼ, wⱼ)` from each null space gives `K = W V⁻¹`, so that
    /// `(A − B K) vⱼ = λⱼ vⱼ`. Successive attempts choose different null-space
    /// columns until the eigenvector matrix `V` has full rank. For a real system
    /// `K` is real in exact arithmetic, and the real part of the computed `K` is
    /// returned.
    ///
    /// The eigenvalues of `A − B K` are then computed independently. The
    /// placement converges when [`PolePlacementReport::maximum_residual`] is at
    /// most the residual tolerance. The accuracy of the placed poles depends on
    /// the condition of `V`, which the report gives. This form uses
    /// [`PolePlacementOptions::default`] and runs the decompositions in a fresh
    /// root scope. [`place_poles_with_options`](Self::place_poles_with_options)
    /// takes options and reports a missed tolerance in its result, and
    /// [`place_poles_scoped`](Self::place_poles_scoped) runs inside the
    /// caller's solve.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_linalg::Matrix;
    /// use scioxide_signal::{LtiTime, StateSpace};
    /// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    /// let b = Matrix::from_vec(1, 1, vec![1.])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let system = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
    /// let report = system.place_poles(&[Complex::new(-2., 0.)])?;
    /// // A − B K = −1 − K = −2.
    /// assert!((report.gain().as_slice()[0] - 1.).abs() < 1e-14);
    /// assert!(report.transfer_condition().is_none());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if the model has no states or no inputs.
    /// - `DimensionMismatch` if `requested` does not have `n` poles.
    /// - `NonFinite` if a requested pole is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if a real system gets a non-real pole
    ///   without a conjugate partner, or a pole occurs more than `rank(B)` times.
    /// - `NumericalInstability` (`LossOfSignificance`) if `(A, B)` is not
    ///   controllable under the rank policy, or no attempt gives a full-rank `V`.
    /// - `ConvergenceFailure` (`Stagnation`) if the achieved poles miss the
    ///   residual tolerance.
    pub fn place_poles(
        &self,
        requested: &[Complex<T::Real>],
    ) -> Result<PolePlacementReport<T>, SciOxideError> {
        self.place_poles_with_options(requested, PolePlacementOptions::default())
            .and_then(require_success)
    }

    /// Computes a state-feedback gain as [`place_poles`](Self::place_poles)
    /// does, with the rank policy and residual tolerance of `options`.
    ///
    /// A placement that misses the residual tolerance is not an error: the
    /// report then has [`PolePlacementTermination::ResidualToleranceNotMet`]
    /// and still holds the gain. Check [`PolePlacementReport::is_success`]
    /// before using the gain.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_linalg::{Matrix, RankPolicy};
    /// use scioxide_signal::{LtiTime, PolePlacementOptions, StateSpace};
    /// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    /// let b = Matrix::from_vec(1, 1, vec![1.])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let system = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
    /// let options = PolePlacementOptions::new().with_rank_policy(RankPolicy::Exact);
    /// let report = system.place_poles_with_options(&[Complex::new(-2., 0.)], options)?;
    /// assert!(report.is_success() && report.maximum_residual() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`place_poles`](Self::place_poles), except that a
    ///   missed residual tolerance is reported in the result.
    /// - `InvalidInput` (`OutOfRange`) if the residual tolerance is not positive.
    /// - `NonFinite` if the residual tolerance is NaN or infinite.
    pub fn place_poles_with_options(
        &self,
        requested: &[Complex<T::Real>],
        options: PolePlacementOptions<T::Real>,
    ) -> Result<PolePlacementReport<T>, SciOxideError> {
        self.place_poles_scoped(requested, options, ExecutionContext::sequential_native())
    }

    /// Computes a state-feedback gain as
    /// [`place_poles_with_options`](Self::place_poles_with_options) does, as
    /// one epoch of `scope`.
    ///
    /// The decompositions and attempts count toward the limits of `scope` and
    /// observe its cancellation and deadline, as described in
    /// [scoped designs](crate#scoped-designs).
    ///
    /// # Errors
    ///
    /// - The errors of [`place_poles_with_options`](Self::place_poles_with_options).
    /// - `ConvergenceFailure` if the attempts or the nested decompositions
    ///   exhaust the limits of `scope`.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn place_poles_scoped<'scope, 'capability>(
        &self,
        requested: &[Complex<T::Real>],
        options: PolePlacementOptions<T::Real>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<PolePlacementReport<T>, SciOxideError> {
        use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};
        let tolerance = options.residual_tolerance;
        crate::util::check_real_slice_finite(&[tolerance])?;
        if tolerance <= T::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.residual_tolerance"),
                "pole residual tolerance must be positive",
            ));
        }
        scope.into().run(
            &mut SolveState::new(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("signal", "pole_placement", 1),
                self.states() as u128,
                0,
            ),
            SolveBudgets::unbounded(),
            |context| self.place_poles_candidate(requested, options, context),
        )
    }

    fn place_poles_candidate(
        &self,
        requested: &[Complex<T::Real>],
        options: PolePlacementOptions<T::Real>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<PolePlacementReport<T>, SciOxideError> {
        let n = self.states();
        let m = self.inputs();
        if n == 0 || m == 0 {
            return Err(empty(
                scioxide_core::argument!("self"),
                "pole placement needs positive state and input counts",
            ));
        }
        exact_len(n, requested.len())?;
        let requested = canonical_poles::<T>(requested)?;
        let controllability_matrix = controllability_matrix(&self.a, &self.b)?;
        let controllability_svd = placement_svd(&controllability_matrix, context)?;
        let controllability = controllability_svd.analyze_rank(options.rank_policy)?;
        let controllability_condition = controllability_svd.condition_estimate()?;
        if !controllability.is_full_rank() {
            return Err(numerical(
                "state pair is not controllable under the selected rank policy",
            ));
        }
        let input_svd = placement_svd(&self.b, context)?;
        let input_condition = input_svd.condition_estimate()?;
        let input_rank = input_svd.analyze_rank(options.rank_policy)?.rank();
        for pole in &requested {
            let count = requested.iter().filter(|other| **other == *pole).count();
            if count > input_rank {
                return Err(inconsistent(
                    scioxide_core::argument!("requested"),
                    "requested pole multiplicity exceeds rank(B)",
                ));
            }
        }
        let (gain, transfer_condition) = if input_rank == n {
            crate::util::admit(context)?;
            (
                full_input_gain(
                    &self.a,
                    &input_svd,
                    &requested,
                    options.rank_policy,
                    context,
                )?,
                None,
            )
        } else {
            let (gain, condition) = eigenstructure_gain::<T>(
                &self.a,
                &self.b,
                &requested,
                options.rank_policy,
                context,
            )?;
            (gain, Some(condition))
        };
        let product = self.b.view().mul_mat(&gain)?;
        let closed = self.a.view().try_sub(&product)?;
        let mut result = scioxide_linalg::GeneralEigenvaluesBuffer::empty();
        scioxide_linalg::GeneralEigenvaluesBuffer::compute_into(
            &closed,
            &GeneralEigenOptions::new(),
            &mut result,
            &mut LinalgWorkspace::new(),
            &mut *context,
        )?;
        let mut achieved = copy_slice(
            result.eigenvalues(),
            scioxide_core::quantity!("achieved_poles"),
        )?;
        sort_poles(&mut achieved);
        let maximum_residual = pole_residual(&requested, &achieved)?;
        let termination = if maximum_residual <= options.residual_tolerance {
            PolePlacementTermination::Converged
        } else {
            PolePlacementTermination::ResidualToleranceNotMet
        };
        Ok(PolePlacementReport {
            gain,
            requested_poles: requested,
            achieved_poles: achieved,
            maximum_residual,
            controllability,
            controllability_condition,
            transfer_condition,
            input_condition,
            termination,
            counts: context.counts(),
        })
    }
}

/// Returns a converged report and maps a missed residual tolerance to the
/// `ConvergenceFailure` of [`StateSpace::place_poles`].
fn require_success<T: NumericScalar>(
    report: PolePlacementReport<T>,
) -> Result<PolePlacementReport<T>, SciOxideError> {
    match report.termination() {
        PolePlacementTermination::Converged => Ok(report),
        PolePlacementTermination::ResidualToleranceNotMet => {
            Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::Stagnation,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
                "placed poles missed the residual tolerance",
            ))
        }
    }
}

fn placement_svd<T: NumericScalar>(
    matrix: &Matrix<T>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<scioxide_linalg::ThinSvdDecomposition<T>, SciOxideError> {
    let mut result = scioxide_linalg::ThinSvdDecomposition::empty();
    scioxide_linalg::ThinSvdDecomposition::compute_into(
        matrix,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        &mut result,
        &mut LinalgWorkspace::new(),
        context,
    )?;
    Ok(result)
}

fn eigenstructure_gain<T: NumericScalar>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    poles: &[Complex<T::Real>],
    policy: RankPolicy<T::Real>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(Matrix<T>, ConditionEstimate<T::Real>), SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let n = a.rows();
    let m = b.cols();
    let ac = complex_matrix(a.view())?;
    let bc = complex_matrix(b.view())?;
    let mut kernels = Vec::new();
    kernels.try_reserve_exact(n).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("pole_placement_kernels"), n)
    })?;
    for &pole in poles {
        let mut equation = Matrix::zeros(n, n + m)?;
        for row in 0..n {
            for col in 0..n {
                let diagonal = if row == col { pole } else { Complex::ZERO };
                equation.as_mut_slice()[row * (n + m) + col] =
                    ac.as_slice()[row * n + col] - diagonal;
            }
            for input in 0..m {
                equation.as_mut_slice()[row * (n + m) + n + input] =
                    -bc.as_slice()[row * m + input];
            }
        }
        let mut kernel = Matrix::empty();
        scioxide_linalg::null_space_into(
            equation.view(),
            policy,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut kernel,
            &mut LinalgWorkspace::new(),
            &mut *context,
        )?;
        if kernel.cols() == 0 {
            return Err(numerical("pole equation has no accepted null space"));
        }
        kernels.push(kernel);
    }

    // Attempt `k` selects column `(j + k) mod cols_j` of each null space, so the
    // selections repeat with the least common multiple of the column counts.
    let period = kernels.iter().try_fold(1_usize, |period, kernel| {
        let columns = kernel.cols();
        let (mut x, mut y) = (period, columns);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        (period / x).checked_mul(columns)
    });
    let attempts = n
        .checked_mul(m.max(1))
        .and_then(|value| value.checked_add(8))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("pole_placement_attempts"))
        })?
        .min(period.unwrap_or(usize::MAX));
    for attempt in 0..attempts {
        crate::util::admit(context)?;
        let mut transfer = Matrix::<Complex<T::Real>>::zeros(n, n)?;
        let mut action = Matrix::<Complex<T::Real>>::zeros(m, n)?;
        for column in 0..n {
            if T::IS_REAL && poles[column].im > T::Real::ZERO {
                let occurrence = (0..column)
                    .filter(|&candidate| poles[candidate] == poles[column])
                    .count();
                let partner = (0..column)
                    .filter(|&candidate| {
                        poles[candidate].re == poles[column].re
                            && poles[candidate].im == -poles[column].im
                    })
                    .nth(occurrence);
                if let Some(partner) = partner {
                    for row in 0..n {
                        transfer.as_mut_slice()[row * n + column] =
                            transfer.as_slice()[row * n + partner].conj();
                    }
                    for row in 0..m {
                        action.as_mut_slice()[row * n + column] =
                            action.as_slice()[row * n + partner].conj();
                    }
                    continue;
                }
            }
            let kernel = &kernels[column];
            let selected = (column + attempt) % kernel.cols();
            let mut scale = T::Real::ZERO;
            for row in 0..n {
                let value = kernel.as_slice()[row * kernel.cols() + selected];
                transfer.as_mut_slice()[row * n + column] = value;
                scale += value.norm_sqr();
            }
            if scale == T::Real::ZERO {
                continue;
            }
            let inverse = T::Real::ONE / scale.sqrt();
            let inverse = Complex::new(inverse, T::Real::ZERO);
            for row in 0..n {
                transfer.as_mut_slice()[row * n + column] =
                    transfer.as_slice()[row * n + column] * inverse;
            }
            for row in 0..m {
                action.as_mut_slice()[row * n + column] =
                    kernel.as_slice()[(n + row) * kernel.cols() + selected] * inverse;
            }
        }
        let transfer_svd = placement_svd(&transfer, context)?;
        if !transfer_svd.analyze_rank(policy)?.is_full_rank() {
            continue;
        }
        let transfer_condition = transfer_svd.condition_estimate()?;
        let mut lu = LuDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        let execution = scioxide_linalg::LinalgExecutionOptions::new();
        LuDecomposition::factor_into(
            &transfer,
            &execution,
            &mut lu,
            &mut workspace,
            &mut *context,
        )?;
        let mut inverse = Matrix::zeros(n, n)?;
        lu.inverse_into(
            &execution,
            inverse.view_mut(),
            &mut workspace,
            &mut *context,
        )?;
        let complex_gain = action.view().mul_mat(&inverse)?;
        let mut gain = Matrix::zeros(m, n)?;
        // Conjugate column pairs make `W V^-1` real in exact arithmetic; a real
        // model keeps the real part and the caller rechecks the spectrum.
        for (destination, &source) in gain.as_mut_slice().iter_mut().zip(complex_gain.as_slice()) {
            *destination = T::from_components(source.re, source.im);
        }
        return Ok((gain, transfer_condition));
    }
    Err(numerical(
        "deterministic pole eigenstructure remained singular",
    ))
}

fn full_input_gain<T: NumericScalar>(
    a: &Matrix<T>,
    decomposition: &scioxide_linalg::ThinSvdDecomposition<T>,
    poles: &[Complex<T::Real>],
    policy: RankPolicy<T::Real>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<Matrix<T>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let target = target_matrix::<T>(poles)?;
    let rhs = a.view().try_sub(&target)?;
    let mut gain = Matrix::zeros(decomposition.cols(), a.cols())?;
    let mut workspace = LinalgWorkspace::new();
    for col in 0..a.cols() {
        let vector = scioxide_linalg::Vector::from_vec(
            (0..a.rows())
                .map(|row| rhs.as_slice()[row * a.cols() + col])
                .collect(),
        );
        let mut solution = scioxide_linalg::Vector::zeros(decomposition.cols())?;
        decomposition.solve_into(
            &vector,
            policy,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            solution.view_mut(),
            &mut workspace,
            &mut *context,
        )?;
        for row in 0..decomposition.cols() {
            gain.as_mut_slice()[row * a.cols() + col] = solution.as_slice()[row];
        }
    }
    Ok(gain)
}

fn target_matrix<T: NumericScalar>(poles: &[Complex<T::Real>]) -> Result<Matrix<T>, SciOxideError> {
    let n = poles.len();
    let mut target = Matrix::zeros(n, n)?;
    let mut index = 0;
    while index < n {
        let pole = poles[index];
        if T::IS_REAL && pole.im != T::Real::ZERO {
            target.as_mut_slice()[index * n + index] = T::from_real(pole.re);
            target.as_mut_slice()[index * n + index + 1] = T::from_real(-pole.im);
            target.as_mut_slice()[(index + 1) * n + index] = T::from_real(pole.im);
            target.as_mut_slice()[(index + 1) * n + index + 1] = T::from_real(pole.re);
            index += 2;
        } else {
            target.as_mut_slice()[index * n + index] = complex_to_scalar::<T>(pole)?;
            index += 1;
        }
    }
    Ok(target)
}

/// Sorts the `requested` poles and, for real systems, pairs near conjugates exactly.
fn canonical_poles<T: NumericScalar>(
    poles: &[Complex<T::Real>],
) -> Result<Vec<Complex<T::Real>>, SciOxideError> {
    let requested = scioxide_core::argument!("requested");
    for pole in poles {
        crate::util::check_real_slice_finite(&[pole.re, pole.im])?;
    }
    let mut ordered = copy_slice(poles, scioxide_core::quantity!("requested_poles"))?;
    sort_poles(&mut ordered);
    if T::IS_REAL {
        let tolerance = T::Real::EPSILON * T::Real::try_from_f64_lossy(32.0)?;
        // Distance of `other` from the conjugate of `pole`, when within tolerance.
        let conjugate_distance = |pole: Complex<T::Real>, other: Complex<T::Real>| {
            ((other.re - pole.re).abs() <= tolerance * (T::Real::ONE + pole.re.abs())
                && (other.im + pole.im).abs() <= tolerance * (T::Real::ONE + pole.im.abs()))
            .then(|| (other - pole.conj()).norm())
        };
        let mut paired = scioxide_core::backend::try_filled(
            ordered.len(),
            false,
            scioxide_core::quantity!("requested_pole_pairs"),
        )?;
        for upper in 0..ordered.len() {
            if ordered[upper].im <= T::Real::ZERO {
                continue;
            }
            let partner = (0..ordered.len())
                .filter(|&lower| !paired[lower] && ordered[lower].im < T::Real::ZERO)
                .filter_map(|lower| {
                    conjugate_distance(ordered[upper], ordered[lower]).map(|d| (lower, d))
                })
                .min_by(|left, right| compare_real(left.1, right.1));
            if let Some((lower, _)) = partner {
                ordered[lower] = ordered[upper].conj();
                paired[upper] = true;
                paired[lower] = true;
            }
        }
        for (pole, &paired) in ordered.iter_mut().zip(&paired) {
            if paired || pole.im == T::Real::ZERO {
                continue;
            }
            if conjugate_distance(*pole, *pole).is_none() {
                return Err(inconsistent(
                    requested,
                    "real systems require complex-conjugate requested poles",
                ));
            }
            pole.im = T::Real::ZERO;
        }
        ordered.sort_by(|left, right| {
            if left.im == T::Real::ZERO && right.im != T::Real::ZERO {
                Ordering::Less
            } else if left.im != T::Real::ZERO && right.im == T::Real::ZERO {
                Ordering::Greater
            } else {
                compare_real(left.re, right.re).then_with(|| compare_real(left.im, right.im))
            }
        });
    }
    Ok(ordered)
}

fn sort_poles<S: RealScalar>(poles: &mut [Complex<S>]) {
    poles.sort_by(|left, right| {
        compare_real(left.re, right.re).then_with(|| compare_real(left.im, right.im))
    });
}

fn compare_real<S: RealScalar>(left: S, right: S) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn pole_residual<S: RealScalar>(
    requested: &[Complex<S>],
    achieved: &[Complex<S>],
) -> Result<S, SciOxideError> {
    let mut remaining = copy_slice(achieved, scioxide_core::quantity!("achieved_pole_matching"))?;
    let mut maximum = S::ZERO;
    for &pole in requested {
        let (index, distance) = remaining
            .iter()
            .enumerate()
            .map(|(index, other)| (index, (*other - pole).norm()))
            .min_by(|left, right| compare_real(left.1, right.1))
            .ok_or_else(|| numerical("achieved pole set is incomplete"))?;
        maximum = maximum.max(distance / (S::ONE + pole.norm()));
        remaining.remove(index);
    }
    Ok(maximum)
}

fn complex_to_scalar<T: NumericScalar>(value: Complex<T::Real>) -> Result<T, SciOxideError> {
    if T::IS_REAL {
        let limit = T::Real::EPSILON
            * T::Real::try_from_f64_lossy(256.0)?
            * (T::Real::ONE + value.re.abs());
        if value.im.abs() > limit {
            return Err(numerical(
                "complex result cannot be represented in a real model",
            ));
        }
    }
    Ok(T::from_components(value.re, value.im))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{ConvergenceReason, SciOxideErrorView};

    /// A controllable single-input system of three states.
    fn system() -> StateSpace<f64> {
        let a = Matrix::from_vec(3, 3, vec![0.1, 1., 0., 0., 0.2, 1., 0., 0., 0.3]).unwrap();
        let b = Matrix::from_vec(3, 1, vec![0.1, 0.2, 1.]).unwrap();
        let c = Matrix::from_vec(1, 3, vec![1., 0., 0.]).unwrap();
        let d = Matrix::zeros(1, 1).unwrap();
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap()
    }

    const REQUESTED: [Complex<f64>; 3] = [
        Complex::new(-0.7, 0.),
        Complex::new(-1.3, 0.),
        Complex::new(-2.9, 0.),
    ];

    #[test]
    fn missed_residual_tolerance_is_stagnation() {
        let options = PolePlacementOptions::new().with_residual_tolerance(f64::MIN_POSITIVE);
        let report = system()
            .place_poles_with_options(&REQUESTED, options)
            .unwrap();
        assert_eq!(
            report.termination(),
            PolePlacementTermination::ResidualToleranceNotMet
        );
        let counts = report.counts();
        let error = require_success(report).unwrap_err();
        let failure = match error.view() {
            SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                detail,
            } => Some((reason, diagnostics.counts(), detail)),
            _ => None,
        };
        assert_eq!(
            failure,
            Some((
                ConvergenceReason::Stagnation,
                counts,
                "placed poles missed the residual tolerance"
            ))
        );
    }

    #[test]
    fn default_placement_succeeds_through_the_one_shot() {
        let system = system();
        let configured = system
            .place_poles_with_options(&REQUESTED, PolePlacementOptions::default())
            .unwrap();
        let placed = system.place_poles(&REQUESTED).unwrap();
        assert!(configured.is_success() && placed.is_success());
        assert_eq!(placed.gain().as_slice(), configured.gain().as_slice());
        let accepted = require_success(configured).unwrap();
        assert_eq!(accepted.achieved_poles(), placed.achieved_poles());
    }
}
