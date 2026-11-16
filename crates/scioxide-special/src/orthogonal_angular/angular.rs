//! Angular-momentum coupling coefficients from exact prime-factor sums.
use crate::complex_support::ScaledForm;
use crate::result::{Either, Evaluation, Form};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::invalid;
use crate::support::real;
use crate::util::scoped;
use crate::{SpecialFunctionResult, orthogonal_angular::angular_integer::Integer};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::algorithm_component;
use scioxide_core::backend::{
    checked_add, checked_mul, scaled::factor_common_scale, try_filled, try_with_capacity,
};
use scioxide_core::{Argument, Budget, RealScalar, SciOxideError, SolveContext, SolveScope};
fn triangle(a: i64, b: i64, c: i64) -> bool {
    a >= 0 && b >= 0 && c >= 0 && (a - b).abs() <= c && c <= a + b && (a + b + c) % 2 == 0
}
fn parity(n: i64) -> bool {
    n.rem_euclid(2) != 0
}
/// Stores an angular momentum `j ∈ {0, 1/2, 1, 3/2, …}` exactly as the integer `2j`.
///
/// # Examples
///
/// ```
/// // j = 1/2
/// let j = scioxide_special::AngularMomentum::new(1)?;
/// assert_eq!(j.twice_j(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AngularMomentum(i32);
impl AngularMomentum {
    /// Creates the angular momentum `j` from `2j`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if `twice_j < 0`.
    pub fn new(twice_j: i32) -> Result<Self, SciOxideError> {
        Self::checked(twice_j, scioxide_core::argument!("twice_j"))
    }
    /// Validates `2j`, naming `argument` if it is negative.
    fn checked(twice_j: i32, argument: Argument) -> Result<Self, SciOxideError> {
        if twice_j < 0 {
            return Err(invalid(argument, "angular momentum must be nonnegative"));
        }
        Ok(Self(twice_j))
    }
    /// Returns `2j`.
    pub const fn twice_j(self) -> i32 {
        self.0
    }
}

/// Stores an angular momentum `j` and its projection `m` exactly as `2j` and `2m`.
///
/// The projection satisfies `|m| ≤ j`, and `j − m` is an integer.
///
/// # Examples
///
/// ```
/// // j = 1/2, m = −1/2
/// let down = scioxide_special::AngularState::new(1, -1)?;
/// assert_eq!((down.momentum().twice_j(), down.twice_m()), (1, -1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AngularState {
    momentum: AngularMomentum,
    m: i32,
}
impl AngularState {
    /// Creates the state `(j, m)` from the doubled quantum numbers `twice_j = 2j` and
    /// `twice_m = 2m`, so that `j = 1/2, m = −1/2` is `new(1, -1)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if `twice_j < 0`.
    /// - `InvalidInput` (`OutOfRange`) with argument `twice_m` if `|m| > j` or `j − m` is not
    ///   an integer.
    pub fn new(twice_j: i32, twice_m: i32) -> Result<Self, SciOxideError> {
        Self::checked(
            twice_j,
            twice_m,
            scioxide_core::argument!("twice_j"),
            scioxide_core::argument!("twice_m"),
        )
    }
    /// Validates `(2j, 2m)`, naming `momentum` or `projection` for the rejected number.
    fn checked(
        twice_j: i32,
        twice_m: i32,
        momentum: Argument,
        projection: Argument,
    ) -> Result<Self, SciOxideError> {
        let momentum = AngularMomentum::checked(twice_j, momentum)?;
        let (j, m) = (i64::from(twice_j), i64::from(twice_m));
        if m.abs() > j || (j - m) % 2 != 0 {
            return Err(invalid(projection, "incompatible magnetic projection"));
        }
        Ok(Self {
            momentum,
            m: twice_m,
        })
    }
    /// Creates the state `(l, m)` of a spherical harmonic from its degree `l` and order `m`.
    fn harmonic(degree: usize, order: i32) -> Result<Self, SciOxideError> {
        let twice_l = degree
            .checked_mul(2)
            .and_then(|twice| i32::try_from(twice).ok())
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("degree"),
                    "twice the degree exceeds i32::MAX",
                )
            })?;
        let twice_m = order.checked_mul(2).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("order"),
                "the order exceeds the degree",
            )
        })?;
        Self::checked(
            twice_l,
            twice_m,
            scioxide_core::argument!("degree"),
            scioxide_core::argument!("order"),
        )
    }
    /// Returns the angular momentum `j`.
    pub const fn momentum(self) -> AngularMomentum {
        self.momentum
    }
    /// Returns `2m`.
    pub const fn twice_m(self) -> i32 {
        self.m
    }
}

/// Sets the work and memory limits of an [`AngularPlan`].
///
/// The coefficients are exact sums, so there is no tolerance. Preparation and each evaluation
/// have their own work budget, 16,777,216 units by default. The byte limits, 64 MiB each by
/// default, cap the prime table built during preparation and the size of an
/// [`AngularWorkspace`]; they exclude allocator overhead.
///
/// # Examples
///
/// ```
/// let options = scioxide_special::AngularOptions::default()
///     .with_iteration_budget(scioxide_core::Budget::try_from(100_000)?);
/// assert_eq!(options.iteration_budget().get(), 100_000);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct AngularOptions {
    iteration_budget: Budget,
    preparation_bytes: Budget,
    workspace_bytes: Budget,
}
impl AngularOptions {
    /// Returns the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the work budget for preparation and for each evaluation.
    pub const fn iteration_budget(self) -> Budget {
        self.iteration_budget
    }
    /// Returns the limit on the memory used by preparation, in bytes.
    pub const fn preparation_bytes(self) -> Budget {
        self.preparation_bytes
    }
    /// Returns the limit on the size of an [`AngularWorkspace`], in bytes.
    pub const fn workspace_bytes(self) -> Budget {
        self.workspace_bytes
    }
    /// Returns a copy with a different work budget.
    pub const fn with_iteration_budget(mut self, budget: Budget) -> Self {
        self.iteration_budget = budget;
        self
    }
    /// Returns a copy with a different preparation memory limit.
    pub const fn with_preparation_bytes(mut self, bytes: Budget) -> Self {
        self.preparation_bytes = bytes;
        self
    }
    /// Returns a copy with a different workspace size limit.
    pub const fn with_workspace_bytes(mut self, bytes: Budget) -> Self {
        self.workspace_bytes = bytes;
        self
    }
}
impl Default for AngularOptions {
    fn default() -> Self {
        Self {
            iteration_budget: Budget::ONE.saturating_add(16_777_215),
            preparation_bytes: Budget::ONE.saturating_add(64 * 1024 * 1024 - 1),
            workspace_bytes: Budget::ONE.saturating_add(64 * 1024 * 1024 - 1),
        }
    }
}
#[derive(Debug)]
struct Coefficient {
    integer: Integer,
    powers: Vec<i64>,
}
impl Coefficient {
    fn new(primes: usize, limbs: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            integer: Integer::new(limbs)?,
            powers: try_filled(primes, 0, scioxide_core::quantity!("coefficient.powers"))?,
        })
    }
    fn clear(&mut self) {
        self.integer.clear();
        self.powers.fill(0);
    }
}
/// Holds the multiword integers and prime-exponent arrays that [`AngularPlan`] evaluations use.
///
/// Created by [`AngularPlan::workspace`] with a size fixed by the plan, and accepted only by
/// plans with the same maximum angular momentum. See
/// [plans and workspaces](scioxide_core#plans-and-workspaces).
#[derive(Debug)]
pub struct AngularWorkspace {
    max_twice_j: i32,
    coefficients: [Coefficient; 3],
    result: Coefficient,
    term: Integer,
    product: Integer,
    scratch_powers: Vec<i64>,
}
/// Evaluates angular-momentum coupling coefficients exactly, for momenta up to a fixed maximum.
///
/// The plan computes Wigner 3j, 6j and 9j symbols, Clebsch–Gordan coefficients, Racah W
/// coefficients and Gaunt coefficients with the conventions of the functions [`wigner_3j`],
/// [`wigner_6j`], [`wigner_9j`], [`clebsch_gordan`], [`racah_w`] and [`gaunt`]. It stores the
/// primes and least prime factors up to `3·2j_max + 2`. Each coefficient is a sum of ratios of
/// factorials (Racah's formulas, DLMF §34.2 and §34.4). The plan writes every factorial as a
/// product of prime powers and adds the terms as exact multiword integers, so the only
/// rounding is in the final conversion to the scalar type (Johansson and Forssén, 2016).
/// The triangle, parity and projection selection rules are checked first, and a coefficient
/// they forbid is exact zero.
///
/// The owned methods, such as [`Self::wigner_3j`], allocate an [`AngularWorkspace`] for each
/// call. The `*_scoped` methods, such as [`Self::wigner_3j_scoped`], reuse one workspace from
/// [`Self::workspace`] and allocate nothing. See
/// [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Accuracy
///
/// The sum is exact. The relative error estimate of each result is `8ε` times the number of
/// floating-point operations in the final conversion. No uniform ulp bound is stated; see the
/// [crate accuracy table](crate#accuracy).
///
/// # References
///
/// - NIST DLMF, Chapter 34. <https://dlmf.nist.gov/34>
/// - H. T. Johansson and C. Forssén, Fast and accurate evaluation of Wigner 3j, 6j, and 9j
///   symbols using prime factorization and multiword integer arithmetic, SIAM J. Sci.
///   Comput. 38 (2016) A376–A384.
#[derive(Debug, Clone)]
pub struct AngularPlan {
    maximum: i32,
    primes: Vec<u32>,
    least_prime: Vec<u32>,
    limbs: usize,
    options: AngularOptions,
    workspace_bytes: usize,
    preparation_work: usize,
}
impl AngularPlan {
    /// Prepares a plan for angular momenta `j ≤ maximum`.
    ///
    /// Preparation sieves the primes up to `3·2j_max + 2`, which costs `O(j_max log log j_max)`
    /// work units, and sizes the workspace for the largest 9j sum.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the prime table exceeds `options.preparation_bytes()`, or the
    ///   workspace would exceed `options.workspace_bytes()`.
    /// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
    /// - `AllocationFailure` if the prime table cannot be allocated.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sieve needs more work than
    ///   `options.iteration_budget()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_special::{AngularMomentum, AngularPlan, AngularState};
    ///
    /// let plan = AngularPlan::new(AngularMomentum::new(2)?, Default::default())?;
    /// let (up, down) = (AngularState::new(1, 1)?, AngularState::new(1, -1)?);
    /// let triplet = AngularState::new(2, 0)?;
    /// // ⟨½ ½; ½ −½ | 1 0⟩ = 1/√2
    /// let cg = plan.clebsch_gordan::<f64>(up, down, triplet)?;
    /// assert!((cg.try_value()? - 0.5_f64.sqrt()).abs() < 1e-14);
    /// // The scoped form reuses one workspace and allocates nothing.
    /// let (mut workspace, scope) = (plan.workspace()?, ExecutionContext::sequential());
    /// let again = plan.clebsch_gordan_scoped::<f64>(up, down, triplet, &mut workspace, scope)?;
    /// assert_eq!(again, cg);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(maximum: AngularMomentum, options: AngularOptions) -> Result<Self, SciOxideError> {
        Self::prepare(maximum, options, None)
    }
    /// Prepares a plan as [`Self::new`] does, inside `scope`.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `ExecutionStopped` if `scope` is cancelled or its
    /// deadline passes.
    pub fn build<'scope, 'cap>(
        maximum: AngularMomentum,
        options: AngularOptions,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<Self, SciOxideError> {
        scoped(scope, options.iteration_budget, |cx| {
            Self::prepare(maximum, options, Some(cx))
        })
    }
    /// Builds the prime table and sizes the workspace.
    fn prepare(
        maximum: AngularMomentum,
        options: AngularOptions,
        context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<Self, SciOxideError> {
        let maximum = maximum.twice_j();
        let maximum_usize = usize::try_from(maximum).map_err(|_| {
            invalid(
                scioxide_core::argument!("maximum"),
                "negative angular bound",
            )
        })?;
        let bound = checked_add(
            checked_mul(maximum_usize, 3, scioxide_core::quantity!("prime_bound"))?,
            2,
            scioxide_core::quantity!("prime_bound"),
        )?;
        if bound > u32::MAX as usize {
            return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "prime_bound"
            )));
        }
        let preparation_bytes = checked_mul(
            checked_add(bound, 1, scioxide_core::quantity!("prime_sieve"))?,
            2 * size_of::<u32>(),
            scioxide_core::quantity!("preparation_bytes"),
        )?;
        if preparation_bytes > options.preparation_bytes.get() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("preparation_bytes"),
                preparation_bytes,
                options.preparation_bytes.get(),
            ));
        }
        let mut work = Work::with_context(options.iteration_budget.get(), context);
        work.check()?;
        let mut least_prime =
            try_filled(bound + 1, u32::MAX, scioxide_core::quantity!("prime_sieve"))?;
        let mut primes = try_with_capacity(bound + 1, scioxide_core::quantity!("primes"))?;
        for n in 2..=bound {
            work.try_step()?;
            if least_prime[n] != u32::MAX {
                continue;
            }
            let index = primes.len() as u32;
            least_prime[n] = index;
            primes.push(n as u32);
            if n <= bound / n {
                for k in (n * n..=bound).step_by(n) {
                    work.try_step()?;
                    if least_prime[k] == u32::MAX {
                        least_prime[k] = index;
                    }
                }
            }
        }
        // Conservative factorial-product capacity for the nested 9j sum.
        let bits = usize::BITS - bound.leading_zeros();
        let limbs = checked_add(
            checked_mul(
                checked_mul(
                    bound,
                    bits as usize,
                    scioxide_core::quantity!("integer_limbs"),
                )?,
                2,
                scioxide_core::quantity!("integer_limbs"),
            )?,
            64,
            scioxide_core::quantity!("integer_limbs"),
        )?;
        let workspace_bytes = checked_add(
            checked_mul(
                checked_mul(limbs, 6, scioxide_core::quantity!("workspace_bytes"))?,
                size_of::<u32>(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            checked_mul(
                checked_mul(primes.len(), 5, scioxide_core::quantity!("workspace_bytes"))?,
                size_of::<i64>(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if workspace_bytes > options.workspace_bytes.get() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                workspace_bytes,
                options.workspace_bytes.get(),
            ));
        }
        Ok(Self {
            maximum,
            primes,
            least_prime,
            limbs,
            options,
            workspace_bytes,
            preparation_work: work.used,
        })
    }

    /// Returns `2j_max`, the largest `2j` the plan accepts.
    pub const fn maximum_twice_j(&self) -> i32 {
        self.maximum
    }
    /// Returns the work units used to build the prime table.
    pub const fn preparation_work(&self) -> usize {
        self.preparation_work
    }
    /// Returns the heap size of an [`AngularWorkspace`] for this plan, in bytes.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    /// Allocates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(&self) -> Result<AngularWorkspace, SciOxideError> {
        let n = self.primes.len();
        let l = self.limbs;
        Ok(AngularWorkspace {
            max_twice_j: self.maximum,
            coefficients: [
                Coefficient::new(n, l)?,
                Coefficient::new(n, l)?,
                Coefficient::new(n, l)?,
            ],
            result: Coefficient::new(n, l)?,
            term: Integer::new(l)?,
            product: Integer::new(l)?,
            scratch_powers: try_filled(n, 0, scioxide_core::quantity!("scratch_powers"))?,
        })
    }

    /// Checks that `scratch` belongs to this plan and every `2j` in `j` is within its maximum.
    fn check(
        &self,
        argument: Argument,
        j: &[i32],
        scratch: &AngularWorkspace,
    ) -> Result<(), SciOxideError> {
        if scratch.max_twice_j != self.maximum {
            return Err(invalid(
                scioxide_core::argument!("scratch"),
                "the workspace was created by a plan with a different maximum",
            ));
        }
        if j.iter().any(|&x| x < 0 || x > self.maximum) {
            return Err(invalid(
                argument,
                "an angular momentum exceeds the maximum of the plan",
            ));
        }
        Ok(())
    }
    /// Evaluates `run` on a new workspace under the plan's work budget.
    fn owned<S: RealScalar>(
        &self,
        run: impl FnOnce(
            &Self,
            &mut AngularWorkspace,
            &mut Work<'_, '_, '_>,
        ) -> Result<Estimate, SciOxideError>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.evaluate_for(
            &mut self.workspace()?,
            None,
            self.options.iteration_budget.get(),
            run,
        )?
        .publish()
    }
    /// Evaluates `run` on `scratch` in an epoch of `scope` under the plan's work budget.
    fn in_scope<'scope, 'cap, S: RealScalar>(
        &self,
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
        run: impl FnOnce(
            &Self,
            &mut AngularWorkspace,
            &mut Work<'_, '_, '_>,
        ) -> Result<Estimate, SciOxideError>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        scoped(scope, self.options.iteration_budget, |cx| {
            self.evaluate_for(scratch, Some(cx), self.options.iteration_budget.get(), run)
        })?
        .publish()
    }
    /// Computes the Wigner 3j symbol of [`crate::wigner_3j`] with this plan.
    ///
    /// `q` holds the three columns `(jᵢ, mᵢ)`. Returns the detailed result with its error
    /// estimate and work counts. Allocates a workspace for the call; see [`Self::wigner_3j_scoped`]
    /// to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `q` if some `j` exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn wigner_3j<S: RealScalar>(
        &self,
        q: [AngularState; 3],
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.three_estimate(q, scratch, work))
    }
    /// Computes [`Self::wigner_3j`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::wigner_3j`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn wigner_3j_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        q: [AngularState; 3],
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.three_estimate(q, scratch, work)
        })
    }
    /// Computes the Clebsch–Gordan coefficient of [`crate::clebsch_gordan`] with this plan.
    ///
    /// `a`, `b` and `c` are the states `(j₁, m₁)`, `(j₂, m₂)` and `(J, M)`. Returns the detailed
    /// result with its error estimate and work counts. Allocates a workspace for the call; see
    /// [`Self::clebsch_gordan_scoped`] to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `a`, `b` or `c`, the first state whose `j`
    ///   exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn clebsch_gordan<S: RealScalar>(
        &self,
        a: AngularState,
        b: AngularState,
        c: AngularState,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.clebsch_estimate(a, b, c, scratch, work))
    }
    /// Computes [`Self::clebsch_gordan`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::clebsch_gordan`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn clebsch_gordan_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        a: AngularState,
        b: AngularState,
        c: AngularState,
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.clebsch_estimate(a, b, c, scratch, work)
        })
    }
    /// Computes the Wigner 6j symbol of [`crate::wigner_6j`] with this plan.
    ///
    /// `j` lists the symbol in row-major order. Returns the detailed result with its error
    /// estimate and work counts. Allocates a workspace for the call; see [`Self::wigner_6j_scoped`]
    /// to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `j` if some `j` exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn wigner_6j<S: RealScalar>(
        &self,
        j: [AngularMomentum; 6],
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.six_estimate(j, scratch, work))
    }
    /// Computes [`Self::wigner_6j`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::wigner_6j`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn wigner_6j_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        j: [AngularMomentum; 6],
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.six_estimate(j, scratch, work)
        })
    }
    /// Computes the Wigner 9j symbol of [`crate::wigner_9j`] with this plan.
    ///
    /// `j` lists the three rows in row-major order. Returns the detailed result with its error
    /// estimate and work counts. Allocates a workspace for the call; see [`Self::wigner_9j_scoped`]
    /// to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `j` if some `j` exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn wigner_9j<S: RealScalar>(
        &self,
        j: [AngularMomentum; 9],
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.nine_estimate(j, scratch, work))
    }
    /// Computes [`Self::wigner_9j`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::wigner_9j`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn wigner_9j_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        j: [AngularMomentum; 9],
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.nine_estimate(j, scratch, work)
        })
    }
    /// Computes the Racah W coefficient of [`crate::racah_w`] with this plan.
    ///
    /// `j` lists `[a, b, c, d, e, f]` of `W(a b c d; e f)`. Returns the detailed result with its
    /// error estimate and work counts. Allocates a workspace for the call; see
    /// [`Self::racah_w_scoped`] to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `j` if some `j` exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn racah_w<S: RealScalar>(
        &self,
        j: [AngularMomentum; 6],
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.racah_estimate(j, scratch, work))
    }
    /// Computes [`Self::racah_w`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::racah_w`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn racah_w_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        j: [AngularMomentum; 6],
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.racah_estimate(j, scratch, work)
        })
    }
    /// Computes the Gaunt coefficient of [`crate::gaunt`] with this plan.
    ///
    /// `q` holds the states `(lᵢ, mᵢ)` with doubled quantum numbers `2lᵢ` and `2mᵢ`, as every
    /// [`AngularState`] does. Returns the detailed result with its error estimate and work
    /// counts. Allocates a workspace for the call; see [`Self::gaunt_scoped`] to reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) with argument `q` if some `l` is a half-integer or
    ///   exceeds the plan's maximum.
    /// - `ConvergenceFailure` (`IterationLimit`) if the sum needs more work than the
    ///   plan's work budget.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn gaunt<S: RealScalar>(
        &self,
        q: [AngularState; 3],
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.owned(|plan, scratch, work| plan.gaunt_estimate(q, scratch, work))
    }
    /// Computes [`Self::gaunt`] on `scratch` inside `scope`, without allocating.
    ///
    /// See [scoped evaluation](crate#scoped-evaluation).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::gaunt`] except `AllocationFailure`; `InvalidInput`
    /// (`OutOfRange`) with argument `scratch` if `scratch` was created by a plan with a
    /// different maximum; and `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    pub fn gaunt_scoped<'scope, 'cap, S: RealScalar>(
        &self,
        q: [AngularState; 3],
        scratch: &mut AngularWorkspace,
        scope: impl Into<SolveScope<'scope, 'cap>>,
    ) -> Result<SpecialFunctionResult<S>, SciOxideError> {
        self.in_scope(scratch, scope, |plan, scratch, work| {
            plan.gaunt_estimate(q, scratch, work)
        })
    }
    fn three_estimate(
        &self,
        q: [AngularState; 3],
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        self.check(
            scioxide_core::argument!("q"),
            &q.map(|x| x.momentum.twice_j()),
            scratch,
        )?;
        three(
            &self.primes,
            &self.least_prime,
            q,
            &mut scratch.result,
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;

        self.finish(scratch, false, work)
    }
    fn clebsch_estimate(
        &self,
        a: AngularState,
        b: AngularState,
        c: AngularState,
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        self.check(
            scioxide_core::argument!("a"),
            &[a.momentum.twice_j()],
            scratch,
        )?;
        self.check(
            scioxide_core::argument!("b"),
            &[b.momentum.twice_j()],
            scratch,
        )?;
        self.check(
            scioxide_core::argument!("c"),
            &[c.momentum.twice_j()],
            scratch,
        )?;
        let third = AngularState {
            momentum: c.momentum,
            m: -c.m,
        };
        three(
            &self.primes,
            &self.least_prime,
            [a, b, third],
            &mut scratch.result,
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;
        scratch.result.integer.negative ^= parity(
            (i64::from(a.momentum.twice_j()) - i64::from(b.momentum.twice_j()) + i64::from(c.m))
                / 2,
        );
        factor(
            &self.primes,
            &self.least_prime,
            &mut scratch.result.powers,
            i64::from(c.momentum.twice_j()) + 1,
            1,
            work,
        )?;

        self.finish(scratch, false, work)
    }
    fn six_estimate(
        &self,
        j: [AngularMomentum; 6],
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let j = j.map(AngularMomentum::twice_j);
        self.check(scioxide_core::argument!("j"), &j, scratch)?;
        six(
            &self.primes,
            &self.least_prime,
            j.map(i64::from),
            &mut scratch.result,
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;

        self.finish(scratch, false, work)
    }
    fn nine_estimate(
        &self,
        j: [AngularMomentum; 9],
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let j = j.map(AngularMomentum::twice_j);
        self.check(scioxide_core::argument!("j"), &j, scratch)?;
        nine(
            &self.primes,
            &self.least_prime,
            j.map(i64::from),
            scratch,
            work,
        )?;
        self.finish(scratch, false, work)
    }
    fn racah_estimate(
        &self,
        j: [AngularMomentum; 6],
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let order = [j[0], j[1], j[4], j[3], j[2], j[5]].map(AngularMomentum::twice_j);
        self.check(scioxide_core::argument!("j"), &order, scratch)?;
        six(
            &self.primes,
            &self.least_prime,
            order.map(i64::from),
            &mut scratch.result,
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;
        scratch.result.integer.negative ^=
            parity(j[..4].iter().map(|x| i64::from(x.twice_j())).sum::<i64>() / 2);
        self.finish(scratch, false, work)
    }
    fn gaunt_estimate(
        &self,
        q: [AngularState; 3],
        scratch: &mut AngularWorkspace,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        if q.iter().any(|x| x.momentum.twice_j() % 2 != 0) {
            return Err(invalid(
                scioxide_core::argument!("q"),
                "Gaunt coefficients require integer angular momenta",
            ));
        }
        self.check(
            scioxide_core::argument!("q"),
            &q.map(|x| x.momentum.twice_j()),
            scratch,
        )?;
        three(
            &self.primes,
            &self.least_prime,
            q.map(|x| AngularState {
                momentum: x.momentum,
                m: 0,
            }),
            &mut scratch.coefficients[0],
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;
        three(
            &self.primes,
            &self.least_prime,
            q,
            &mut scratch.coefficients[1],
            &mut scratch.term,
            &mut scratch.scratch_powers,
            work,
        )?;
        Integer::product(
            &scratch.coefficients[0].integer,
            &scratch.coefficients[1].integer,
            &mut scratch.result.integer,
            work,
        )?;
        for (index, out) in scratch.result.powers.iter_mut().enumerate() {
            *out = scratch.coefficients[0].powers[index] + scratch.coefficients[1].powers[index];
        }
        for x in q {
            factor(
                &self.primes,
                &self.least_prime,
                &mut scratch.result.powers,
                i64::from(x.momentum.twice_j()) + 1,
                1,
                work,
            )?;
        }

        self.finish(scratch, true, work)
    }
    fn finish(
        &self,
        scratch: &AngularWorkspace,
        gaunt: bool,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let (mut value, rounding) = convert(&self.primes, &scratch.result, work)?;
        if gaunt {
            value = value.mul(Scaled::from_value(
                1.0 / (4.0 * core::f64::consts::PI).sqrt(),
            )?)?;
        }
        let (parts, exponent) = factor_common_scale([value])?;
        Ok(Estimate::from(crate::support::JetData {
            derivative_error: None,
            y: Jet::constant(real(parts[0].round()?)),
            scale: Jet::constant(real(exponent as f64 * core::f64::consts::LN_2)),
            error: rounding * parts[0].round()?.abs(),
            method: algorithm_component!("prime_factor_exact_sum"),
        }))
    }

    fn evaluate_for<S: RealScalar>(
        &self,
        scratch: &mut AngularWorkspace,
        context: Option<&mut SolveContext<'_, '_>>,
        allowance: usize,
        run: impl FnOnce(
            &Self,
            &mut AngularWorkspace,
            &mut Work<'_, '_, '_>,
        ) -> Result<Estimate, SciOxideError>,
    ) -> Result<AngularEvaluation<S>, SciOxideError> {
        let mut work = Work::with_context(allowance, context);
        work.check()?;
        let result = run(self, scratch, &mut work);
        if result.as_ref().is_ok_and(|e| e.y.v == real(0.0)) {
            return Ok(Evaluation::finite(
                S::ZERO,
                crate::result::Bound::exact(),
                None,
                scioxide_core::algorithm_id!("angular-momentum", "exact_angular_zero", 1),
                work.used,
                crate::SpecialFunctionTermination::Exact,
            )
            .map_form(Either::Right));
        }
        Ok(crate::complex_support::report(
            result,
            false,
            &work,
            algorithm_component!("angular-momentum"),
        )?
        .map_form(Either::Left))
    }
}
/// Evaluation of an angular coefficient.
type AngularEvaluation<S> = Evaluation<S, Either<ScaledForm, Form<S>>>;
fn valuation(mut n: i64, p: u32, work: &mut Work<'_, '_, '_>) -> Result<i64, SciOxideError> {
    let mut result = 0;
    while n >= i64::from(p) {
        work.try_step()?;
        n /= i64::from(p);
        result += n;
    }
    Ok(result)
}
fn factorials(
    primes: &[u32],
    output: &mut [i64],
    positive: &[i64],
    negative: &[i64],
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if positive.iter().chain(negative).any(|&n| n < 0) {
        return Err(invalid(
            scioxide_core::argument!("query"),
            "negative factorial argument",
        ));
    }
    let maximum = positive.iter().chain(negative).copied().max().unwrap_or(0);
    output.fill(0);
    for (&prime, out) in primes
        .iter()
        .zip(output)
        .take_while(|&(&p, _)| i64::from(p) <= maximum)
    {
        let mut e = 0;
        for &n in positive {
            e += valuation(n, prime, work)?;
        }
        for &n in negative {
            e -= valuation(n, prime, work)?;
        }
        *out = e;
    }
    Ok(())
}
fn delta(
    primes: &[u32],
    powers: &mut [i64],
    a: i64,
    b: i64,
    c: i64,
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let maximum = (a + b + c) / 2 + 1;
    for (&p, out) in primes
        .iter()
        .zip(powers)
        .take_while(|&(&p, _)| i64::from(p) <= maximum)
    {
        *out += valuation((a + b - c) / 2, p, work)?
            + valuation((a - b + c) / 2, p, work)?
            + valuation((-a + b + c) / 2, p, work)?
            - valuation((a + b + c) / 2 + 1, p, work)?;
    }
    Ok(())
}
// Add the prime exponents of n^multiplier using the prepared factor indices.
fn factor(
    primes: &[u32],
    least_prime: &[u32],
    powers: &mut [i64],
    n: i64,
    multiplier: i64,
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let mut n = usize::try_from(n).unwrap_or(usize::MAX);
    if n == 0 || n >= least_prime.len() {
        return Err(invalid(
            scioxide_core::argument!("query"),
            "factor outside prepared prime table",
        ));
    }
    while n > 1 {
        work.try_step()?;
        let index = least_prime[n] as usize;
        powers[index] += multiplier;
        n /= primes[index] as usize;
    }
    Ok(())
}

// Add term * sqrt(product(p^powers)) to out without rounding. Lowering a
// common exponent multiplies the accumulated integer, never divides it.
fn add_factored(
    primes: &[u32],
    powers: &[i64],
    out: &mut Coefficient,
    term: &mut Integer,
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if term.is_zero() {
        return Ok(());
    }
    if out.integer.is_zero() {
        out.integer.copy_from(term)?;
        out.powers.copy_from_slice(powers);
        return Ok(());
    }
    for ((&p, &e), m) in primes.iter().zip(powers).zip(&mut out.powers) {
        let difference = e - *m;
        if parity(difference) {
            return Err(invalid(
                scioxide_core::argument!("query"),
                "inconsistent exact-sum square-root factors",
            ));
        }
        if difference < 0 {
            for _ in 0..-difference / 2 {
                out.integer.mul_small(p, work)?;
            }
            *m = e;
        } else {
            for _ in 0..difference / 2 {
                term.mul_small(p, work)?;
            }
        }
    }
    out.integer.add(term, work)
}
#[allow(clippy::too_many_arguments)]
fn racah_sum(
    primes: &[u32],
    least_prime: &[u32],
    start: i64,
    end: i64,
    term_factors: impl Fn(i64) -> (i64, [i64; 7]),
    out: &mut Coefficient,
    term: &mut Integer,
    powers: &mut [i64],
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    out.clear();
    if start > end {
        return Ok(());
    }
    let (n, d) = term_factors(start);
    factorials(primes, powers, &[n], &d, work)?;
    for e in powers.iter_mut() {
        *e *= 2;
    }
    for k in start..=end {
        term.one()?;
        term.negative = parity(k);
        add_factored(primes, powers, out, term, work)?;
        if k < end {
            let (n, d) = term_factors(k);
            let (next_n, next_d) = term_factors(k + 1);
            // Every factorial argument changes by zero or one. The numerator
            // is constant (3j) or increasing (6j); denominators move either way.
            if next_n != n {
                factor(primes, least_prime, powers, next_n, 2, work)?;
            }
            for (previous, next) in d.into_iter().zip(next_d) {
                if next > previous {
                    factor(primes, least_prime, powers, next, -2, work)?;
                } else if next < previous {
                    factor(primes, least_prime, powers, previous, 2, work)?;
                }
            }
        }
    }
    Ok(())
}
fn three(
    primes: &[u32],
    least_prime: &[u32],
    q: [AngularState; 3],
    out: &mut Coefficient,
    term: &mut Integer,
    powers: &mut [i64],
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    out.clear();
    let [a, b, c] = q.map(|x| i64::from(x.momentum.twice_j()));
    let [u, v, t] = q.map(|x| i64::from(x.m));
    if !triangle(a, b, c) || u + v + t != 0 {
        return Ok(());
    }
    let (ab, au, bv, cu, cv) = (
        (a + b - c) / 2,
        (a - u) / 2,
        (b + v) / 2,
        (c - b + u) / 2,
        (c - a - v) / 2,
    );
    let start = 0.max(-cu).max(-cv);
    let end = ab.min(au).min(bv);
    racah_sum(
        primes,
        least_prime,
        start,
        end,
        |k| (0, [k, ab - k, au - k, bv - k, cu + k, cv + k, 0]),
        out,
        term,
        powers,
        work,
    )?;
    delta(primes, &mut out.powers, a, b, c, work)?;
    factorials(
        primes,
        powers,
        &[
            (a + u) / 2,
            (a - u) / 2,
            (b + v) / 2,
            (b - v) / 2,
            (c + t) / 2,
            (c - t) / 2,
        ],
        &[],
        work,
    )?;
    for (e, &x) in out.powers.iter_mut().zip(powers.iter()) {
        *e += x;
    }
    out.integer.negative ^= parity((a - b - t) / 2);
    Ok(())
}
fn six(
    primes: &[u32],
    least_prime: &[u32],
    j: [i64; 6],
    out: &mut Coefficient,
    term: &mut Integer,
    powers: &mut [i64],
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    out.clear();
    let [a, b, c, d, e, f] = j;
    let triangles = [(a, b, c), (a, e, f), (d, b, f), (d, e, c)];
    if triangles.iter().any(|&(a, b, c)| !triangle(a, b, c)) {
        return Ok(());
    }
    let low = [
        (a + b + c) / 2,
        (a + e + f) / 2,
        (d + b + f) / 2,
        (d + e + c) / 2,
    ];
    let high = [
        (a + b + d + e) / 2,
        (b + c + e + f) / 2,
        (c + a + f + d) / 2,
    ];
    let start = low.into_iter().fold(0, i64::max);
    let end = high.into_iter().fold(i64::MAX, i64::min);
    racah_sum(
        primes,
        least_prime,
        start,
        end,
        |k| {
            (
                k + 1,
                [
                    k - low[0],
                    k - low[1],
                    k - low[2],
                    k - low[3],
                    high[0] - k,
                    high[1] - k,
                    high[2] - k,
                ],
            )
        },
        out,
        term,
        powers,
        work,
    )?;
    for (a, b, c) in triangles {
        delta(primes, &mut out.powers, a, b, c, work)?;
    }
    Ok(())
}
fn nine(
    primes: &[u32],
    least_prime: &[u32],
    j: [i64; 9],
    s: &mut AngularWorkspace,
    work: &mut Work<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    s.result.clear();
    if (0..3).any(|k| {
        !triangle(j[3 * k], j[3 * k + 1], j[3 * k + 2]) || !triangle(j[k], j[k + 3], j[k + 6])
    }) {
        return Ok(());
    }
    if j[8] == 0 {
        // https://dlmf.nist.gov/34.7.E1; the selection rules above imply
        // j[2]=j[5] and j[6]=j[7].
        // Apply the phase and inverse square root in the exact representation.
        six(
            primes,
            least_prime,
            [j[0], j[1], j[2], j[4], j[3], j[6]],
            &mut s.result,
            &mut s.term,
            &mut s.scratch_powers,
            work,
        )?;
        s.result.integer.negative ^= parity((j[1] + j[3] + j[2] + j[6]) / 2);
        for n in [j[2] + 1, j[6] + 1] {
            factor(primes, least_prime, &mut s.result.powers, n, -1, work)?;
        }
        return Ok(());
    }
    let start = (j[0] - j[8])
        .abs()
        .max((j[3] - j[7]).abs())
        .max((j[1] - j[5]).abs());
    let end = (j[0] + j[8]).min(j[3] + j[7]).min(j[1] + j[5]);
    // All three variable triangles have the same parity after row/column
    // validation. Their lower bounds therefore already have the required parity.
    for k in (start..=end).step_by(2) {
        work.try_step()?;
        let requests = [
            [j[0], j[3], j[6], j[7], j[8], k],
            [j[1], j[4], j[7], j[3], k, j[5]],
            [j[2], j[5], j[8], k, j[0], j[1]],
        ];
        for (out, request) in s.coefficients.iter_mut().zip(requests) {
            six(
                primes,
                least_prime,
                request,
                out,
                &mut s.term,
                &mut s.scratch_powers,
                work,
            )?;
            if out.integer.is_zero() {
                break;
            }
        }
        if s.coefficients.iter().any(|c| c.integer.is_zero()) {
            continue;
        }
        for (index, out) in s.scratch_powers.iter_mut().enumerate() {
            *out = s.coefficients.iter().map(|c| c.powers[index]).sum();
        }
        Integer::product(
            &s.coefficients[0].integer,
            &s.coefficients[1].integer,
            &mut s.product,
            work,
        )?;
        Integer::product(&s.product, &s.coefficients[2].integer, &mut s.term, work)?;
        s.term.mul_small(
            u32::try_from(k + 1).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("summation_index"))
            })?,
            work,
        )?;
        s.term.negative ^= parity(k);
        add_factored(primes, &s.scratch_powers, &mut s.result, &mut s.term, work)?;
    }
    Ok(())
}
fn convert(
    primes: &[u32],
    c: &Coefficient,
    work: &mut Work<'_, '_, '_>,
) -> Result<(Scaled<f64>, f64), SciOxideError> {
    if c.integer.is_zero() {
        return Ok((Scaled::ZERO, 0.0));
    }

    let mut value = c.integer.scaled()?;
    let mut operations = 4_u64;
    for (&p, &power) in primes.iter().zip(&c.powers) {
        if power == 0 {
            continue;
        }
        let half = power.div_euclid(2);
        let mut exponent = half.unsigned_abs();
        let mut factor = Scaled::from_value(f64::from(p))?;
        let mut product = Scaled::from_value(1.0)?;
        while exponent > 0 {
            work.try_step()?;
            if exponent & 1 != 0 {
                product = product.mul(factor)?;
                operations += 1;
            }
            exponent >>= 1;
            if exponent > 0 {
                factor = factor.mul(factor)?;
                operations += 1;
            }
        }
        value = if half < 0 {
            value.div(product)?
        } else {
            value.mul(product)?
        };
        operations += 1;
        if power.rem_euclid(2) != 0 {
            work.try_step()?;
            value = value.mul(Scaled::from_value(f64::from(p))?.sqrt()?)?;
            operations += 2;
        }
    }
    Ok((value, 8.0 * f64::EPSILON * operations as f64))
}

// Preparation and execution consume one default allowance without mutating a plan.
fn convenient<S: RealScalar>(
    momenta: impl IntoIterator<Item = AngularMomentum>,
    run: impl FnOnce(
        &AngularPlan,
        &mut AngularWorkspace,
        &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError>,
) -> Result<S, SciOxideError> {
    let maximum = momenta.into_iter().max().unwrap_or(AngularMomentum(0));
    let options = AngularOptions::default();
    let plan = AngularPlan::new(maximum, options)?;
    let preparation = plan.preparation_work;
    let remaining = options.iteration_budget.get().saturating_sub(preparation);
    if remaining == 0 {
        return Err(crate::result::exhausted_work(preparation));
    }
    let result = plan.evaluate_for::<S>(&mut plan.workspace()?, None, remaining, run)?;
    let total = preparation
        .checked_add(result.counts().iterations())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("iterations")))?;
    result
        .with_counts(scioxide_core::EvaluationCounts::from_iterations(total))
        .value()
}
/// Builds the states `state(0)`, `state(1)` and `state(2)`, reporting the first error.
fn states(
    state: impl FnMut(usize) -> Result<AngularState, SciOxideError>,
) -> Result<[AngularState; 3], SciOxideError> {
    let [a, b, c] = core::array::from_fn(state);
    Ok([a?, b?, c?])
}
/// Validates doubled angular momenta, reporting the first negative one.
fn momenta<const N: usize>(twice_j: [i32; N]) -> Result<[AngularMomentum; N], SciOxideError> {
    twice_j
        .into_iter()
        .try_for_each(|j| AngularMomentum::new(j).map(drop))?;
    Ok(twice_j.map(AngularMomentum))
}
/// Computes the Wigner 3j symbol `(j₁ j₂ j₃; m₁ m₂ m₃)` from the doubled quantum numbers
/// `twice_j = [2j₁, 2j₂, 2j₃]` and `twice_m = [2m₁, 2m₂, 2m₃]`.
///
/// Doubling makes half-integer quantum numbers exact: `j = ½` is `1`. The symbol is given by
/// Racah's formula (DLMF §34.2) in the Condon–Shortley phase convention. It vanishes unless
/// `m₁ + m₂ + m₃ = 0`, `|j₁ − j₂| ≤ j₃ ≤ j₁ + j₂` and `j₁ + j₂ + j₃` is an integer; such
/// forbidden symbols are returned as exact zero.
///
/// Each call builds an [`AngularPlan`] for the largest `j` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::wigner_3j`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if some `2jᵢ < 0`, or with argument
///   `twice_m` if some `|mᵢ| > jᵢ` or `jᵢ − mᵢ` is not an integer.
/// - `ResourceLimit` if the largest `j` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// // (1 1 0; 1 −1 0) = 1/√3: 2j = [2, 2, 0] and 2m = [2, −2, 0]
/// let value = scioxide_special::wigner_3j::<f64>([2, 2, 0], [2, -2, 0])?;
/// assert!((value - 1.0 / 3.0_f64.sqrt()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.2–34.3. <https://dlmf.nist.gov/34.2>
pub fn wigner_3j<S: RealScalar>(twice_j: [i32; 3], twice_m: [i32; 3]) -> Result<S, SciOxideError> {
    let q = states(|i| AngularState::new(twice_j[i], twice_m[i]))?;
    convenient(q.map(AngularState::momentum), |plan, scratch, work| {
        plan.three_estimate(q, scratch, work)
    })
}
/// Computes the Clebsch–Gordan coefficient `⟨j₁ m₁; j₂ m₂ | J M⟩` from the doubled quantum
/// numbers `twice_j = [2j₁, 2j₂, 2J]` and `twice_m = [2m₁, 2m₂, 2M]`.
///
/// Doubling makes half-integer quantum numbers exact: `j = ½` is `1`. The coefficient is
///
/// ```text
/// ⟨j₁ m₁; j₂ m₂ | J M⟩ = (−1)^(j₁−j₂+M) √(2J+1) (j₁ j₂ J; m₁ m₂ −M),
/// ```
///
/// where the bracket on the right is [`wigner_3j`] (DLMF §34.1). The coefficient vanishes
/// unless `M = m₁ + m₂` and `(j₁, j₂, J)` satisfy the triangle conditions; such forbidden
/// coefficients are returned as exact zero.
///
/// Each call builds an [`AngularPlan`] for the largest `j` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::clebsch_gordan`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if some `2jᵢ < 0`, or with argument
///   `twice_m` if some `|mᵢ| > jᵢ` or `jᵢ − mᵢ` is not an integer.
/// - `ResourceLimit` if the largest `j` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// // ⟨½ ½; ½ −½ | 1 0⟩ = 1/√2: 2j = [1, 1, 2] and 2m = [1, −1, 0]
/// let value = scioxide_special::clebsch_gordan::<f64>([1, 1, 2], [1, -1, 0])?;
/// assert!((value - 0.5_f64.sqrt()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.1–34.2. <https://dlmf.nist.gov/34>
pub fn clebsch_gordan<S: RealScalar>(
    twice_j: [i32; 3],
    twice_m: [i32; 3],
) -> Result<S, SciOxideError> {
    let [a, b, c] = states(|i| AngularState::new(twice_j[i], twice_m[i]))?;
    convenient(
        [a, b, c].map(AngularState::momentum),
        |plan, scratch, work| plan.clebsch_estimate(a, b, c, scratch, work),
    )
}
/// Computes the Wigner 6j symbol `{j₁ j₂ j₃; j₄ j₅ j₆}` from the doubled momenta
/// `twice_j = [2j₁, 2j₂, 2j₃, 2j₄, 2j₅, 2j₆]`, listed in row-major order.
///
/// Doubling makes half-integer momenta exact: `j = ½` is `1`. The symbol is given by Racah's
/// sum (DLMF §34.4). It vanishes unless each of the triads `(j₁ j₂ j₃)`, `(j₁ j₅ j₆)`,
/// `(j₄ j₂ j₆)` and `(j₄ j₅ j₃)` satisfies the triangle conditions and has an integer sum;
/// such forbidden symbols are returned as exact zero.
///
/// Each call builds an [`AngularPlan`] for the largest `j` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::wigner_6j`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if some `2jᵢ < 0`.
/// - `ResourceLimit` if the largest `j` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// // {0 j j; 0 j j} = (−1)^(2j)/(2j+1); here j = ½, so 2j = 1
/// let value = scioxide_special::wigner_6j::<f64>([0, 1, 1, 0, 1, 1])?;
/// assert!((value + 0.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.4. <https://dlmf.nist.gov/34.4>
pub fn wigner_6j<S: RealScalar>(twice_j: [i32; 6]) -> Result<S, SciOxideError> {
    let j = momenta(twice_j)?;
    convenient(j, |plan, scratch, work| plan.six_estimate(j, scratch, work))
}
/// Computes the Wigner 9j symbol `{j₁ j₂ j₃; j₄ j₅ j₆; j₇ j₈ j₉}` from the doubled momenta
/// `twice_j = [2j₁, 2j₂, …, 2j₉]`, listed row by row.
///
/// Doubling makes half-integer momenta exact: `j = ½` is `1`. The symbol is the sum over `k`
/// of `(−1)^(2k) (2k+1)` times three 6j symbols (DLMF §34.6), each evaluated exactly, and
/// reduces to a single 6j symbol when `j₉ = 0`. It vanishes unless every row and every column
/// satisfies the triangle conditions; such forbidden symbols are returned as exact zero. The
/// cost grows with the number of terms `k`, at most `2j_max + 1`.
///
/// Each call builds an [`AngularPlan`] for the largest `j` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::wigner_9j`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if some `2jᵢ < 0`.
/// - `ResourceLimit` if the largest `j` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// // {1 1 0; 1 1 0; 0 0 0} = {1 1 0; 1 1 0} = 1/3, with 2j = 2 for every j = 1
/// let value = scioxide_special::wigner_9j::<f64>([2, 2, 0, 2, 2, 0, 0, 0, 0])?;
/// assert!((value - 1.0 / 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.6. <https://dlmf.nist.gov/34.6>
pub fn wigner_9j<S: RealScalar>(twice_j: [i32; 9]) -> Result<S, SciOxideError> {
    let j = momenta(twice_j)?;
    convenient(j, |plan, scratch, work| {
        plan.nine_estimate(j, scratch, work)
    })
}
/// Computes the Racah coefficient `W(a b c d; e f)` from the doubled momenta
/// `twice_j = [2a, 2b, 2c, 2d, 2e, 2f]`.
///
/// Doubling makes half-integer momenta exact: `j = ½` is `1`. The coefficient is
///
/// ```text
/// W(a b c d; e f) = (−1)^(a+b+c+d) {a b e; d c f},
/// ```
///
/// where the brace is [`wigner_6j`] (DLMF §34.4). Forbidden couplings are returned as exact
/// zero.
///
/// Each call builds an [`AngularPlan`] for the largest `j` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::racah_w`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `twice_j` if some `2jᵢ < 0`.
/// - `ResourceLimit` if the largest `j` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `3·2j_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// // W(½ ½ ½ ½; 0 0) = (−1)^2 {½ ½ 0; ½ ½ 0} = −1/2, with 2j = 1 for every j = ½
/// let value = scioxide_special::racah_w::<f64>([1, 1, 1, 1, 0, 0])?;
/// assert!((value + 0.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.4. <https://dlmf.nist.gov/34.4>
pub fn racah_w<S: RealScalar>(twice_j: [i32; 6]) -> Result<S, SciOxideError> {
    let j = momenta(twice_j)?;
    convenient(j, |plan, scratch, work| {
        plan.racah_estimate(j, scratch, work)
    })
}
/// Computes the Gaunt coefficient, the integral of a product of three spherical harmonics, from
/// the degrees `degree = [l₁, l₂, l₃]` and orders `order = [m₁, m₂, m₃]`, which are not doubled.
///
/// ```text
/// ∫ Y_l₁^m₁ Y_l₂^m₂ Y_l₃^m₃ dΩ
///     = √((2l₁+1)(2l₂+1)(2l₃+1)/(4π)) (l₁ l₂ l₃; 0 0 0) (l₁ l₂ l₃; m₁ m₂ m₃),
/// ```
///
/// where the integral runs over the unit sphere without complex conjugation, `Y_l^m` is
/// [`crate::spherical_harmonic`], whose degree and order have the same types, and the brackets
/// are [`wigner_3j`] (DLMF §34.3). Forbidden couplings, including an odd `l₁ + l₂ + l₃`, are
/// returned as exact zero.
///
/// Each call builds an [`AngularPlan`] for the largest `l` and evaluates the exact sum once;
/// preparation and evaluation share the default work budget of [`AngularOptions`]. Use
/// [`AngularPlan::gaunt`] to evaluate many coefficients or to get the error estimate.
///
/// # Accuracy
///
/// The sum is exact; only the final conversion to the scalar type rounds. No uniform ulp bound
/// is stated; see the [crate accuracy table](crate#accuracy).
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) with argument `degree` if some `2lᵢ` exceeds `i32::MAX`, or
///   with argument `order` if some `|mᵢ| > lᵢ`.
/// - `ResourceLimit` if the largest `l` needs more memory than the default limits of
///   [`AngularOptions`].
/// - `SizeOverflow` if `6l_max + 2` exceeds `u32::MAX`.
/// - `ConvergenceFailure` (`IterationLimit`) if preparation and evaluation together need more
///   than the default work budget.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
///
/// // ∫ Y₁⁰ Y₁⁰ Y₀⁰ dΩ = Y₀⁰ = (4π)^(−1/2), because Y₁⁰ is real and normalized
/// let value = scioxide_special::gaunt::<f64>([1, 1, 0], [0, 0, 0])?;
/// assert!((value - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §34.3. <https://dlmf.nist.gov/34.3>
pub fn gaunt<S: RealScalar>(degree: [usize; 3], order: [i32; 3]) -> Result<S, SciOxideError> {
    let q = states(|i| AngularState::harmonic(degree[i], order[i]))?;
    convenient(q.map(AngularState::momentum), |plan, scratch, work| {
        plan.gaunt_estimate(q, scratch, work)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn changing_common_factors_preserves_multiword_cancellation() {
        let primes = [2, 3];
        let mut out = Coefficient::new(2, 8).unwrap();
        let mut term = Integer::new(8).unwrap();
        let mut work = Work::new(10_000);
        // 3*2^100 - 45 - 3*2^100 + 45, followed by -sqrt(6) + 2sqrt(6).
        // The small residual must survive both rescaling and exact cancellation.
        for (magnitude, negative, powers, expected) in [
            (3, false, [200, 0], None),
            (5, true, [0, 4], None),
            (3, true, [200, 0], Some(-45.0)),
            (5, false, [0, 4], Some(0.0)),
            (1, true, [1, 1], Some(-6.0_f64.sqrt())),
            (2, false, [1, 1], Some(6.0_f64.sqrt())),
        ] {
            term.one().unwrap();
            term.mul_small(magnitude, &mut work).unwrap();
            term.negative = negative;
            add_factored(&primes, &powers, &mut out, &mut term, &mut work).unwrap();
            if let Some(expected) = expected {
                let actual = convert(&primes, &out, &mut work)
                    .unwrap()
                    .0
                    .try_value()
                    .unwrap();
                assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
            }
        }
    }

    #[test]
    fn prepared_factor_indices_cover_bound_and_reject_outside() {
        let plan = AngularPlan::new(AngularMomentum::new(20).unwrap(), Default::default()).unwrap();
        let mut powers = alloc::vec![0; plan.primes.len()];
        let mut work = Work::new(10_000);
        for n in 1..plan.least_prime.len() {
            powers.fill(0);
            factor(
                &plan.primes,
                &plan.least_prime,
                &mut powers,
                n as i64,
                1,
                &mut work,
            )
            .unwrap();
            let product = plan
                .primes
                .iter()
                .zip(&powers)
                .fold(1_u64, |product, (&p, &e)| {
                    product * u64::from(p).pow(e as u32)
                });
            assert_eq!(product, n as u64);
        }
        for n in [-1, 0, plan.least_prime.len() as i64, i64::MAX] {
            assert!(
                factor(
                    &plan.primes,
                    &plan.least_prime,
                    &mut powers,
                    n,
                    1,
                    &mut work
                )
                .is_err()
            );
        }
    }
}
