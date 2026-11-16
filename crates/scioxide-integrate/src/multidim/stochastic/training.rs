//! Training report, importance norm and the training driver shared by VEGAS and MISER.

use scioxide_core::{
    AffineValue, AlgorithmId, EvaluationRole, RealScalar, Scaled, ScaledAccumulator, SciOxideError,
    SolveBudgets, SolveContext, SolveScope,
};

use crate::report::IntegrationTermination;
use crate::workspace::Attempts;

/// Outcome of training a VEGAS map or MISER strata.
///
/// The training observations do not enter any later estimate. The report is
/// successful when training ran to its end. When a budget, cancellation or
/// deadline stops training after at least one update, training returns `Ok`
/// with the map or strata of the last completed update and the stop as an
/// unsuccessful [`termination`](Self::termination); see
/// [termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{RandomStreams, VegasOptions, VegasTrainer};
/// use scioxide_random::{Philox4x64, SeedableRng};
/// let plan = VegasTrainer::new(&[Interval::new(0.0, 1.0)?], VegasOptions::default())?;
/// let root = Philox4x64::from_seed(1);
/// let (_, report) = plan.train(|x: &[f64]| Ok(x[0]), RandomStreams::new(&root, 0))?;
/// assert!(report.is_success());
/// // The default is 5 iterations of 2,048 pilot samples.
/// assert_eq!(report.updates(), 5);
/// assert_eq!(report.evaluations(), 5 * 2048);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct TrainingReport {
    termination: IntegrationTermination,
    evaluations: usize,
    updates: usize,
}

impl TrainingReport {
    /// Returns how training ended.
    ///
    /// The status is `SampleBudgetCompleted` when training ran to its end,
    /// `FixedRuleCompleted` for a box of dimension zero (or MISER training of zero
    /// components), `DegenerateDomain` for a box of zero width, and `BudgetLimit`
    /// or `ExecutionStopped` after a stop.
    pub const fn termination(&self) -> IntegrationTermination {
        self.termination
    }

    /// Returns whether training ran to its end.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Returns the number of calls of the integrand.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Returns the number of completed VEGAS iterations or MISER splits.
    #[must_use]
    pub const fn updates(&self) -> usize {
        self.updates
    }
}

/// Returns the Euclidean norm of the callback's real components, or the absolute
/// value of a single component; a non-finite component is a `NonFinite` error.
pub(super) fn importance<S: RealScalar, V: AffineValue<Scale = S> + ?Sized>(
    value: &V,
) -> Result<Scaled<S>, SciOxideError> {
    let components = value.dimension()?;
    if components == 1 {
        return Scaled::from_value(
            crate::kernel::checked_callback_value(value.component(0)?)?.abs(),
        );
    }
    let mut squares = ScaledAccumulator::ZERO;
    for index in 0..components {
        let component = crate::kernel::checked_callback_value(value.component(index)?)?;
        squares.add_scaled(Scaled::from_value(component)?.square()?)?;
    }
    squares.total()?.sqrt()
}

/// State returned by a training kernel.
pub(super) struct Trained {
    /// How training ended; `SampleBudgetCompleted` when every pass completed.
    pub(super) termination: IntegrationTermination,
    /// Completed updates; the kernel wrote the map or strata to the output exactly
    /// when this is positive or the termination is successful.
    pub(super) updates: usize,
}

/// Runs one training call and records it in `attempts`. A stop before the first
/// update becomes `ConvergenceFailure` or `ExecutionStopped`.
pub(super) fn train<'capability>(
    attempts: &mut Attempts<TrainingReport>,
    scope: SolveScope<'_, 'capability>,
    algorithm: AlgorithmId,
    dimension: usize,
    kernel: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<Trained, SciOxideError>,
) -> Result<TrainingReport, SciOxideError> {
    attempts.run(
        scope,
        crate::workspace::identity(algorithm, dimension),
        SolveBudgets::unbounded(),
        |context| {
            let trained = kernel(context)?;
            let counts = context.counts();
            if trained.updates == 0 && !trained.termination.is_success() {
                return Err(crate::workspace::unpublished(
                    trained.termination,
                    algorithm,
                    counts,
                ));
            }
            Ok(TrainingReport {
                termination: trained.termination,
                evaluations: counts.total()?.provider(EvaluationRole::Function),
                updates: trained.updates,
            })
        },
    )
}
