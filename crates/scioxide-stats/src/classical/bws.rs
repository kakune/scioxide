//! Baumgartner–Weiss–Schindler test with exact or Monte Carlo permutation p-values.
use super::*;
use crate::{
    PermutationKind, PermutationMode, ResampledIndices, ResamplingMethod, ResamplingOptions,
    ResamplingPlan, ResamplingWorkspace,
};
use alloc::vec::Vec;

/// Prepared Baumgartner–Weiss–Schindler (BWS) two-sample test for permutation
/// p-values.
///
/// The plan stores the pooled midranks of the two samples. [`BwsPlan::test`]
/// evaluates the statistic of [`bws_test_with_options`] over exact or Monte Carlo label
/// permutations, so it handles samples too large for full enumeration. Each
/// evaluation of the statistic costs O(n² + m²) time.
///
/// # References
///
/// W. Baumgartner, P. Weiß and H. Schindler, "A nonparametric test for the
/// general two-sample problem", *Biometrics* 54 (1998), 1129–1135.
#[derive(Clone, Debug)]
pub struct BwsPlan {
    pool: Vec<f64>,
    identity: Vec<usize>,
    first: usize,
    alternative: TestAlternative,
}
impl BwsPlan {
    /// Ranks the pooled samples `x` and `y` for the given alternative.
    ///
    /// A two-sided alternative selects the omnibus statistic. `Less` and
    /// `Greater` select the signed directional statistic.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if either sample has fewer than two observations.
    /// - `NonFinite` if `x` or `y` contains NaN or infinity.
    pub fn new<S: RealScalar>(
        x: &[S],
        y: &[S],
        alternative: TestAlternative,
    ) -> Result<Self, SciOxideError> {
        sample(x, 2, scioxide_core::argument!("x"))?;
        sample(y, 2, scioxide_core::argument!("y"))?;
        let pool = ranks::pooled_ranks(x, y)?;
        let mut identity = scioxide_core::backend::try_with_capacity(
            pool.len(),
            scioxide_core::quantity!("bws_identity"),
        )?;
        identity.extend(0..pool.len());
        Ok(Self {
            pool,
            identity,
            first: x.len(),
            alternative,
        })
    }
    /// Returns the BWS statistic of the observed samples.
    pub fn observed(&self) -> f64 {
        ranks::bws_score(
            &self.pool,
            &self.identity[..self.first],
            &self.identity[self.first..],
            self.alternative == TestAlternative::TwoSided,
        )
    }
    /// Returns the BWS statistic for one assignment of the pooled observations to
    /// the two groups.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `indices` does not have two groups of the original
    ///   sizes, or contains a duplicate or out-of-range index.
    pub fn statistic(&self, indices: ResampledIndices<'_>) -> Result<f64, SciOxideError> {
        if indices.groups() != 2
            || indices.group(0)?.len() != self.first
            || indices.group(1)?.len() != self.pool.len() - self.first
            || indices
                .indices()
                .iter()
                .enumerate()
                .any(|(i, &v)| v >= self.pool.len() || indices.indices()[..i].contains(&v))
        {
            return Err(invalid(
                scioxide_core::argument!("indices"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid BWS pooled assignment",
            ));
        }
        Ok(ranks::bws_score(
            &self.pool,
            indices.group(0)?,
            indices.group(1)?,
            self.alternative == TestAlternative::TwoSided,
        ))
    }
    /// Creates the matching permutation plan for [`Self::test`].
    ///
    /// The plan permutes labels between two independent groups. The two-sided
    /// statistic uses the upper tail, and the directional statistics keep their
    /// alternative.
    ///
    /// # Errors
    ///
    /// - Any error of [`ResamplingPlan::new`], for example an invalid replicate
    ///   count or an exact enumeration whose size overflows.
    pub fn permutation_plan(
        &self,
        mode: PermutationMode,
        options: ResamplingOptions<f64>,
    ) -> Result<ResamplingPlan<f64>, SciOxideError> {
        ResamplingPlan::new(
            &[self.first, self.pool.len() - self.first],
            ResamplingMethod::Permutation {
                kind: PermutationKind::Independent,
                mode,
                alternative: self.tail(),
            },
            options,
        )
    }
    fn tail(&self) -> TestAlternative {
        if self.alternative == TestAlternative::TwoSided {
            TestAlternative::Greater
        } else {
            self.alternative
        }
    }
    fn check_plan(&self, plan: &ResamplingPlan<f64>) -> Result<(), SciOxideError> {
        if plan.offsets.as_slice() != [0, self.first, self.pool.len()]
            || !matches!(plan.method(),ResamplingMethod::Permutation{kind:PermutationKind::Independent,alternative,..} if alternative==self.tail())
        {
            return Err(invalid(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "BWS permutation plan mismatch",
            ));
        }
        Ok(())
    }
    /// Computes the permutation p-value of the BWS test.
    ///
    /// `plan` must come from [`Self::permutation_plan`]. The remaining arguments
    /// have the same meaning as in [`classical_permutation_test`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `plan` does not match the group sizes or tail of this
    ///   test.
    /// - Any error of [`classical_permutation_test`].
    #[allow(clippy::too_many_arguments)]
    pub fn test_into<'scope, 'capability>(
        &self,
        plan: &ResamplingPlan<f64>,
        replicates: impl crate::ReplicateExecution<f64>,
        output: &mut [f64],
        workspace: &mut ResamplingWorkspace<f64>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ClassicalPermutationResult<f64>, SciOxideError> {
        workspace.reject_attempt();
        self.check_plan(plan)?;
        let report = plan.evaluate_into(
            crate::ResamplingProblem::Indexed(&|indices| self.statistic(indices)),
            replicates,
            output,
            workspace,
            scope,
        )?;
        let test = match (report.observed(), report.inference()) {
            (Some(statistic), Some(crate::ResamplingInference::Test(p))) => Some(result(
                statistic,
                p.p_value,
                ClassicalMethod::Permutation {
                    exact: p.exact,
                    replicates: report.completed(),
                },
                TestSamplePolicy::IndependentGroups,
                [self.first, self.pool.len() - self.first],
            )?),
            _ => None,
        };
        Ok(crate::ClassicalPermutationResult {
            test,
            resampling: report,
        })
    }
    /// Evaluates the permutation test with owned replicate storage.
    ///
    /// # Errors
    ///
    /// Returns allocation errors and the errors of the caller-output form.
    pub fn test(
        &self,
        plan: &ResamplingPlan<f64>,
        replicates: impl crate::ReplicateExecution<f64>,
    ) -> Result<ClassicalPermutationResult<f64>, SciOxideError> {
        let mut output = scioxide_core::backend::try_filled(
            plan.replicates(),
            0.0,
            scioxide_core::quantity!("permutation_output"),
        )?;
        let mut workspace = plan.workspace()?;
        self.test_into(
            plan,
            replicates,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
