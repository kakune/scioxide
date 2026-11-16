//! Distribution tests applied to each batch of a [`BatchView`](crate::BatchView).
use super::*;
use crate::{ObservationData, StatisticsOptions};
use alloc::vec::Vec;

use scioxide_core::{ExecutionContext, backend::try_filled};

/// Selects the distribution test that [`distribution_tests_into`] applies to
/// each batch.
///
/// Each variant calls the named function with the batch's samples. Two-sample
/// variants, including the variance and k-sample tests applied to two groups,
/// need a second view `y`.
#[derive(Clone, Copy)]
pub enum DistributionBatchTest<'a, S: RealScalar> {
    /// [`skew_test_with_options`].
    Skew,
    /// [`kurtosis_test_with_options`].
    Kurtosis,
    /// [`normality_test_with_options`].
    Normality,
    /// [`jarque_bera_test_with_options`].
    JarqueBera,
    /// [`shapiro_wilk_test_with_options`].
    Shapiro,
    /// [`ks_one_sample_test_with_options`] against the given CDF.
    KsOne(&'a (dyn Fn(S) -> Result<S, SciOxideError> + Sync)),
    /// [`cramer_von_mises_test_with_options`] against the given CDF.
    CramerVonMisesOne(&'a (dyn Fn(S) -> Result<S, SciOxideError> + Sync)),
    /// [`ks_two_sample_test_with_options`].
    KsTwo,
    /// [`cramer_von_mises_two_sample_test_with_options`].
    CramerVonMisesTwo,
    /// [`epps_singleton_two_sample_test_with_options`] at the given frequencies.
    EppsSingleton([f64; 2]),
    /// [`mood_scale_test_with_options`].
    Mood,
    /// [`ansari_bradley_test_with_options`].
    AnsariBradley,
    /// [`bartlett_test_with_options`] of the two samples.
    Bartlett,
    /// [`levene_test_with_options`] of the two samples with the given center.
    Levene(VarianceCenter),
    /// [`fligner_test_with_options`] of the two samples with the given center.
    Fligner(VarianceCenter),
    /// [`anderson_k_sample_test_with_options`] of the two samples.
    AndersonK(AndersonKVariant),
}
impl<S: RealScalar> DistributionBatchTest<'_, S> {
    fn two_samples(self) -> bool {
        matches!(
            self,
            Self::KsTwo
                | Self::CramerVonMisesTwo
                | Self::EppsSingleton(_)
                | Self::Mood
                | Self::AnsariBradley
                | Self::Bartlett
                | Self::Levene(_)
                | Self::Fligner(_)
                | Self::AndersonK(_)
        )
    }
}

struct Worker<S: RealScalar> {
    x: Vec<S>,
    y: Vec<S>,
}

/// Workspace for [`distribution_tests_into`].
///
/// It holds per-worker copies of one batch and the results of all batches. The
/// tests still allocate their own scratch on each call.
pub struct DistributionBatchWorkspace<S: RealScalar> {
    workers: Vec<Worker<S>>,
    results: Vec<Option<DistributionTestResult<S>>>,
}
impl<S: RealScalar> DistributionBatchWorkspace<S> {
    /// Allocates a workspace for the shapes of `x` and `y` and for the number of
    /// workers that the [`StatisticsOptions`] and `context` select.
    ///
    /// Pass the same statistics options and context to [`distribution_tests_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a view has vector-valued observations, or `y` has a
    ///   different batch count from `x`.
    /// - `ResourceLimit` if the workspace exceeds
    ///   [`StatisticsOptions::max_bytes`].
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn new(
        x: crate::BatchView<'_, S>,
        y: Option<crate::BatchView<'_, S>>,
        admission: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let x = ObservationData::from_batch(x);
        let y = y.map(ObservationData::from_batch);
        if x.dimensions() != 1
            || y.is_some_and(|v| v.dimensions() != 1 || v.batches() != x.batches())
        {
            return Err(invalid(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "distribution batch shape invalid",
            ));
        }
        let workers = crate::estimation::worker_count(admission, context, x.batches())?.max(1);
        let per_worker = x
            .samples()
            .checked_add(y.map_or(0, |v| v.samples()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        let bytes = crate::estimation::storage_bytes(&[
            (workers, core::mem::size_of::<Worker<S>>()),
            (
                per_worker.checked_mul(workers).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                })?,
                core::mem::size_of::<S>(),
            ),
            (
                x.batches(),
                core::mem::size_of::<Option<DistributionTestResult<S>>>(),
            ),
        ])?;
        crate::error_helpers::limit(
            bytes,
            admission.max_bytes(),
            scioxide_core::quantity!("batch_bytes"),
        )?;
        let mut slots = scioxide_core::backend::try_with_capacity(
            workers,
            scioxide_core::quantity!("distribution_batch_workers"),
        )?;
        for _ in 0..workers {
            slots.push(Worker {
                x: try_filled(
                    x.samples(),
                    S::ZERO,
                    scioxide_core::quantity!("distribution_batch_x"),
                )?,
                y: try_filled(
                    y.map_or(0, |v| v.samples()),
                    S::ZERO,
                    scioxide_core::quantity!("distribution_batch_y"),
                )?,
            });
        }
        Ok(Self {
            workers: slots,
            results: try_filled(
                x.batches(),
                None,
                scioxide_core::quantity!("distribution_batch_results"),
            )?,
        })
    }
}

/// Applies one distribution test to every batch of `x` (and `y`), writing one
/// result per batch to `output`.
///
/// Missing and non-finite observations follow the policies of the
/// [`StatisticsOptions`]; see
/// [Data, weights and missing values](crate#data-weights-and-missing-values).
/// `options` configures the test itself.
/// Each sample drops its own left-out observations. Batches may run in parallel
/// as the statistics options and `execution` allow. Weighted views and vector-valued
/// observations are not supported. Each test keeps the work and byte limits of
/// its scalar function.
///
/// # Errors
///
/// - `InvalidInput` if `y` is given for a one-sample test or missing for a
///   two-sample test, a view has weights or vector-valued observations, the
///   views have different batch counts, `output.len()` differs from the batch
///   count, or `workspace` was sized for smaller views.
/// - `InvalidInput` or `NonFinite` if a batch has a missing or non-finite
///   observation under a `Reject` policy.
/// - Any error of the selected test on any batch.
#[allow(clippy::too_many_arguments)]
pub fn distribution_tests_into<S: RealScalar>(
    test: DistributionBatchTest<'_, S>,
    x: crate::BatchView<'_, S>,
    y: Option<crate::BatchView<'_, S>>,
    options: ClassicalOptions,
    admission: StatisticsOptions,
    output: &mut [Option<DistributionTestResult<S>>],
    workspace: &mut DistributionBatchWorkspace<S>,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let x = ObservationData::from_batch(x);
    let y = y.map(ObservationData::from_batch);
    if x.weights.is_some()
        || y.is_some_and(|v| v.weights.is_some())
        || test.two_samples() != y.is_some()
        || x.dimensions() != 1
        || y.is_some_and(|v| v.dimensions() != 1 || v.batches() != x.batches())
        || output.len() != x.batches()
        || workspace.results.len() < x.batches()
        || workspace
            .workers
            .iter()
            .any(|w| w.x.len() < x.samples() || w.y.len() < y.map_or(0, |v| v.samples()))
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "distribution batch shape, weights or workspace mismatch",
        ));
    }
    crate::estimation::execute(
        admission,
        execution,
        x.batches(),
        1,
        &mut workspace.workers,
        &mut workspace.results,
        |batch, work, staged| {
            let mut nx = 0;
            for i in 0..x.samples() {
                if x.admit(batch, i, admission, scioxide_core::argument!("x"))?
                    .is_some()
                {
                    work.x[nx] = x.value(batch, i, 0);
                    nx += 1;
                }
            }
            let mut ny = 0;
            if let Some(y) = y {
                for i in 0..y.samples() {
                    if y.admit(batch, i, admission, scioxide_core::argument!("y"))?
                        .is_some()
                    {
                        work.y[ny] = y.value(batch, i, 0);
                        ny += 1;
                    }
                }
            }
            let a = &work.x[..nx];
            let b = &work.y[..ny];
            let result = match test {
                DistributionBatchTest::Skew => skew_test_with_options(a, options),
                DistributionBatchTest::Kurtosis => kurtosis_test_with_options(a, options),
                DistributionBatchTest::Normality => normality_test_with_options(a, options),
                DistributionBatchTest::JarqueBera => jarque_bera_test_with_options(a, options),
                DistributionBatchTest::Shapiro => shapiro_wilk_test_with_options(a, options),
                DistributionBatchTest::KsOne(cdf) => {
                    ks_one_sample_test_with_options(a, cdf, options)
                }
                DistributionBatchTest::CramerVonMisesOne(cdf) => {
                    cramer_von_mises_test_with_options(a, cdf, options)
                }
                DistributionBatchTest::KsTwo => ks_two_sample_test_with_options(a, b, options),
                DistributionBatchTest::CramerVonMisesTwo => {
                    cramer_von_mises_two_sample_test_with_options(a, b, options)
                }
                DistributionBatchTest::EppsSingleton(points) => {
                    epps_singleton_two_sample_test_with_options(a, b, points, options)
                }
                DistributionBatchTest::Mood => mood_scale_test_with_options(a, b, options),
                DistributionBatchTest::AnsariBradley => {
                    ansari_bradley_test_with_options(a, b, options)
                }
                DistributionBatchTest::Bartlett => bartlett_test_with_options(&[a, b], options),
                DistributionBatchTest::Levene(center) => {
                    levene_test_with_options(&[a, b], center, options)
                }
                DistributionBatchTest::Fligner(center) => {
                    fligner_test_with_options(&[a, b], center, options)
                }
                DistributionBatchTest::AndersonK(variant) => {
                    anderson_k_sample_test_with_options(&[a, b], variant, options)
                }
            }?;
            staged[0] = Some(result);
            Ok(())
        },
    )?;
    output.copy_from_slice(&workspace.results[..output.len()]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::StridedLayout;
    #[test]
    fn borrowed_lanes_commit_only_after_all_succeed() {
        let values = [1.0_f64, 2.0, 3.0, 2.0, 3.0, 4.0];
        let view = crate::BatchView::new(
            &values,
            StridedLayout::new(&[2, 3], &[3, 1]).unwrap(),
            crate::Axis::Index(1),
        )
        .unwrap();
        let admission = StatisticsOptions::default().with_max_bytes(1024);
        let mut workspace =
            DistributionBatchWorkspace::new(view, None, admission, ExecutionContext::sequential())
                .unwrap();
        // The admission byte limit is the workspace's only budget.
        let error = DistributionBatchWorkspace::new(
            view,
            None,
            admission.with_max_bytes(8),
            ExecutionContext::sequential(),
        )
        .err()
        .unwrap();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { required, limit: 8, .. } if required > 8
        ));
        let mut output = [None, None];
        distribution_tests_into(
            DistributionBatchTest::Shapiro,
            view,
            None,
            ClassicalOptions::default(),
            StatisticsOptions::default(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert!((output[0].unwrap().statistic - 1.0).abs() < 1e-14);
        assert!((output[1].unwrap().statistic - 1.0).abs() < 1e-14);
        let bad = [1.0_f64, 2.0, 3.0, 2.0, 2.0, 2.0];
        let bad_view = crate::BatchView::new(
            &bad,
            StridedLayout::new(&[2, 3], &[3, 1]).unwrap(),
            crate::Axis::Index(1),
        )
        .unwrap();
        let previous = output;
        assert!(
            distribution_tests_into(
                DistributionBatchTest::Shapiro,
                bad_view,
                None,
                ClassicalOptions::default(),
                StatisticsOptions::default(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, previous);
    }
}
