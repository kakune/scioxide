//! Classical tests applied to each batch of a [`BatchView`](crate::BatchView).
use super::*;
use crate::{ObservationData, StatisticsOptions};
use alloc::vec::Vec;

use scioxide_core::{ExecutionContext, backend::try_filled};

/// Selects the classical test that [`classical_tests_into`] applies to each batch.
///
/// Each variant calls the function of the same name with the batch's samples.
/// Two-sample variants need a second view `y`; one-sample variants and `Page`
/// take none.
#[derive(Debug, Clone, Copy)]
pub enum ClassicalBatchTest<'a, S: RealScalar> {
    /// [`t_test_one_sample_with_options`] against the given null mean.
    OneSampleT(S),
    /// [`t_test_paired_with_options`].
    PairedT,
    /// [`t_test_independent_with_options`] with the given variance model.
    IndependentT(TTestVariance),
    /// [`pearson_test_with_options`].
    Pearson,
    /// [`point_biserial_test_with_options`], with the binary variable in `x`.
    PointBiserial,
    /// [`spearman_test_with_options`].
    Spearman,
    /// [`kendall_test_with_options`] with the given variant.
    Kendall(KendallVariant),
    /// [`chatterjee_test_with_options`]; the alternative must be `Greater`.
    Chatterjee,
    /// [`somers_test_with_options`]: Somers' `D(y|x)`.
    Somers,
    /// [`mann_whitney_test_with_options`].
    MannWhitney,
    /// [`rank_sum_test_with_options`].
    RankSum,
    /// [`brunner_munzel_test_with_options`] with the given reference distribution.
    BrunnerMunzel(BrunnerMunzelCalibration),
    /// [`bws_test_with_options`].
    Bws,
    /// [`wilcoxon_test_with_options`]; `x` holds the paired differences.
    Wilcoxon(SignedRankZeros),
    /// [`quantile_test_with_options`].
    Quantile {
        /// Quantile value under the null hypothesis.
        value: S,
        /// Probability of the quantile, in `(0, 1)`.
        probability: f64,
    },
    /// [`page_trend_test_with_options`]: each sample of `x` is a block and its vector
    /// coordinates are the treatments.
    Page {
        /// Treatment indices in increasing order of expected response.
        treatment_order: &'a [usize],
    },
}
impl<S: RealScalar> ClassicalBatchTest<'_, S> {
    fn independent(self) -> bool {
        matches!(
            self,
            Self::IndependentT(_)
                | Self::MannWhitney
                | Self::RankSum
                | Self::BrunnerMunzel(_)
                | Self::Bws
        )
    }
    fn two_samples(self) -> bool {
        !matches!(
            self,
            Self::OneSampleT(_) | Self::Wilcoxon(_) | Self::Quantile { .. } | Self::Page { .. }
        )
    }
}
struct Worker<S: RealScalar> {
    x: Vec<S>,
    y: Vec<S>,
}

/// Workspace for [`classical_tests_into`].
///
/// It holds per-worker copies of one batch and the results of all batches. The
/// rank and exact tests allocate their own scratch on each call.
pub struct ClassicalBatchWorkspace<S: RealScalar> {
    workers: Vec<Worker<S>>,
    results: Vec<Option<ClassicalTestResult<S>>>,
}
impl<S: RealScalar> ClassicalBatchWorkspace<S> {
    /// Allocates a workspace for the shapes of `x` and `y` and for the number of
    /// workers that the [`StatisticsOptions`] and `context` select.
    ///
    /// Pass the same statistics options and context to [`classical_tests_into`].
    /// [`StatisticsOptions::max_bytes`] bounds the size of this workspace,
    /// excluding the scratch the tests allocate themselves.
    ///
    /// # Errors
    ///
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
        let workers = crate::estimation::worker_count(admission, context, x.batches())?.max(1);
        let nx = x.samples().checked_mul(x.dimensions()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        let ny = y.map_or(Ok(0), |v| {
            v.samples().checked_mul(v.dimensions()).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
        })?;
        let scalar = nx.checked_add(ny).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        let bytes = crate::estimation::storage_bytes(&[
            (workers, core::mem::size_of::<Worker<S>>()),
            (
                scalar.checked_mul(workers).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                })?,
                core::mem::size_of::<S>(),
            ),
            (
                x.batches(),
                core::mem::size_of::<Option<ClassicalTestResult<S>>>(),
            ),
        ])?;
        crate::error_helpers::limit(
            bytes,
            admission.max_bytes(),
            scioxide_core::quantity!("batch_bytes"),
        )?;
        let mut slots = scioxide_core::backend::try_with_capacity(
            workers,
            scioxide_core::quantity!("classical_workers"),
        )?;
        for _ in 0..workers {
            slots.push(Worker {
                x: try_filled(
                    nx,
                    S::ZERO,
                    scioxide_core::quantity!("classical_first_sample"),
                )?,
                y: try_filled(
                    ny,
                    S::ZERO,
                    scioxide_core::quantity!("classical_second_sample"),
                )?,
            });
        }
        Ok(Self {
            workers: slots,
            results: try_filled(
                x.batches(),
                None,
                scioxide_core::quantity!("classical_result_staging"),
            )?,
        })
    }
}

/// Applies one classical test to every batch of `x` (and `y`), writing one
/// result per batch to `output`.
///
/// Missing and non-finite observations follow the policies of the
/// [`StatisticsOptions`]; see
/// [Data, weights and missing values](crate#data-weights-and-missing-values).
/// `options` configures the test itself.
/// Paired tests drop a whole row when either member is left out, and Page's
/// test drops a whole block. Independent samples drop observations separately.
/// [`ClassicalTestResult::omitted`] reports the counts. Batches may run in
/// parallel as the statistics options and `context` allow. Weighted views are not
/// supported. Each test has the cost of its scalar function.
///
/// # Errors
///
/// - `InvalidInput` if `y` is given for a one-sample test or missing for a
///   two-sample test, either view has weights, the views have different batch
///   counts or incompatible shapes, or `output.len()` differs from the batch
///   count.
/// - `InvalidInput` if `workspace` was sized for smaller views.
/// - `InvalidInput` or `NonFinite` if a batch has a missing or non-finite
///   observation under a `Reject` policy.
/// - Any error of the selected test on any batch.
#[allow(clippy::too_many_arguments)]
pub fn classical_tests_into<S: RealScalar>(
    test: ClassicalBatchTest<'_, S>,
    x: crate::BatchView<'_, S>,
    y: Option<crate::BatchView<'_, S>>,
    options: ClassicalOptions,
    admission: StatisticsOptions,
    output: &mut [Option<ClassicalTestResult<S>>],
    workspace: &mut ClassicalBatchWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let x = ObservationData::from_batch(x);
    let y = y.map(ObservationData::from_batch);
    let is_page = matches!(test, ClassicalBatchTest::Page { .. });
    if x.weights.is_some()
        || y.is_some_and(|v| v.weights.is_some())
        || test.two_samples() != y.is_some()
        || (!is_page && x.dimensions() != 1)
        || y.is_some_and(|v| v.dimensions() != 1 || v.batches() != x.batches())
        || (!test.independent() && y.is_some_and(|v| v.samples() != x.samples()))
        || output.len() != x.batches()
    {
        return Err(invalid(
            scioxide_core::argument!("y"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "classical batch shape/weight mismatch",
        ));
    }
    let size = x
        .samples()
        .checked_mul(x.dimensions())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if workspace.results.len() < x.batches()
        || workspace
            .workers
            .iter()
            .any(|w| w.x.len() < size || w.y.len() < y.map_or(0, |v| v.samples()))
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "classical batch workspace is too small",
        ));
    }
    crate::estimation::execute(
        admission,
        context,
        x.batches(),
        1,
        &mut workspace.workers,
        &mut workspace.results,
        |batch, work, staged| {
            let (mut nx, mut ny) = (0, 0);
            for i in 0..x.samples() {
                if !test.independent()
                    && (!x.present(batch, i) || y.is_some_and(|v| !v.present(batch, i)))
                {
                    if admission.missing_policy() == crate::MissingPolicy::Skip {
                        continue;
                    }
                    return Err(invalid(
                        scioxide_core::argument!("admission"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "missing member of a classical observation row",
                    ));
                }
                let ax = x
                    .admit(batch, i, admission, scioxide_core::argument!("x"))?
                    .is_some();
                let ay = if !test.independent() {
                    y.map_or(Ok(true), |v| {
                        v.admit(batch, i, admission, scioxide_core::argument!("y"))
                            .map(|v| v.is_some())
                    })?
                } else {
                    true
                };
                if ax && ay {
                    for j in 0..x.dimensions() {
                        work.x[nx * x.dimensions() + j] = x.value(batch, i, j);
                    }
                    nx += 1;
                    if !test.independent()
                        && let Some(y) = y
                    {
                        work.y[ny] = y.value(batch, i, 0);
                        ny += 1;
                    }
                }
            }
            if test.independent()
                && let Some(y) = y
            {
                for i in 0..y.samples() {
                    if y.admit(batch, i, admission, scioxide_core::argument!("y"))?
                        .is_some()
                    {
                        work.y[ny] = y.value(batch, i, 0);
                        ny += 1;
                    }
                }
            }
            let a = &work.x[..nx * x.dimensions()];
            let b = &work.y[..ny];
            let mut result = match test {
                ClassicalBatchTest::OneSampleT(null) => {
                    t_test_one_sample_with_options(a, null, options)
                }
                ClassicalBatchTest::PairedT => t_test_paired_with_options(a, b, options),
                ClassicalBatchTest::IndependentT(variance) => {
                    t_test_independent_with_options(a, b, variance, options)
                }
                ClassicalBatchTest::Pearson => pearson_test_with_options(a, b, options),
                ClassicalBatchTest::PointBiserial => {
                    point_biserial_test_with_options(a, b, options)
                }
                ClassicalBatchTest::Spearman => spearman_test_with_options(a, b, options),
                ClassicalBatchTest::Kendall(variant) => {
                    kendall_test_with_options(a, b, variant, options)
                }
                ClassicalBatchTest::Chatterjee => chatterjee_test_with_options(a, b, options),
                ClassicalBatchTest::Somers => somers_test_with_options(a, b, options),
                ClassicalBatchTest::MannWhitney => mann_whitney_test_with_options(a, b, options),
                ClassicalBatchTest::RankSum => rank_sum_test_with_options(a, b, options),
                ClassicalBatchTest::BrunnerMunzel(calibration) => {
                    brunner_munzel_test_with_options(a, b, calibration, options)
                }
                ClassicalBatchTest::Bws => bws_test_with_options(a, b, options),
                ClassicalBatchTest::Wilcoxon(zeros) => {
                    wilcoxon_test_with_options(a, zeros, options)
                }
                ClassicalBatchTest::Quantile { value, probability } => {
                    quantile_test_with_options(a, value, probability, options)
                }
                ClassicalBatchTest::Page { treatment_order } => page_trend_test_with_options(
                    scioxide_linalg::MatrixView::from_slice(nx, x.dimensions(), a)?,
                    treatment_order,
                    options,
                ),
            }?;
            result.omitted = [x.samples() - nx, y.map_or(0, |v| v.samples() - ny)];
            result.missing_policy = admission.missing_policy();
            result.non_finite_policy = admission.non_finite_policy();
            staged[0] = Some(result);
            Ok(())
        },
    )?;
    output.copy_from_slice(&workspace.results[..output.len()]);
    Ok(())
}
