//! Combination of independent p-values and multiple-testing adjustment.
use super::invalid;
use alloc::vec::Vec;

use crate::StatisticsOptions;
#[cfg(not(feature = "std"))]
#[cfg_attr(test, allow(unused_imports))]
use scioxide_core::RealScalar;
use scioxide_core::{
    ExecutionContext, SciOxideError,
    backend::{try_filled, try_with_capacity},
};
use scioxide_random::{Cdf, ChiSquared, Normal, Quantile, StudentT, Survival};

/// Selects how [`combine_pvalues`] combines `k` independent p-values `pᵢ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PValueCombination {
    /// Fisher's method: `X = −2 Σ ln pᵢ`, chi-square with `2k` degrees of
    /// freedom, upper tail.
    Fisher,
    /// Pearson's method: `X = 2 Σ ln(1 − pᵢ)`. The p-value is
    /// `P(χ²₂ₖ ≤ −X)`, small when the `pᵢ` are close to 0.
    Pearson,
    /// Tippett's method: `X = min pᵢ`, with p-value `1 − (1 − X)ᵏ`.
    Tippett,
    /// Stouffer's method: `Z = Σ wᵢ Φ⁻¹(1 − pᵢ) / √(Σ wᵢ²)`, standard normal,
    /// upper tail. The only method that accepts weights.
    Stouffer,
    /// Mudholkar and George's logit method: `L = Σ ln((1 − pᵢ)/pᵢ)`, with
    /// `L √(3(5k + 4)/(k π² (5k + 2)))` compared with Student's t on `5k + 4`
    /// degrees of freedom, upper tail.
    MudholkarGeorge,
}

/// Result of [`combine_pvalues`]: the combined statistic and its p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombinedPValue {
    /// Statistic of the selected method. Input p-values of 0 or 1 can make it
    /// infinite.
    statistic: f64,
    /// Combined p-value, in `[0, 1]`.
    p_value: f64,
}
impl CombinedPValue {
    /// Returns the statistic.
    pub const fn statistic(&self) -> f64 {
        self.statistic
    }
    /// Returns the p value.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
}

fn validate(pvalues: &[f64]) -> Result<(), SciOxideError> {
    if pvalues.is_empty()
        || pvalues
            .iter()
            .any(|&p| !p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        Err(invalid(
            scioxide_core::argument!("pvalues"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "p-values must be nonempty and lie in [0,1]",
        ))
    } else {
        Ok(())
    }
}

/// Combines independent p-values of tests of the same null hypothesis.
///
/// The methods are described on [`PValueCombination`]. The null distributions
/// assume independent p-values that are uniform on `[0, 1]` under the null, as
/// from continuous test statistics. For discrete or dependent p-values the
/// result is not calibrated. `weights` applies only to Stouffer's method. The
/// cost is O(k) time and O(1) memory. Stouffer scores and squared weights
/// use compensated extended-range sums; every positive weight retains the
/// infinite score of an endpoint p-value. Zero weights exclude those scores.
///
/// # Errors
///
/// - `InvalidInput` if `pvalues` is empty or a p-value is not in `[0, 1]`.
/// - `InvalidInput` if `weights` is given for a method other than Stouffer's,
///   has the wrong length, has a negative or non-finite entry, or is all zero.
/// - `InvalidInput` for Stouffer's and the logit method if one p-value is 0
///   and another is 1.
/// - `NumericalInstability` if the combined p-value is not finite or lies
///   outside `[0, 1]`.
/// - Errors from evaluating the null distribution.
pub fn combine_pvalues(
    pvalues: &[f64],
    method: PValueCombination,
    weights: Option<&[f64]>,
) -> Result<CombinedPValue, SciOxideError> {
    validate(pvalues)?;
    if method != PValueCombination::Stouffer && weights.is_some() {
        return Err(invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "weights apply only to Stouffer combination",
        ));
    }
    if weights.is_some_and(|w| {
        w.len() != pvalues.len()
            || w.iter().any(|&value| !value.is_finite() || value < 0.0)
            || w.iter().all(|&value| value == 0.0)
    }) {
        return Err(invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid Stouffer weights",
        ));
    }
    let count = pvalues.len();
    let df = count
        .checked_mul(2)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let result = match method {
        PValueCombination::Fisher => {
            let statistic = -2.0 * pvalues.iter().map(|&p| p.ln()).sum::<f64>();
            let p = if statistic.is_infinite() {
                0.0
            } else {
                ChiSquared::<f64>::new(df as f64)?.sf(statistic)?
            };
            CombinedPValue {
                statistic,
                p_value: p,
            }
        }
        PValueCombination::Pearson => {
            let statistic = 2.0 * pvalues.iter().map(|&p| (-p).ln_1p()).sum::<f64>();
            let p = if statistic.is_infinite() {
                1.0
            } else {
                ChiSquared::<f64>::new(df as f64)?.cdf(-statistic)?
            };
            CombinedPValue {
                statistic,
                p_value: p,
            }
        }
        PValueCombination::Tippett => {
            let statistic = pvalues.iter().copied().fold(1.0_f64, f64::min);
            CombinedPValue {
                statistic,
                p_value: -((count as f64) * (-statistic).ln_1p()).exp_m1(),
            }
        }
        PValueCombination::Stouffer => {
            let normal = Normal::<f64>::new(0.0, 1.0)?;
            let mut sum = scioxide_core::ScaledAccumulator::ZERO;
            let mut square = scioxide_core::ScaledAccumulator::ZERO;
            let mut positive_infinite = false;
            let mut negative_infinite = false;
            for (index, &p) in pvalues.iter().enumerate() {
                let weight = weights.map_or(1.0, |w| w[index]);
                if weight == 0.0 {
                    continue;
                }
                let weight = scioxide_core::Scaled::from_value(weight)?;
                square.add_scaled(weight.square()?)?;
                if p == 0.0 {
                    positive_infinite = true;
                } else if p == 1.0 {
                    negative_infinite = true;
                } else {
                    sum.add_product(
                        weight,
                        scioxide_core::Scaled::from_value(-normal.quantile(p)?)?,
                    )?;
                }
            }
            if positive_infinite && negative_infinite {
                return Err(invalid(
                    scioxide_core::argument!("pvalues"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "opposed infinite normal scores",
                ));
            }
            let statistic = if positive_infinite {
                f64::INFINITY
            } else if negative_infinite {
                f64::NEG_INFINITY
            } else {
                sum.total()?.div(square.total()?.sqrt()?)?.round()?
            };
            let p = if statistic == f64::INFINITY {
                0.0
            } else if statistic == f64::NEG_INFINITY {
                1.0
            } else {
                normal.sf(statistic)?
            };
            CombinedPValue {
                statistic,
                p_value: p,
            }
        }
        PValueCombination::MudholkarGeorge => {
            if pvalues.contains(&0.0) && pvalues.contains(&1.0) {
                return Err(invalid(
                    scioxide_core::argument!("pvalues"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "opposed infinite logistic scores",
                ));
            }
            let statistic = pvalues
                .iter()
                .map(|&p| (-p.ln()) + (-p).ln_1p())
                .sum::<f64>();
            let nu = count
                .checked_mul(5)
                .and_then(|n| n.checked_add(4))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                })? as f64;
            let scale =
                (3.0 / count as f64).sqrt() / core::f64::consts::PI * (nu / (nu - 2.0)).sqrt();
            let p = if statistic == f64::INFINITY {
                0.0
            } else if statistic == f64::NEG_INFINITY {
                1.0
            } else {
                StudentT::<f64>::new(nu)?.sf(statistic * scale)?
            };
            CombinedPValue {
                statistic,
                p_value: p,
            }
        }
    };
    if !result.p_value.is_finite() || !(0.0..=1.0).contains(&result.p_value) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "combined p-value outside [0,1]",
        ));
    }
    Ok(result)
}

/// Selects a p-value adjustment that controls the familywise error rate.
///
/// `m` is the number of p-values and `p₍₁₎ ≤ … ≤ p₍ₘ₎` the sorted p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilywiseMethod {
    /// Bonferroni: `p̃ᵢ = min(1, m pᵢ)`. Valid under any dependence.
    Bonferroni,
    /// Holm's step-down method:
    /// `p̃₍ᵢ₎ = min(1, maxⱼ≤ᵢ (m − j + 1) p₍ⱼ₎)`. Valid under any dependence
    /// and never larger than Bonferroni.
    Holm,
}

/// Selects a p-value adjustment that controls the false discovery rate.
///
/// `m` is the number of p-values and `p₍₁₎ ≤ … ≤ p₍ₘ₎` the sorted p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FalseDiscoveryMethod {
    /// Benjamini–Hochberg: `p̃₍ᵢ₎ = min(1, minⱼ≥ᵢ m p₍ⱼ₎ / j)`. Valid for
    /// independent or positively regression dependent tests.
    BenjaminiHochberg,
    /// Benjamini–Yekutieli: the Benjamini–Hochberg values multiplied by
    /// `Σₖ₌₁ᵐ 1/k`. Valid under any dependence.
    BenjaminiYekutieli,
}

fn order(pvalues: &[f64], indices: &mut [usize]) {
    for (index, slot) in indices.iter_mut().enumerate() {
        *slot = index;
    }
    indices.sort_unstable_by(|&a, &b| pvalues[a].total_cmp(&pvalues[b]).then(a.cmp(&b)));
}

/// Writes Bonferroni or Holm adjusted p-values of one lane, in input order.
fn adjust_familywise_lane(
    pvalues: &[f64],
    method: FamilywiseMethod,
    output: &mut [f64],
    indices: &mut [usize],
) -> Result<(), SciOxideError> {
    validate(pvalues)?;
    if output.len() != pvalues.len() || indices.len() != pvalues.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "p-value adjustment shape mismatch",
        ));
    }
    if method == FamilywiseMethod::Bonferroni {
        for (slot, &p) in output.iter_mut().zip(pvalues) {
            *slot = (p * pvalues.len() as f64).min(1.0);
        }
    } else {
        order(pvalues, indices);
        let mut running = 0.0_f64;
        for (rank, &index) in indices.iter().enumerate() {
            running = running.max((pvalues.len() - rank) as f64 * pvalues[index]);
            output[index] = running.min(1.0);
        }
    }
    Ok(())
}

/// Writes Benjamini–Hochberg or Benjamini–Yekutieli adjusted p-values of one
/// lane, in input order.
fn adjust_discovery_lane(
    pvalues: &[f64],
    method: FalseDiscoveryMethod,
    output: &mut [f64],
    indices: &mut [usize],
) -> Result<(), SciOxideError> {
    validate(pvalues)?;
    if output.len() != pvalues.len() || indices.len() != pvalues.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "p-value adjustment shape mismatch",
        ));
    }
    order(pvalues, indices);
    let multiplier = match method {
        FalseDiscoveryMethod::BenjaminiHochberg => 1.0,
        FalseDiscoveryMethod::BenjaminiYekutieli => {
            (1..=pvalues.len()).map(|i| 1.0 / i as f64).sum()
        }
    };
    let mut running = 1.0_f64;
    for rank in (0..pvalues.len()).rev() {
        let index = indices[rank];
        running =
            running.min((pvalues.len() as f64 * multiplier / (rank + 1) as f64) * pvalues[index]);
        output[index] = running.min(1.0);
    }
    Ok(())
}

/// Adjusts p-values for multiple testing and returns them in input order.
///
/// Adjusts `pvalues` as a single family using `method`. This is the allocating
/// form of [`adjust_pvalues_into`]. It runs sequentially, and its scratch is
/// bounded by [`StatisticsOptions::max_bytes`]. The cost is O(n log n) time
/// for `n` p-values.
///
/// # Errors
///
/// - `InvalidInput` if `pvalues` is empty or a p-value is not in `[0, 1]`.
/// - `ResourceLimit` if the scratch exceeds `max_bytes`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{FamilywiseMethod, PValueAdjustment, StatisticsOptions, adjust_pvalues_with_options};
///
/// // Bonferroni multiplies each p-value by the number of tests, here 2.
/// let method = PValueAdjustment::Familywise(FamilywiseMethod::Bonferroni);
/// let adjusted = adjust_pvalues_with_options(&[0.01, 0.04], method, StatisticsOptions::new())?;
/// assert!((adjusted[0] - 0.02).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. Holm, "A simple sequentially rejective multiple test procedure", *Scand.
/// J. Statist.* 6 (1979), 65–70.
///
/// Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: a
/// practical and powerful approach to multiple testing", *J. Roy. Statist.
/// Soc. Ser. B* 57 (1995), 289–300.
///
/// Y. Benjamini and D. Yekutieli, "The control of the false discovery rate in
/// multiple testing under dependency", *Ann. Statist.* 29 (2001), 1165–1188.
pub fn adjust_pvalues_with_options(
    pvalues: &[f64],
    method: PValueAdjustment,
    options: StatisticsOptions,
) -> Result<Vec<f64>, SciOxideError> {
    let width = pvalues.len();
    if width == 0 {
        return Err(invalid(
            scioxide_core::argument!("pvalues"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "p-values must be nonempty and lie in [0,1]",
        ));
    }
    let context = ExecutionContext::sequential_native();
    let mut workspace = PValueAdjustmentWorkspace::new(1, width, options, context)?;
    let mut output = try_filled(
        pvalues.len(),
        0.0,
        scioxide_core::quantity!("adjusted_p_values"),
    )?;
    adjust_pvalues_into(
        pvalues,
        width,
        method,
        options,
        &mut output,
        &mut workspace,
        context,
    )?;
    Ok(output)
}

/// Selects a multiple-testing adjustment of p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PValueAdjustment {
    /// Control of the familywise error rate.
    Familywise(FamilywiseMethod),
    /// Control of the false discovery rate.
    FalseDiscovery(FalseDiscoveryMethod),
}

/// Workspace for [`adjust_pvalues_into`].
pub struct PValueAdjustmentWorkspace {
    workers: Vec<Vec<usize>>,
    staged: Vec<f64>,
    width: usize,
}

impl PValueAdjustmentWorkspace {
    /// Allocates a workspace for `lanes` lanes of `width` p-values and for the
    /// number of workers that `options` and `context` select.
    ///
    /// Use the same `options` and `context` in [`adjust_pvalues_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `lanes` or `width` is zero.
    /// - `ResourceLimit` if the workspace exceeds
    ///   [`StatisticsOptions::max_bytes`].
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn new(
        lanes: usize,
        width: usize,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if lanes == 0 || width == 0 {
            return Err(invalid(
                scioxide_core::argument!("width"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid multiple-testing batch shape",
            ));
        }
        let workers = crate::estimation::worker_count(options, context, lanes)?.max(1);
        let total = lanes.checked_mul(width).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        let bytes = total
            .checked_mul(core::mem::size_of::<f64>())
            .and_then(|v| {
                v.checked_add(
                    workers
                        .checked_mul(width)?
                        .checked_mul(core::mem::size_of::<usize>())?,
                )
            })
            .and_then(|v| v.checked_add(workers.checked_mul(core::mem::size_of::<Vec<usize>>())?))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        crate::error_helpers::limit(
            bytes,
            options.max_bytes(),
            scioxide_core::quantity!("multiple_bytes"),
        )?;
        let mut slots = try_with_capacity(
            workers,
            scioxide_core::quantity!("multiple_testing_workers"),
        )?;
        for _ in 0..workers {
            slots.push(try_filled(
                width,
                0,
                scioxide_core::quantity!("multiple_testing_sort_indices"),
            )?);
        }
        Ok(Self {
            workers: slots,
            staged: try_filled(
                total,
                0.0,
                scioxide_core::quantity!("multiple_testing_staged_results"),
            )?,
            width,
        })
    }
}

/// Adjusts p-values for multiple testing, writing them to `output` in input
/// order.
///
/// `pvalues` and `output` hold lanes of `width` values each, one lane after
/// another, and each lane is adjusted separately by `method`. Lanes may run in
/// parallel as `options` and `context` allow. Missing and non-finite p-values
/// are not accepted. See [`adjust_pvalues_with_options`] for the allocating form.
///
/// # Errors
///
/// - `InvalidInput` if `pvalues` is empty, `width` is zero or does not divide
///   `pvalues.len()`, or `output.len() != pvalues.len()`.
/// - `InvalidInput` if `workspace` was created for a different width or fewer
///   values.
/// - `InvalidInput` if a p-value is not in `[0, 1]`.
/// - An error if execution is cancelled or the executor fails.
pub fn adjust_pvalues_into(
    pvalues: &[f64],
    width: usize,
    method: PValueAdjustment,
    options: StatisticsOptions,
    output: &mut [f64],
    workspace: &mut PValueAdjustmentWorkspace,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if width == 0
        || pvalues.is_empty()
        || !pvalues.len().is_multiple_of(width)
        || output.len() != pvalues.len()
        || workspace.width != width
        || workspace.staged.len() < pvalues.len()
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "multiple-testing batch shape mismatch",
        ));
    }
    let lanes = pvalues.len() / width;
    crate::estimation::execute(
        options,
        context,
        lanes,
        width,
        &mut workspace.workers,
        &mut workspace.staged,
        |lane, indices, staged| {
            let input = &pvalues[lane * width..(lane + 1) * width];
            match method {
                PValueAdjustment::Familywise(name) => {
                    adjust_familywise_lane(input, name, staged, indices)
                }
                PValueAdjustment::FalseDiscovery(name) => {
                    adjust_discovery_lane(input, name, staged, indices)
                }
            }
        },
    )?;
    output.copy_from_slice(&workspace.staged[..pvalues.len()]);
    Ok(())
}

/// Computes the result of [`adjust_pvalues_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the configured form's errors for these defaults.
pub fn adjust_pvalues(
    pvalues: &[f64],
    method: PValueAdjustment,
) -> Result<Vec<f64>, SciOxideError> {
    adjust_pvalues_with_options(pvalues, method, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &[
                "options",
                "weights",
                "left_weights",
                "right_weights",
                "direction",
                "ties",
            ],
            scioxide_core::argument!("pvalues"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use core::num::NonZeroUsize;
    #[cfg(feature = "std")]
    use scioxide_core::ExecutionPolicy;

    fn adjust(p: &[f64], method: PValueAdjustment) -> Vec<f64> {
        adjust_pvalues_with_options(p, method, StatisticsOptions::new()).unwrap()
    }

    #[test]
    fn combinations_match_simple_analytic_cases() {
        let p = [0.1, 0.2];
        let fisher = combine_pvalues(&p, PValueCombination::Fisher, None).unwrap();
        assert!((fisher.statistic + 2.0 * 0.02_f64.ln()).abs() < 1e-13);
        let tippett = combine_pvalues(&p, PValueCombination::Tippett, None).unwrap();
        assert!((tippett.p_value - 0.19).abs() < 1e-14);
        let stouffer =
            combine_pvalues(&[0.5, 0.5], PValueCombination::Stouffer, Some(&[1.0, 2.0])).unwrap();
        assert!((stouffer.p_value - 0.5).abs() < 1e-12);
        let wide_weights = combine_pvalues(
            &[0.5, 0.5],
            PValueCombination::Stouffer,
            Some(&[1e300, 2e300]),
        )
        .unwrap();
        assert!((wide_weights.p_value - 0.5).abs() < 1e-12);
        assert!(combine_pvalues(&p, PValueCombination::Pearson, None).is_ok());
        assert!(combine_pvalues(&p, PValueCombination::MudholkarGeorge, None).is_ok());
    }

    #[test]
    fn multiplicity_preserves_order_and_rejects_atomically() {
        let p = [0.04, 0.01, 0.03, 0.2];
        let bh = adjust(
            &p,
            PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg),
        );
        assert!((bh[0] - 0.05333333333333334).abs() < 1e-14);
        assert!((bh[1] - 0.04).abs() < 1e-14);
        let holm = adjust(&p, PValueAdjustment::Familywise(FamilywiseMethod::Holm));
        assert_eq!(holm, [0.09, 0.04, 0.09, 0.2]);
        let bonferroni = adjust(
            &p,
            PValueAdjustment::Familywise(FamilywiseMethod::Bonferroni),
        );
        let by = adjust(
            &p,
            PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiYekutieli),
        );
        assert!(bonferroni.iter().zip(&holm).all(|(&b, &h)| b >= h));
        assert!(by.iter().zip(&bh).all(|(&b, &h)| b >= h));
        let mut output = [9.0; 4];
        let bad = [0.1, f64::NAN, 0.2, 0.3];
        let context = ExecutionContext::sequential_native();
        let options = StatisticsOptions::new();
        let mut workspace = PValueAdjustmentWorkspace::new(1, bad.len(), options, context).unwrap();
        assert!(
            adjust_pvalues_into(
                &bad,
                bad.len(),
                PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiYekutieli),
                options,
                &mut output,
                &mut workspace,
                context,
            )
            .is_err()
        );
        assert_eq!(output, [9.0; 4]);
    }

    #[test]
    fn owned_adjustment_honors_the_configured_byte_limit() {
        // One sequential worker: four sort indices, one index-vector header and
        // four staged outputs.
        let p = [0.04, 0.01, 0.03, 0.2];
        let required = 4 * core::mem::size_of::<f64>()
            + 4 * core::mem::size_of::<usize>()
            + core::mem::size_of::<Vec<usize>>();
        let method = PValueAdjustment::Familywise(FamilywiseMethod::Holm);
        let exact = StatisticsOptions::new().with_max_bytes(required);
        assert_eq!(
            adjust_pvalues_with_options(&p, method, exact).unwrap(),
            [0.09, 0.04, 0.09, 0.2]
        );
        let error = adjust_pvalues_with_options(&p, method, exact.with_max_bytes(required - 1))
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { required: r, limit, .. }
                if r == required && limit == required - 1
        ));
    }

    #[test]
    fn simulated_global_null_control() {
        let mut state = 548_u64;
        let mut holm_rejections = 0;
        let mut bh_rejections = 0;
        for _ in 0..5000 {
            let mut p = [0.0; 8];
            for value in &mut p {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *value = ((state >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64);
            }
            let holm = adjust(&p, PValueAdjustment::Familywise(FamilywiseMethod::Holm));
            let bh = adjust(
                &p,
                PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg),
            );
            let bonferroni = adjust(
                &p,
                PValueAdjustment::Familywise(FamilywiseMethod::Bonferroni),
            );
            let by = adjust(
                &p,
                PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiYekutieli),
            );
            assert!(bonferroni.iter().zip(&holm).all(|(&b, &h)| b >= h));
            assert!(by.iter().zip(&bh).all(|(&b, &h)| b >= h));
            holm_rejections += usize::from(holm.iter().any(|&v| v <= 0.05));
            bh_rejections += usize::from(bh.iter().any(|&v| v <= 0.05));
        }
        assert!((150..=350).contains(&holm_rejections), "{holm_rejections}");
        assert!((150..=350).contains(&bh_rejections), "{bh_rejections}");
    }

    #[test]
    fn independent_batch_publishes_transactionally() {
        let input = [0.01, 0.04, 0.2, 0.03, 0.06, 0.1];
        let mut output = [9.0; 6];
        let options = StatisticsOptions::new().with_max_bytes(4096);
        let context = ExecutionContext::sequential();
        let mut workspace = PValueAdjustmentWorkspace::new(2, 3, options, context).unwrap();
        let method = PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg);
        adjust_pvalues_into(
            &input,
            3,
            method,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        for lane in 0..2 {
            let expected = adjust(&input[lane * 3..lane * 3 + 3], method);
            assert_eq!(&output[lane * 3..lane * 3 + 3], expected);
        }
        output.fill(9.0);
        let mut bad = input;
        bad[4] = f64::NAN;
        assert!(
            adjust_pvalues_into(
                &bad,
                3,
                PValueAdjustment::Familywise(FamilywiseMethod::Holm),
                options,
                &mut output,
                &mut workspace,
                context,
            )
            .is_err()
        );
        assert_eq!(output, [9.0; 6]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn injected_executor_matches_sequential_lanes() {
        let executor = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
        let input = [0.01, 0.04, 0.2, 0.03, 0.06, 0.1];
        let mut output = [9.0; 6];
        let options = StatisticsOptions::new()
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()))
            .with_max_bytes(4096);
        let context = ExecutionContext::new(&executor);
        let mut workspace = PValueAdjustmentWorkspace::new(2, 3, options, context).unwrap();
        let method = PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg);
        adjust_pvalues_into(
            &input,
            3,
            method,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        for lane in 0..2 {
            let expected = adjust(&input[lane * 3..lane * 3 + 3], method);
            assert_eq!(&output[lane * 3..lane * 3 + 3], expected);
        }
    }
}
