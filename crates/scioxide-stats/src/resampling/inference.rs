//! Confidence intervals, jackknife estimates and p-values from the replicate
//! statistics.
use super::*;
use crate::error_helpers::finite;
use core::cmp::Ordering;
use scioxide_core::Scaled;

fn rounded<S: RealScalar>(x: Scaled<f64>) -> Result<S, SciOxideError> {
    let y = x.round_to()?;
    if !x.is_zero() && y == S::ZERO {
        return Err(breakdown("inference diagnostic underflow"));
    }
    finite(y)
}
/// Returns the mean of `values` and the sum of squared deviations from it.
fn centered_summary<S: RealScalar>(
    values: &[S],
) -> Result<(Scaled<f64>, Scaled<f64>), SciOxideError> {
    let (center, population) = crate::arithmetic::scaled_second(
        values.len(),
        |i| {
            let x = values[i].into_f64();
            crate::arithmetic::native(x, 64).then_some(x)
        },
        |i| Scaled::<f64>::from_value(values[i].into_f64()),
    )?;
    Ok((
        center.total()?,
        population.mul(Scaled::from_value(values.len() as f64)?)?,
    ))
}
fn jackknife<S: RealScalar>(
    plan: &ResamplingPlan<S>,
    observed: S,
    values: &[S],
) -> Result<(JackknifeEstimate<S>, Option<f64>), SciOxideError> {
    let observed = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(observed))?;
    let mut bias = Scaled::<f64>::ZERO;
    let mut variance = Scaled::<f64>::ZERO;
    let mut second = Scaled::<f64>::ZERO;
    let mut third = Scaled::<f64>::ZERO;
    let groups = if plan.paired() { 1 } else { plan.groups() };
    for pair in plan.offsets.windows(2).take(groups) {
        let sample = &values[pair[0]..pair[1]];
        let n = sample.len();
        let (center, squares) = centered_summary(sample)?;
        let nm1 = Scaled::<f64>::from_value((n - 1) as f64)?;
        let factor = nm1.div(Scaled::<f64>::from_value((n) as f64)?)?;
        bias = bias.add(center.sub(observed)?.mul(nm1)?)?;
        variance = variance.add(squares.mul(factor)?)?;
        for &value in sample {
            // Stratified/multisample influence: (n_j-1)(mean_j - theta_ji)/n_j.
            let influence = center
                .sub(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(value),
                )?)?
                .mul(factor)?;
            let square = influence.square()?;
            second = second.add(square)?;
            third = third.add(square.mul(influence)?)?;
        }
    }
    let acceleration = if second.is_zero() {
        None
    } else {
        Some(
            third
                .div(
                    second
                        .mul(second.sqrt()?)?
                        .mul(Scaled::<f64>::from_value(6.0)?)?,
                )?
                .round_to()?,
        )
    };
    Ok((
        JackknifeEstimate {
            bias: rounded(bias)?,
            bias_corrected: rounded(observed.sub(bias)?)?,
            standard_error: rounded(variance.sqrt()?)?,
        },
        acceleration,
    ))
}
fn normal_quantile(p: f64) -> Result<f64, SciOxideError> {
    if !(0.0..1.0).contains(&p) || p == 0.0 {
        return Err(breakdown("BCa probability saturated"));
    }
    let tail = if p <= 0.5 { p } else { 1.0 - p };
    let z = core::f64::consts::SQRT_2 * scioxide_special::erfcinv(2.0 * tail)?;
    finite(if p <= 0.5 { -z } else { z })
}
fn adjusted(p: f64, z0: f64, a: f64) -> Result<f64, SciOxideError> {
    let z = normal_quantile(p)?;
    let denominator = 1.0 - a * (z0 + z);
    if denominator <= 0.0 {
        return Err(breakdown("singular BCa probability adjustment"));
    }
    let score = z0 + (z0 + z) / denominator;
    let p = 0.5 * scioxide_special::erfc(-score / core::f64::consts::SQRT_2)?;
    if p <= 0.0 || p >= 1.0 || !p.is_finite() {
        return Err(breakdown("adjusted BCa probability saturated"));
    }
    Ok(p)
}
fn bootstrap_interval<S: RealScalar>(
    plan: &ResamplingPlan<S>,
    observed: S,
    values: &[S],
    deletes: &[S],
    sorted: &mut [S],
) -> Result<BootstrapInterval<S>, SciOxideError> {
    let ResamplingMethod::Bootstrap {
        method, confidence, ..
    } = plan.method
    else {
        return Err(invalid(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected bootstrap method",
        ));
    };
    let (center, squares) = centered_summary(values)?;
    let error = squares
        .div(Scaled::<f64>::from_value((values.len() - 1) as f64)?)?
        .sqrt()?;
    let degenerate = values.iter().all(|&v| v == values[0]);
    let alpha = (1.0 - confidence.get()) / 2.0;
    let (lower, upper, bias_correction, acceleration) = if method == BootstrapMethod::Bca {
        let (_, acceleration) = jackknife(plan, observed, deletes)?;
        let a = acceleration.ok_or_else(|| {
            breakdown("BCa acceleration undefined for constant jackknife statistics")
        })?;
        let less = values.iter().filter(|&&v| v < observed).count();
        let equal = values.iter().filter(|&&v| v == observed).count();
        let p = (less as f64 + 0.5 * equal as f64) / values.len() as f64;
        let z0 = normal_quantile(p)?;
        (
            adjusted(alpha, z0, a)?,
            adjusted(1.0 - alpha, z0, a)?,
            Some(z0),
            Some(a),
        )
    } else {
        (alpha, 1.0 - alpha, None, None)
    };
    if lower > upper {
        return Err(breakdown("BCa quantile order reversed"));
    }
    sorted.copy_from_slice(values);
    crate::ranking::sort_finite(sorted);
    let quantile = |p| {
        crate::empirical::quantile_sorted_by(
            sorted.len(),
            crate::QuantileProbability::new(p)?,
            crate::QuantileMethod::Linear,
            |i| sorted[i],
        )
        .map(|q| q.value())
    };
    let lo = quantile(lower)?;
    let hi = quantile(upper)?;
    let (lo, hi) = if method == BootstrapMethod::Basic {
        let twice = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(observed))?
            .mul(Scaled::<f64>::from_value(2.0)?)?;
        (
            rounded(twice.sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(hi),
            )?)?)?,
            rounded(twice.sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(lo),
            )?)?)?,
        )
    } else {
        (lo, hi)
    };
    Ok(BootstrapInterval {
        bounds: Interval::new(lo, hi)?,
        standard_error: rounded(error)?,
        mean_monte_carlo_error: rounded(
            error.div(Scaled::<f64>::from_value((values.len()) as f64)?.sqrt()?)?,
        )?,
        bias: rounded(center.sub(Scaled::<f64>::from_value(
            scioxide_core::RealScalar::into_f64(observed),
        )?)?)?,
        degenerate,
        bias_correction,
        acceleration,
    })
}
fn wilson(count: usize, total: usize) -> Result<Interval<f64>, SciOxideError> {
    let n = total as f64;
    let p = count as f64 / n;
    let z = ConfidenceProbability::default().normal_critical_value::<f64>()?;
    let z2 = z * z;
    let center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let radius = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
    Interval::new((center - radius).max(0.0), (center + radius).min(1.0))
}
fn test<S: RealScalar>(
    plan: &ResamplingPlan<S>,
    observed: S,
    values: &[S],
) -> Result<ResamplingTest, SciOxideError> {
    let (alternative, exact) = match plan.method {
        ResamplingMethod::Permutation {
            alternative, mode, ..
        } => (alternative, mode == PermutationMode::Exact),
        ResamplingMethod::MonteCarlo { alternative } => (alternative, false),
        _ => {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected test method",
            ));
        }
    };
    let scale = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(observed))?.abs();
    let threshold = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
        plan.tolerance.absolute(),
    ))?
    .add(scale.mul(Scaled::<f64>::from_value(
        scioxide_core::RealScalar::into_f64(plan.tolerance.relative()),
    )?)?)?;
    let (mut less_equal, mut greater_equal) = (0, 0);
    for &value in values {
        let tied = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(value))?
            .sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(observed),
            )?)?
            .abs()
            .abs_cmp(threshold)
            != Ordering::Greater;
        less_equal += usize::from(value <= observed || tied);
        greater_equal += usize::from(value >= observed || tied);
    }
    let correction = usize::from(!exact);
    let denominator = values.len() as f64 + correction as f64;
    let lower = (less_equal as f64 + correction as f64) / denominator;
    let upper = (greater_equal as f64 + correction as f64) / denominator;
    let p_value = match alternative {
        TestAlternative::Less => lower,
        TestAlternative::Greater => upper,
        TestAlternative::TwoSided => (2.0 * lower.min(upper)).min(1.0),
    };
    Ok(ResamplingTest {
        p_value,
        less_equal,
        greater_equal,
        exact,
        resolution: 1.0 / denominator,
        tail_uncertainty: if exact {
            None
        } else {
            Some([
                wilson(less_equal, values.len())?,
                wilson(greater_equal, values.len())?,
            ])
        },
    })
}
pub(super) fn infer<S: RealScalar>(
    plan: &ResamplingPlan<S>,
    observed: S,
    values: &[S],
    deletes: &[S],
    sorted: &mut [S],
) -> Result<ResamplingInference<S>, SciOxideError> {
    match plan.method {
        ResamplingMethod::Bootstrap { .. } => {
            bootstrap_interval(plan, observed, values, deletes, sorted)
                .map(ResamplingInference::Bootstrap)
        }
        ResamplingMethod::Jackknife { .. } => jackknife(plan, observed, values)
            .map(|(estimate, _)| ResamplingInference::Jackknife(estimate)),
        _ => test(plan, observed, values).map(ResamplingInference::Test),
    }
}
