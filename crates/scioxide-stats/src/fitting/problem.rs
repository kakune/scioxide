use super::{FitModel, FitObservation, FitTruncation, finite, invalid, number};
use crate::{QuantileMethod, QuantileProbability, RawEmpiricalDistribution};
use alloc::vec::Vec;

use scioxide_core::{Bounds, RealScalar, SciOxideError};

/// Selects the objective that [`PreparedFit`] minimizes.
///
/// The matching methods use residuals `rᵢ = (model value − sample value)/sᵢ`
/// with positive scales `sᵢ`, and minimize `Σ rᵢ²`, without a factor ½.
#[derive(Clone, Copy, Debug)]
pub enum FitMethod<'a, S: RealScalar> {
    /// Maximum likelihood: minimizes `−Σ log p(oᵢ | θ)`, where `p` is the density
    /// or mass of an exact observation or the probability of a censoring
    /// interval, conditional on the truncation interval if there is one.
    Likelihood,
    /// Method of moments: matches the raw moments `E[Xᵏ]` to the sample raw
    /// moments `(1/n) Σ xᵢᵏ`.
    Moments {
        /// Moment orders `k`, each positive.
        orders: &'a [usize],
        /// Positive scale of each moment residual.
        scales: &'a [S],
    },
    /// Quantile matching: matches the quantiles `Q(pᵢ)` to the sample quantiles.
    Quantiles {
        /// Probabilities `pᵢ`, each in `(0, 1)`.
        probabilities: &'a [S],
        /// Positive scale of each quantile residual.
        scales: &'a [S],
        /// Definition of the sample quantile.
        convention: QuantileMethod,
    },
}
/// An objective function of a parameter vector, minimized by a [`FitSolver`](crate::FitSolver).
///
/// [`PreparedFit`], [`PreparedPowerFit`](crate::PreparedPowerFit),
/// [`PreparedPpcc`](crate::PreparedPpcc) and [`PreparedOdr`](crate::PreparedOdr)
/// implement it. An objective does not count its own evaluations; the solver
/// counts each call in its solve context.
pub trait FitObjective<S: RealScalar>: Sync {
    /// Scratch storage for one evaluation. Concurrent evaluations each need their own.
    type Workspace: Send;
    /// Returns the bytes that [`Self::workspace`] allocates, which callers check
    /// against their byte limit.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the size overflows `usize`.
    fn workspace_bytes(&self) -> Result<usize, SciOxideError>;
    /// Creates scratch storage of at most [`Self::workspace_bytes`] bytes.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if memory cannot be reserved.
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError>;
    /// Returns the number of parameters.
    fn dimension(&self) -> usize;
    /// Checks that `parameters` lie in the domain of the objective, including its
    /// bounds and integer constraints.
    ///
    /// # Errors
    ///
    /// An error describing the first violated constraint.
    fn validate(
        &self,
        parameters: &[S],
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;
    /// Returns the objective value at `parameters`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::validate`], and an error if the value cannot be
    /// computed or overflows.
    fn value(&self, parameters: &[S], workspace: &mut Self::Workspace) -> Result<S, SciOxideError>;
    /// Returns the number of residuals of a sum-of-squares objective, or `None`
    /// if the objective is not a sum of squares.
    fn residual_count(&self) -> Option<usize> {
        None
    }
    /// Writes the residuals `rᵢ` at `parameters`, whose squares sum to the objective.
    ///
    /// On error, `output` may be partly written. The default returns `NotImplemented`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::value`], and `InvalidInput` if `output` does not have
    /// [`Self::residual_count`] entries.
    fn residuals(
        &self,
        _parameters: &[S],
        _output: &mut [S],
        _workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented("fitting residuals"))
    }
}
/// Objective for fitting a parametric family to a sample, by maximum likelihood
/// or by matching moments or quantiles.
///
/// The sample may contain censored observations and come from a truncated
/// distribution when fitted by likelihood. Each parameter has bounds and may
/// be restricted to integers. The plan computes the sample moments or
/// quantiles once. Minimize it with [`fit`](crate::fit) or
/// [`fit_into`](crate::fit_into).
pub struct PreparedFit<'a, S: RealScalar> {
    model: &'a dyn FitModel<S>,
    observations: &'a [FitObservation<S>],
    method: FitMethod<'a, S>,
    truncation: Option<FitTruncation<S>>,
    bounds: &'a Bounds<S>,
    integer: &'a [bool],
    targets: Vec<S>,
}
impl<'a, S: RealScalar> PreparedFit<'a, S> {
    /// Checks the sample and the method and computes the sample moments or quantiles.
    ///
    /// `bounds` and `integer` have one entry per model parameter.
    /// [`StatisticsOptions::max_bytes`](crate::StatisticsOptions::max_bytes)
    /// limits the memory for the sample copy and the matching targets.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `observations` is empty, the model has no parameters,
    ///   or `bounds` or `integer` has the wrong length.
    /// - `InvalidInput` if the bounds of an integer parameter contain no integer.
    /// - `NonFinite` if an observation or truncation endpoint is NaN or infinite,
    ///   and `InvalidInput` if a censoring or truncation interval is empty.
    /// - `InvalidInput` if an observation lies outside the truncation interval.
    /// - `InvalidInput` if a moment order is zero, a quantile probability is not
    ///   in `(0, 1)`, or the scales are empty, of the wrong length or not positive.
    /// - `NotImplemented` for moment or quantile matching with censored
    ///   observations or truncation.
    /// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
    pub fn new(
        model: &'a dyn FitModel<S>,
        observations: &'a [FitObservation<S>],
        method: FitMethod<'a, S>,
        truncation: Option<FitTruncation<S>>,
        bounds: &'a Bounds<S>,
        integer: &'a [bool],
        options: crate::StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        let d = model.parameter_count();
        if d == 0 || d != bounds.len() || d != integer.len() || observations.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("observations"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fitting metadata dimensions or empty observations",
            ));
        }
        for (&is_integer, &(a, b)) in integer.iter().zip(bounds.as_slice()) {
            if is_integer && a.zip(b).is_some_and(|(a, b)| a.ceil() > b.floor()) {
                return Err(invalid(
                    scioxide_core::argument!("bounds"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "bounds contain no integer",
                ));
            }
        }
        for &o in observations {
            o.validate()?;
        }
        if let Some(t) = truncation {
            let (a, b) = t.interval()?;
            for &o in observations {
                if let FitObservation::Exact(x) = o {
                    if x <= a || x > b {
                        return Err(invalid(
                            scioxide_core::argument!("observations"),
                            scioxide_core::InvalidInputReason::OutOfRange,
                            "exact observation outside truncation",
                        ));
                    }
                } else if let Some((l, u)) = o.interval()
                    && l.max(a) >= u.min(b)
                {
                    return Err(invalid(
                        scioxide_core::argument!("observations"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "censor event outside truncation",
                    ));
                }
            }
        }
        let count = match method {
            FitMethod::Likelihood => 0,
            FitMethod::Moments { orders, scales } => {
                if orders.contains(&0) {
                    return Err(invalid(
                        scioxide_core::argument!("method"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "moment orders must be positive",
                    ));
                }
                check_scales(orders.len(), scales)?;
                orders.len()
            }
            FitMethod::Quantiles {
                probabilities,
                scales,
                ..
            } => {
                check_scales(probabilities.len(), scales)?;
                for &q in probabilities {
                    if !q.is_finite() || q <= S::ZERO || q >= S::ONE {
                        return Err(invalid(
                            scioxide_core::argument!("method"),
                            scioxide_core::InvalidInputReason::OutOfRange,
                            "quantile probabilities must be interior",
                        ));
                    }
                }
                probabilities.len()
            }
        };
        let elements = count
            .checked_add(if count > 0 {
                observations.len().checked_mul(2).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("fit_sample_storage"))
                })?
            } else {
                0
            })
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("fit_storage")))?;
        crate::error_helpers::limit(
            scioxide_core::backend::checked_mul(
                elements,
                core::mem::size_of::<S>(),
                scioxide_core::quantity!("fit_preparation_bytes"),
            )?,
            options.max_bytes(),
            scioxide_core::quantity!("fit_preparation_bytes"),
        )?;

        let mut targets = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("fit_targets"),
        )?;
        if count > 0 {
            if truncation.is_some()
                || observations
                    .iter()
                    .any(|o| !matches!(o, FitObservation::Exact(_)))
            {
                return Err(SciOxideError::not_implemented(
                    "moment/quantile matching with censoring or truncation",
                ));
            }
            let mut samples = scioxide_core::backend::try_with_capacity(
                observations.len(),
                scioxide_core::quantity!("fit_exact_samples"),
            )?;
            for o in observations {
                if let FitObservation::Exact(x) = o {
                    samples.push(*x);
                }
            }
            match method {
                FitMethod::Moments { orders, .. } => {
                    for &order in orders {
                        // Existing arbitrary-moment core provides extended-range accumulation.
                        let sum = samples.iter().try_fold(
                            crate::arithmetic::Compensated::<scioxide_core::Scaled<f64>>::zero(),
                            |sum, &x| {
                                sum.added(crate::arithmetic::integer_power(
                                    scioxide_core::Scaled::<f64>::from_value(
                                        scioxide_core::RealScalar::into_f64(x),
                                    )?,
                                    order,
                                )?)
                            },
                        )?;
                        targets.push(
                            sum.total()?
                                .div(scioxide_core::Scaled::<f64>::from_value(
                                    (samples.len()) as f64,
                                )?)?
                                .round_to()?,
                        );
                    }
                }
                FitMethod::Quantiles {
                    probabilities,
                    convention,
                    ..
                } => {
                    let empirical = RawEmpiricalDistribution::new(&samples)?;
                    for &q in probabilities {
                        targets.push(
                            empirical
                                .quantile(
                                    QuantileProbability::new(q.try_to_f64_exact()?)?,
                                    convention,
                                )?
                                .value(),
                        );
                    }
                }
                FitMethod::Likelihood => {}
            }
        }
        Ok(Self {
            model,
            observations,
            method,
            truncation,
            bounds,
            integer,
            targets,
        })
    }
    /// Returns the parameter bounds.
    pub fn bounds(&self) -> &Bounds<S> {
        self.bounds
    }
    /// Returns the flags that mark integer parameters.
    pub fn integrality(&self) -> &[bool] {
        self.integer
    }
    /// Returns the fitting method.
    pub fn method(&self) -> FitMethod<'a, S> {
        self.method
    }
    fn residual(&self, p: &[S], i: usize) -> Result<S, SciOxideError> {
        match self.method {
            FitMethod::Moments { orders, scales } => {
                finite((self.model.moment(p, orders[i])? - self.targets[i]) / scales[i])
            }
            FitMethod::Quantiles {
                probabilities,
                scales,
                ..
            } => finite((self.model.quantile(p, probabilities[i])? - self.targets[i]) / scales[i]),
            FitMethod::Likelihood => Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "likelihood has no matching residuals",
            )),
        }
    }
}
fn check_scales<S: RealScalar>(n: usize, scales: &[S]) -> Result<(), SciOxideError> {
    if n == 0 || scales.len() != n || scales.iter().any(|&x| !x.is_finite() || x <= S::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("method"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty or invalid matching scales",
        ));
    }
    Ok(())
}
pub(super) fn validate_parameters<S: RealScalar>(
    p: &[S],
    bounds: &Bounds<S>,
    integer: &[bool],
) -> Result<(), SciOxideError> {
    if p.len() != bounds.len() || integer.len() != p.len() {
        return Err(invalid(
            scioxide_core::argument!("p"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "parameter metadata shape",
        ));
    }
    for ((&x, &(a, b)), &i) in p.iter().zip(bounds.as_slice()).zip(integer) {
        finite(x)?;
        if a.is_some_and(|a| x < a) || b.is_some_and(|b| x > b) || (i && x.floor() != x) {
            return Err(invalid(
                scioxide_core::argument!("p"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "parameters violate bounds or integrality",
            ));
        }
    }
    Ok(())
}
impl<S: RealScalar> FitObjective<S> for PreparedFit<'_, S> {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        self.bounds.len()
    }
    fn validate(&self, p: &[S], _workspace: &mut ()) -> Result<(), SciOxideError> {
        validate_parameters(p, self.bounds, self.integer)?;
        self.model.validate(p)
    }
    fn value(&self, p: &[S], workspace: &mut ()) -> Result<S, SciOxideError> {
        self.validate(p, workspace)?;
        if !matches!(self.method, FitMethod::Likelihood) {
            return (0..self.targets.len()).try_fold(S::ZERO, |sum, i| {
                let r = self.residual(p, i)?;
                finite(sum + r * r)
            });
        }
        let truncation = self.truncation.map(|t| t.interval()).transpose()?;
        let normalizer = if let Some((a, b)) = truncation {
            if a == S::NEG_INFINITY && b == S::INFINITY {
                S::ZERO
            } else {
                finite(self.model.log_observation(p, interval_observation(a, b))?)?
            }
        } else {
            S::ZERO
        };
        // Compensated reduction uses the shared statistics accumulator.
        let sum = self.observations.iter().try_fold(
            crate::arithmetic::Compensated::<f64>::zero(),
            |sum, &o| {
                let event = match (o, truncation) {
                    (FitObservation::Exact(_), _) | (_, None) => o,
                    (_, Some((a, b))) => {
                        let (l, u) = o.interval().ok_or_else(|| {
                            invalid(
                                scioxide_core::argument!("observations"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "censor interval",
                            )
                        })?;
                        interval_observation(l.max(a), u.min(b))
                    }
                };
                let log = self.model.log_observation(p, event)?;
                if log == S::NEG_INFINITY {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("observation"),
                        scioxide_core::DomainReason::OutsideDomain,
                        "observation has zero model probability",
                    ));
                }
                let value = finite(log - normalizer)?;
                sum.added(-value.try_to_f64_exact()?)
            },
        )?;
        finite(number(sum.total()?)?)
    }
    fn residual_count(&self) -> Option<usize> {
        if matches!(self.method, FitMethod::Likelihood) {
            None
        } else {
            Some(self.targets.len())
        }
    }
    fn residuals(
        &self,
        p: &[S],
        output: &mut [S],
        workspace: &mut (),
    ) -> Result<(), SciOxideError> {
        self.validate(p, workspace)?;
        if self.residual_count() != Some(output.len()) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "residual output shape",
            ));
        }
        for (i, x) in output.iter_mut().enumerate() {
            *x = self.residual(p, i)?;
        }
        Ok(())
    }
}
fn interval_observation<S: RealScalar>(a: S, b: S) -> FitObservation<S> {
    if a == S::NEG_INFINITY {
        FitObservation::LeftCensored(b)
    } else if b == S::INFINITY {
        FitObservation::RightCensored(a)
    } else {
        FitObservation::IntervalCensored { lower: a, upper: b }
    }
}

/// Evaluates an objective at a batch of parameter vectors, in parallel.
///
/// `parameters` holds `m` vectors of length `d = objective.dimension()`, stored
/// one after another. `output[i]` receives the value at vector `i`. `scratch`
/// has length `m`, and `workers` holds one objective workspace for each worker
/// of the execution plan, at most 64. `options` sets the execution policy, and
/// `context` supplies the executor. The values do not depend on the number of
/// workers.
///
/// These evaluations are not counted in any solve context. A solver that uses
/// this function counts them itself.
///
/// # Errors
///
/// - `InvalidInput` if `d` is zero, `parameters.len() ≠ m d`, `scratch` has the
///   wrong length, or `workers` is too short.
/// - The errors of [`FitObjective::value`], and an error if a value is not finite.
/// - `ExecutionStopped` on cancellation or deadline.
pub fn fit_objectives_into<S: RealScalar, W: Send>(
    objective: &dyn FitObjective<S, Workspace = W>,
    parameters: &[S],
    options: crate::StatisticsOptions,
    output: &mut [S],
    scratch: &mut [S],
    workers: &mut [W],
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let d = objective.dimension();
    if d == 0
        || output.len().checked_mul(d) != Some(parameters.len())
        || scratch.len() != output.len()
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "fitting batch shape",
        ));
    }
    crate::estimation::execute(
        options,
        context,
        output.len(),
        1,
        workers,
        scratch,
        |i, workspace, out| {
            out[0] = finite(objective.value(&parameters[i * d..(i + 1) * d], workspace)?)?;
            Ok(())
        },
    )?;
    output.copy_from_slice(scratch);
    Ok(())
}
