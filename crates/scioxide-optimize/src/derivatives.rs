//! Derivative selection and finite-difference fallback for the solvers.

use crate::storage::resize;
use alloc::vec::Vec;
use scioxide_core::{Bounds, ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::{
    DerivativeCandidate, DerivativeOperation, DerivativeRequest, DerivativeShape,
    DerivativeStrategy, DifferenceScheme, FiniteDifference, FiniteDifferenceWorkspace,
    FixedCoordinatePolicy,
};

/// Computes gradients and Jacobians from a supplied derivative or by finite differences.
///
/// The workspace holds the finite-difference storage for one derivative
/// operation and one shape, for forward and central differences. The solvers
/// use it to obtain derivatives in the order that a
/// [`DerivativeStrategy`] lists, and custom line searches can use it the same
/// way. Finite differences run under the execution context of the call.
///
/// ```
/// use scioxide_differentiate::DerivativeOperation;
/// use scioxide_optimize::DerivativeWorkspace;
///
/// let storage = DerivativeWorkspace::<f64>::new(DerivativeOperation::Gradient, 2, 1)?;
/// assert_eq!(storage.input_dimension(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct DerivativeWorkspace<S: RealScalar> {
    operation: DerivativeOperation,
    outputs: usize,
    staging: Vec<S>,
    storage: FiniteDifferenceWorkspace<S>,
}

impl<S: RealScalar> DerivativeWorkspace<S> {
    /// Creates a workspace for `operation` with `inputs` coordinates and `outputs` components.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for a shape that `operation` does not accept, and
    /// `SizeOverflow` or `AllocationFailure` if the storage cannot be reserved.
    pub fn new(
        operation: DerivativeOperation,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self, SciOxideError> {
        let shape = match (operation, inputs, outputs) {
            (DerivativeOperation::First, 1, 1) | (DerivativeOperation::Gradient, _, 1) => {
                DerivativeShape::gradient(inputs)
            }
            (DerivativeOperation::Second, 1, 1) | (DerivativeOperation::Hessian, _, 1) => {
                DerivativeShape::hessian(inputs)
            }
            (DerivativeOperation::Jacobian, _, _) => DerivativeShape::jacobian(inputs, outputs),
            _ => {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("operation"),
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "the derivative operation does not accept these dimensions",
                ));
            }
        };
        // The requirements depend on the shape only, not on the scheme.
        let storage = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
            &FiniteDifference::<S>::default(),
            shape,
        ))?;
        let staging = scioxide_core::backend::try_filled(
            shape.output_len()?,
            S::ZERO,
            scioxide_core::quantity!("derivative_staging"),
        )?;
        Ok(Self {
            operation,
            outputs,
            staging,
            storage,
        })
    }

    /// Returns the number of input coordinates.
    pub fn input_dimension(&self) -> usize {
        self.storage.input_dimension()
    }
    /// Returns the number of components the callback returns.
    pub fn output_dimension(&self) -> usize {
        self.storage.output_dimension()
    }

    /// Writes the derivative at `x` to `output`, from the first source in `strategy` that is available.
    ///
    /// For an analytic or automatic source, `supplied` is called with `true`
    /// (analytic) or `false` (automatic). It writes the derivative and returns
    /// `Ok(true)`, or returns `Ok(false)` to move on to the next source. For a
    /// finite-difference source, `sample` evaluates the function at a point.
    /// `base` is the function value at `x` when it is already known; it is not
    /// evaluated again. With `bounds`, finite-difference steps stay inside the
    /// box. An error from a callback is returned unchanged and does not move on
    /// to the next source. `output` is written only when a complete, finite
    /// derivative is available. The call returns the source used and the
    /// number of function evaluations.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_differentiate::{DerivativeOperation, DerivativeStrategy};
    /// use scioxide_optimize::DerivativeWorkspace;
    ///
    /// // d(x²)/dx = 4 at x = 2
    /// let mut work = DerivativeWorkspace::<f64>::new(DerivativeOperation::Gradient, 1, 1)?;
    /// let mut gradient = [0.0];
    /// let _ = work.acquire_into(
    ///     DerivativeStrategy::Central,
    ///     &[2.0],
    ///     None,
    ///     Some(&[4.0]),
    ///     |_, _| Ok(false),
    ///     |x, y| {
    ///         y[0] = x[0] * x[0];
    ///         Ok(())
    ///     },
    ///     &mut gradient,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!((gradient[0] - 4.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `output` does not match the workspace.
    /// - `NonFinite` if `x` or the derivative contains NaN or infinity.
    /// - `InvalidInput` if no source in `strategy` is available.
    /// - `ExecutionStopped` on cancellation or deadline, `ConvergenceFailure`
    ///   if a finite-difference evaluation limit is reached, and
    ///   `NumericalInstability` if the finite difference cannot be completed.
    #[allow(clippy::too_many_arguments)]
    pub fn acquire_into(
        &mut self,
        strategy: DerivativeStrategy,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        base: Option<&[S]>,
        mut supplied: impl FnMut(bool, &mut [S]) -> Result<bool, SciOxideError>,
        mut sample: impl FnMut(&[S], &mut [S]) -> Result<(), SciOxideError>,
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(DerivativeCandidate, usize), SciOxideError> {
        if x.len() != self.input_dimension() || output.len() != self.staging.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.input_dimension(), self.staging.len()],
                [x.len(), output.len()],
            ));
        }
        crate::validation::finite_input(x)?;
        execution.check()?;
        let (candidate, evaluations) = strategy.select(|candidate| match candidate {
            DerivativeCandidate::FiniteDifference(scheme) => {
                let gradient = self.operation == DerivativeOperation::Gradient;
                let method = FiniteDifference::default().with_scheme(scheme);
                let method = if gradient && bounds.is_some() {
                    method.with_fixed_coordinates(FixedCoordinatePolicy::FeasibleZero)
                } else {
                    method
                };
                let scalar_base = || {
                    base.map(|b| {
                        if b.len() == 1 {
                            Ok(b[0])
                        } else {
                            Err(SciOxideError::dimension_mismatch([1], [b.len()]))
                        }
                    })
                    .transpose()
                };
                let report = match self.operation {
                    DerivativeOperation::First | DerivativeOperation::Gradient => method
                        .gradient_into(
                            &mut sample,
                            x,
                            bounds,
                            scalar_base()?,
                            &mut self.staging,
                            &mut self.storage,
                            execution,
                        )?,
                    DerivativeOperation::Second | DerivativeOperation::Hessian => method
                        .hessian_into(
                            &mut sample,
                            x,
                            bounds,
                            scalar_base()?,
                            &mut self.staging,
                            &mut self.storage,
                            execution,
                        )?,
                    DerivativeOperation::Jacobian => method.jacobian_into(
                        &mut sample,
                        x,
                        self.outputs,
                        bounds,
                        base,
                        &mut self.staging,
                        &mut self.storage,
                        execution,
                    )?,
                };
                report.require_success()?;
                Ok(Some(report.evaluations()))
            }
            source => {
                self.staging.fill(S::NAN);
                supplied(source == DerivativeCandidate::Analytic, &mut self.staging)
                    .map(|available| available.then_some(0))
            }
        })?;
        crate::utils::validate_finite(&self.staging)?;
        execution.check()?;
        output.copy_from_slice(&self.staging);
        Ok((candidate, evaluations))
    }
}

pub(crate) fn gradient_into<S, F>(
    f: &F,
    x: &[S],
    value: S,
    strategy: DerivativeStrategy,
    output: &mut [S],
    storage: &mut DerivativeWorkspace<S>,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    F: crate::ObjectiveFunction<S> + ?Sized,
{
    let _ = storage.acquire_into(
        strategy,
        x,
        None,
        Some(&[value]),
        |analytic, out| {
            if analytic {
                f.gradient_into(x, out)
            } else {
                f.automatic_gradient_into(x, out)
            }
        },
        |point, values| {
            values[0] = f.eval(point)?;
            Ok(())
        },
        output,
        execution,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix_into<S: RealScalar + NumericScalar<Real = S>>(
    strategy: DerivativeStrategy,
    x: &[S],
    base: &[S],
    output: &mut scioxide_linalg::Matrix<S>,
    storage: &mut DerivativeWorkspace<S>,
    mut supplied: impl FnMut(bool, &mut scioxide_linalg::Matrix<S>) -> Result<bool, SciOxideError>,
    sample: impl FnMut(&[S], &mut [S]) -> Result<(), SciOxideError>,
    execution: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    let rows = if storage.operation == DerivativeOperation::Hessian {
        storage.input_dimension()
    } else {
        storage.output_dimension()
    };
    crate::validation::matrix_shape(rows, storage.input_dimension(), output)?;
    crate::validation::input(storage.input_dimension(), x)?;
    execution.check()?;
    let (_, numerical) = strategy.select(|source| match source {
        DerivativeCandidate::FiniteDifference(scheme) => Ok(Some(Some(scheme))),
        source => {
            output.as_mut_slice().fill(<S as RealScalar>::NAN);
            supplied(source == DerivativeCandidate::Analytic, output).map(|yes| yes.then_some(None))
        }
    })?;
    let evaluations = if let Some(scheme) = numerical {
        let strategy = if scheme == DifferenceScheme::Forward {
            DerivativeStrategy::Forward
        } else {
            DerivativeStrategy::Central
        };
        storage
            .acquire_into(
                strategy,
                x,
                None,
                Some(base),
                |_, _| Ok(false),
                sample,
                output.as_mut_slice(),
                execution,
            )?
            .1
    } else {
        0
    };
    crate::validation::matrix_shape(rows, storage.input_dimension(), output)?;
    crate::utils::validate_finite(output.as_slice())?;
    execution.check()?;
    Ok(evaluations)
}

pub(crate) fn hessian_into<S, F>(
    f: &F,
    x: &[S],
    value: S,
    strategy: DerivativeStrategy,
    output: &mut scioxide_linalg::Matrix<S>,
    storage: &mut DerivativeWorkspace<S>,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: crate::ObjectiveFunction<S> + ?Sized,
{
    matrix_into(
        strategy,
        x,
        &[value],
        output,
        storage,
        |analytic, out| {
            if analytic {
                f.hessian_into(x, out)
            } else {
                f.automatic_hessian_into(x, out)
            }
        },
        |point, out| {
            out[0] = f.eval(point)?;
            Ok(())
        },
        execution,
    )?;
    Ok(())
}

#[derive(Debug)]
pub(crate) struct Differences<S: RealScalar + NumericScalar<Real = S>> {
    bounds: Option<Bounds<S>>,
    free: Vec<usize>,
    point: Vec<S>,
    gradient: Vec<S>,
    full: Vec<S>,
    work: Option<DerivativeWorkspace<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Differences<S> {
    pub(crate) fn new() -> Self {
        Self {
            bounds: None,
            free: Vec::new(),
            point: Vec::new(),
            gradient: Vec::new(),
            full: Vec::new(),
            work: None,
        }
    }
    pub(crate) fn prepare(
        &mut self,
        bounds: &[(Option<S>, Option<S>)],
    ) -> Result<(), SciOxideError> {
        let same = self.full.len() == bounds.len()
            && self.free.iter().enumerate().all(|(k, &j)| {
                self.bounds
                    .as_ref()
                    .is_some_and(|b| b.as_slice()[k] == bounds[j])
            })
            && self.free.len() == bounds.iter().filter(|(l, u)| l.is_none() || l != u).count();
        if same {
            return Ok(());
        }
        self.free.clear();
        scioxide_core::backend::try_reserve(
            &mut self.free,
            bounds.len(),
            scioxide_core::quantity!("sqp_free_coordinates"),
        )?;
        self.free.extend(
            bounds
                .iter()
                .enumerate()
                .filter(|(_, (l, u))| l.is_none() || l != u)
                .map(|(j, _)| j),
        );
        let mut reduced = Vec::new();
        scioxide_core::backend::try_reserve(
            &mut reduced,
            self.free.len(),
            scioxide_core::quantity!("sqp_reduced_bounds"),
        )?;
        reduced.extend(self.free.iter().map(|&j| bounds[j]));
        self.bounds = if reduced.is_empty() {
            None
        } else {
            Some(Bounds::new(reduced)?)
        };
        resize(
            &mut self.point,
            self.free.len(),
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        resize(
            &mut self.gradient,
            self.free.len(),
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        resize(
            &mut self.full,
            bounds.len(),
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        self.work = if self.free.is_empty() {
            None
        } else {
            Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Gradient,
                self.free.len(),
                1,
            )?)
        };
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gradient(
        &mut self,
        x: &[S],
        base: S,
        strategy: DerivativeStrategy,
        output: &mut [S],
        mut supplied: impl FnMut(bool, &mut [S]) -> Result<bool, SciOxideError>,
        mut evaluate: impl FnMut(&[S]) -> Result<S, SciOxideError>,
        execution: ExecutionContext<'_>,
    ) -> Result<bool, SciOxideError> {
        let available = if let Some(work) = &mut self.work {
            self.full.copy_from_slice(x);
            for (value, &j) in self.point.iter_mut().zip(&self.free) {
                *value = x[j];
            }
            // Supplied derivatives have full shape; finite differences use the
            // prepared free-coordinate shape and are scattered only on success.
            let mut numerical = None;
            let _ = strategy.select(|candidate| match candidate {
                DerivativeCandidate::FiniteDifference(scheme) => {
                    numerical = Some(scheme);
                    Ok(Some(()))
                }
                source => {
                    output.fill(<S as RealScalar>::NAN);
                    supplied(source == DerivativeCandidate::Analytic, output)
                        .map(|yes| yes.then_some(()))
                }
            })?;
            if let Some(scheme) = numerical {
                let strategy = if scheme == DifferenceScheme::Forward {
                    DerivativeStrategy::Forward
                } else {
                    DerivativeStrategy::Central
                };
                let free = &self.free;
                let full = &mut self.full;
                let _ = work.acquire_into(
                    strategy,
                    &self.point,
                    self.bounds.as_ref(),
                    Some(&[base]),
                    |_, _| Ok(false),
                    |point, out| {
                        for (&j, &value) in free.iter().zip(point) {
                            full[j] = value;
                        }
                        out[0] = evaluate(full)?;
                        Ok(())
                    },
                    &mut self.gradient,
                    execution,
                )?;
                output.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
                for (&j, &value) in self.free.iter().zip(&self.gradient) {
                    output[j] = value;
                }
                self.free.len() == x.len()
            } else {
                true
            }
        } else {
            let (_, available) = strategy.select(|source| {
                if matches!(source, DerivativeCandidate::FiniteDifference(_)) {
                    output.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
                    Ok(Some(false))
                } else {
                    supplied(source == DerivativeCandidate::Analytic, output)
                        .map(|yes| yes.then_some(true))
                }
            })?;
            available
        };
        crate::utils::validate_finite(output)?;
        Ok(available)
    }
}

pub(crate) fn prepared<S: RealScalar>(
    workspace: &mut Option<DerivativeWorkspace<S>>,
) -> Result<&mut DerivativeWorkspace<S>, SciOxideError> {
    workspace.as_mut().ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "derivative workspace was not prepared",
        )
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use scioxide_core::{CancellationToken, ExecutionStopReason, SciOxideErrorView};

    #[test]
    fn injected_cancellation_during_a_stencil_preserves_output() {
        let mut workspace =
            DerivativeWorkspace::<f64>::new(DerivativeOperation::Gradient, 2, 1).unwrap();
        let token = CancellationToken::new();
        let mut output = [71.0, 73.0];
        let samples = Cell::new(0);
        let error = workspace
            .acquire_into(
                DerivativeStrategy::Central,
                &[1.0, 2.0],
                None,
                Some(&[5.0]),
                |_, _| Ok(false),
                |x, y| {
                    samples.set(samples.get() + 1);
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    token.cancel();
                    Ok(())
                },
                &mut output,
                ExecutionContext::sequential().with_cancellation(&token),
            )
            .unwrap_err();
        assert!(samples.get() > 0);
        assert_eq!(output, [71.0, 73.0]);
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::Cancelled,
                ..
            }
        ));
        token.reset();
        let _ = workspace
            .acquire_into(
                DerivativeStrategy::Central,
                &[1.0, 2.0],
                None,
                Some(&[5.0]),
                |_, _| Ok(false),
                |x, y| {
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    Ok(())
                },
                &mut output,
                ExecutionContext::sequential().with_cancellation(&token),
            )
            .unwrap();
        assert!((output[0] - 2.0).abs() < 1e-8 && (output[1] - 4.0).abs() < 1e-8);
    }

    #[test]
    fn supplied_matrix_derivatives_cannot_change_shape_or_leave_old_entries() {
        let mut work = DerivativeWorkspace::<f64>::new(DerivativeOperation::Hessian, 2, 1).unwrap();
        for wrong_shape in [true, false] {
            let mut output = scioxide_linalg::Matrix::zeros(2, 2).unwrap();
            let error = matrix_into(
                DerivativeStrategy::Analytic,
                &[1.0, 2.0],
                &[5.0],
                &mut output,
                &mut work,
                |_, out| {
                    if wrong_shape {
                        out.try_resize_flat(1, 4)?;
                        out.as_mut_slice().fill(1.0);
                    } else {
                        out.as_mut_slice()[0] = 1.0;
                    }
                    Ok(true)
                },
                |_, _| Ok(()),
                ExecutionContext::sequential(),
            )
            .unwrap_err();
            assert!(if wrong_shape {
                matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
            } else {
                matches!(error.view(), SciOxideErrorView::NonFinite { .. })
            });
        }
    }

    #[test]
    fn injected_deadline_during_a_stencil_is_not_derivative_absence() {
        use core::sync::atomic::{AtomicBool, Ordering};
        use core::time::Duration;
        use scioxide_core::{CapabilityError, Clock, ClockBinding, Deadline, MonotonicInstant};
        struct TestClock(AtomicBool);
        impl Clock for TestClock {
            fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
                Ok(MonotonicInstant::from_duration_since_epoch(
                    Duration::from_secs(u64::from(self.0.load(Ordering::Relaxed))),
                ))
            }
        }
        let clock = TestClock(AtomicBool::new(false));
        let binding = ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_deadline(Deadline::at(
                &binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
            ))
            .unwrap();
        let mut workspace =
            DerivativeWorkspace::<f64>::new(DerivativeOperation::Gradient, 1, 1).unwrap();
        let mut output = [99.0];
        let error = workspace
            .acquire_into(
                DerivativeStrategy::Forward,
                &[2.0],
                None,
                Some(&[4.0]),
                |_, _| Ok(false),
                |x, y| {
                    y[0] = x[0] * x[0];
                    clock.0.store(true, Ordering::Relaxed);
                    Ok(())
                },
                &mut output,
                execution,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::DeadlineReached,
                ..
            }
        ));
        assert_eq!(output, [99.0]);
    }

    #[test]
    fn unavailable_derivatives_fall_back_but_errors_and_unwritten_values_do_not() {
        let mut workspace =
            DerivativeWorkspace::<f64>::new(DerivativeOperation::Gradient, 1, 1).unwrap();
        let samples = Cell::new(0);
        let mut output = [99.0];
        let _ = workspace
            .acquire_into(
                DerivativeStrategy::AnalyticOrAutomaticOrForward,
                &[2.0],
                None,
                Some(&[4.0]),
                |_, _| Ok(false),
                |x, y| {
                    samples.set(samples.get() + 1);
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &mut output,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(samples.get(), 1);
        assert!((output[0] - 4.0).abs() < 1e-5);
        for fail in [false, true] {
            output = [99.0];
            let error = workspace
                .acquire_into(
                    DerivativeStrategy::AnalyticOrForward,
                    &[2.0],
                    None,
                    Some(&[4.0]),
                    |_, _| {
                        if fail {
                            Err(SciOxideError::singular(None))
                        } else {
                            Ok(true)
                        }
                    },
                    |_, _| {
                        samples.set(samples.get() + 1);
                        Ok(())
                    },
                    &mut output,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(samples.get(), 1);
            assert_eq!(output, [99.0]);
            assert!(if fail {
                matches!(error.view(), SciOxideErrorView::Singular { .. })
            } else {
                matches!(error.view(), SciOxideErrorView::NonFinite { .. })
            });
        }
    }
}
