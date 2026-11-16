//! Test-only dispatch across the operations of `AdaptiveDifference`, without
//! numerical kernels. Scalar operations use the scalar methods in owned form
//! and the one-coordinate gradient or Hessian in workspace form.
#![allow(clippy::too_many_arguments)]
use scioxide_core::{
    Bounds, ExecutionPolicy, InvalidInputReason, NumericScalar, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, argument,
};
use scioxide_differentiate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOperation {
    First,
    Second,
    Gradient,
    Jacobian,
    Hessian,
    Jvp(ZeroDirectionPolicy),
}
pub struct TestAdaptivePlan<S: RealScalar> {
    method: AdaptiveDifference<S>,
    operation: TestOperation,
    inputs: usize,
    outputs: usize,
}
impl<S: RealScalar> TestAdaptivePlan<S> {
    pub fn new(
        operation: TestOperation,
        inputs: usize,
        outputs: usize,
        options: AdaptiveDifference<S>,
    ) -> Result<Self, SciOxideError> {
        let method = match operation {
            TestOperation::Jvp(policy) => options.with_zero_direction(policy),
            _ => options,
        };
        Ok(Self {
            method,
            operation,
            inputs,
            outputs,
        })
    }
    fn shape(&self) -> DerivativeShape {
        match self.operation {
            TestOperation::First | TestOperation::Gradient => {
                DerivativeShape::gradient(self.inputs)
            }
            TestOperation::Second | TestOperation::Hessian => DerivativeShape::hessian(self.inputs),
            TestOperation::Jacobian => DerivativeShape::jacobian(self.inputs, self.outputs),
            TestOperation::Jvp(_) => DerivativeShape::jvp(self.inputs, self.outputs),
        }
    }
    pub fn output_len(&self) -> usize {
        self.shape().output_len().unwrap()
    }
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
    ) -> Result<AdaptiveDifferenceWorkspace<T>, SciOxideError> {
        AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(&self.method, self.shape()))
    }
    pub fn workspace_requirements<T: NumericScalar<Real = S>>(
        &self,
    ) -> Result<AdaptiveDifferenceRequirements<T>, SciOxideError> {
        AdaptiveDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(
            &self.method,
            self.shape(),
        ))
    }
    pub fn evaluate<
        T: NumericScalar<Real = S>,
        F: FnMut(&[S], &mut [T]) -> Result<(), SciOxideError>,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        mut function: F,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        let m = &self.method;
        let scalar = |function: &mut F, t: S| {
            let mut y = [T::ZERO];
            function(&[t], &mut y)?;
            Ok(y[0])
        };
        match (self.operation, direction) {
            (TestOperation::First, None) if bounds.is_none() && x.len() == 1 => {
                m.derivative(|t| scalar(&mut function, t), x[0])
            }
            (TestOperation::Second, None) if bounds.is_none() && x.len() == 1 => {
                m.second_derivative(|t| scalar(&mut function, t), x[0])
            }
            (TestOperation::First | TestOperation::Gradient, None) => {
                m.gradient(function, x, bounds)
            }
            (TestOperation::Second | TestOperation::Hessian, None) => {
                m.hessian(function, x, bounds)
            }
            (TestOperation::Jacobian, None) => m.jacobian(function, x, self.outputs, bounds),
            (TestOperation::Jvp(_), Some(d)) => m.jvp(function, x, d, self.outputs, bounds),
            _ => Err(mismatch()),
        }
    }
    fn evaluate_sampled<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        let m = &self.method;
        match (self.operation, direction) {
            (TestOperation::First | TestOperation::Gradient, None) => {
                m.gradient_into(function, x, bounds, out, errors, workspace, cx)
            }
            (TestOperation::Second | TestOperation::Hessian, None) => {
                m.hessian_into(function, x, bounds, out, errors, workspace, cx)
            }
            (TestOperation::Jacobian, None) => m.jacobian_into(
                function,
                x,
                self.outputs,
                bounds,
                out,
                errors,
                workspace,
                cx,
            ),
            (TestOperation::Jvp(_), Some(d)) => {
                m.jvp_into(function, x, d, bounds, out, errors, workspace, cx)
            }
            _ => Err(mismatch()),
        }
    }
    pub fn evaluate_into<
        T: NumericScalar<Real = S>,
        F: FnMut(&[S], &mut [T]) -> Result<(), SciOxideError>,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.evaluate_sampled(x, bounds, direction, function, out, errors, workspace, cx)
    }
    pub fn evaluate_contextual<
        T: NumericScalar<Real = S>,
        F: FnMut(&[S], &mut [T], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.evaluate_sampled(
            x,
            bounds,
            direction,
            Contextual(function),
            out,
            errors,
            workspace,
            cx,
        )
    }
    pub fn with_budgets(&self, budgets: SolveBudgets) -> Result<Self, SciOxideError> {
        Self::new(
            self.operation,
            self.inputs,
            self.outputs,
            self.method.clone().with_budgets(budgets),
        )
    }
    pub fn configured(&self, policy: ExecutionPolicy) -> Result<Self, SciOxideError> {
        Self::new(
            self.operation,
            self.inputs,
            self.outputs,
            self.method.clone().with_execution(policy),
        )
    }
    pub fn evaluate_parallel<
        T: NumericScalar<Real = S>,
        F: Fn(&[S], &mut [T]) -> Result<(), SciOxideError> + Sync,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        policy: ExecutionPolicy,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.configured(policy)?.evaluate_sampled(
            x,
            bounds,
            direction,
            Parallel(function),
            out,
            errors,
            workspace,
            cx,
        )
    }
}
fn mismatch() -> SciOxideError {
    SciOxideError::invalid_input(
        argument!("direction"),
        InvalidInputReason::Inconsistent,
        "direction must match the prepared capability",
    )
}
