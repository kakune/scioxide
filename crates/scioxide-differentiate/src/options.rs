//! Derivative operations, difference schemes and step selection.

use alloc::vec::Vec;
use scioxide_core::{Dimensions, InvalidInputReason, RealScalar, SciOxideError};
/// Selects the derivative that a node plan computes and the layout of its
/// output.
///
/// [`PhysicalDifferencePlan`](crate::PhysicalDifferencePlan) and
/// [`RelativeDifference`](crate::RelativeDifference) take it; the method values
/// name the operation by their methods instead. For `f: ℝⁿ → ℝᵐ`, outputs are
/// row-major; see [Conventions](crate#conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOperation {
    /// First derivative `f′(x)`, with `n = m = 1`.
    First,
    /// Second derivative `f″(x)`, with `n = m = 1`.
    Second,
    /// Gradient `∇f(x)` of length `n`, with `m = 1`.
    Gradient,
    /// Jacobian, `m × n`; entry `i·n + j` is `∂fᵢ/∂xⱼ`.
    Jacobian,
    /// Hessian, symmetric `n × n`, with `m = 1`.
    Hessian,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DifferenceShape {
    First,
    Second,
    Gradient,
    Jacobian,
    Hessian,
    Jvp(ZeroDirectionPolicy),
}
impl From<DerivativeOperation> for DifferenceShape {
    fn from(operation: DerivativeOperation) -> Self {
        match operation {
            DerivativeOperation::First => Self::First,
            DerivativeOperation::Second => Self::Second,
            DerivativeOperation::Gradient => Self::Gradient,
            DerivativeOperation::Jacobian => Self::Jacobian,
            DerivativeOperation::Hessian => Self::Hessian,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeKind {
    Gradient,
    Jacobian,
    Hessian,
    Jvp,
}
/// Names the operation and dimensions of one evaluation, to size a workspace.
///
/// For `f: ℝⁿ → ℝᵐ` with `n = inputs` and `m = outputs`, the output has `n`
/// entries for a gradient, `m·n` for a Jacobian, `n²` for a Hessian and `m` for
/// a Jacobian-vector product.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::DerivativeShape;
///
/// assert_eq!(DerivativeShape::jacobian(2, 3).output_len()?, 6);
/// assert_eq!(DerivativeShape::hessian(4).output_len()?, 16);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivativeShape {
    kind: ShapeKind,
    inputs: usize,
    outputs: usize,
}
impl DerivativeShape {
    /// Returns the shape of the gradient of `f: ℝⁿ → ℝ` with `n = inputs`.
    pub const fn gradient(inputs: usize) -> Self {
        Self {
            kind: ShapeKind::Gradient,
            inputs,
            outputs: 1,
        }
    }
    /// Returns the shape of the Jacobian of `f: ℝⁿ → ℝᵐ` with `n = inputs` and
    /// `m = outputs`.
    pub const fn jacobian(inputs: usize, outputs: usize) -> Self {
        Self {
            kind: ShapeKind::Jacobian,
            inputs,
            outputs,
        }
    }
    /// Returns the shape of the Hessian of `f: ℝⁿ → ℝ` with `n = inputs`.
    pub const fn hessian(inputs: usize) -> Self {
        Self {
            kind: ShapeKind::Hessian,
            inputs,
            outputs: 1,
        }
    }
    /// Returns the shape of the Jacobian-vector product of `f: ℝⁿ → ℝᵐ` with
    /// `n = inputs` and `m = outputs`.
    pub const fn jvp(inputs: usize, outputs: usize) -> Self {
        Self {
            kind: ShapeKind::Jvp,
            inputs,
            outputs,
        }
    }
    /// Returns the number of output entries: `n`, `m·n`, `n²` or `m`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` or `n²` does not fit in
    ///   `usize`.
    pub fn output_len(&self) -> Result<usize, SciOxideError> {
        crate::validation::shape(
            self.difference(ZeroDirectionPolicy::MathematicalZero),
            self.inputs,
            self.outputs,
        )
    }
    pub(crate) const fn inputs(&self) -> usize {
        self.inputs
    }
    pub(crate) const fn outputs(&self) -> usize {
        self.outputs
    }
    pub(crate) const fn is_hessian(&self) -> bool {
        matches!(self.kind, ShapeKind::Hessian)
    }
    pub(crate) const fn difference(&self, zero_direction: ZeroDirectionPolicy) -> DifferenceShape {
        match self.kind {
            ShapeKind::Gradient => DifferenceShape::Gradient,
            ShapeKind::Jacobian => DifferenceShape::Jacobian,
            ShapeKind::Hessian => DifferenceShape::Hessian,
            ShapeKind::Jvp => DifferenceShape::Jvp(zero_direction),
        }
    }
}
/// Pairs a method value with the [`DerivativeShape`] of an evaluation, to size
/// its workspace.
///
/// [`FiniteDifferenceWorkspace`](crate::FiniteDifferenceWorkspace),
/// [`AdaptiveDifferenceWorkspace`](crate::AdaptiveDifferenceWorkspace) and
/// [`ComplexStepWorkspace`](crate::ComplexStepWorkspace) take it in
/// `workspace_requirements` and `workspace`. The method supplies the settings
/// that change the size, such as the number of levels and the byte limit.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::{DerivativeRequest, DerivativeShape};
/// use scioxide_differentiate::{FiniteDifference, FiniteDifferenceWorkspace};
///
/// let method = FiniteDifference::<f64>::default();
/// let request = DerivativeRequest::new(&method, DerivativeShape::jacobian(2, 3));
/// let workspace = FiniteDifferenceWorkspace::workspace(request)?;
/// assert_eq!(workspace.input_dimension(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct DerivativeRequest<'a, M> {
    method: &'a M,
    shape: DerivativeShape,
}
impl<M> Clone for DerivativeRequest<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M> Copy for DerivativeRequest<'_, M> {}
impl<'a, M> DerivativeRequest<'a, M> {
    /// Returns the request for `method` and `shape`.
    pub const fn new(method: &'a M, shape: DerivativeShape) -> Self {
        Self { method, shape }
    }
    pub(crate) const fn method(&self) -> &'a M {
        self.method
    }
    pub(crate) const fn shape(&self) -> DerivativeShape {
        self.shape
    }
}

/// Selects the finite-difference formula of [`FiniteDifference`](crate::FiniteDifference).
///
/// If neither direction fits the nominal stencil, the step shrinks toward the
/// side with more room (forward on ties). If rounding prevents a one-sided
/// central stencil, a smaller symmetric stencil is tried. All nodes remain
/// distinct and finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DifferenceScheme {
    /// One-sided differences with truncation error `O(h)`: nodes `x, x + h`
    /// for first derivatives and `x, x + h, x + 2h` for second derivatives.
    /// If the forward nodes leave the bounds, the step is taken backward.
    Forward,
    /// Central differences with truncation error `O(h²)`: nodes
    /// `x − h, x, x + h`. Near a bound, a one-sided stencil of the same order
    /// with nodes `x, x ± h, x ± 2h` (and `x ± 3h` for second derivatives).
    Central,
}

/// Selects what a Jacobian-vector product does when every component of the
/// direction is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[must_use]
pub enum ZeroDirectionPolicy {
    /// Returns `J v = 0` without calling the function.
    #[default]
    MathematicalZero,
    /// Calls the function once at `x` and checks its values before returning
    /// zero. A product with no outputs never calls the function.
    ValidateBase,
}

// Validated, exclusive step selection shared by real-node difference families.

#[derive(Debug, Clone)]
enum StepValues<S> {
    Scalar(S),
    Coordinates(Vec<S>),
}
#[derive(Debug, Clone, Default)]
enum StepKind<S> {
    #[default]
    Default,
    Absolute(StepValues<S>),
    Relative(StepValues<S>),
}

/// Selects the step of [`FiniteDifference`](crate::FiniteDifference) or the
/// initial step of [`AdaptiveDifference`](crate::AdaptiveDifference).
///
/// The default is the method's own step (see [Conventions](crate#conventions)).
/// An absolute step `h` is used as given. A relative step `h` becomes
/// `h · max(1, |xᵢ|)` at coordinate `xᵢ`. A single value applies to every
/// coordinate; a per-coordinate selection needs one value per input, and each
/// evaluation returns `DimensionMismatch` otherwise. Steps are magnitudes: the
/// direction of each displacement comes from the scheme and the bounds.
#[derive(Debug, Clone)]
pub struct DifferenceStep<S: RealScalar>(StepKind<S>);
impl<S: RealScalar> Default for DifferenceStep<S> {
    fn default() -> Self {
        Self(StepKind::Default)
    }
}
impl<S: RealScalar> DifferenceStep<S> {
    /// Returns the absolute step `step` for every coordinate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `step` is not finite and positive.
    pub fn absolute(step: S) -> Result<Self, SciOxideError> {
        positive(step, scioxide_core::argument!("step"))?;
        Ok(Self(StepKind::Absolute(StepValues::Scalar(step))))
    }
    /// Returns the relative step `step · max(1, |xᵢ|)` for every coordinate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `step` is not finite and positive.
    pub fn relative(step: S) -> Result<Self, SciOxideError> {
        positive(step, scioxide_core::argument!("step"))?;
        Ok(Self(StepKind::Relative(StepValues::Scalar(step))))
    }
    /// Returns the absolute step `steps[i]` for coordinate `i`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if a step is not finite and positive.
    /// - `SizeOverflow` or `AllocationFailure` if the steps cannot be copied.
    pub fn absolute_coordinates(steps: &[S]) -> Result<Self, SciOxideError> {
        steps
            .iter()
            .try_for_each(|&step| positive(step, scioxide_core::argument!("steps")))?;
        Ok(Self(StepKind::Absolute(StepValues::Coordinates(
            scioxide_core::backend::try_copy_slice(
                steps,
                scioxide_core::quantity!("difference_steps"),
            )?,
        ))))
    }
    /// Returns the relative step `steps[i] · max(1, |xᵢ|)` for coordinate `i`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if a step is not finite and positive.
    /// - `SizeOverflow` or `AllocationFailure` if the steps cannot be copied.
    pub fn relative_coordinates(steps: &[S]) -> Result<Self, SciOxideError> {
        steps
            .iter()
            .try_for_each(|&step| positive(step, scioxide_core::argument!("steps")))?;
        Ok(Self(StepKind::Relative(StepValues::Coordinates(
            scioxide_core::backend::try_copy_slice(
                steps,
                scioxide_core::quantity!("difference_steps"),
            )?,
        ))))
    }
    pub(super) fn is_default(&self) -> bool {
        matches!(self.0, StepKind::Default)
    }
    pub(super) fn validate(&self, dimension: usize) -> Result<(), SciOxideError> {
        if let StepKind::Absolute(StepValues::Coordinates(v))
        | StepKind::Relative(StepValues::Coordinates(v)) = &self.0
            && v.len() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(dimension),
                Dimensions::one(v.len()),
            ));
        }
        Ok(())
    }
    pub(super) fn magnitude(&self, x: S, index: usize, default: S) -> S {
        let value = |values: &StepValues<S>| match values {
            StepValues::Scalar(step) => *step,
            StepValues::Coordinates(steps) => steps[index],
        };
        match &self.0 {
            StepKind::Default => default,
            StepKind::Absolute(values) => value(values),
            StepKind::Relative(values) => value(values) * x.abs().max(S::ONE),
        }
    }
}
fn positive<S: RealScalar>(
    step: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !step.is_finite() || step <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "step magnitudes must be finite and positive",
        ));
    }
    Ok(())
}
