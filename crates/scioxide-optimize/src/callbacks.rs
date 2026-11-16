//! Closure adapters for the objective, constraint and least-squares traits.

use core::marker::PhantomData;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

use crate::traits::{ConstraintFunction, LeastSquaresProblem, ObjectiveFunction};
use crate::validation::{matrix_shape, slice_shape};

type VectorCallback<S> = fn(&[S], &mut [S]) -> Result<(), SciOxideError>;
type ProductCallback<S> = fn(&[S], &[S], &mut [S]) -> Result<(), SciOxideError>;
type MatrixCallback<S> = fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>;

fn validate_dimension(
    argument: scioxide_core::Argument,
    dimension: usize,
) -> Result<(), SciOxideError> {
    if dimension == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Empty,
            alloc::format!("{argument} must be positive"),
        ));
    }
    Ok(())
}

/// Builds an [`ObjectiveFunction`] from closures.
///
/// The value closure takes `x` and returns `f(x)`. Gradient and Hessian
/// closures write every entry of the supplied buffers. The adapter checks only
/// lengths and allocates nothing per call; the solver validates finite values
/// before publishing output.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{Bfgs, ObjectiveFn};
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + 4.0 * x[1] * x[1]))
///     .with_gradient(|x, g| {
///         g[0] = 2.0 * x[0];
///         g[1] = 8.0 * x[1];
///         Ok(())
///     });
/// let result = Bfgs::<f64>::default().minimize(&objective, &[1.0, -1.0])?;
/// result.require_success()?;
/// assert!(result.x().iter().all(|v| v.abs() < 1e-4));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct ObjectiveFn<
    E,
    S = f64,
    G = VectorCallback<S>,
    H = MatrixCallback<S>,
    P = ProductCallback<S>,
    AG = VectorCallback<S>,
    AH = MatrixCallback<S>,
> {
    dimension: usize,
    evaluate: E,
    gradient: Option<G>,
    hessian: Option<H>,
    product: Option<P>,
    automatic_gradient: Option<AG>,
    automatic_hessian: Option<AH>,
    scalar: PhantomData<fn() -> S>,
}

impl<E> ObjectiveFn<E> {
    /// Creates an objective of `dimension` variables from a value closure.
    /// A zero dimension is accepted; the solver validates the problem.
    pub const fn new<S>(dimension: usize, evaluate: E) -> ObjectiveFn<E, S>
    where
        S: RealScalar + NumericScalar<Real = S>,
        E: Fn(&[S]) -> Result<S, SciOxideError>,
    {
        ObjectiveFn {
            dimension,
            evaluate,
            gradient: None,
            hessian: None,
            product: None,
            automatic_gradient: None,
            automatic_hessian: None,
            scalar: PhantomData,
        }
    }
}

impl<E, S, G, H, P, AG, AH> ObjectiveFn<E, S, G, H, P, AG, AH> {
    /// Adds a closure that writes `∇²f(x) d` into the supplied output slice.
    #[must_use]
    pub fn with_hessian_vector_product<P2>(self, product: P2) -> ObjectiveFn<E, S, G, H, P2, AG, AH>
    where
        P2: Fn(&[S], &[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ObjectiveFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            gradient: self.gradient,
            hessian: self.hessian,
            product: Some(product),
            automatic_gradient: self.automatic_gradient,
            automatic_hessian: self.automatic_hessian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes the analytic gradient.
    #[must_use]
    pub fn with_gradient<G2>(self, gradient: G2) -> ObjectiveFn<E, S, G2, H, P, AG, AH>
    where
        G2: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ObjectiveFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            gradient: Some(gradient),
            hessian: self.hessian,
            product: self.product,
            automatic_gradient: self.automatic_gradient,
            automatic_hessian: self.automatic_hessian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes the analytic Hessian.
    #[must_use]
    pub fn with_hessian<H2>(self, hessian: H2) -> ObjectiveFn<E, S, G, H2, P, AG, AH>
    where
        H2: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    {
        ObjectiveFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            gradient: self.gradient,
            hessian: Some(hessian),
            product: self.product,
            automatic_gradient: self.automatic_gradient,
            automatic_hessian: self.automatic_hessian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes a gradient computed by automatic
    /// differentiation.
    #[must_use]
    pub fn with_automatic_gradient<AG2>(
        self,
        automatic_gradient: AG2,
    ) -> ObjectiveFn<E, S, G, H, P, AG2, AH>
    where
        AG2: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ObjectiveFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            gradient: self.gradient,
            hessian: self.hessian,
            product: self.product,
            automatic_gradient: Some(automatic_gradient),
            automatic_hessian: self.automatic_hessian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes a Hessian computed by automatic
    /// differentiation.
    #[must_use]
    pub fn with_automatic_hessian<AH2>(
        self,
        automatic_hessian: AH2,
    ) -> ObjectiveFn<E, S, G, H, P, AG, AH2>
    where
        AH2: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    {
        ObjectiveFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            gradient: self.gradient,
            hessian: self.hessian,
            product: self.product,
            automatic_gradient: self.automatic_gradient,
            automatic_hessian: Some(automatic_hessian),
            scalar: PhantomData,
        }
    }
}

impl<S, E, G, H, P, AG, AH> ObjectiveFunction<S> for ObjectiveFn<E, S, G, H, P, AG, AH>
where
    S: RealScalar + NumericScalar<Real = S>,
    E: Fn(&[S]) -> Result<S, SciOxideError>,
    P: Fn(&[S], &[S], &mut [S]) -> Result<(), SciOxideError>,
    G: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    H: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    AG: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    AH: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
{
    fn dim(&self) -> usize {
        self.dimension
    }

    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        slice_shape(self.dimension, x)?;
        // Solvers reject a non-finite trial value and refuse a non-finite start.
        (self.evaluate)(x)
    }

    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        slice_shape(self.dimension, output)?;
        let Some(callback) = &self.gradient else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }

    fn automatic_gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        slice_shape(self.dimension, output)?;
        let Some(callback) = &self.automatic_gradient else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }

    fn has_hessian_vector_product(&self) -> bool {
        self.product.is_some()
    }

    fn hessian_vector_product_into(
        &self,
        x: &[S],
        direction: &[S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        slice_shape(self.dimension, x)?;
        slice_shape(self.dimension, direction)?;
        slice_shape(self.dimension, output)?;
        let callback = self.product.as_ref().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "objective has no Hessian-vector product",
            )
        })?;
        callback(x, direction, output)
    }

    fn hessian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        matrix_shape(self.dimension, self.dimension, output)?;
        let Some(callback) = &self.hessian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }

    fn automatic_hessian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        matrix_shape(self.dimension, self.dimension, output)?;
        let Some(callback) = &self.automatic_hessian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }
}

/// Builds a [`ConstraintFunction`] from closures.
///
/// The value closure returns `c(x)`; whether it is an equality or an
/// inequality is set when it is added to [`crate::Constraints`]. Values are
/// passed through unchanged. Jacobian rows from the optional closures are
/// written into the supplied buffer after checking its length.
pub struct ConstraintFn<E, S = f64, J = VectorCallback<S>, AJ = VectorCallback<S>> {
    dimension: usize,
    evaluate: E,
    jacobian: Option<J>,
    automatic_jacobian: Option<AJ>,
    scalar: PhantomData<fn() -> S>,
}

impl<E> ConstraintFn<E> {
    /// Creates a constraint of `dimension` variables from a value closure.
    /// A zero dimension is accepted; the solver validates the problem.
    pub const fn new<S>(dimension: usize, evaluate: E) -> ConstraintFn<E, S>
    where
        S: RealScalar + NumericScalar<Real = S>,
        E: Fn(&[S]) -> Result<S, SciOxideError>,
    {
        ConstraintFn {
            dimension,
            evaluate,
            jacobian: None,
            automatic_jacobian: None,
            scalar: PhantomData,
        }
    }
}

impl<E, S, J, AJ> ConstraintFn<E, S, J, AJ> {
    /// Adds a closure that writes the analytic gradient `∇c(x)`.
    #[must_use]
    pub fn with_jacobian<J2>(self, jacobian: J2) -> ConstraintFn<E, S, J2, AJ>
    where
        J2: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ConstraintFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            jacobian: Some(jacobian),
            automatic_jacobian: self.automatic_jacobian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes `∇c(x)` computed by automatic
    /// differentiation.
    #[must_use]
    pub fn with_automatic_jacobian<AJ2>(self, automatic_jacobian: AJ2) -> ConstraintFn<E, S, J, AJ2>
    where
        AJ2: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        ConstraintFn {
            dimension: self.dimension,
            evaluate: self.evaluate,
            jacobian: self.jacobian,
            automatic_jacobian: Some(automatic_jacobian),
            scalar: PhantomData,
        }
    }
}

impl<S, E, J, AJ> ConstraintFunction<S> for ConstraintFn<E, S, J, AJ>
where
    S: RealScalar + NumericScalar<Real = S>,
    E: Fn(&[S]) -> Result<S, SciOxideError>,
    J: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    AJ: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    fn dim(&self) -> usize {
        self.dimension
    }

    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        slice_shape(self.dimension, x)?;
        // Solvers reject a non-finite trial value and refuse a non-finite start.
        (self.evaluate)(x)
    }

    fn jacobian_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        slice_shape(self.dimension, output)?;
        let Some(callback) = &self.jacobian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }

    fn automatic_jacobian_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        slice_shape(self.dimension, x)?;
        slice_shape(self.dimension, output)?;
        let Some(callback) = &self.automatic_jacobian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }
}

/// Builds a [`LeastSquaresProblem`] from closures.
///
/// The residual closure writes every entry of `r(x)` into the supplied slice.
/// Optional Jacobian closures write the supplied `m × n` matrix. The adapter
/// checks only shapes and allocates nothing per call; the solver checks finiteness.
pub struct LeastSquaresFn<R, S = f64, J = MatrixCallback<S>, AJ = MatrixCallback<S>> {
    parameter_dimension: usize,
    residual_dimension: usize,
    residual: R,
    jacobian: Option<J>,
    automatic_jacobian: Option<AJ>,
    scalar: PhantomData<fn() -> S>,
}

impl<R> LeastSquaresFn<R> {
    /// Creates a problem with `parameter_dimension` parameters and
    /// `residual_dimension` residuals from a residual closure.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` (`Empty`) if either dimension is zero.
    pub fn new<S>(
        parameter_dimension: usize,
        residual_dimension: usize,
        residual: R,
    ) -> Result<LeastSquaresFn<R, S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        R: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    {
        validate_dimension(
            scioxide_core::argument!("parameter_dimension"),
            parameter_dimension,
        )?;
        validate_dimension(
            scioxide_core::argument!("residual_dimension"),
            residual_dimension,
        )?;
        Ok(LeastSquaresFn {
            parameter_dimension,
            residual_dimension,
            residual,
            jacobian: None,
            automatic_jacobian: None,
            scalar: PhantomData,
        })
    }
}

impl<R, S, J, AJ> LeastSquaresFn<R, S, J, AJ> {
    /// Adds a closure that writes the analytic `m × n` Jacobian.
    #[must_use]
    pub fn with_jacobian<J2>(self, jacobian: J2) -> LeastSquaresFn<R, S, J2, AJ>
    where
        J2: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    {
        LeastSquaresFn {
            parameter_dimension: self.parameter_dimension,
            residual_dimension: self.residual_dimension,
            residual: self.residual,
            jacobian: Some(jacobian),
            automatic_jacobian: self.automatic_jacobian,
            scalar: PhantomData,
        }
    }

    /// Adds a closure that writes the Jacobian computed by automatic
    /// differentiation.
    #[must_use]
    pub fn with_automatic_jacobian<AJ2>(
        self,
        automatic_jacobian: AJ2,
    ) -> LeastSquaresFn<R, S, J, AJ2>
    where
        AJ2: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    {
        LeastSquaresFn {
            parameter_dimension: self.parameter_dimension,
            residual_dimension: self.residual_dimension,
            residual: self.residual,
            jacobian: self.jacobian,
            automatic_jacobian: Some(automatic_jacobian),
            scalar: PhantomData,
        }
    }
}

impl<S, R, J, AJ> LeastSquaresProblem<S> for LeastSquaresFn<R, S, J, AJ>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
    J: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
    AJ: Fn(&[S], &mut Matrix<S>) -> Result<(), SciOxideError>,
{
    fn param_dim(&self) -> usize {
        self.parameter_dimension
    }

    fn residual_dim(&self) -> usize {
        self.residual_dimension
    }

    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        slice_shape(self.parameter_dimension, x)?;
        slice_shape(self.residual_dimension, output)?;
        (self.residual)(x, output)
    }

    fn jacobian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        slice_shape(self.parameter_dimension, x)?;
        matrix_shape(self.residual_dimension, self.parameter_dimension, output)?;
        let Some(callback) = &self.jacobian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }

    fn automatic_jacobian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        slice_shape(self.parameter_dimension, x)?;
        matrix_shape(self.residual_dimension, self.parameter_dimension, output)?;
        let Some(callback) = &self.automatic_jacobian else {
            return Ok(false);
        };
        callback(x, output)?;
        Ok(true)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct NativeObjective;

    impl ObjectiveFunction<f64> for NativeObjective {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1])
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 1.0]);
            Ok(true)
        }

        fn hessian_into(
            &self,
            _x: &[f64],
            output: &mut Matrix<f64>,
        ) -> Result<bool, SciOxideError> {
            output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 0.0]);
            Ok(true)
        }
    }

    struct NativeLeastSquares;

    impl LeastSquaresProblem<f64> for NativeLeastSquares {
        fn param_dim(&self) -> usize {
            2
        }

        fn residual_dim(&self) -> usize {
            2
        }

        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0]);
            Ok(())
        }

        fn jacobian_into(
            &self,
            _x: &[f64],
            output: &mut Matrix<f64>,
        ) -> Result<bool, SciOxideError> {
            output.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            Ok(true)
        }
    }

    #[test]
    fn shapes_are_checked_before_closures_run() {
        let calls = core::cell::Cell::new(0);
        let objective = ObjectiveFn::new(2, |_: &[f64]| {
            calls.set(calls.get() + 1);
            Ok(0.0)
        })
        .with_gradient(|_, g| {
            calls.set(calls.get() + 1);
            g.fill(1.0);
            Ok(())
        })
        .with_hessian(|_, h| {
            calls.set(calls.get() + 1);
            h.as_mut_slice().fill(1.0);
            Ok(())
        })
        .with_hessian_vector_product(|_, _, p| {
            calls.set(calls.get() + 1);
            p.fill(1.0);
            Ok(())
        });
        let mut output = [7.0];
        let mut matrix = Matrix::from_vec(1, 1, alloc::vec![7.0]).unwrap();
        for error in [
            objective.eval(&[1.0]).unwrap_err(),
            objective
                .gradient_into(&[1.0, 2.0], &mut output)
                .unwrap_err(),
            objective
                .hessian_into(&[1.0, 2.0], &mut matrix)
                .unwrap_err(),
            objective
                .hessian_vector_product_into(&[1.0, 2.0], &[1.0], &mut [0.0, 0.0])
                .unwrap_err(),
            objective
                .hessian_vector_product_into(&[1.0, 2.0], &[1.0, 2.0], &mut output)
                .unwrap_err(),
        ] {
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
        }
        let residual = LeastSquaresFn::new(2, 2, |_: &[f64], _: &mut [f64]| {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            residual
                .residual_into(&[0.0, 0.0], &mut output)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let constraint = ConstraintFn::new(2, |_: &[f64]| {
            calls.set(calls.get() + 1);
            Ok(0.0)
        })
        .with_jacobian(|_, g| {
            calls.set(calls.get() + 1);
            g.fill(1.0);
            Ok(())
        });
        assert!(constraint.jacobian_into(&[0.0, 0.0], &mut output).is_err());
        assert_eq!(calls.get(), 0);
        assert_eq!(output, [7.0]);
    }

    #[test]
    fn absent_derivatives_and_hessian_products() {
        let objective = ObjectiveFn::new(2, |_: &[f64]| Ok(0.0));
        let x = [0.0, 0.0];
        let mut g = [0.0, 0.0];
        let mut h = Matrix::from_vec(2, 2, alloc::vec![0.0;4]).unwrap();
        assert!(!objective.gradient_into(&x, &mut g).unwrap());
        assert!(!objective.automatic_gradient_into(&x, &mut g).unwrap());
        assert!(!objective.hessian_into(&x, &mut h).unwrap());
        assert!(!objective.automatic_hessian_into(&x, &mut h).unwrap());
        assert!(!objective.has_hessian_vector_product());
        assert!(
            objective
                .hessian_vector_product_into(&x, &x, &mut g)
                .is_err()
        );
        let objective = objective.with_hessian_vector_product(|_, d, out| {
            out[0] = 2.0 * d[0];
            out[1] = 0.0;
            Ok(())
        });
        assert!(objective.has_hessian_vector_product());
        objective
            .hessian_vector_product_into(&x, &[3.0, 4.0], &mut g)
            .unwrap();
        assert_eq!(g, [6.0, 0.0]);
        let constraint = ConstraintFn::new(2, |_: &[f64]| Ok(0.0));
        assert!(!constraint.jacobian_into(&x, &mut g).unwrap());
        assert!(!constraint.automatic_jacobian_into(&x, &mut g).unwrap());
        let problem = LeastSquaresFn::new(2, 2, |_: &[f64], _: &mut [f64]| Ok(())).unwrap();
        assert!(!problem.jacobian_into(&x, &mut h).unwrap());
        assert!(!problem.automatic_jacobian_into(&x, &mut h).unwrap());
    }

    #[test]
    fn automatic_callbacks_write_the_supplied_storage() {
        let x = [3.0, 4.0];
        let mut g = [0.0, 0.0];
        let mut h = Matrix::from_vec(2, 2, alloc::vec![0.0;4]).unwrap();
        let objective = ObjectiveFn::new(2, |_: &[f64]| Ok(0.0))
            .with_automatic_gradient(|x, g| {
                g.copy_from_slice(&[2.0 * x[0], 1.0]);
                Ok(())
            })
            .with_automatic_hessian(|_, h| {
                h.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 0.0]);
                Ok(())
            });
        assert!(objective.automatic_gradient_into(&x, &mut g).unwrap());
        assert_eq!(Some(g.to_vec()), NativeObjective.gradient(&x).unwrap());
        assert!(objective.automatic_hessian_into(&x, &mut h).unwrap());
        assert_eq!(Some(h.clone()), NativeObjective.hessian(&x).unwrap());
        let constraint = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1]))
            .with_jacobian(|_, g| {
                g.fill(1.0);
                Ok(())
            })
            .with_automatic_jacobian(|_, g| {
                g.fill(1.0);
                Ok(())
            });
        assert!((constraint.eval(&x).unwrap() - 7.0).abs() < 1e-14);
        assert!(constraint.jacobian_into(&x, &mut g).unwrap());
        assert_eq!(g, [1.0, 1.0]);
        assert!(constraint.automatic_jacobian_into(&x, &mut g).unwrap());
        assert_eq!(g, [1.0, 1.0]);
        let problem = LeastSquaresFn::new(2, 2, |_: &[f64], _: &mut [f64]| Ok(()))
            .unwrap()
            .with_automatic_jacobian(|_, j| {
                j.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            });
        assert!(problem.automatic_jacobian_into(&x, &mut h).unwrap());
        assert_eq!(Some(h), NativeLeastSquares.jacobian(&x).unwrap());
    }

    #[test]
    fn callback_error_is_not_converted_to_a_domain_value() {
        let objective = ObjectiveFn::new(1, |_: &[f64]| {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "injected failure",
            ))
        });

        assert!(matches!(
            (objective.eval(&[0.0]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }

    #[test]
    fn zero_dimension_and_values_are_passed_through() {
        let objective = ObjectiveFn::new(0, |_: &[f64]| Ok(f64::NAN));
        assert!(objective.eval(&[]).unwrap().is_nan());
        let constraint = ConstraintFn::new(0, |_: &[f64]| Ok(f64::INFINITY));
        assert!(constraint.eval(&[]).unwrap().is_infinite());
        assert!(LeastSquaresFn::new(0, 1, |_: &[f64], _: &mut [f64]| Ok(())).is_err());
        assert!(LeastSquaresFn::new(1, 0, |_: &[f64], _: &mut [f64]| Ok(())).is_err());
    }

    #[test]
    fn pure_and_native_problem_callbacks_are_equivalent() {
        let pure_objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1]))
            .with_gradient(|x: &[f64], output| {
                output.copy_from_slice(&[2.0 * x[0], 1.0]);
                Ok(())
            })
            .with_hessian(|_: &[f64], output| {
                output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 0.0]);
                Ok(())
            });
        let point = [3.0, 4.0];
        assert_eq!(
            pure_objective.eval(&point).unwrap(),
            NativeObjective.eval(&point).unwrap()
        );
        assert_eq!(
            pure_objective.gradient(&point).unwrap(),
            NativeObjective.gradient(&point).unwrap()
        );
        assert_eq!(
            pure_objective.hessian(&point).unwrap(),
            NativeObjective.hessian(&point).unwrap()
        );

        let pure_least_squares = LeastSquaresFn::new(2, 2, |x: &[f64], output| {
            output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0]);
            Ok(())
        })
        .unwrap()
        .with_jacobian(|_: &[f64], output| {
            output.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        });
        assert_eq!(
            pure_least_squares.residual(&point).unwrap(),
            NativeLeastSquares.residual(&point).unwrap()
        );
        assert_eq!(
            pure_least_squares.jacobian(&point).unwrap(),
            NativeLeastSquares.jacobian(&point).unwrap()
        );
    }
}
