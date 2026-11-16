//! Neville's algorithm for polynomial interpolation with an error estimate.

#![allow(clippy::many_single_char_names)]

use super::tableau::tableau;
use alloc::vec::Vec;

use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, check_value_shape,
    write_value,
};

use crate::checked::{intermediate, output};
use crate::protocol::{
    ExtrapolationPolicy, Interpolant, NodeSeparation, SampleOptions, domain_from_nodes,
    resolve_coordinate,
};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, gather, zeroed};
use scioxide_poly::backend::polynomial_basis::linear_difference_ratio;

/// Evaluates the interpolating polynomial of degree at most `n − 1` with Neville's algorithm.
///
/// Each query builds the tableau of interpolants on consecutive nodes from
/// scratch, as corrections `C` and `D` between successive orders, starting at
/// the node nearest the query. Nothing is precomputed, so construction is
/// cheap and a query costs O(n² d) operations for `d` real components. For
/// many queries on fixed data, [`crate::BarycentricInterpolator`] costs O(n d)
/// per query.
///
/// [`Self::evaluate_with_error_into`] also returns the magnitude of the last
/// correction, which estimates the interpolation error at the query. It is an
/// estimate, not a bound. A query that coincides with a node under the
/// [`NodeSeparation`] returns that node's value with error estimate zero. With
/// [`ExtrapolationPolicy::MethodDefined`] the polynomial is evaluated outside
/// the node interval.
///
/// # Errors
///
/// Queries fail as described in the [crate docs](crate#errors), and with
/// `NonFinite` if an intermediate or the result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{ExtrapolationPolicy, Interpolant, NevilleInterpolator};
///
/// let neville = NevilleInterpolator::new(
///     vec![0.0_f64, 1.0, 2.0, 3.0],
///     vec![0.0_f64, 1.0, 4.0, 9.0], // y = x^2
/// )?;
/// let mut workspace = neville.workspace()?;
/// let (mut value, mut error) = (0.0, 0.0);
/// neville.evaluate_with_error_into(
///     [1.5],
///     ExtrapolationPolicy::Reject,
///     &mut value,
///     &mut error,
///     &mut workspace,
/// )?;
/// assert!((value - 2.25_f64).abs() < 1e-14);
/// assert!(error.abs() < 1e-14);
/// assert_eq!(neville.evaluate([1.5], ExtrapolationPolicy::Reject)?, value);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed.,
/// Springer, 2002, §2.1.
#[derive(Debug, Clone)]
#[must_use]
pub struct NevilleInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    nodes: Vec<S>,
    /// `[node][component]` values.
    values: Vec<S>,
    template: V,
    dimension: usize,
    separation: NodeSeparation<S>,
    domain: Interval<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> NevilleInterpolator<S, V> {
    /// Stores the samples `(x[j], y[j])`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// must be strictly increasing.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty or not strictly increasing.
    /// - `NonFinite` if a node or value is NaN or infinite.
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, SampleOptions::new())
    }

    /// Stores the samples `(x[j], y[j])`, validating the nodes under `options`.
    ///
    /// The nodes must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`]. A query within that separation of a
    /// node returns the node's value.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty, or not strictly increasing and distinct
    ///   under the separation of `options`.
    /// - `NonFinite` if a node or value is NaN or infinite.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let separation = options.node_separation();
        if x.len() != y.len() {
            return Err(crate::error::shape(x.len(), y.len()));
        }
        validate_nodes(&x, 1, separation, scioxide_core::argument!("x"))?;
        let flat = flatten(y, x.len())?;
        Ok(Self {
            domain: domain_from_nodes(&x)?,
            nodes: x,
            values: flat.components,
            template: flat.template,
            dimension: flat.dimension,
            separation,
        })
    }

    /// Returns the nodes in increasing order.
    pub fn nodes(&self) -> &[S] {
        &self.nodes
    }

    /// Returns the values as real components, node by node (`[node][component]`).
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Evaluates the interpolant and an estimate of its error at `point`.
    ///
    /// `value_out` and `error_out` each hold [`Interpolant::value_dimension`]
    /// components. The error estimate of a component is the magnitude of its
    /// last tableau correction. It is zero at a node and for a single sample.
    ///
    /// # Errors
    ///
    /// - The query errors described in the [crate docs](crate#errors).
    /// - `NonFinite` if an intermediate or the result overflows.
    pub fn evaluate_with_error_into<O, P>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        value_out: &mut O,
        error_out: &mut P,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError>
    where
        O: AffineValueMut<Scale = S> + ?Sized,
        P: AffineValueMut<Scale = S> + ?Sized,
    {
        check_value_shape(value_out, self.dimension)?;
        check_value_shape(error_out, self.dimension)?;
        let scratch = workspace.scratch(self.scratch_len())?;
        let (values, rest) = scratch.split_at_mut(self.dimension);
        let (errors, tables) = rest.split_at_mut(self.dimension);
        self.stage(point, policy, values, Some(errors), tables)?;
        write_value(values, value_out)?;
        write_value(errors, error_out)
    }

    /// Evaluates the interpolant and an estimate of its error at `point`,
    /// returning `(value, error)`.
    ///
    /// This is [`Self::evaluate_with_error_into`] with newly allocated outputs
    /// and scratch, and returns its results bit for bit.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_with_error_into`].
    /// - `AllocationFailure` if the outputs or the scratch cannot be allocated.
    ///
    /// ```
    /// use scioxide_interp::{ExtrapolationPolicy, NevilleInterpolator};
    /// // y = x² on four nodes: the last correction, of degree three, vanishes.
    /// let neville = NevilleInterpolator::new(vec![0.0_f64, 1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0, 9.0])?;
    /// let (value, error) = neville.evaluate_with_error([1.5], ExtrapolationPolicy::Reject)?;
    /// assert!((value - 2.25).abs() < 1e-14 && error < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_with_error(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<(V::Owned, V::Owned), SciOxideError> {
        let mut value = zeroed(&self.template)?;
        let mut error = zeroed(&self.template)?;
        let mut workspace = QueryWorkspace::new(self.scratch_len())?;
        self.evaluate_with_error_into(point, policy, &mut value, &mut error, &mut workspace)?;
        Ok((value, error))
    }

    /// Scratch length: values and errors plus two correction tables.
    fn scratch_len(&self) -> usize {
        2 * self.dimension + 2 * self.nodes.len()
    }

    /// Computes every component's value, and its error estimate when requested.
    fn stage(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        values: &mut [S],
        mut errors: Option<&mut [S]>,
        tables: &mut [S],
    ) -> Result<(), SciOxideError> {
        let t = resolve_coordinate(&self.domain, point[0], policy)?;
        let n = self.nodes.len();
        let (c, rest) = tables.split_at_mut(n);
        let d = &mut rest[..n];
        for (component, slot) in values.iter_mut().enumerate() {
            gather(&self.values, self.dimension, component, c);
            d.copy_from_slice(c);
            let (value, error) = neville_kernel(&self.nodes, t, self.separation, c, d)?;
            *slot = value;
            if let Some(errors) = errors.as_deref_mut() {
                errors[component] = error;
            }
        }
        Ok(())
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for NevilleInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 1] {
        [self.domain]
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.scratch_len())
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.scratch_len())
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let scratch = workspace.scratch(self.dimension + 2 * self.nodes.len())?;
        let (values, tables) = scratch.split_at_mut(self.dimension);
        self.stage(point, policy, values, None, tables)?;
        write_value(values, out)
    }
}

/// Neville correction tableau for one component; `c` and `d` hold its samples.
fn neville_kernel<S: RealScalar>(
    x: &[S],
    t: S,
    separation: NodeSeparation<S>,
    c: &mut [S],
    d: &mut [S],
) -> Result<(S, S), SciOxideError> {
    // With w = C_{m-1,i+1} - D_{m-1,i}:
    //   C_{m,i} = (x_i - t) w / (x_i - x_{i+m}),
    //   D_{m,i} = (x_{i+m} - t) w / (x_i - x_{i+m}).
    // `linear_difference_ratio` evaluates each ratio of differences without
    // first forming a possibly overflowing raw difference.
    let (value, correction) = tableau(x, t, separation, c, d, |c, d, i, m| {
        let w_val = intermediate(c[i + 1] - d[i])?;
        let c_ratio = linear_difference_ratio(x[i], t, x[i], x[i + m])?;
        let d_ratio = linear_difference_ratio(x[i + m], t, x[i], x[i + m])?;
        c[i] = intermediate(w_val * c_ratio)?;
        d[i] = intermediate(w_val * d_ratio)?;
        Ok(())
    })?;
    // The last, most centred correction is the error estimate.
    Ok((value, output(correction.abs())?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};
    // The crate's unit-test allocator is installed in the barycentric tests.

    use scioxide_core::Complex;

    fn with_error(x: &[f64], y: &[f64], t: f64) -> Result<(f64, f64), SciOxideError> {
        let neville = NevilleInterpolator::new(x.to_vec(), y.to_vec())?;
        let mut workspace = neville.workspace()?;
        let (mut value, mut error) = (0.0, 0.0);
        neville.evaluate_with_error_into(
            [t],
            ExtrapolationPolicy::Reject,
            &mut value,
            &mut error,
            &mut workspace,
        )?;
        Ok((value, error))
    }

    #[test]
    fn test_interpolate_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 5.0];
        for (xi, yi) in x.iter().zip(y.iter()) {
            let (v, _err) = with_error(&x, &y, *xi).unwrap();
            assert_close(v, *yi, 5e-13, 5e-13);
        }
    }

    #[test]
    fn test_quadratic_exact() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let (v, err) = with_error(&x, &y, 1.5).unwrap();
        assert_close(v, 2.25, 5e-14, 5e-14);
        assert!(err < 1e-12);
    }

    #[test]
    fn test_single_point() {
        let (v, err) = with_error(&[1.0], &[5.0], 1.0).unwrap();
        assert_close(v, 5.0, 5e-15, 5e-15);
        assert_close(err, 0.0, 5e-15, 5e-15);
    }

    #[test]
    fn test_empty_input() {
        let result = NevilleInterpolator::<f64>::new(vec![], vec![]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_dimension_mismatch() {
        let result = with_error(&[0.0, 1.0], &[1.0], 0.5);
        assert_error(&result, scioxide_core::SciOxideErrorKind::DimensionMismatch);
    }

    #[test]
    fn test_nan_query() {
        let result = with_error(&[0.0, 1.0], &[0.0, 1.0], f64::NAN);
        assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_two_points_midpoint() {
        let (v, _err) = with_error(&[0.0, 2.0], &[1.0, 5.0], 1.0).unwrap();
        assert_close(v, 3.0, 5e-15, 5e-15);
    }

    #[test]
    fn policies_errors_and_reuse_after_failure() {
        let neville = NevilleInterpolator::new(vec![0.0_f64, 1.0], vec![1.0, 3.0]).unwrap();
        let mut workspace = neville.workspace().unwrap();
        let (mut value, mut error) = (7.0, 8.0);
        let mut run = |policy, value: &mut f64, error: &mut f64| {
            neville.evaluate_with_error_into([2.0], policy, value, error, &mut workspace)
        };
        assert!(run(ExtrapolationPolicy::Reject, &mut value, &mut error).is_err());
        assert_eq!((value, error), (7.0, 8.0));
        run(ExtrapolationPolicy::Clamp, &mut value, &mut error).unwrap();
        assert_eq!((value, error), (3.0, 0.0));
        run(ExtrapolationPolicy::MethodDefined, &mut value, &mut error).unwrap();
        assert_close(value, 5.0, 5e-15, 5e-15);
        let mut short = [0.0; 2];
        assert!(
            neville
                .evaluate_with_error_into(
                    [0.5],
                    ExtrapolationPolicy::Reject,
                    &mut value,
                    &mut short,
                    &mut workspace
                )
                .is_err()
        );
    }

    #[test]
    fn undersized_workspace_is_a_dimension_mismatch_and_queries_do_not_allocate() {
        let small = NevilleInterpolator::new(vec![0.0_f64], vec![1.0]).unwrap();
        let nodes: Vec<f64> = (0..12).map(f64::from).collect();
        let large = NevilleInterpolator::new(nodes.clone(), nodes).unwrap();
        let mut workspace = small.workspace().unwrap();
        let mut out = 0.0;
        assert_eq!(
            large
                .evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut workspace)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let mut workspace = large.workspace().unwrap();
        {
            large
                .evaluate_into([3.5], ExtrapolationPolicy::Reject, &mut out, &mut workspace)
                .unwrap();
        };
        assert_close(out, 3.5, 5e-13, 5e-13);
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        let x = vec![0.0_f64, 0.5, 1.5, 3.0];
        let re = [1.0, -1.0, 2.0, 0.25];
        let im = [3.0, 0.5, -2.0, 1.0];
        let values: Vec<Complex<f64>> = re
            .iter()
            .zip(&im)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect();
        let complex = NevilleInterpolator::new(x.clone(), values).unwrap();
        let mut workspace = complex.workspace().unwrap();
        for query in [0.25, 1.0, 2.9] {
            let (mut value, mut error) = (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            complex
                .evaluate_with_error_into(
                    [query],
                    ExtrapolationPolicy::Reject,
                    &mut value,
                    &mut error,
                    &mut workspace,
                )
                .unwrap();
            let (real, real_error) = with_error(&x, &re, query).unwrap();
            let (imaginary, imaginary_error) = with_error(&x, &im, query).unwrap();
            assert_eq!((value.re, error.re), (real, real_error));
            assert_eq!((value.im, error.im), (imaginary, imaginary_error));
        }
    }
}
