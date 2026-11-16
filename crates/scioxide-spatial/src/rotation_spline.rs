//! Interpolation of rotations: spherical linear interpolation and cubic
//! rotation splines.
use crate::math3::{Mat, add, cross, dot, mul, n, norm, product, scale, sub};
use crate::validation::{finite, input, out_of_range as invalid};
use crate::{
    Rotation, SpatialExecutionContext, TransformOptions, TransformWorkspace,
    TransformWorkspaceRequest,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    BudgetKind, ExecutionContext, RealScalar, ResidualTolerance, SciOxideError, SolveBudgets,
    SolveContext, SolveScope,
};
mod factor;
use factor::Factor;
use scioxide_core::backend::{KernelBackend, KernelDispatch};

fn matrix<S: RealScalar>(action: impl Fn([S; 3]) -> [S; 3]) -> Mat<S> {
    let columns: [[_; 3]; 3] = core::array::from_fn(|i| {
        let mut e = [S::ZERO; 3];
        e[i] = S::ONE;
        action(e)
    });
    core::array::from_fn(|i| core::array::from_fn(|j| columns[j][i]))
}
// SO(3) right-Jacobian coefficients and derivatives. The series avoids subtracting
// nearby trigonometric values; derivative coefficients are derived from the same series.
//
// j1 = (1 - cos t) / t^2 and j2 = (t - sin t) / t^3 are the coefficients of
// [r]x and [r]x^2 in J(r). k1 and k2 are the matching (1/t) d/dt terms,
// with the sign convention `nonlinear` expects. Below t = 1 the alternating
// Taylor series converges fast. Twelve terms reach t^24 / 26!, far below f64
// epsilon.
fn coefficients<S: RealScalar>(theta: S) -> [S; 4] {
    let x = theta * theta;
    if theta <= S::ONE {
        let mut a = S::HALF;
        let mut b = S::ONE / n::<S>(6);
        let mut j1 = a;
        let mut j2 = b;
        let mut k1 = S::ZERO;
        let mut k2 = S::ZERO;
        let mut power = S::ONE;
        for k in 1..=12u16 {
            a = -a / n::<S>((2 * k + 1) * (2 * k + 2));
            b = -b / n::<S>((2 * k + 2) * (2 * k + 3));
            k1 -= n::<S>(2 * k) * a * power;
            k2 += n::<S>(2 * k) * b * power;
            power *= x;
            j1 += a * power;
            j2 += b * power;
        }
        [j1, j2, k1, k2]
    } else {
        let (s, c) = theta.sin_cos();
        [
            (S::ONE - c) / x,
            (theta - s) / (x * theta),
            (-theta * s - n::<S>(2) * (c - S::ONE)) / (x * x),
            (-n::<S>(2) * theta + n::<S>(3) * s - theta * c) / (x * x * theta),
        ]
    }
}
// Body angular velocity of R_i exp(r(t)): omega = J(r) r_dot, where
// J(r) = I - j1 [r]x + j2 [r]x^2. The cross products apply [r]x to a vector
// without forming a matrix.
fn jacobian<S: RealScalar>(r: [S; 3], v: [S; 3]) -> [S; 3] {
    let [a, b, _, _] = coefficients(norm(r));
    let c = cross(r, v);
    add(sub(v, scale(c, a)), scale(cross(r, c), b))
}
// J(r)^-1 = I + [r]x / 2 + (1 - (t/2) cot(t/2)) / t^2 [r]x^2. The last
// coefficient cancels catastrophically as t -> 0, so a Taylor series
// (1/12 + t^2/720 + t^4/30240 + t^6/1209600) replaces it below t = 0.1.
fn inverse_jacobian<S: RealScalar>(r: [S; 3], v: [S; 3]) -> [S; 3] {
    let theta = norm(r);
    let x = theta * theta;
    let a = if theta < S::ONE / n::<S>(10) {
        S::ONE / n::<S>(12)
            + x / n::<S>(720)
            + x * x / n::<S>(30240)
            + x * x * x / (n::<S>(30240) * n::<S>(40))
    } else {
        (S::ONE - theta * S::HALF / (theta * S::HALF).tan()) / x
    };
    let c = cross(r, v);
    add(add(v, scale(c, S::HALF)), scale(cross(r, c), a))
}
// The (dJ/dt) r_dot term in omega_dot = J(r) r_ddot + (dJ/dt) r_dot. It is
// the only part of angular acceleration that is nonlinear in the rate
// unknowns.
fn nonlinear<S: RealScalar>(r: [S; 3], v: [S; 3]) -> [S; 3] {
    let [_, b, k1, k2] = coefficients(norm(r));
    let c = cross(r, v);
    add(
        scale(add(scale(c, k1), scale(cross(r, c), k2)), dot(r, v)),
        scale(cross(v, c), b),
    )
}
#[derive(Debug, Clone)]
struct Knots<S: RealScalar> {
    times: Vec<S>,
    rotations: Vec<Rotation<S>>,
    vectors: Vec<[S; 3]>,
}
impl<S: RealScalar> Knots<S> {
    // Validates `times` and `rotations`.
    fn new(times: &[S], rotations: &[Rotation<S>]) -> Result<Self, SciOxideError> {
        const TIMES: scioxide_core::Argument = scioxide_core::argument!("times");
        if times.len() < 2 {
            return Err(invalid(TIMES, "interpolation needs at least two times"));
        }
        if times.len() != rotations.len() {
            return Err(SciOxideError::dimension_mismatch(
                [times.len()],
                [rotations.len()],
            ));
        }
        for pair in times.windows(2) {
            input([pair[0], pair[1]])?;
            finite([pair[1] - pair[0]])?;
            if pair[1] <= pair[0] {
                return Err(invalid(TIMES, "rotation times must be strictly increasing"));
            }
        }
        let mut saved_times =
            try_filled(times.len(), S::ZERO, scioxide_core::quantity!("knot_times"))?;
        saved_times.copy_from_slice(times);
        let mut saved_rotations = try_filled(
            rotations.len(),
            Rotation::identity(),
            scioxide_core::quantity!("knot_rotations"),
        )?;
        saved_rotations.copy_from_slice(rotations);
        let mut vectors = try_filled(
            times.len() - 1,
            [S::ZERO; 3],
            scioxide_core::quantity!("knot_rotation_vectors"),
        )?;
        // Each segment is parameterized by its relative rotation
        // R_i^-1 R_{i+1}. The principal rotation vector (angle <= pi) makes
        // interpolation follow the shortest arc.
        for (v, pair) in vectors.iter_mut().zip(rotations.windows(2)) {
            *v = pair[0].inverse().compose(pair[1])?.as_rotation_vector();
        }
        Ok(Self {
            times: saved_times,
            rotations: saved_rotations,
            vectors,
        })
    }
    fn rotations_into(
        &self,
        times: &[S],
        options: TransformOptions,
        output: &mut [Rotation<S>],
        workspace: &mut TransformWorkspace<S>,
        context: ExecutionContext<'_>,
        orientation_vector: impl Fn(usize, S) -> [S; 3] + Sync,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        if output.len() != times.len() {
            return Err(SciOxideError::dimension_mismatch(
                [times.len()],
                [output.len()],
            ));
        }
        workspace.require(TransformWorkspaceRequest::rotations(times.len(), options))?;
        let first = self.times[0];
        let last = self.times[self.times.len() - 1];
        for block in times.chunks(256) {
            context.check()?;
            for &time in block {
                input([time])?;
                if time < first || time > last {
                    return Err(invalid(
                        scioxide_core::argument!("times"),
                        "rotation interpolation query outside knot interval",
                    ));
                }
            }
        }
        let staging = &mut workspace.rotations[..times.len()];
        let policy = options.execution_policy();
        // Auto estimate: knot search plus an exponential map per time.
        crate::batch::dispatch_chunks(
            staging,
            policy,
            context,
            crate::batch::work(128),
            |offset, values| {
                for (block, values) in values.chunks_mut(256).enumerate() {
                    context.check()?;
                    let start = offset + block * 256;
                    for (value, &time) in values.iter_mut().zip(&times[start..]) {
                        let (i, u) = self.locate_valid(time);
                        let r = finite(orientation_vector(i, u))?;
                        *value = self.rotation(i, u, r)?;
                    }
                }
                Ok(())
            },
        )?;
        context.check()?;
        output.copy_from_slice(staging);
        Ok(())
    }
    // Locates the `time` argument.
    fn locate(&self, time: S) -> Result<(usize, S), SciOxideError> {
        input([time])?;
        if time < self.times[0] || time > self.times[self.times.len() - 1] {
            return Err(invalid(
                scioxide_core::argument!("time"),
                "rotation interpolation query outside knot interval",
            ));
        }
        Ok(self.locate_valid(time))
    }
    // The batch boundary has proved the closed interval and finiteness.
    fn locate_valid(&self, time: S) -> (usize, S) {
        let i = self
            .times
            .partition_point(|&x| x <= time)
            .saturating_sub(1)
            .min(self.times.len() - 2);
        (
            i,
            (time - self.times[i]) / (self.times[i + 1] - self.times[i]),
        )
    }
    fn rotation(&self, i: usize, u: S, r: [S; 3]) -> Result<Rotation<S>, SciOxideError> {
        // Knots are returned exactly. Composing with exp(r) at u = 1 would
        // only reproduce R_{i+1} up to rounding.
        if u == S::ZERO {
            Ok(self.rotations[i])
        } else if u == S::ONE {
            Ok(self.rotations[i + 1])
        } else {
            self.rotations[i].compose(Rotation::from_rotation_vector(r)?)
        }
    }
}
/// Spherical linear interpolation (slerp) between rotations given at knot
/// times.
///
/// Between knots `tᵢ < tᵢ₊₁` with rotations `Rᵢ` and `Rᵢ₊₁`, the interpolant is
/// `R(t) = Rᵢ exp(u vᵢ)`, where `u = (t − tᵢ) / (tᵢ₊₁ − tᵢ)` and `vᵢ` is the
/// rotation vector of `Rᵢ⁻¹ Rᵢ₊₁`, of angle at most π. This is rotation at
/// constant angular velocity along the shorter arc; for an exact half turn
/// the sign convention of [`Rotation`] picks the direction. At the knots the
/// given rotations are returned exactly. Storage is O(n) for `n` knots, and
/// evaluation costs O(log n) without allocation.
///
/// # References
///
/// K. Shoemake, "Animating rotation with quaternion curves", *SIGGRAPH
/// Comput. Graph.* 19(3), 1985.
#[derive(Debug, Clone)]
pub struct Slerp<S: RealScalar> {
    knots: Knots<S>,
}
impl<S: RealScalar> Slerp<S> {
    /// Creates the interpolant through `rotations[i]` at `times[i]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two times or the times are
    ///   not strictly increasing.
    /// - `NonFinite` if a time is NaN or infinite.
    /// - `DimensionMismatch` if `times` and `rotations` differ in length.
    /// - `AllocationFailure` if storage cannot be reserved.
    ///
    /// ```
    /// use scioxide_spatial::{Rotation, Slerp};
    /// let interpolant = Slerp::new(&[0.0_f64, 1.0], &[Rotation::identity(); 2])?;
    /// assert_eq!(interpolant.evaluate(0.5)?, Rotation::identity());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(times: &[S], rotations: &[Rotation<S>]) -> Result<Self, SciOxideError> {
        Ok(Self {
            knots: Knots::new(times, rotations)?,
        })
    }
    /// Returns the interpolated rotation at `time`, which must lie between the
    /// first and last knot. There is no extrapolation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `time` is NaN or infinite.
    /// - `InvalidInput` if `time` lies outside the knot interval.
    ///
    /// ```
    /// use scioxide_spatial::{Rotation, Slerp};
    /// let interpolant = Slerp::new(&[0.0_f64, 1.0], &[Rotation::identity(); 2])?;
    /// assert_eq!(interpolant.evaluate(0.5)?, Rotation::identity());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, time: S) -> Result<Rotation<S>, SciOxideError> {
        let (i, u) = self.knots.locate(time)?;
        self.knots.rotation(i, u, scale(self.knots.vectors[i], u))
    }
    /// Evaluates the interpolant at many times and writes the rotations to
    /// `output`.
    ///
    /// The times may be in any order and may repeat. `output` must have one
    /// entry per time, and `workspace` must meet a
    /// [`TransformWorkspaceRequest::rotations`] request for that count. See the
    /// [crate documentation](crate#execution) for parallel execution and
    /// cancellation.
    ///
    /// # Errors
    ///
    /// - As [`Self::evaluate`] for each time.
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace`
    ///   is too small.
    /// - `ResourceLimit` if the workspace exceeds its byte limit.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_spatial::{
    ///     Rotation, RotationSpline, RotationSplineOptions, Slerp, TransformOptions, TransformWorkspace,
    ///     TransformWorkspaceRequest,
    /// };
    /// let interpolant = Slerp::new(&[0.0_f64, 1.0], &[Rotation::identity(); 2])?;
    /// let options = TransformOptions::new();
    /// let mut workspace =
    ///     TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(3, options))?;
    /// let mut output = [Rotation::identity(); 3];
    /// interpolant.evaluate_rotations_into(
    ///     &[1.0, 0.5, 0.0],
    ///     options,
    ///     &mut output,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(output, [Rotation::identity(); 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_rotations_into<'scope, 'capability>(
        &self,
        times: &[S],
        options: TransformOptions,
        output: &mut [Rotation<S>],
        workspace: &mut TransformWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.knots.rotations_into(
            times,
            options,
            output,
            workspace,
            context.into().execution(),
            |i, u| scale(self.knots.vectors[i], u),
        )
    }
}
/// Options for [`RotationSpline::build`]: the iteration limit and the
/// continuity tolerance.
///
/// Batch evaluation uses [`TransformOptions`] instead.
///
/// ```
/// use scioxide_core::ResidualTolerance;
/// use scioxide_spatial::RotationSplineOptions;
/// let options = RotationSplineOptions::<f64>::new()
///     .with_max_iterations(64)
///     .with_continuity_tolerance(ResidualTolerance::default());
/// assert_eq!(options.max_iterations(), 64);
/// let _ = options.continuity_tolerance();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RotationSplineOptions<S: RealScalar> {
    /// Maximum fixed-point iterations; zero allows none.
    max_iterations: usize,
    /// Tolerance on the jump in angular acceleration at interior knots.
    continuity_tolerance: ResidualTolerance<S>,
}
impl<S: RealScalar> Default for RotationSplineOptions<S> {
    fn default() -> Self {
        // An absolute part is measured in radians per squared time unit, so a
        // nonzero default would make acceptance depend on the time unit.
        let relative = scioxide_core::Tolerance::from_sqrt_epsilon()
            .with_absolute(S::ZERO)
            .unwrap_or_else(|_| scioxide_core::Tolerance::from_sqrt_epsilon());
        Self {
            max_iterations: 32,
            continuity_tolerance: ResidualTolerance::from_tolerance(relative),
        }
    }
}
impl<S: RealScalar> RotationSplineOptions<S> {
    /// Returns the defaults: at most 32 iterations, and a relative continuity
    /// tolerance of `√ε` with no absolute part, so that acceptance does not
    /// depend on the time unit. An absolute part, if set, is in radians per
    /// squared time unit.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the iteration limit. With zero, preparation succeeds only if the
    /// initial guess is already continuous.
    pub const fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }
    /// Sets the tolerance on the jump in angular acceleration.
    pub const fn with_continuity_tolerance(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.continuity_tolerance = tolerance;
        self
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }
    /// Returns the continuity tolerance.
    pub const fn continuity_tolerance(self) -> ResidualTolerance<S> {
        self.continuity_tolerance
    }
}
/// How [`RotationSpline::build`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationSplineTermination {
    /// The angular acceleration is continuous within the tolerance.
    Converged,
    /// The iteration limit was reached first; no spline is returned.
    IterationLimit,
}
/// Result of [`RotationSpline::build`].
#[derive(Debug, Clone)]
pub struct RotationSplineResult<S: RealScalar> {
    /// The spline, present only when `termination` is `Converged`.
    pub spline: Option<RotationSpline<S>>,
    /// How preparation ended.
    pub termination: RotationSplineTermination,
    /// Number of fixed-point iterations performed.
    pub iterations: usize,
    /// Largest jump in angular acceleration at an interior knot; zero for two
    /// knots.
    pub continuity_residual: S,
}
/// Value and derivatives of a [`RotationSpline`] at one time, in the time unit
/// of the knots.
#[derive(Debug, Clone, Copy)]
pub struct RotationSplineSample<S: RealScalar> {
    /// Interpolated rotation.
    pub rotation: Rotation<S>,
    /// Body-frame angular velocity, radians per time unit.
    pub angular_velocity: [S; 3],
    /// Body-frame angular acceleration, radians per squared time unit.
    pub angular_acceleration: [S; 3],
}
/// Cubic rotation spline with continuous angular velocity and acceleration.
///
/// Between knots `tᵢ < tᵢ₊₁` the spline is `R(t) = Rᵢ exp(rᵢ(t))`, where
/// `rᵢ` is a cubic polynomial in `t` with `rᵢ(tᵢ) = 0` and `exp(rᵢ(tᵢ₊₁)) =
/// Rᵢ⁻¹ Rᵢ₊₁`. The coefficients are chosen so that angular velocity and
/// acceleration are continuous at the knots, which requires a nonlinear
/// solve; see [`Self::build`]. Storage is O(n) for `n` knots, and
/// evaluation costs O(log n) without allocation.
#[derive(Debug, Clone)]
pub struct RotationSpline<S: RealScalar> {
    knots: Knots<S>,
    coefficients: Vec<[[S; 3]; 3]>,
}
impl<S: RealScalar> RotationSpline<S> {
    /// Returns the rotation and its body-frame angular velocity and
    /// acceleration at `time`, which must lie between the first and last knot.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `time` is NaN or infinite.
    /// - `InvalidInput` if `time` lies outside the knot interval.
    /// - `NumericalInstability` if a derivative is not representable.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_spatial::{Rotation, RotationSpline, RotationSplineOptions};
    /// let result = RotationSpline::build(
    ///     &[0.0_f64, 1.0],
    ///     &[Rotation::identity(); 2],
    ///     RotationSplineOptions::new(),
    ///     ExecutionContext::sequential(),
    /// )?;
    /// let spline = result.spline.unwrap();
    /// assert_eq!(spline.evaluate(0.5)?.rotation, Rotation::identity());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, time: S) -> Result<RotationSplineSample<S>, SciOxideError> {
        let (i, u) = self.knots.locate(time)?;
        let [r, v, a] = self.segment(i, u)?;
        Ok(RotationSplineSample {
            rotation: self.knots.rotation(i, u, r)?,
            angular_velocity: v,
            angular_acceleration: a,
        })
    }
    /// Evaluates the rotation, without derivatives, at many times and writes
    /// the rotations to `output`.
    ///
    /// The times may be in any order and may repeat. At the knots the given
    /// rotations are returned exactly. `output` must have one entry per time,
    /// and `workspace` must meet a [`TransformWorkspaceRequest::rotations`]
    /// request for that count. See the [crate documentation](crate#execution)
    /// for parallel execution and cancellation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` or `InvalidInput` as in [`Self::evaluate`].
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace`
    ///   is too small.
    /// - `ResourceLimit` if the workspace exceeds its byte limit.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_spatial::{
    ///     Rotation, RotationSpline, RotationSplineOptions, Slerp, TransformOptions, TransformWorkspace,
    ///     TransformWorkspaceRequest,
    /// };
    /// let interpolant = RotationSpline::build(
    ///     &[0.0_f64, 1.0],
    ///     &[Rotation::identity(); 2],
    ///     RotationSplineOptions::new(),
    ///     ExecutionContext::sequential(),
    /// )?
    /// .spline
    /// .unwrap();
    /// let options = TransformOptions::new();
    /// let mut workspace =
    ///     TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(3, options))?;
    /// let mut output = [Rotation::identity(); 3];
    /// interpolant.evaluate_rotations_into(
    ///     &[1.0, 0.5, 0.0],
    ///     options,
    ///     &mut output,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(output, [Rotation::identity(); 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_rotations_into<'scope, 'capability>(
        &self,
        times: &[S],
        options: TransformOptions,
        output: &mut [Rotation<S>],
        workspace: &mut TransformWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.knots.rotations_into(
            times,
            options,
            output,
            workspace,
            context.into().execution(),
            |i, u| self.orientation_vector(i, u),
        )
    }
    fn orientation_vector(&self, i: usize, u: S) -> [S; 3] {
        let [a, b, c] = self.coefficients[i];
        scale(add(scale(add(scale(a, u), b), u), c), u)
    }
    fn segment(&self, i: usize, u: S) -> Result<[[S; 3]; 3], SciOxideError> {
        // r(u) = a u^3 + b u^2 + c u in Horner form. The u-derivatives are
        // divided by dt (and by dt^2) to give time derivatives, then mapped
        // through the Jacobian to body angular rates.
        let [a, b, c] = self.coefficients[i];
        let dt = self.knots.times[i + 1] - self.knots.times[i];
        let r = self.orientation_vector(i, u);
        let v = add(
            scale(add(scale(a, n::<S>(3) * u), scale(b, n::<S>(2))), u),
            c,
        )
        .map(|x| x / dt);
        let a = add(scale(a, n::<S>(6) * u), scale(b, n::<S>(2))).map(|x| (x / dt) / dt);
        let velocity = finite(jacobian(r, v))?;
        let acceleration = finite(add(jacobian(r, a), nonlinear(r, v)))?;
        Ok([finite(r)?, velocity, acceleration])
    }
    // Hermite conditions per segment: r(0) = 0, r(1) = rotation vector,
    // r'(0) = dt * omega_i (J(0) = I), and r'(1) = dt * J(r)^-1 omega_{i+1}.
    // Solving the cubic gives c = start, a = start + end - 2r and
    // b = 3r - 2 start - end.
    fn set_rates(&mut self, rates: &[[S; 3]]) -> Result<(), SciOxideError> {
        for (i, coeff) in self.coefficients.iter_mut().enumerate() {
            let dt = self.knots.times[i + 1] - self.knots.times[i];
            let r = self.knots.vectors[i];
            let start = scale(rates[i], dt);
            let end = scale(inverse_jacobian(r, rates[i + 1]), dt);
            *coeff = [
                finite(sub(add(start, end), scale(r, n::<S>(2))))?,
                finite(sub(sub(scale(r, n::<S>(3)), scale(start, n::<S>(2))), end))?,
                finite(start)?,
            ];
        }
        Ok(())
    }
    // Magnitude of the terms that form a segment's acceleration, before
    // cancellation: the Hermite second derivatives combine 6r and at most
    // 4|start| + 4|end| (start = c, end = a + 2r - c), divided by dt^2. For
    // near-constant rates the acceleration itself cancels to almost zero, so
    // it cannot scale the rounding noise of the jump.
    fn acceleration_scale(&self, i: usize) -> S {
        let [a, _, c] = self.coefficients[i];
        let dt = self.knots.times[i + 1] - self.knots.times[i];
        let r = self.knots.vectors[i];
        let end = sub(add(a, scale(r, n::<S>(2))), c);
        let magnitude = n::<S>(6) * norm(r) + n::<S>(4) * (norm(c) + norm(end));
        ((magnitude / dt) / dt).min(S::MAX)
    }
    fn continuity(&self, tolerance: ResidualTolerance<S>) -> Result<(S, bool), SciOxideError> {
        let mut residual = S::ZERO;
        let mut accepted = true;
        for i in 1..self.coefficients.len() {
            let [_, _, left] = self.segment(i - 1, S::ONE)?;
            let [_, _, right] = self.segment(i, S::ZERO)?;
            let jump = norm(sub(left, right));
            finite([jump])?;
            residual = residual.max(jump);
            let scale = norm(left)
                .max(norm(right))
                .max(self.acceleration_scale(i - 1))
                .max(self.acceleration_scale(i));
            accepted &= tolerance.is_satisfied(jump, scale)?;
        }
        Ok((residual, accepted))
    }
}
impl<S: RealScalar> RotationSpline<S> {
    /// Computes the spline through `rotations[i]` at `times[i]`.
    ///
    /// The angular velocity at the first and last knot equals the rotation
    /// vector of the adjacent segment divided by its length in time. The
    /// interior angular velocities are found by fixed-point iteration, each
    /// step solving a block tridiagonal linear system, until the jump in
    /// angular acceleration at every interior knot passes the continuity
    /// tolerance. If [`RotationSplineOptions::max_iterations`] is reached
    /// first, the result has termination `IterationLimit` and no spline.
    ///
    /// With a solve context, the iterations count toward the enclosing solve
    /// as described in [Solve accounting](scioxide_core#solve-accounting); its
    /// own iteration limit is not used. Cancellation and deadlines are checked
    /// at each iteration.
    ///
    /// # Errors
    ///
    /// - As [`Slerp::new`] for the knots.
    /// - `Singular` if a linear system of the iteration is singular.
    /// - `NumericalInstability` if an intermediate result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_spatial::{Rotation, RotationSpline, RotationSplineOptions};
    /// let result = RotationSpline::build(
    ///     &[0.0_f64, 1.0],
    ///     &[Rotation::identity(); 2],
    ///     RotationSplineOptions::new(),
    ///     ExecutionContext::sequential(),
    /// )?;
    /// let spline = result.spline.unwrap();
    /// assert_eq!(spline.evaluate(0.5)?.rotation, Rotation::identity());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn build<'scope, 'capability>(
        times: &[S],
        rotations: &[Rotation<S>],
        options: RotationSplineOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<RotationSplineResult<S>, SciOxideError> {
        let mut budgets = SolveBudgets::unbounded();
        if let Some(iterations) = core::num::NonZeroUsize::new(options.max_iterations) {
            budgets = budgets.with_limit(BudgetKind::Iteration, iterations.into());
        }
        crate::context::scoped(
            scope,
            scioxide_core::algorithm_id!("spatial", "rotation-spline", 1),
            budgets,
            |context| prepare(times, rotations, options, context),
        )
    }
}
// Block-tridiagonal (block Thomas) factorization of the part of the
// acceleration-continuity equations that is linear in the interior rates.
// Row i has diagonal 4(1/dt_i + 1/dt_{i+1}) I, sub-diagonal
// (2/dt_i) J(r_i) and super-diagonal (2/dt_{i+1}) J(r_{i+1})^-1. `upper`
// holds the eliminated D'_i^-1 U_i. The system depends only on the knots,
// so it is factored once and reused by every fixed-point iteration.
struct Blocks<S: RealScalar> {
    factors: Vec<Factor<S>>,
    upper: Vec<Mat<S>>,
    lower: Vec<Mat<S>>,
    backend: KernelBackend,
}
impl<S: RealScalar> Blocks<S> {
    fn new(
        dt: &[S],
        vectors: &[[S; 3]],
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let count = dt.len() - 1;
        let mut factors = Vec::new();
        factors.try_reserve_exact(count).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("spline_factors"), count)
        })?;
        let mut upper = try_filled(
            count.saturating_sub(1),
            [[S::ZERO; 3]; 3],
            scioxide_core::quantity!("spline_upper_blocks"),
        )?;
        let mut lower = try_filled(
            count.saturating_sub(1),
            [[S::ZERO; 3]; 3],
            scioxide_core::quantity!("spline_lower_blocks"),
        )?;
        let backend = KernelDispatch::detect(context.processor()).backend();
        for i in 0..count {
            context.check()?;
            let diagonal = n::<S>(4) * (S::ONE / dt[i] + S::ONE / dt[i + 1]);
            finite([diagonal])?;
            let mut block = matrix(|v| scale(v, diagonal));
            if i > 0 {
                lower[i - 1] = matrix(|v| scale(jacobian(vectors[i], v), n::<S>(2) / dt[i]));
                let correction = product(lower[i - 1], upper[i - 1]);
                block = core::array::from_fn(|j| sub(block[j], correction[j]));
            }
            let factor = Factor::new(block)?;
            if i + 1 < count {
                let next =
                    matrix(|v| scale(inverse_jacobian(vectors[i + 1], v), n::<S>(2) / dt[i + 1]));
                for j in 0..3 {
                    let rhs = core::array::from_fn::<S, 3, _>(|k| next[k][j]);
                    let out = factor.solve(rhs, backend)?;
                    for (k, row) in upper[i].iter_mut().enumerate() {
                        row[j] = out[k];
                    }
                }
            }
            factors.push(factor);
        }
        Ok(Self {
            factors,
            upper,
            lower,
            backend,
        })
    }
    fn solve(
        &self,
        rhs: &mut [[S; 3]],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        for i in 0..rhs.len() {
            context.check()?;
            let input = if i == 0 {
                rhs[i]
            } else {
                sub(rhs[i], mul(self.lower[i - 1], rhs[i - 1]))
            };
            rhs[i] = self.factors[i].solve(input, self.backend)?;
        }
        for i in (0..rhs.len().saturating_sub(1)).rev() {
            rhs[i] = sub(rhs[i], mul(self.upper[i], rhs[i + 1]));
        }
        Ok(())
    }
}
// Scoped spline preparation.
fn prepare<S: RealScalar>(
    times: &[S],
    rotations: &[Rotation<S>],
    options: RotationSplineOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<RotationSplineResult<S>, SciOxideError> {
    let execution = context.execution();
    execution.check()?;
    let knots = Knots::new(times, rotations)?;
    let segments = times.len() - 1;
    let mut spline = RotationSpline {
        knots,
        coefficients: try_filled(
            segments,
            [[S::ZERO; 3]; 3],
            scioxide_core::quantity!("spline_coefficients"),
        )?,
    };
    let mut dt = try_filled(
        segments,
        S::ZERO,
        scioxide_core::quantity!("segment_durations"),
    )?;
    let mut rates = try_filled(
        times.len(),
        [S::ZERO; 3],
        scioxide_core::quantity!("knot_rates"),
    )?;
    for (i, pair) in times.windows(2).enumerate() {
        dt[i] = pair[1] - pair[0];
        rates[i + 1] = finite(spline.knots.vectors[i].map(|x| x / dt[i]))?;
    }
    // Initial guess: every rate is its incoming secant, and the first rate
    // uses the first segment's secant. The endpoint rates stay fixed at
    // these secants. Only interior rates are solved for.
    rates[0] = rates[1];
    spline.set_rates(&rates)?;
    if segments == 1 {
        return Ok(RotationSplineResult {
            spline: Some(spline),
            termination: RotationSplineTermination::Converged,
            iterations: 0,
            continuity_residual: S::ZERO,
        });
    }
    let (mut residual, accepted) = spline.continuity(options.continuity_tolerance)?;
    if accepted {
        return Ok(RotationSplineResult {
            spline: Some(spline),
            termination: RotationSplineTermination::Converged,
            iterations: 0,
            continuity_residual: residual,
        });
    }
    if options.max_iterations == 0 {
        return Ok(RotationSplineResult {
            spline: None,
            termination: RotationSplineTermination::IterationLimit,
            iterations: 0,
            continuity_residual: residual,
        });
    }
    // The right-hand side has a constant part (`base`): the 6 r / dt^2 terms,
    // plus the known endpoint-rate couplings moved across in the first and
    // last rows. Each iteration subtracts the nonlinear term evaluated at the
    // previous rates and re-solves the fixed linear system (a fixed-point
    // iteration). Acceptance is the measured continuity residual, not a
    // step-size heuristic.
    let blocks = Blocks::new(&dt, &spline.knots.vectors, execution)?;
    let mut base = try_filled(
        segments - 1,
        [S::ZERO; 3],
        scioxide_core::quantity!("continuity_base"),
    )?;
    for (i, b) in base.iter_mut().enumerate() {
        *b = add(
            spline.knots.vectors[i].map(|x| ((n::<S>(6) * x) / dt[i]) / dt[i]),
            spline.knots.vectors[i + 1].map(|x| ((n::<S>(6) * x) / dt[i + 1]) / dt[i + 1]),
        );
    }
    base[0] = sub(
        base[0],
        scale(
            jacobian(spline.knots.vectors[0], rates[0]),
            n::<S>(2) / dt[0],
        ),
    );
    let last = segments - 2;
    base[last] = sub(
        base[last],
        scale(
            inverse_jacobian(spline.knots.vectors[segments - 1], rates[segments]),
            n::<S>(2) / dt[segments - 1],
        ),
    );
    let mut rhs = try_filled(
        base.len(),
        [S::ZERO; 3],
        scioxide_core::quantity!("continuity_rhs"),
    )?;
    let mut iterations = 0;
    while iterations < options.max_iterations {
        execution.check()?;
        if !context.record_iteration()?.is_granted() {
            break;
        }
        iterations += 1;
        for (i, b) in rhs.iter_mut().enumerate() {
            let r = spline.knots.vectors[i];
            *b = finite(sub(
                base[i],
                nonlinear(r, inverse_jacobian(r, rates[i + 1])),
            ))?;
        }
        blocks.solve(&mut rhs, execution)?;
        rates[1..segments].copy_from_slice(&rhs);
        spline.set_rates(&rates)?;
        let (next, accepted) = spline.continuity(options.continuity_tolerance)?;
        residual = next;
        if accepted {
            execution.check()?;
            return Ok(RotationSplineResult {
                spline: Some(spline),
                termination: RotationSplineTermination::Converged,
                iterations,
                continuity_residual: residual,
            });
        }
    }
    Ok(RotationSplineResult {
        spline: None,
        termination: RotationSplineTermination::IterationLimit,
        iterations,
        continuity_residual: residual,
    })
}

impl<S: RealScalar> Slerp<S> {
    /// Evaluates rotations at every query time with default transform options.
    /// Allocates `times.len()` rotations and a workspace; cost and accuracy match [`Self::evaluate_rotations_into`].
    ///
    /// # Errors
    /// As [`Self::evaluate_rotations_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn evaluate_rotations(&self, times: &[S]) -> Result<Vec<Rotation<S>>, SciOxideError> {
        let options = TransformOptions::new();
        let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
            times.len(),
            options,
        ))?;
        let mut output = scioxide_core::backend::try_filled(
            times.len(),
            Rotation::identity(),
            scioxide_core::quantity!("rotation_output"),
        )?;
        self.evaluate_rotations_into(
            times,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S: RealScalar> RotationSpline<S> {
    /// Evaluates rotations at every query time with default transform options.
    /// Allocates `times.len()` rotations and a workspace; cost and accuracy match [`Self::evaluate_rotations_into`].
    ///
    /// # Errors
    /// As [`Self::evaluate_rotations_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn evaluate_rotations(&self, times: &[S]) -> Result<Vec<Rotation<S>>, SciOxideError> {
        let options = TransformOptions::new();
        let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
            times.len(),
            options,
        ))?;
        let mut output = scioxide_core::backend::try_filled(
            times.len(),
            Rotation::identity(),
            scioxide_core::quantity!("rotation_output"),
        )?;
        self.evaluate_rotations_into(
            times,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S: RealScalar> RotationSpline<S> {
    /// Constructs an interpolating rotation spline with default continuity tolerance and 32 iterations.
    /// Uses [`Self::with_options`] with its allocation, cost and accuracy.
    ///
    /// # Errors
    /// As [`Self::with_options`]; iteration exhaustion is `ConvergenceFailure`.
    ///
    /// ```
    /// use scioxide_spatial::{Rotation, RotationSpline};
    /// let turn = Rotation::from_axis_angle([0.0_f64, 0.0, 1.0], 0.5)?;
    /// let spline = RotationSpline::new(&[0.0, 1.0], &[Rotation::identity(), turn])?;
    /// let angle = spline.evaluate(0.5)?.rotation.as_axis_angle().1;
    /// assert!((angle - 0.25).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(times: &[S], rotations: &[Rotation<S>]) -> Result<Self, SciOxideError> {
        into_spline(Self::with_options(
            times,
            rotations,
            RotationSplineOptions::default(),
        )?)
    }
    /// Fits a rotation spline using sequential native execution.
    /// Allocates and iterates as [`Self::build`], reporting non-convergence in the result.
    ///
    /// # Errors
    /// As [`Self::build`].
    pub fn with_options(
        times: &[S],
        rotations: &[Rotation<S>],
        options: RotationSplineOptions<S>,
    ) -> Result<RotationSplineResult<S>, SciOxideError> {
        Self::build(
            times,
            rotations,
            options,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
fn into_spline<S: RealScalar>(
    result: RotationSplineResult<S>,
) -> Result<RotationSpline<S>, SciOxideError> {
    result.spline.ok_or_else(|| {
        SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new().with_counts(
                scioxide_core::EvaluationCounts::from_iterations(result.iterations).into(),
            ),
            "rotation spline did not reach continuous angular acceleration",
        )
    })
}

#[cfg(test)]
mod owned_tests {
    use super::*;
    #[test]
    #[allow(clippy::unwrap_used)]
    fn spline_conversion_preserves_success_and_iteration_diagnostics() {
        fn check<S: RealScalar>() {
            let value = |x| S::try_from_f64_lossy(x).unwrap();
            let times = [0., 0.7, 2., 3., 5.].map(value);
            let rotations = [
                [0.1, 0.2, 0.3],
                [0.4, -0.1, 0.6],
                [-0.7, 0.6, 0.9],
                [0.8, 0.2, -0.1],
                [0.4, 1., 0.6],
            ]
            .map(|r| Rotation::from_rotation_vector(r.map(value)).unwrap());
            let result = RotationSpline::with_options(
                &times,
                &rotations,
                RotationSplineOptions::new().with_max_iterations(0),
            )
            .unwrap();
            assert_eq!(
                result.termination,
                RotationSplineTermination::IterationLimit
            );
            let error = into_spline(result).unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::ConvergenceFailure
            );
            if let scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } = error.view()
            {
                assert_eq!(reason, scioxide_core::ConvergenceReason::IterationLimit);
                assert_eq!(diagnostics.counts().own().iterations(), 0);
            } else {
                unreachable!();
            }
            let result =
                RotationSpline::with_options(&times, &rotations, RotationSplineOptions::new())
                    .unwrap();
            let expected = result
                .spline
                .as_ref()
                .unwrap()
                .evaluate(value(0.5))
                .unwrap()
                .rotation;
            assert_eq!(
                into_spline(result)
                    .unwrap()
                    .evaluate(value(0.5))
                    .unwrap()
                    .rotation,
                expected
            );
        }
        check::<f32>();
        check::<f64>();
    }
}
