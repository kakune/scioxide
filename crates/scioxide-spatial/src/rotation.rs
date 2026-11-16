//! Rotations in three dimensions, stored as unit quaternions.
use crate::math3::{add, cross, dot, n, norm, scale};
use crate::validation::{finite, input, out_of_range as invalid};
use crate::{
    SpatialExecutionContext, TransformOptions, TransformWorkspace, TransformWorkspaceRequest,
};
use scioxide_core::{Argument, RealScalar, SciOxideError, Tolerance};

// Evaluate each affine component before narrowing; independent coordinate scales
// and cancellation with translation must survive intermediate products.
pub(crate) fn apply_affine<S: RealScalar>(
    matrix: &[[S; 3]; 3],
    point: [S; 3],
    translation: [S; 3],
) -> Result<[S; 3], SciOxideError> {
    let mut output = [S::ZERO; 3];
    for ((value, row), shift) in output.iter_mut().zip(matrix).zip(translation) {
        *value = scioxide_core::checked_sum_of_products(
            &[row[0], row[1], row[2], S::ONE],
            &[point[0], point[1], point[2], shift],
        )?;
    }
    Ok(output)
}

/// A coordinate axis of an Euler sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    /// X axis.
    X,
    /// Y axis.
    Y,
    /// Z axis.
    Z,
}
impl RotationAxis {
    fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}
/// Whether the axes of an Euler sequence move with the body or stay fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerFrame {
    /// Each rotation is about an axis of the body, rotated by the previous
    /// rotations.
    Intrinsic,
    /// Each rotation is about a fixed axis of the reference frame.
    Extrinsic,
}
/// An Euler-angle convention: three rotation axes and a frame.
///
/// Consecutive axes must differ, which allows the six proper Euler sequences
/// (such as Z-X-Z) and the six Tait–Bryan sequences (such as X-Y-Z), each
/// intrinsic or extrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EulerSequence {
    axes: [RotationAxis; 3],
    frame: EulerFrame,
}
impl EulerSequence {
    /// Creates a convention from three axes and a frame.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if two consecutive axes are equal.
    pub fn new(axes: [RotationAxis; 3], frame: EulerFrame) -> Result<Self, SciOxideError> {
        if axes[0] == axes[1] || axes[1] == axes[2] {
            return Err(invalid(
                scioxide_core::argument!("axes"),
                "adjacent Euler axes must differ",
            ));
        }
        Ok(Self { axes, frame })
    }
    /// Returns the axes in the order the rotations are applied.
    pub const fn axes(self) -> [RotationAxis; 3] {
        self.axes
    }
    /// Returns whether the axes are intrinsic or extrinsic.
    pub const fn frame(self) -> EulerFrame {
        self.frame
    }
}
/// Euler angles of a rotation, as returned by [`Rotation::as_euler`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAngles<S: RealScalar> {
    /// The three angles in radians, in sequence order. At gimbal lock the
    /// third angle is zero.
    pub angles: [S; 3],
    /// Whether the rotation is at gimbal lock, where only the sum or difference
    /// of the first and third angles is determined.
    pub gimbal_lock: bool,
}

/// A rotation of three-dimensional space.
///
/// Rotations are active: [`Self::apply`] rotates a vector in a fixed frame.
/// They are right-handed, so a positive angle about the z axis turns the x
/// axis toward the y axis. The rotation is stored as a unit quaternion
/// `[x, y, z, w]` with the scalar part last. Of the two quaternions `q` and
/// `−q` that represent the same rotation, the stored one has the first
/// nonzero component of `(w, x, y, z)` positive.
///
/// ```
/// use scioxide_spatial::Rotation;
///
/// // A quarter turn about z maps the x axis to the y axis.
/// let turn = Rotation::from_axis_angle([0.0_f64, 0.0, 1.0], core::f64::consts::FRAC_PI_2)?;
/// let [x, y, z] = turn.apply([1.0, 0.0, 0.0])?;
/// assert!(x.abs() < 1e-15 && (y - 1.0).abs() < 1e-15 && z.abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// K. Shoemake, "Euler angle conversion", in *Graphics Gems IV*, Academic
/// Press, 1994.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation<S: RealScalar> {
    q: [S; 4],
}
impl<S: RealScalar> Rotation<S> {
    /// Returns the identity rotation.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(
    ///     Rotation::<f64>::identity().as_quaternion(),
    ///     [0.0, 0.0, 0.0, 1.0]
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn identity() -> Self {
        Self {
            q: [S::ZERO, S::ZERO, S::ZERO, S::ONE],
        }
    }
    /// Creates a rotation from a quaternion `[x, y, z, w]` of any nonzero
    /// length, which is normalized.
    ///
    /// The quaternion is scaled by its largest component before
    /// normalization, so no length overflows or underflows.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the quaternion is zero.
    /// - `NonFinite` if a component is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(
    ///     Rotation::from_quaternion([0.0_f64, 0.0, 0.0, 2.0])?,
    ///     Rotation::identity()
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_quaternion(q: [S; 4]) -> Result<Self, SciOxideError> {
        input(q)?;
        let max = q.iter().fold(S::ZERO, |m, x| m.max(x.abs()));
        if max == S::ZERO {
            return Err(invalid(scioxide_core::argument!("q"), "zero quaternion"));
        }
        // Prescaling by the largest component puts it at magnitude 1, so the
        // sum of squares lies in [1, 4] and can neither overflow nor underflow.
        let q = q.map(|x| x / max);
        let length = q.iter().map(|&x| x * x).sum::<S>().sqrt();
        let mut q = q.map(|x| x / length);
        // Choose a deterministic representative, including exact half turns.
        // q and -q are the same rotation. The first nonzero component in
        // (w, x, y, z) order is made positive, so w >= 0, and half turns
        // (w = 0) fall back to the vector part.
        let sign = [q[3], q[0], q[1], q[2]].into_iter().find(|&x| x != S::ZERO);
        if sign.is_some_and(|x| x < S::ZERO) {
            q = q.map(|x| -x);
        }
        Ok(Self { q })
    }
    /// Returns the unit quaternion `[x, y, z, w]`, with the sign convention
    /// described on [`Rotation`].
    pub const fn as_quaternion(self) -> [S; 4] {
        self.q
    }
    /// Creates the rotation by `angle` radians about `axis`, which need not
    /// have unit length.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis` is zero and `angle` is not. A zero angle
    ///   gives the identity for any axis.
    /// - `NonFinite` if an input is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// let turn = Rotation::from_axis_angle([0.0_f64, 0.0, 1.0], 0.5)?;
    /// assert!((turn.as_axis_angle().1 - 0.5).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_axis_angle(axis: [S; 3], angle: S) -> Result<Self, SciOxideError> {
        input(axis)?;
        input([angle])?;
        if angle == S::ZERO {
            return Ok(Self::identity());
        }
        let m = axis.iter().fold(S::ZERO, |m, x| m.max(x.abs()));
        if m == S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                "zero rotation axis",
            ));
        }
        // Prescale so `norm(axis)` cannot overflow or underflow for extreme
        // axis magnitudes.
        let axis = axis.map(|x| x / m);
        let (sin, cos) = (angle * S::HALF).sin_cos();
        let v = scale(axis, sin / norm(axis));
        Self::from_quaternion([v[0], v[1], v[2], cos])
    }
    /// Creates the rotation about `v` by `|v|` radians.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component is NaN or infinite.
    /// - `NumericalInstability` if `|v|` is not representable.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// let turn = Rotation::from_rotation_vector([0.0_f64, 0.0, 0.5])?;
    /// assert!((turn.as_rotation_vector()[2] - 0.5).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_rotation_vector(v: [S; 3]) -> Result<Self, SciOxideError> {
        input(v)?;
        Self::from_axis_angle(v, finite([norm(v)])?[0])
    }
    /// Returns a unit axis and an angle in `[0, π]` radians. The identity
    /// returns the x axis and angle zero.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(Rotation::<f64>::identity().as_axis_angle().1, 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_axis_angle(self) -> ([S; 3], S) {
        let v = [self.q[0], self.q[1], self.q[2]];
        let length = norm(v);
        if length == S::ZERO {
            ([S::ONE, S::ZERO, S::ZERO], S::ZERO)
        } else {
            // theta = 2 atan2(|v|, w) stays well conditioned at all angles,
            // whereas acos(w) loses accuracy near 0 and pi. The canonical
            // w >= 0 puts the angle in [0, pi].
            (v.map(|x| x / length), n::<S>(2) * length.atan2(self.q[3]))
        }
    }
    /// Returns the rotation vector: the axis scaled by the angle in `[0, π]`.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(Rotation::<f64>::identity().as_rotation_vector(), [0.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_rotation_vector(self) -> [S; 3] {
        let (axis, angle) = self.as_axis_angle();
        scale(axis, angle)
    }
    /// Returns the rotation matrix, row-major, acting on column vectors.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(Rotation::<f64>::identity().as_matrix()[0], [1.0, 0.0, 0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_matrix(self) -> [[S; 3]; 3] {
        let [x, y, z, w] = self.q;
        let t = n::<S>(2);
        let o = S::ONE;
        [
            [
                o - t * (y * y + z * z),
                t * (x * y - z * w),
                t * (x * z + y * w),
            ],
            [
                t * (x * y + z * w),
                o - t * (x * x + z * z),
                t * (y * z - x * w),
            ],
            [
                t * (x * z - y * w),
                t * (y * z + x * w),
                o - t * (x * x + y * y),
            ],
        ]
    }
    /// Creates a rotation from a row-major rotation matrix.
    ///
    /// The matrix must be orthogonal with determinant one, up to `tolerance`:
    /// each entry of `MMᵀ − I` and `det M − 1` must pass the residual test of
    /// `tolerance` at scale one. The matrix is not projected onto the nearest
    /// rotation. The quaternion is computed by Shepperd's method, which avoids
    /// cancellation near half turns.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the matrix is not orthogonal within `tolerance`, or
    ///   if it is a reflection.
    /// - `NonFinite` if an entry is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// let matrix = Rotation::<f64>::identity().as_matrix();
    /// assert_eq!(
    ///     Rotation::from_matrix(matrix, scioxide_core::Tolerance::from_sqrt_epsilon())?,
    ///     Rotation::identity()
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_matrix(m: [[S; 3]; 3], tolerance: Tolerance<S>) -> Result<Self, SciOxideError> {
        const MATRIX: Argument = scioxide_core::argument!("m");
        for row in m {
            input(row)?;
        }
        for (i, a) in m.iter().enumerate() {
            for (j, b) in m.iter().enumerate() {
                let expected = if i == j { S::ONE } else { S::ZERO };
                // An unrepresentable residual is far from orthogonal, not a
                // numerical failure of this check.
                let residual = (dot(*a, *b) - expected).abs();
                if !residual.is_finite() || !tolerance.is_residual_ok(residual, S::ONE)? {
                    return Err(invalid(MATRIX, "matrix is not orthogonal"));
                }
            }
        }
        let det = dot(m[0], cross(m[1], m[2]));
        if !det.is_finite()
            || det <= S::ZERO
            || !tolerance.is_residual_ok((det - S::ONE).abs(), S::ONE)?
        {
            return Err(invalid(MATRIX, "matrix is not a proper rotation"));
        }
        // Shepperd's method. Each candidate pivot (trace or one diagonal
        // entry) gives 4 * q_p^2 up to a constant. Picking the largest keeps
        // the pivot component away from zero, so there is no cancellation
        // near half turns. The other components come from off-diagonal
        // sums and differences, which are all scaled by 4 * q_p. The common
        // factor is removed by the normalization in `from_quaternion`.
        let diagonal = [m[0][0], m[1][1], m[2][2], m[0][0] + m[1][1] + m[2][2]];
        let index = (1..4).fold(0, |best, i| {
            if diagonal[i] > diagonal[best] {
                i
            } else {
                best
            }
        });
        let mut q = [S::ZERO; 4];
        if index == 3 {
            q = [
                m[2][1] - m[1][2],
                m[0][2] - m[2][0],
                m[1][0] - m[0][1],
                S::ONE + diagonal[3],
            ];
        } else {
            let i = index;
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            q[i] = S::ONE + m[i][i] - m[j][j] - m[k][k];
            q[j] = m[j][i] + m[i][j];
            q[k] = m[k][i] + m[i][k];
            q[3] = m[k][j] - m[j][k];
        }
        Self::from_quaternion(q)
    }
    /// Returns the inverse rotation.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// let turn = Rotation::from_rotation_vector([0.0_f64, 0.0, 0.5])?;
    /// assert!(turn.compose(turn.inverse())?.as_axis_angle().1 < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn inverse(self) -> Self {
        // A half turn is its own inverse. Negating the vector part would
        // give -q, the same rotation but not the canonical sign.
        if self.q[3] == S::ZERO {
            return self;
        }
        Self {
            q: [-self.q[0], -self.q[1], -self.q[2], self.q[3]],
        }
    }
    /// Returns the rotation that applies `other` first, then `self`.
    ///
    /// The quaternion product is normalized again, so repeated composition
    /// does not drift from unit length.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::from_quaternion`] for the product, which
    /// do not occur for finite unit quaternions.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// let turn = Rotation::from_rotation_vector([0.0_f64, 0.0, 0.5])?;
    /// assert!((turn.compose(turn)?.as_axis_angle().1 - 1.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn compose(self, other: Self) -> Result<Self, SciOxideError> {
        let a = [self.q[0], self.q[1], self.q[2]];
        let b = [other.q[0], other.q[1], other.q[2]];
        // Hamilton product. Renormalizing through `from_quaternion` stops
        // drift from the unit sphere under repeated composition, and
        // restores the canonical sign.
        let v = add(add(scale(b, self.q[3]), scale(a, other.q[3])), cross(a, b));
        Self::from_quaternion([v[0], v[1], v[2], self.q[3] * other.q[3] - dot(a, b)])
    }
    /// Rotates a vector using range-safe sums of products.
    ///
    /// Each matrix-vector component is rounded once from its exact sum of
    /// products; the rotation matrix itself has native scalar rounding.
    /// Evaluation uses constant storage and does not allocate.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `v` is NaN or infinite.
    /// - `NumericalInstability` if the result is not representable.
    ///
    /// ```
    /// use scioxide_spatial::Rotation;
    /// assert_eq!(
    ///     Rotation::<f64>::identity().apply([1.0, 2.0, 3.0])?,
    ///     [1.0, 2.0, 3.0]
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply(self, v: [S; 3]) -> Result<[S; 3], SciOxideError> {
        input(v)?;
        apply_affine(&self.as_matrix(), v, [S::ZERO; 3])
    }
    /// Creates the rotation with the given Euler angles in radians.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an angle is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::{EulerFrame, EulerSequence, Rotation, RotationAxis::*};
    /// let sequence = EulerSequence::new([X, Y, Z], EulerFrame::Extrinsic)?;
    /// let rotation = Rotation::from_euler(sequence, [0.0_f64, 0.0, 0.5])?;
    /// let angles = rotation.as_euler(sequence, scioxide_core::Tolerance::from_sqrt_epsilon())?;
    /// assert!((angles.angles[2] - 0.5).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn from_euler(sequence: EulerSequence, angles: [S; 3]) -> Result<Self, SciOxideError> {
        input(angles)?;
        sequence.axes.into_iter().zip(angles).try_fold(
            Self::identity(),
            |rotation, (axis, angle)| {
                let mut v = [S::ZERO; 3];
                v[axis.index()] = S::ONE;
                let step = Self::from_axis_angle(v, angle)?;
                // Intrinsic axes move with the body, so later steps apply
                // first (R1 R2 R3). Extrinsic steps apply in order about
                // fixed axes (R3 R2 R1).
                match sequence.frame {
                    EulerFrame::Intrinsic => rotation.compose(step),
                    EulerFrame::Extrinsic => step.compose(rotation),
                }
            },
        )
    }
    /// Returns the Euler angles of the rotation in a given convention.
    ///
    /// The middle angle lies in `[0, π]` for proper Euler sequences and in
    /// `[−π/2, π/2]` for Tait–Bryan sequences; the outer angles lie in
    /// `[−π, π]`. Gimbal lock is reported when `|sin β|` (proper Euler) or
    /// `|cos β|` (Tait–Bryan) of the middle angle `β` is at most the threshold
    /// of `tolerance` at one. Then the third angle is set to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the threshold of `tolerance` overflows.
    ///
    /// ```
    /// use scioxide_spatial::{EulerFrame, EulerSequence, Rotation, RotationAxis::*};
    /// let sequence = EulerSequence::new([X, Y, Z], EulerFrame::Extrinsic)?;
    /// let rotation = Rotation::from_euler(sequence, [0.0_f64, 0.0, 0.5])?;
    /// let angles = rotation.as_euler(sequence, scioxide_core::Tolerance::from_sqrt_epsilon())?;
    /// assert!((angles.angles[2] - 0.5).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_euler(
        self,
        sequence: EulerSequence,
        tolerance: Tolerance<S>,
    ) -> Result<EulerAngles<S>, SciOxideError> {
        Ok(EulerExtraction::new(sequence, tolerance)?.extract(self))
    }

    /// Computes the Euler angles of many rotations, as [`Self::as_euler`] does,
    /// and writes them to `output`.
    ///
    /// `output` must have one entry per rotation, and `workspace` must meet a
    /// [`TransformWorkspaceRequest::euler`] request for that count. See the
    /// [crate documentation](crate#execution) for parallel execution and
    /// cancellation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace`
    ///   is too small.
    /// - `ResourceLimit` if the workspace exceeds its byte limit.
    /// - As [`Self::as_euler`] for the threshold.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, Tolerance};
    /// use scioxide_spatial::{
    ///     EulerAngles, EulerFrame, EulerSequence, Rotation, RotationAxis::*, TransformOptions,
    ///     TransformWorkspace, TransformWorkspaceRequest,
    /// };
    /// let sequence = EulerSequence::new([X, Y, Z], EulerFrame::Extrinsic)?;
    /// let options = TransformOptions::new();
    /// let mut output = [EulerAngles {
    ///     angles: [0.0_f64; 3],
    ///     gimbal_lock: false,
    /// }; 2];
    /// let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::euler(2, options))?;
    /// Rotation::as_euler_batch_into(
    ///     &[Rotation::identity(); 2],
    ///     sequence,
    ///     Tolerance::from_sqrt_epsilon(),
    ///     options,
    ///     &mut output,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(output[0].angles, [0.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_euler_batch_into<'scope, 'capability>(
        rotations: &[Self],
        sequence: EulerSequence,
        tolerance: Tolerance<S>,
        options: TransformOptions,
        output: &mut [EulerAngles<S>],
        workspace: &mut TransformWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().execution();
        context.check()?;
        let extraction = EulerExtraction::new(sequence, tolerance)?;
        if output.len() != rotations.len() {
            return Err(SciOxideError::dimension_mismatch(
                [rotations.len()],
                [output.len()],
            ));
        }
        workspace.require(TransformWorkspaceRequest::euler(rotations.len(), options))?;
        let staging = &mut workspace.euler[..rotations.len()];
        let policy = options.execution_policy();
        // Auto estimate: a matrix and up to three arctangents per rotation.
        crate::batch::dispatch_chunks(
            staging,
            policy,
            context,
            crate::batch::work(64),
            |offset, values| {
                for (block, values) in values.chunks_mut(256).enumerate() {
                    context.check()?;
                    let start = offset + block * 256;
                    for (value, &rotation) in values.iter_mut().zip(&rotations[start..]) {
                        *value = extraction.extract(rotation);
                    }
                }
                Ok(())
            },
        )?;
        context.check()?;
        output.copy_from_slice(staging);
        Ok(())
    }
}

struct EulerExtraction<S> {
    intrinsic: bool,
    parity: usize,
    repetition: bool,
    indices: [usize; 3],
    threshold: S,
}
impl<S: RealScalar> EulerExtraction<S> {
    fn new(sequence: EulerSequence, tolerance: Tolerance<S>) -> Result<Self, SciOxideError> {
        let [a, b, c] = sequence.axes.map(RotationAxis::index);
        let parity = usize::from(b != (a + 1) % 3);
        Ok(Self {
            intrinsic: sequence.frame == EulerFrame::Intrinsic,
            parity,
            repetition: a == c,
            indices: [a, (a + 1 + parity) % 3, (a + 2 - parity) % 3],
            threshold: tolerance.threshold(S::ONE)?,
        })
    }
    // Batch extraction reuses this geometry; avoid a per-rotation call boundary.
    #[inline(always)]
    fn extract(&self, rotation: Rotation<S>) -> EulerAngles<S> {
        // Static-axis matrix extraction (Shoemake); intrinsic extraction uses the inverse.
        let Self {
            intrinsic,
            parity,
            repetition,
            indices: [i, j, k],
            ..
        } = *self;
        let m = if intrinsic {
            rotation.inverse().as_matrix()
        } else {
            rotation.as_matrix()
        };
        // `measure` is |sin(beta)| for proper Euler (repeated) sequences and
        // |cos(beta)| for Tait-Bryan sequences. Gimbal lock is where it
        // vanishes. It is also the well-conditioned atan2 argument for the
        // middle angle.
        let measure = if repetition {
            m[i][j].hypot(m[i][k])
        } else {
            m[i][i].hypot(m[j][i])
        };
        let lock = measure <= self.threshold;
        let mut angles = if repetition {
            if lock {
                [(-m[j][k]).atan2(m[j][j]), measure.atan2(m[i][i]), S::ZERO]
            } else {
                [
                    m[i][j].atan2(m[i][k]),
                    measure.atan2(m[i][i]),
                    m[j][i].atan2(-m[k][i]),
                ]
            }
        } else if lock {
            [
                (-m[j][k]).atan2(m[j][j]),
                (-m[k][i]).atan2(measure),
                S::ZERO,
            ]
        } else {
            [
                m[k][j].atan2(m[k][k]),
                (-m[k][i]).atan2(measure),
                m[j][i].atan2(m[i][i]),
            ]
        };
        if (parity != 0) ^ intrinsic {
            angles = angles.map(|x| -x);
        }
        // (alpha, beta, gamma) and (alpha + pi, -beta, gamma - pi) are the
        // same rotation for repeated sequences. This picks beta in [0, pi].
        if repetition && angles[1] < S::ZERO {
            angles = [angles[0] + S::PI, -angles[1], angles[2] - S::PI];
        }
        if lock && repetition && angles[2] != S::ZERO {
            // At the singularity select the equivalent representative with third angle zero.
            // Only alpha + gamma (beta near 0) or alpha - gamma (beta near pi)
            // is identifiable, so gamma is folded into alpha with that sign.
            angles[0] += if angles[1] > S::HALF * S::PI {
                -angles[2]
            } else {
                angles[2]
            };
            angles[2] = S::ZERO;
        }
        for index in [0, 2] {
            if angles[index] > S::PI {
                angles[index] -= S::TAU;
            }
            if angles[index] < -S::PI {
                angles[index] += S::TAU;
            }
        }
        EulerAngles {
            angles,
            gimbal_lock: lock,
        }
    }
}
