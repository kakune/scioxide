use super::*;
use scioxide_core::AlgebraicScalar;

/// Closed disk `{z : |z − center| ≤ radius}` in the complex plane.
///
/// A proved root count for a disk also proves that no root lies on its
/// boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDisk<R: RealScalar> {
    /// Center; finite.
    pub center: Complex<R>,
    /// Radius; finite and positive.
    pub radius: R,
}
impl<R: RealScalar> RootDisk<R> {
    /// Validate this disk as the `argument`.
    /// # Errors
    /// Returns `NonFinite` for a NaN or infinite center or radius and invalid input
    /// for a radius that is not positive.
    fn validate(self, argument: Argument) -> Result<()> {
        if !self.center.re.is_finite() || !self.center.im.is_finite() || !self.radius.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "disk center and radius must be finite",
            ));
        }
        if self.radius <= R::ZERO {
            return Err(invalid(argument, "disk radius must be positive"));
        }
        Ok(())
    }
}
/// Claim that a disk contains `count` roots of a polynomial, counted with
/// multiplicity, checked by [`PolynomialRootPlan::verify`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialRootCertificate<R: RealScalar> {
    /// Disk in the variable of the polynomial.
    pub disk: RootDisk<R>,
    /// Claimed number of roots in the disk, counted with multiplicity.
    pub count: usize,
}
/// Proves the number of roots of a polynomial in complex disks, and isolates
/// all its roots in disjoint disks.
///
/// The polynomial is `p(z) = a₀ + a₁z + … + a_d zᵈ` with real or complex
/// coefficients in increasing order of degree, taken exactly as stored. Roots
/// are counted with multiplicity.
///
/// # Method
///
/// For a disk with center `c` and radius `r`, the plan first tries a Rouché
/// test for one simple root: `|p(c)| + r²M₂ < r|p′(c)|`, where `M₂` bounds
/// `|p″(z)|/2` on the disk. It evaluates `p(c)` and `p′(c)` by Horner's rule in
/// midpoint-radius complex arithmetic, which avoids the overestimation of
/// rectangular intervals, at O(d) cost for degree `d`. If that test fails, it
/// computes the Taylor shift `p(c + rz) = Σⱼ bⱼ zʲ` in interval arithmetic,
/// at O(d²) cost, and Pellet's test `|bₖ| > Σ_{j≠k} |bⱼ|` proves that the disk
/// contains exactly `k` roots. All comparisons are strict and rounded outward;
/// if neither test succeeds the count is inconclusive.
///
/// A disk costs one callback evaluation of the budget, its Rouché test one
/// iteration and each row of the Taylor shift another.
///
/// # References
///
/// - P. Henrici, *Applied and Computational Complex Analysis*, Vol. 1, Wiley,
///   1974 (Rouché's theorem and root inclusion).
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, chapter 3.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_validated::{PolynomialRootCertificate, PolynomialRootPlan, RootDisk, ValidatedOptions};
///
/// // p(z) = z² − 2 has the roots ±√2 ≈ ±1.4142.
/// let coefficients = [-2.0_f64, 0.0, 1.0];
/// let plan = PolynomialRootPlan::new(&coefficients, ValidatedOptions::new())?;
/// let disk = |re: f64| RootDisk {
///     center: Complex::new(re, 0.0),
///     radius: 0.125,
/// };
/// let cover = [
///     PolynomialRootCertificate {
///         disk: disk(1.41),
///         count: 1,
///     },
///     PolynomialRootCertificate {
///         disk: disk(-1.41),
///         count: 1,
///     },
/// ];
/// let report = plan.verify(cover[0])?;
/// assert_eq!(report.count(), Some(1));
/// assert!(plan.verify_cover(&cover)?.is_proved());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct PolynomialRootPlan<S: NumericScalar> {
    coefficients: Vec<PreparedCoefficient<S::Real>>,
    options: ValidatedOptions,
}
/// Immutable enclosures of one coefficient after exact power-of-two scaling.
#[derive(Clone, Copy)]
struct PreparedCoefficient<R: RealScalar> {
    enclosure: Box2<R>,
    ball: Ball<R>,
    magnitude: R,
}
impl<S: NumericScalar> core::fmt::Debug for PolynomialRootPlan<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PolynomialRootPlan")
            .field("degree", &self.degree())
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<S: NumericScalar> PolynomialRootPlan<S> {
    /// Creates a plan for the polynomial with coefficients `a₀, a₁, …`.
    ///
    /// Leading coefficients that are exactly zero are dropped, so the degree
    /// is that of the stored polynomial. The plan copies the coefficients and
    /// scales them by a power of two, which is exact except for subnormal
    /// results, whose rounding is enclosed. This costs O(d) storage for degree
    /// `d`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if every coefficient is zero, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - `ResourceLimit` if the storage exceeds the byte limit of `options`.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or reserved.
    pub fn new(coefficients: &[S], options: ValidatedOptions) -> Result<Self> {
        crate::environment::check()?;
        if coefficients.iter().any(|x| !x.has_finite_components()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "polynomial coefficients must be finite",
            ));
        }
        let len = coefficients
            .iter()
            .rposition(|x| *x != S::ZERO)
            .map(|i| i + 1)
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("coefficients"),
                    "zero polynomial has no isolated-root count",
                )
            })?;
        let coefficients = &coefficients[..len];
        let magnitude = coefficients.iter().fold(S::Real::ZERO, |m, x| {
            m.max(x.real_component().abs())
                .max(x.imaginary_component().abs())
        });
        let scale = 1 - scioxide_core::backend::standard::exponent(magnitude)?;
        let bytes = crate::shell::mul(len, core::mem::size_of::<PreparedCoefficient<S::Real>>())?;
        crate::shell::within_limit(bytes, options.byte_limit())?;
        let mut prepared = Vec::new();
        scioxide_core::backend::try_reserve(
            &mut prepared,
            len,
            scioxide_core::quantity!("coefficients"),
        )?;
        for &coefficient in coefficients {
            let enclosure = crate::interval_math::scale(Box2::point(coefficient)?, scale)?;
            let (center, remainder) = enclosure
                .midpoint::<Complex<S::Real>>()?
                .ok_or_else(|| crate::shell::breakdown("unbounded normalized coefficient"))?;
            prepared.push(PreparedCoefficient {
                enclosure,
                ball: Ball {
                    center,
                    radius: upper(remainder.modulus()?),
                },
                magnitude: upper(enclosure.modulus()?),
            });
        }
        Ok(Self {
            coefficients: prepared,
            options,
        })
    }
    /// Returns the degree `d`, after dropping leading coefficients that are
    /// exactly zero.
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }
    /// Returns the options.
    pub const fn options(&self) -> &ValidatedOptions {
        &self.options
    }
    const fn identity(&self) -> SolveIdentity {
        identity(POLYNOMIAL_COUNT, self.coefficients.len())
    }
    /// Returns the workspace size for `request`: independent proofs, or an
    /// isolation from `request.capacity()` seed disks. To verify a cover with
    /// more disks than the degree, request `ProofRequest::refinement(cover.len())`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(&self, request: ProofRequest) -> Result<RootRequirements> {
        let shape = workspace::Shape {
            terms: self.coefficients.len(),
            projections: self.coefficients.len().max(request.capacity()),
            disks: request.capacity(),
            ..workspace::Shape::default()
        };
        RootRequirements::new::<S>(shape, request.proof_count(), self.options.byte_limit())
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if it cannot be sized or reserved.
    pub fn workspace(&self, request: ProofRequest) -> Result<RootWorkspace<S>> {
        let mut workspace = RootWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Proves or refutes that the disk of `certificate` contains `count` roots.
    ///
    /// The claim is `Proved` if the tests prove that count and `Refuted` if
    /// they prove a different one; [`PolynomialVerificationReport::count`]
    /// returns the proved count either way. Allocates a workspace and runs
    /// sequentially in a new solve.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the center or radius is NaN or infinite.
    /// - `InvalidInput` if the radius is not positive, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    pub fn verify(
        &self,
        certificate: PolynomialRootCertificate<S::Real>,
    ) -> Result<PolynomialVerificationReport> {
        self.verify_scoped(
            certificate,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify`], and those of
    ///   [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_scoped<'s, 'c>(
        &self,
        certificate: PolynomialRootCertificate<S::Real>,
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<PolynomialVerificationReport> {
        workspace.core.run(
            scope.into(),
            self.identity(),
            &self.options,
            || {
                let requirements = self.workspace_requirements(ProofRequest::proofs(1))?;
                certificate
                    .disk
                    .validate(scioxide_core::argument!("certificate.disk"))?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, _, _, _, context| self.verify_in(certificate, &mut lanes[0], context),
            PolynomialVerificationReport::is_proved,
        )
    }
    /// Verify inside an existing epoch.
    fn verify_in(
        &self,
        certificate: PolynomialRootCertificate<S::Real>,
        lane: &mut workspace::Lane<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PolynomialVerificationReport> {
        let mut scratch = lane.polynomial(self.coefficients.len());
        let result = self
            .count_raw(
                certificate.disk,
                &mut scratch,
                context,
                scioxide_core::argument!("certificate.disk"),
            )
            .map(|counted| report(certificate.count, counted));
        finish(result, context)
    }
    /// Verifies independent disk counts, writing one report per certificate.
    ///
    /// The disks may overlap; to prove that disjoint disks hold every root, use
    /// [`Self::verify_cover`]. The proofs run in parallel under the options'
    /// execution policy, and the reports do not depend on the policy or the
    /// number of workers. On error, `reports` is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `reports` and `certificates` differ in length.
    /// - The errors of [`Self::verify_scoped`] for any certificate, and errors
    ///   of the executor.
    pub fn verify_batch_into<'s, 'c>(
        &self,
        certificates: &[PolynomialRootCertificate<S::Real>],
        reports: &mut [PolynomialVerificationReport],
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<()> {
        let count = reports.len();
        workspace
            .core
            .verify_batch(
                scope.into(),
                self.identity(),
                &self.options,
                || {
                    let requirements =
                        self.workspace_requirements(ProofRequest::proofs(certificates.len()))?;
                    crate::shell::exact(certificates.len(), count)?;
                    certificates.iter().try_for_each(|c| {
                        c.disk.validate(scioxide_core::argument!("certificates"))
                    })?;
                    Ok(requirements)
                },
                reports,
                |_, _| Ok(crate::shell::Prepared::Ready(())),
                &|index, lane, _, _, context| self.verify_in(certificates[index], lane, context),
            )
            .map(|_| ())
    }
    /// Count roots in the caller's `disk_argument`, using workspace `w`.
    fn count_raw(
        &self,
        disk: RootDisk<S::Real>,
        w: &mut workspace::PolynomialScratch<'_, S::Real>,
        context: &mut SolveContext<'_, '_>,
        disk_argument: Argument,
    ) -> Result<Counted> {
        crate::environment::check()?;
        disk.validate(disk_argument)?;
        if let Some(reason) = admit(context, EvaluationRole::Function)? {
            return Ok(Err(reason));
        }
        match self.simple_root(disk, context)? {
            Err(RootInconclusive::Reason(RootProofFailure::Inequality)) => {}
            decided => return Ok(decided),
        }
        let center = Box2::point(disk.center)?;
        let radius = Box2::point(disk.radius)?;
        w.coefficients.fill(Box2::zero()?);
        // Affine Horner expansion: coefficients of p(c+r*z), not rounded
        // derivatives or a silently converted ordinary polynomial.
        for (degree, &coefficient) in self.coefficients.iter().rev().enumerate() {
            if let Some(reason) = iteration(context)? {
                return Ok(Err(reason));
            }
            for j in (1..=degree).rev() {
                w.coefficients[j] = w.coefficients[j]
                    .mul(center)?
                    .add(w.coefficients[j - 1].mul(radius)?)?;
            }
            w.coefficients[0] = w.coefficients[0].mul(center)?.add(coefficient.enclosure)?;
            if w.coefficients[..=degree].iter().any(|v| !v.finite()) {
                return Ok(Err(RootInconclusive::Unbounded));
            }
        }
        let mut sum = S::Real::ZERO;
        for (m, c) in w.magnitudes.iter_mut().zip(w.coefficients.iter()) {
            *m = c.modulus()?;
            sum = add_up(sum, upper(*m));
        }
        if !sum.is_finite() {
            return Ok(Err(RootInconclusive::Unbounded));
        }
        for (k, &m) in w.magnitudes.iter().enumerate() {
            let (lower, upper_bound) = bounded(m, scioxide_core::argument!("self"))?;
            let others = upper(D::point(sum)?.sub_admitted(&D::point(upper_bound)?)?);
            if lower > others {
                return Ok(Ok(k));
            }
        }
        Ok(Err(RootInconclusive::Reason(RootProofFailure::Inequality)))
    }
    /// Prove exactly one root by comparing the linear Taylor term with the
    /// residual and a second-derivative majorant on the boundary.
    fn simple_root(
        &self,
        disk: RootDisk<S::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Counted> {
        const UNDECIDED: Counted = Err(RootInconclusive::Reason(RootProofFailure::Inequality));
        if let Some(reason) = iteration(context)? {
            return Ok(Err(reason));
        }
        let center = disk.center;
        let norm = point_modulus(center)?.1;
        let extent = add_up(norm, disk.radius);
        if !extent.is_finite() {
            return Ok(UNDECIDED);
        }
        let leading = self.coefficients[self.degree()];
        let mut value = leading.ball;
        let mut derivative = Ball::zero();
        let (mut b0, mut b1, mut b2) = (leading.magnitude, S::Real::ZERO, S::Real::ZERO);
        for &coefficient in self.coefficients[..self.degree()].iter().rev() {
            // These are Horner recurrences for B(t), B′(t), B″(t)/2,
            // where B has the absolute values of the scaled coefficients.
            b2 = add_up(mul_up(b2, extent), b1);
            b1 = add_up(mul_up(b1, extent), b0);
            b0 = add_up(mul_up(b0, extent), coefficient.magnitude);
            let Some(next_derivative) = derivative.horner(center, norm, value)? else {
                return Ok(UNDECIDED);
            };
            // The addend's center is used above; its disk error is added once.
            derivative = next_derivative;
            let Some(next_value) = value.horner(center, norm, coefficient.ball)? else {
                return Ok(UNDECIDED);
            };
            value = next_value;
            if !b0.is_finite() || !b1.is_finite() || !b2.is_finite() {
                return Ok(UNDECIDED);
            }
        }
        let residual = add_up(point_modulus(value.center)?.1, value.radius);
        let slope =
            (-add_up(-point_modulus(derivative.center)?.0, derivative.radius)).max(S::Real::ZERO);
        let left = add_up(residual, mul_up(disk.radius, mul_up(disk.radius, b2)));
        let right = crate::rounding::endpoint(
            disk.radius,
            slope,
            crate::rounding::Operation::Multiply,
            false,
        );
        Ok(if left.is_finite() && right.is_finite() && left < right {
            Ok(1)
        } else {
            UNDECIDED
        })
    }
    /// Proves that the disks of `certificates` are pairwise disjoint and
    /// together contain all `d` roots, counted with multiplicity.
    ///
    /// Every disk count is verified first; if one is not proved, the report is
    /// that disk's. Then the disks are checked for strict separation, and an
    /// overlap makes the result inconclusive. Finally the claim is `Proved` if
    /// the counts sum to `d` and `Refuted` if they sum to less;
    /// [`PolynomialVerificationReport::count`] returns the sum. A disk with a
    /// simple root costs O(d). The separation test sorts the disks by their
    /// projections on the real axis and costs O(k log k + m) for `k` disks and
    /// `m` pairs whose projections overlap, O(k²) in the worst case. Storage is
    /// O(d + k).
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a center or radius is NaN or infinite.
    /// - `InvalidInput` if a radius is not positive, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    pub fn verify_cover(
        &self,
        certificates: &[PolynomialRootCertificate<S::Real>],
    ) -> Result<PolynomialVerificationReport> {
        self.verify_cover_scoped(
            certificates,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// [`Self::verify_cover`] with your workspace, run in `scope`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::verify_cover`], and those of
    ///   [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn verify_cover_scoped<'s, 'c>(
        &self,
        certificates: &[PolynomialRootCertificate<S::Real>],
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<PolynomialVerificationReport> {
        workspace.core.run(
            scope.into(),
            self.identity(),
            &self.options,
            || {
                let requirements = RootRequirements::new::<S>(
                    workspace::Shape {
                        terms: self.coefficients.len(),
                        projections: self.coefficients.len().max(certificates.len()),
                        ..workspace::Shape::default()
                    },
                    1,
                    self.options.byte_limit(),
                )?;
                certificates
                    .iter()
                    .try_for_each(|c| c.disk.validate(scioxide_core::argument!("certificates")))?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, _, _, _, context| self.cover(certificates, &mut lanes[0], context),
            PolynomialVerificationReport::is_proved,
        )
    }
    fn cover(
        &self,
        certificates: &[PolynomialRootCertificate<S::Real>],
        lane: &mut workspace::Lane<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PolynomialVerificationReport> {
        let mut count = 0usize;
        for &certificate in certificates {
            let report = self.verify_in(certificate, lane, context)?;
            if !report.is_proved() {
                return Ok(report);
            }
            count = checked_add(
                count,
                certificate.count,
                scioxide_core::quantity!("root_count"),
            )?;
        }
        match sweep(
            lane.projections(certificates.len()),
            |i| certificates[i].disk,
            context,
        )? {
            Separation::Disjoint => {}
            Separation::Overlap(_, _) => {
                return Ok(PolynomialVerificationReport::unknown(
                    RootInconclusive::Reason(RootProofFailure::Inequality),
                ));
            }
            Separation::Stopped(reason) => {
                return Ok(PolynomialVerificationReport::unknown(reason));
            }
        }
        finish(
            Ok(PolynomialVerificationReport::counted(self.degree(), count)),
            context,
        )
    }
}
/// An established root count, including multiplicities, or why none was established.
type Counted = core::result::Result<usize, RootInconclusive>;
/// The report of a `counted` disk against its `claim`ed count.
fn report(claim: usize, counted: Counted) -> PolynomialVerificationReport {
    counted.map_or_else(PolynomialVerificationReport::unknown, |count| {
        PolynomialVerificationReport::counted(claim, count)
    })
}
/// Output of [`PolynomialRootPlan::isolate`]: disjoint disks with proved root
/// counts, up to a fixed capacity.
///
/// The storage proves nothing by itself: keep the report of the isolation, or
/// pass [`Self::view`] to [`PolynomialRootPlan::verify_cover`].
///
/// ```
/// use scioxide_validated::PolynomialRootCover;
///
/// let cover = PolynomialRootCover::<f64>::new(2)?;
/// assert_eq!(cover.capacity(), 2);
/// assert!(cover.view().is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct PolynomialRootCover<R: RealScalar> {
    certificates: Vec<PolynomialRootCertificate<R>>,
    used: usize,
}
/// Covers are equal when they have the same capacity and written certificates.
impl<R: RealScalar> PartialEq for PolynomialRootCover<R> {
    fn eq(&self, other: &Self) -> bool {
        self.capacity() == other.capacity() && self.view() == other.view()
    }
}
impl<R: RealScalar> core::fmt::Debug for PolynomialRootCover<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PolynomialRootCover")
            .field("capacity", &self.capacity())
            .field("certificates", &self.view())
            .finish()
    }
}
impl<R: RealScalar> PolynomialRootCover<R> {
    /// Allocates an empty cover with room for `capacity` disks.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            certificates: try_filled(
                capacity,
                workspace::blank_certificate(),
                scioxide_core::quantity!("certificates"),
            )?,
            used: 0,
        })
    }
    /// Returns the largest number of disks the cover can hold.
    pub fn capacity(&self) -> usize {
        self.certificates.len()
    }
    /// Returns the disks and counts written by the last isolation.
    pub fn view(&self) -> &[PolynomialRootCertificate<R>] {
        &self.certificates[..self.used]
    }
    /// Write staged certificates that fit this output.
    fn publish(&mut self, staged: &[PolynomialRootCertificate<R>]) {
        self.certificates[..staged.len()].copy_from_slice(staged);
        self.used = staged.len();
    }
}
/// A complex point and an absolute Euclidean error bound.
#[derive(Clone, Copy)]
struct Ball<R: RealScalar> {
    center: Complex<R>,
    radius: R,
}
impl<R: RealScalar> Ball<R> {
    fn zero() -> Self {
        Self {
            center: Complex::new(R::ZERO, R::ZERO),
            radius: R::ZERO,
        }
    }
    /// Enclose `self * center + addend`, propagating disk errors by |center|.
    /// Each component has two products and two additions, with depth three.
    /// Its forward error is bounded by gamma_3 times the exact absolute mass,
    /// plus underflow. Summing the six rounded absolute terms has depth five.
    /// 16*epsilon times this mass bounds the sum of both component errors,
    /// including mass reduction and bound rounding; 16*eta covers underflow,
    /// where eta is the smallest positive subnormal. The L1 error bounds the
    /// Euclidean error, including without hardware FMA. Nonfinite ranges decline
    /// this test and let the outward Taylor expansion decide the disk instead.
    fn horner(self, center: Complex<R>, norm: R, addend: Self) -> Result<Option<Self>> {
        let ac = self.center.re * center.re;
        let bd = self.center.im * center.im;
        let ad = self.center.re * center.im;
        let bc = self.center.im * center.re;
        let center = Complex::new((ac - bd) + addend.center.re, (ad + bc) + addend.center.im);
        let mass = (((ac.abs() + bd.abs()) + ad.abs()) + bc.abs())
            + (addend.center.re.abs() + addend.center.im.abs());
        let sixteen = R::from_f64_ieee(16.0);
        let eta = R::ZERO.next_after(R::ONE);
        let error = add_up(mul_up(mass, sixteen * R::EPSILON), sixteen * eta);
        let radius = add_up(add_up(mul_up(self.radius, norm), addend.radius), error);
        Ok(
            (radius.is_finite() && center.re.is_finite() && center.im.is_finite())
                .then_some(Self { center, radius }),
        )
    }
}

/// Euclidean norm bounds of a finite point, with an exact axis case. Directed
/// squares avoid general interval dispatch; exponent extremes use scaled hypot.
fn point_modulus<R: RealScalar>(z: Complex<R>) -> Result<(R, R)> {
    let re = z.re.abs();
    let im = z.im.abs();
    if re == R::ZERO || im == R::ZERO {
        let norm = re.max(im);
        return Ok((norm, norm));
    }
    let (rl, rh) = crate::rounding::square_bounds(re, re);
    let (il, ih) = crate::rounding::square_bounds(im, im);
    let lo = -add_up(-rl, -il);
    let hi = add_up(rh, ih);
    if lo >= R::MIN_POSITIVE && hi.is_finite() {
        use crate::rounding::{Operation, endpoint};
        return Ok((
            endpoint(lo, R::ZERO, Operation::Sqrt, false),
            endpoint(hi, R::ZERO, Operation::Sqrt, true),
        ));
    }
    Box2::point(z)?
        .modulus()?
        .bounds()
        .ok_or_else(|| crate::shell::breakdown("empty point modulus"))
}

/// An outward horizontal projection and its original disk index.
#[derive(Debug, Clone, Copy)]
pub(super) struct Projection<R: RealScalar> {
    lower: R,
    upper: R,
    index: usize,
}
impl<R: RealScalar> Projection<R> {
    pub(super) fn zero() -> Self {
        Self {
            lower: R::ZERO,
            upper: R::ZERO,
            index: 0,
        }
    }
}
enum Separation {
    Disjoint,
    Overlap(usize, usize),
    Stopped(RootInconclusive),
}
/// Find the first overlapping pair in input order, pruning disjoint projections.
fn sweep<R: RealScalar>(
    projections: &mut [Projection<R>],
    disk: impl Fn(usize) -> RootDisk<R>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Separation> {
    for (index, projection) in projections.iter_mut().enumerate() {
        if let Some(reason) = iteration(context)? {
            return Ok(Separation::Stopped(reason));
        }
        let d = disk(index);
        *projection = Projection {
            lower: -add_up(-d.center.re, d.radius),
            upper: add_up(d.center.re, d.radius),
            index,
        };
    }
    projections.sort_unstable_by(|a, b| {
        a.lower
            .partial_cmp(&b.lower)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
    let mut first = None;
    for (i, a) in projections.iter().enumerate() {
        for b in &projections[i + 1..] {
            if b.lower > a.upper {
                break;
            }
            if let Some(reason) = iteration(context)? {
                return Ok(Separation::Stopped(reason));
            }
            let pair = (a.index.min(b.index), a.index.max(b.index));
            if first.is_none_or(|old| pair < old) && !separated(disk(a.index), disk(b.index))? {
                first = Some(pair);
            }
        }
    }
    Ok(first.map_or(Separation::Disjoint, |(i, j)| Separation::Overlap(i, j)))
}

/// Strict disk separation.
fn separated<R: RealScalar>(a: RootDisk<R>, b: RootDisk<R>) -> Result<bool> {
    let distance = Box2::point(a.center)?
        .sub(Box2::point(b.center)?)?
        .modulus()?;
    let radius = add_up(a.radius, b.radius);
    Ok(distance.bounds().is_some_and(|(lo, _)| lo > radius))
}

/// Summary of a polynomial root isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct PolynomialIsolationReport {
    certificates: usize,
    roots: usize,
    termination: RootIsolationTermination,
}
impl PolynomialIsolationReport {
    /// Returns the number of disks written to the cover.
    pub const fn certificates(&self) -> usize {
        self.certificates
    }
    /// Returns the sum of their proved counts, with multiplicity.
    pub const fn roots(&self) -> usize {
        self.roots
    }
    /// Returns whether the disks contain every root of the polynomial.
    pub const fn is_complete(&self) -> bool {
        matches!(self.termination, RootIsolationTermination::Complete)
    }
    /// Returns the termination status.
    pub const fn termination(&self) -> RootIsolationTermination {
        self.termination
    }
}
impl<S: NumericScalar> PolynomialRootPlan<S> {
    /// Isolates the roots near `seeds` in disjoint disks with proved counts;
    /// see [`Self::isolate_into`].
    ///
    /// Allocates a cover with one slot per seed, which every isolation fits,
    /// and a workspace, and runs sequentially.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_validated::{PolynomialRootPlan, RootDisk, ValidatedOptions};
    ///
    /// // Seed disks around approximate roots of z² − 2.
    /// let coefficients = [-2.0_f64, 0.0, 1.0];
    /// let plan = PolynomialRootPlan::new(&coefficients, ValidatedOptions::new())?;
    /// let seed = |re: f64| RootDisk {
    ///     center: Complex::new(re, 0.0),
    ///     radius: 0.125,
    /// };
    /// let (report, cover) = plan.isolate(&[seed(1.4), seed(-1.4)])?;
    /// assert!(report.is_complete() && report.roots() == 2);
    /// assert!(cover.view().iter().all(|c| c.count == 1));
    /// assert!(plan.verify_cover(cover.view())?.is_proved());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::isolate_into`].
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn isolate(
        &self,
        seeds: &[RootDisk<S::Real>],
    ) -> Result<(PolynomialIsolationReport, PolynomialRootCover<S::Real>)> {
        let mut cover = PolynomialRootCover::new(seeds.len())?;
        let report = self.isolate_into(
            seeds,
            &mut cover,
            &mut RootWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok((report, cover))
    }
    /// Isolates the roots near `seeds` in disjoint disks with proved counts,
    /// and writes them to `out`.
    ///
    /// Seeds are disks around approximate roots, for example from
    /// `scioxide-rootfind`. The plan counts the roots in each disk. A disk whose
    /// count is not proved doubles its radius, and disks that overlap are
    /// merged, in input order; a changed disk is counted again. A radius
    /// that would overflow ends the isolation with
    /// [`RefinementLimit::Resolution`](crate::RefinementLimit::Resolution). The
    /// isolation is complete when the counts sum to the degree; with too few
    /// seeds it may not be. `out` must hold one disk per seed. Every `Ok`
    /// result, including a stop, writes the proved disks to `out`; on error
    /// `out` is unchanged.
    ///
    /// ```
    /// use scioxide_core::{Complex, ExecutionContext};
    /// use scioxide_validated::{
    ///     PolynomialRootCover, PolynomialRootPlan, ProofRequest, RootDisk, ValidatedOptions,
    /// };
    ///
    /// let coefficients = [-2.0_f64, 0.0, 1.0];
    /// let plan = PolynomialRootPlan::new(&coefficients, ValidatedOptions::new())?;
    /// let seed = |re: f64| RootDisk {
    ///     center: Complex::new(re, 0.0),
    ///     radius: 0.125,
    /// };
    /// let seeds = [seed(1.4), seed(-1.4)];
    /// let mut cover = PolynomialRootCover::new(seeds.len())?;
    /// let mut workspace = plan.workspace(ProofRequest::refinement(seeds.len()))?;
    /// let report = plan.isolate_into(
    ///     &seeds,
    ///     &mut cover,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!(report.is_complete() && report.roots() == 2);
    /// assert!(plan.verify_cover(cover.view())?.is_proved());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` has room for fewer disks than there are
    ///   seeds.
    /// - `NonFinite` if a seed center or radius is NaN or infinite.
    /// - `InvalidInput` if a seed radius is not positive, or if the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    /// - Errors of [`RootWorkspace::try_reserve`] if the workspace must grow.
    pub fn isolate_into<'s, 'c>(
        &self,
        seeds: &[RootDisk<S::Real>],
        out: &mut PolynomialRootCover<S::Real>,
        workspace: &mut RootWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<PolynomialIsolationReport> {
        let capacity = seeds.len();
        let report = workspace.core.run(
            scope.into(),
            self.identity(),
            &self.options,
            || {
                let requirements =
                    self.workspace_requirements(ProofRequest::refinement(capacity))?;
                crate::shell::at_least(capacity, out.capacity())?;
                seeds
                    .iter()
                    .try_for_each(|x| x.validate(scioxide_core::argument!("seeds")))?;
                Ok(requirements)
            },
            |_, _| Ok(()),
            |lanes, refinement, _, _, context| {
                let lane = &mut lanes[0];
                self.refine(seeds, &mut refinement.disks(capacity), lane, context)
            },
            PolynomialIsolationReport::is_complete,
        )?;
        out.publish(workspace.core.shared.staged_disks(report.certificates));
        Ok(report)
    }
    /// Grow, merge and count disks into the staged certificates.
    fn refine(
        &self,
        seeds: &[RootDisk<S::Real>],
        w: &mut workspace::DiskScratch<'_, S::Real>,
        lane: &mut workspace::Lane<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PolynomialIsolationReport> {
        // A merged or doubled radius beyond the finite range leaves no
        // representable radius increase.
        const RESOLUTION: Option<RootIsolationTermination> = Some(RootIsolationTermination::Limit(
            crate::RefinementLimit::Resolution,
        ));
        w.disks.copy_from_slice(seeds);
        w.counts.fill(None);
        let mut active = seeds.len();
        let reason = loop {
            if let Some(reason) = iteration(context)? {
                break Some(reason.into());
            }
            // Merge before counting, so published disks are always disjoint.
            loop {
                let (i, j) = match sweep(lane.projections(active), |i| w.disks[i], context)? {
                    Separation::Disjoint => break,
                    Separation::Overlap(i, j) => (i, j),
                    Separation::Stopped(reason) => {
                        return self.stage(w, active, Some(reason.into()), context);
                    }
                };
                let distance = upper(
                    Box2::point(w.disks[i].center)?
                        .sub(Box2::point(w.disks[j].center)?)?
                        .modulus()?,
                );
                let radius = w.disks[i].radius.max(add_up(distance, w.disks[j].radius));
                if !radius.is_finite() {
                    return self.stage(w, active, RESOLUTION, context);
                }
                w.disks[i].radius = radius;
                w.counts[i] = None;
                w.disks.copy_within(j + 1..active, j);
                w.counts.copy_within(j + 1..active, j);
                active -= 1;
            }
            for i in 0..active {
                if w.counts[i].is_some() {
                    continue;
                }
                let mut scratch = lane.polynomial(self.coefficients.len());
                let counted = self.count_raw(
                    w.disks[i],
                    &mut scratch,
                    context,
                    scioxide_core::argument!("seeds"),
                );
                // Claim the established count, so a counted disk is proved and committed.
                let counted = counted.map(|counted| report(counted.unwrap_or(0), counted));
                let report = finish(counted, context)?;
                match (report.count(), report.status()) {
                    (Some(n), _) if n > 0 => w.counts[i] = Some(n),
                    (_, RootVerificationStatus::Inconclusive(reason))
                        if reason != RootInconclusive::Reason(RootProofFailure::Inequality) =>
                    {
                        return self.stage(w, active, Some(reason.into()), context);
                    }
                    _ => {}
                }
            }
            let count = w.counts[..active].iter().try_fold(0usize, |sum, n| {
                checked_add(sum, n.unwrap_or(0), scioxide_core::quantity!("root_count"))
            })?;
            if count == self.degree() {
                break None;
            }
            if w.counts[..active].iter().all(Option::is_some) {
                break Some(RootInconclusive::Reason(RootProofFailure::Inequality).into());
            }
            for (disk, count) in w.disks[..active].iter_mut().zip(&mut w.counts[..active]) {
                if count.is_none() {
                    let radius = mul_up(disk.radius, S::Real::TWO);
                    if !radius.is_finite() {
                        return self.stage(w, active, RESOLUTION, context);
                    }
                    disk.radius = radius;
                }
            }
        };
        self.stage(w, active, reason, context)
    }
    /// Stage the proved counts of the first `active` disks.
    fn stage(
        &self,
        w: &mut workspace::DiskScratch<'_, S::Real>,
        active: usize,
        reason: Option<RootIsolationTermination>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PolynomialIsolationReport> {
        let mut used = 0;
        let mut roots = 0;
        for (&disk, &count) in w.disks[..active].iter().zip(&w.counts[..active]) {
            if let Some(count) = count {
                w.staged[used] = PolynomialRootCertificate { disk, count };
                used += 1;
                roots = checked_add(roots, count, scioxide_core::quantity!("root_count"))?;
            }
        }
        // Only uncounted disks grow; a merge invalidates its destination count.
        // The remaining counted disks therefore stay pairwise disjoint, including
        // when interrupted mid-merge. No new proof work is performed after a stop.
        let reason = stop(context)?
            .map(RootIsolationTermination::Inconclusive)
            .or(reason);
        if used > 0 {
            context.commit_accepted(PartialResultTrust::CertificateVerified)?;
        }
        Ok(PolynomialIsolationReport {
            certificates: used,
            roots,
            termination: reason
                .or_else(|| {
                    (roots != self.degree())
                        .then_some(RootInconclusive::Reason(RootProofFailure::Inequality).into())
                })
                .unwrap_or(RootIsolationTermination::Complete),
        })
    }
}

pub(crate) const POLYNOMIAL_COUNT: scioxide_core::AlgorithmId =
    scioxide_core::algorithm_id!("validated-roots", "polynomial-count", 1);

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_horner<R: RealScalar>() {
        let exponents = if R::MANTISSA_DIGITS == 24 {
            [-149, -126, -60, -1, 0, 1, 60, 126]
        } else {
            [-1074, -1022, -500, -1, 0, 1, 500, 1022]
        };
        let mut proved = 0;
        for a in exponents {
            for b in exponents {
                let x = Complex::new(
                    (R::ONE + R::EPSILON).scalbn(a),
                    -(R::ONE - R::EPSILON).scalbn(a),
                );
                let z = Complex::new(
                    (R::ONE - R::EPSILON).scalbn(b),
                    R::from_f64_ieee(0.7).scalbn(b),
                );
                let norm = point_modulus(z).unwrap().1;
                for cancel in [false, true] {
                    let addend = if cancel {
                        Complex::new(-(x.re * z.re - x.im * z.im), -(x.re * z.im + x.im * z.re))
                    } else {
                        Complex::new(R::ONE, -R::ONE)
                    };
                    if !addend.re.is_finite() || !addend.im.is_finite() {
                        continue;
                    }
                    let input = Ball {
                        center: x,
                        radius: R::ZERO,
                    };
                    let Some(output) = input
                        .horner(
                            z,
                            norm,
                            Ball {
                                center: addend,
                                radius: R::ZERO,
                            },
                        )
                        .unwrap()
                    else {
                        continue;
                    };
                    // Independent exact product sums enclose the represented
                    // expression, including cancellation and underflow.
                    let mut exact = BoxSum::new(
                        Box2::point(addend).unwrap(),
                        scioxide_core::argument!("addend"),
                    )
                    .unwrap();
                    exact
                        .product(Box2::point(x).unwrap(), Box2::point(z).unwrap(), false)
                        .unwrap();
                    let delta = exact
                        .finish()
                        .unwrap()
                        .sub(Box2::point(output.center).unwrap())
                        .unwrap();
                    assert!(
                        upper(delta.modulus().unwrap()) <= output.radius,
                        "width {}, exponents {a}, {b}, cancel {cancel}",
                        R::MANTISSA_DIGITS
                    );
                    proved += 1;
                }
            }
        }
        assert!(proved >= 64);
        let input = Ball {
            center: Complex::new(R::from_f64_ieee(0.75), -R::HALF),
            radius: R::from_f64_ieee(0.0625),
        };
        let addend = Ball {
            center: Complex::new(R::from_f64_ieee(0.25), -R::from_f64_ieee(0.75)),
            radius: R::from_f64_ieee(0.03125),
        };
        let z = Complex::new(R::from_f64_ieee(1.125), R::from_f64_ieee(0.875));
        let output = input
            .horner(z, point_modulus(z).unwrap().1, addend)
            .unwrap()
            .unwrap();
        // Cardinal boundary points independently exercise both disk errors.
        let offsets = |r| {
            [
                Complex::new(r, R::ZERO),
                Complex::new(-r, R::ZERO),
                Complex::new(R::ZERO, r),
                Complex::new(R::ZERO, -r),
            ]
        };
        for a in offsets(input.radius) {
            for b in offsets(addend.radius) {
                let mut exact = BoxSum::new(
                    Box2::point(addend.center + b).unwrap(),
                    scioxide_core::argument!("addend"),
                )
                .unwrap();
                exact
                    .product(
                        Box2::point(input.center + a).unwrap(),
                        Box2::point(z).unwrap(),
                        false,
                    )
                    .unwrap();
                let delta = exact
                    .finish()
                    .unwrap()
                    .sub(Box2::point(output.center).unwrap())
                    .unwrap();
                assert!(upper(delta.modulus().unwrap()) <= output.radius);
            }
        }
        let large = Ball {
            center: Complex::new(R::MAX, R::MAX),
            radius: R::ZERO,
        };
        assert!(
            large
                .horner(
                    Complex::new(R::TWO, R::TWO),
                    R::from_f64_ieee(3.0),
                    Ball::zero()
                )
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn point_horner_encloses_exact_sums_at_both_precisions() {
        disk_horner::<f32>();
        disk_horner::<f64>();
    }

    #[test]
    fn coefficient_scaling_keeps_destructive_underflow() {
        let eta = f64::from_bits(1);
        let plan = PolynomialRootPlan::new(&[eta, f64::MAX], ValidatedOptions::new()).unwrap();
        let coefficient = plan.coefficients[0];
        assert!(coefficient.ball.radius > 0.0);
        assert!(coefficient.magnitude > 0.0);
        let (lo, hi) = coefficient.enclosure.re.bounds().unwrap();
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0);
    }
}
