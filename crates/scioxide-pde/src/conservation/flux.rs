use super::{finite, invalid, number};
use crate::fvm::geometry::dot;
use scioxide_core::{RealScalar, Scaled, SciOxideError};

/// A homogeneous real conservation law with `N` conserved components in `D` dimensions.
///
/// Providers are pure and must return finite fluxes and enclosing normal wave speeds.
/// Admissibility is a convex set, checked at stage inputs and accepted Euler/Heun candidates.
/// Custom laws own the validity of their wave bounds and convexity claims.
pub trait ConservationLaw<R: RealScalar, const D: usize, const N: usize>: Sync {
    /// Validate dimension and immutable equation parameters before preparation.
    fn validate(&self) -> Result<(), SciOxideError>;
    /// Whether a finite conserved state belongs to the supported physical domain.
    fn admissible(&self, state: &[R; N]) -> bool;
    /// Physical flux in the supplied unit normal direction.
    /// # Errors
    /// Rejects inadmissible states or unrepresentable arithmetic.
    fn normal_flux(&self, state: &[R; N], normal: [R; D]) -> Result<[R; N], SciOxideError>;
    /// Enclosing left/right signal velocities for two states.
    /// # Errors
    /// Rejects inadmissible states or unrepresentable arithmetic.
    fn wave_bounds(
        &self,
        left: &[R; N],
        right: &[R; N],
        normal: [R; D],
    ) -> Result<(R, R), SciOxideError>;
    /// Compute both physical normal fluxes and enclosing signal velocities.
    ///
    /// The default validates the Riemann inputs and calls [`Self::normal_flux`]
    /// and [`Self::wave_bounds`]. Laws may share primitive recovery between them;
    /// the Euler implementation uses `O(D + N)` arithmetic and constant storage.
    /// Accuracy and bounds are those of the individual law methods.
    /// # Errors
    /// Rejects invalid equation parameters, states, normals and unrepresentable results.
    /// # References
    /// Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., ch. 10.
    ///
    /// ```
    /// use scioxide_pde::{ConservationLaw, IdealGasEuler};
    /// let gas = IdealGasEuler::new(1.4_f64)?;
    /// let u: [f64; 3] = gas.conserved(1.0, [0.0], 1.0)?;
    /// let data = gas.riemann_data(&u, &u, [1.0])?;
    /// assert!((data.left_flux[1] - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn riemann_data(
        &self,
        left: &[R; N],
        right: &[R; N],
        normal: [R; D],
    ) -> Result<RiemannData<R, N>, SciOxideError> {
        check(self, left, right, normal)?;
        Ok(RiemannData {
            left_flux: self.normal_flux(left, normal)?,
            right_flux: self.normal_flux(right, normal)?,
            wave_bounds: self.wave_bounds(left, right, normal)?,
        })
    }
    /// Reflect a state at an impermeable slip wall, when supported.
    /// # Errors
    /// The default rejects laws without a reflection contract.
    fn reflect(&self, _state: &[R; N], _normal: [R; D]) -> Result<[R; N], SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("self"),
            "this equation does not support slip walls",
        ))
    }
}

/// Supplies physical fluxes and enclosing signal velocities for one oriented face.
///
/// This immutable value shares a law's state recovery between numerical fluxes.
#[derive(Debug, Clone, Copy)]
pub struct RiemannData<R, const N: usize> {
    /// Physical flux of the left state in the face normal direction.
    pub left_flux: [R; N],
    /// Physical flux of the right state in the same normal direction.
    pub right_flux: [R; N],
    /// Ordered finite lower and upper signal velocities.
    pub wave_bounds: (R, R),
}

/// One oriented numerical flux per unit face measure, with an enclosing speed.
#[derive(Debug, Clone, Copy)]
pub struct RiemannReport<R, const N: usize> {
    pub(crate) flux: [R; N],
    pub(crate) maximum_speed: R,
    pub(crate) used_hlle_fallback: bool,
}
impl<R: Copy, const N: usize> RiemannReport<R, N> {
    /// Report the flux from the left toward the right state, the largest magnitude of
    /// the enclosing signal velocities and whether an HLLE fallback replaced HLLC.
    pub const fn new(flux: [R; N], maximum_speed: R, used_hlle_fallback: bool) -> Self {
        Self {
            flux,
            maximum_speed,
            used_hlle_fallback,
        }
    }
    /// Flux from the left state toward the right state.
    pub const fn flux(&self) -> [R; N] {
        self.flux
    }
    /// Largest magnitude of the enclosing left/right signal velocities.
    pub const fn maximum_speed(&self) -> R {
        self.maximum_speed
    }
    /// Whether HLLC replaced an invalid star state by HLLE.
    pub const fn used_hlle_fallback(&self) -> bool {
        self.used_hlle_fallback
    }
}

/// Pure numerical flux capability paired with a compatible equation.
pub trait NumericalFlux<R: RealScalar, E, const D: usize, const N: usize>: Sync
where
    E: ConservationLaw<R, D, N>,
{
    /// Evaluate the same flux for both incident cells.
    /// # Errors
    /// Rejects invalid states, non-unit normals, and numerical breakdown.
    fn evaluate(
        &self,
        equation: &E,
        left: &[R; N],
        right: &[R; N],
        normal: [R; D],
    ) -> Result<RiemannReport<R, N>, SciOxideError>;
}

fn check<R: RealScalar, E: ConservationLaw<R, D, N> + ?Sized, const D: usize, const N: usize>(
    equation: &E,
    left: &[R; N],
    right: &[R; N],
    normal: [R; D],
) -> Result<(), SciOxideError> {
    equation.validate()?;
    let left_valid = equation.admissible(left);
    if !left_valid || !equation.admissible(right) {
        return Err(invalid(
            if left_valid {
                scioxide_core::argument!("right")
            } else {
                scioxide_core::argument!("left")
            },
            "inadmissible Riemann state",
        ));
    }
    check_normal(normal)
}
fn check_normal<R: RealScalar, const D: usize>(normal: [R; D]) -> Result<(), SciOxideError> {
    let norm = dot(normal, normal);
    if normal.iter().any(|x| !x.is_finite())
        || !norm.is_finite()
        || (norm - R::ONE).abs() > number::<R>(128)? * R::EPSILON
    {
        return Err(invalid(
            scioxide_core::argument!("normal"),
            "Riemann normal must be a finite unit vector",
        ));
    }
    Ok(())
}
fn report<R: RealScalar, const N: usize>(
    flux: [R; N],
    speed: R,
    fallback: bool,
) -> Result<RiemannReport<R, N>, SciOxideError> {
    for &x in &flux {
        finite(x, scioxide_core::NonFiniteOrigin::Intermediate)?;
    }
    if !speed.is_finite() || speed < R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("equation"),
            "invalid Riemann speed bound",
        ));
    }
    Ok(RiemannReport {
        flux,
        maximum_speed: speed,
        used_hlle_fallback: fallback,
    })
}

/// Constant-velocity scalar transport `u_t + velocity · grad(u) = 0`.
#[derive(Debug, Clone, Copy)]
pub struct LinearAdvection<R, const D: usize> {
    /// Finite transport velocity.
    pub velocity: [R; D],
}
/// Shared validation of an advection velocity or Burgers direction field.
fn validate_velocity<R: RealScalar, const D: usize>(
    velocity: &[R; D],
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !(1..=3).contains(&D) || velocity.iter().any(|x| !x.is_finite()) {
        return Err(invalid(argument, "invalid advection dimension or velocity"));
    }
    Ok(())
}
impl<R: RealScalar, const D: usize> ConservationLaw<R, D, 1> for LinearAdvection<R, D> {
    fn validate(&self) -> Result<(), SciOxideError> {
        validate_velocity(&self.velocity, scioxide_core::argument!("self.velocity"))
    }
    fn admissible(&self, u: &[R; 1]) -> bool {
        u[0].is_finite()
    }
    fn normal_flux(&self, u: &[R; 1], n: [R; D]) -> Result<[R; 1], SciOxideError> {
        Ok([finite(
            dot(self.velocity, n) * u[0],
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?])
    }
    fn wave_bounds(&self, _: &[R; 1], _: &[R; 1], n: [R; D]) -> Result<(R, R), SciOxideError> {
        let a = finite(
            dot(self.velocity, n),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        Ok((a, a))
    }
}

/// Directional scalar Burgers law with flux `direction * u² / 2`.
#[derive(Debug, Clone, Copy)]
pub struct Burgers<R, const D: usize> {
    /// Finite flux direction, not necessarily a unit vector.
    pub direction: [R; D],
}
impl<R: RealScalar, const D: usize> ConservationLaw<R, D, 1> for Burgers<R, D> {
    fn validate(&self) -> Result<(), SciOxideError> {
        validate_velocity(&self.direction, scioxide_core::argument!("self.direction"))
    }
    fn admissible(&self, u: &[R; 1]) -> bool {
        u[0].is_finite()
    }
    fn normal_flux(&self, u: &[R; 1], n: [R; D]) -> Result<[R; 1], SciOxideError> {
        Ok([finite(
            (dot(self.direction, n) * u[0]) * (u[0] * R::HALF),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?])
    }
    fn wave_bounds(&self, l: &[R; 1], r: &[R; 1], n: [R; D]) -> Result<(R, R), SciOxideError> {
        let a = dot(self.direction, n);
        let x = finite(a * l[0], scioxide_core::NonFiniteOrigin::Intermediate)?;
        let y = finite(a * r[0], scioxide_core::NonFiniteOrigin::Intermediate)?;
        Ok((x.min(y), x.max(y)))
    }
}

/// Exact scalar Godunov flux for advection and directional Burgers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Godunov;
impl<R: RealScalar, const D: usize> NumericalFlux<R, LinearAdvection<R, D>, D, 1> for Godunov {
    fn evaluate(
        &self,
        e: &LinearAdvection<R, D>,
        l: &[R; 1],
        r: &[R; 1],
        n: [R; D],
    ) -> Result<RiemannReport<R, 1>, SciOxideError> {
        check(e, l, r, n)?;
        let a = dot(e.velocity, n);
        report(
            e.normal_flux(if a >= R::ZERO { l } else { r }, n)?,
            a.abs(),
            false,
        )
    }
}
impl<R: RealScalar, const D: usize> NumericalFlux<R, Burgers<R, D>, D, 1> for Godunov {
    fn evaluate(
        &self,
        e: &Burgers<R, D>,
        l: &[R; 1],
        r: &[R; 1],
        n: [R; D],
    ) -> Result<RiemannReport<R, 1>, SciOxideError> {
        check(e, l, r, n)?;
        let a = dot(e.direction, n);
        let (sl, sr) = e.wave_bounds(l, r, n)?;
        let fl = e.normal_flux(l, n)?[0];
        let fr = e.normal_flux(r, n)?[0];
        // Godunov selects min over [l,r] for l<=r, max over [r,l] otherwise,
        // including the stationary point of a concave or convex quadratic.
        let mut f = if l[0] <= r[0] { fl.min(fr) } else { fl.max(fr) };
        if l[0].min(r[0]) <= R::ZERO && l[0].max(r[0]) >= R::ZERO {
            if l[0] <= r[0] && a >= R::ZERO {
                f = R::ZERO;
            }
            if l[0] > r[0] && a < R::ZERO {
                f = R::ZERO;
            }
        }
        report([f], sl.abs().max(sr.abs()), false)
    }
}

/// Local Lax–Friedrichs flux using the equation's enclosing wave speeds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rusanov;
impl<R: RealScalar, E: ConservationLaw<R, D, N>, const D: usize, const N: usize>
    NumericalFlux<R, E, D, N> for Rusanov
{
    fn evaluate(
        &self,
        e: &E,
        l: &[R; N],
        r: &[R; N],
        n: [R; D],
    ) -> Result<RiemannReport<R, N>, SciOxideError> {
        let RiemannData {
            left_flux: fl,
            right_flux: fr,
            wave_bounds: (sl, sr),
        } = e.riemann_data(l, r, n)?;
        if !sl.is_finite() || !sr.is_finite() || sl > sr {
            return Err(invalid(
                scioxide_core::argument!("equation"),
                "unordered wave speeds",
            ));
        }
        let a = sl.abs().max(sr.abs());
        if l == r {
            return report(fl, a, false);
        }
        let mut flux = core::array::from_fn(|i| {
            fl[i] * R::HALF + fr[i] * R::HALF - (a * R::HALF) * (r[i] - l[i])
        });
        if !(a * R::HALF).is_normal() && a != R::ZERO || flux.iter().any(|x| !x.is_finite()) {
            let half = Scaled::from_value(R::HALF)?;
            let speed = Scaled::from_value(a)?;
            for i in 0..N {
                flux[i] = Scaled::from_value(fl[i])?
                    .add(Scaled::from_value(fr[i])?)?
                    .sub(speed.mul(Scaled::from_value(r[i])?.sub(Scaled::from_value(l[i])?)?)?)?
                    .mul(half)?
                    .round()?;
            }
        }
        report(flux, a, false)
    }
}

/// Calorically perfect Euler gas. States are `[rho, rho*v_0, ..., rho*v_(D-1), E]`.
/// `N` must equal `D+2`. Exact vacuum is outside this API's domain.
#[derive(Debug, Clone, Copy)]
pub struct IdealGasEuler<R> {
    gamma: R,
    density_minimum: R,
    pressure_minimum: R,
}
/// Recovered once for each state of an Euler Riemann problem.
#[derive(Clone, Copy)]
struct EulerPrimitive<R, const D: usize> {
    pressure: R,
    velocity: [R; D],
    normal_velocity: R,
    sound_speed: R,
}
impl<R: RealScalar, const D: usize> EulerPrimitive<R, D> {
    fn signals(self) -> Result<(R, R), SciOxideError> {
        Ok((
            finite(
                self.normal_velocity - self.sound_speed,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
            finite(
                self.normal_velocity + self.sound_speed,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
        ))
    }
    fn flux<const N: usize>(self, u: &[R; N], n: [R; D]) -> Result<[R; N], SciOxideError> {
        let mut flux = core::array::from_fn(|i| u[i] * self.normal_velocity);
        // Mass flux is the normal momentum itself, including tiny velocities
        // whose product with density remains representable.
        flux[0] = (0..D).fold(R::ZERO, |sum, j| sum + u[j + 1] * n[j]);
        for j in 0..D {
            flux[j + 1] += self.pressure * n[j];
        }
        flux[N - 1] = (u[N - 1] + self.pressure) * self.normal_velocity;
        if flux.iter().all(|x| x.is_finite())
            && (self.normal_velocity.is_normal() || flux[0] == R::ZERO)
        {
            return Ok(flux);
        }
        let mut momentum = Scaled::ZERO;
        for j in 0..D {
            momentum =
                momentum.add(Scaled::from_value(u[j + 1])?.mul(Scaled::from_value(n[j])?)?)?;
        }
        let v = momentum.div(Scaled::from_value(u[0])?)?;
        let p = Scaled::from_value(self.pressure)?;
        for i in 0..N {
            flux[i] = if i == N - 1 {
                Scaled::from_value(u[i])?.add(p)?.mul(v)?.round()?
            } else if i > 0 {
                Scaled::from_value(u[i])?
                    .mul(v)?
                    .add(p.mul(Scaled::from_value(n[i - 1])?)?)?
                    .round()?
            } else {
                momentum.round()?
            };
        }
        Ok(flux)
    }
}
impl<R: RealScalar> IdealGasEuler<R> {
    /// Construct a gas with strict positive density/pressure and finite `gamma > 1`.
    /// # Errors
    /// Rejects an invalid heat-capacity ratio.
    pub fn new(gamma: R) -> Result<Self, SciOxideError> {
        if !gamma.is_finite() || gamma <= R::ONE {
            return Err(invalid(
                scioxide_core::argument!("gamma"),
                "Euler gamma must be finite and greater than one",
            ));
        }
        Ok(Self {
            gamma,
            density_minimum: R::ZERO,
            pressure_minimum: R::ZERO,
        })
    }
    /// Set nonnegative finite lower bounds; admissible values must exceed both.
    /// # Errors
    /// Rejects negative or nonfinite bounds.
    pub fn with_minima(mut self, density: R, pressure: R) -> Result<Self, SciOxideError> {
        if !density.is_finite() || !pressure.is_finite() || density < R::ZERO || pressure < R::ZERO
        {
            return Err(invalid(
                if !density.is_finite() || density < R::ZERO {
                    scioxide_core::argument!("density")
                } else {
                    scioxide_core::argument!("pressure")
                },
                "invalid Euler admissibility minima",
            ));
        }
        self.density_minimum = density;
        self.pressure_minimum = pressure;
        Ok(self)
    }
    /// Heat-capacity ratio.
    pub const fn gamma(&self) -> R {
        self.gamma
    }
    /// Pressure derived from a conserved state, including transverse kinetic energy.
    /// # Errors
    /// Rejects malformed, nonfinite or inadmissible states and cancellation to zero pressure.
    pub fn pressure<const D: usize, const N: usize>(&self, u: &[R; N]) -> Result<R, SciOxideError> {
        self.checked_pressure::<D, N>(u, scioxide_core::argument!("u"))
    }
    /// Shared pressure kernel; `argument` names the caller's conserved state.
    fn checked_pressure<const D: usize, const N: usize>(
        &self,
        u: &[R; N],
        argument: scioxide_core::Argument,
    ) -> Result<R, SciOxideError> {
        if !(1..=3).contains(&D)
            || N != D + 2
            || u.iter().any(|x| !x.is_finite())
            || u[0] <= self.density_minimum
        {
            return Err(invalid(argument, "inadmissible Euler density or shape"));
        }
        let mut kinetic = R::ZERO;
        let mut native = true;
        for &m in &u[1..=D] {
            let v = m / u[0];
            let half_m = m * R::HALF;
            let term = v * half_m;
            native &= m == R::ZERO || (v.is_normal() && half_m.is_normal() && term.is_normal());
            kinetic += term;
        }
        let internal = u[N - 1] - kinetic;
        let pressure = (self.gamma - R::ONE) * internal;
        let p = if native && kinetic.is_finite() && internal.is_finite() && pressure.is_finite() {
            pressure
        } else {
            self.pressure_extended::<D, N>(u)?
        };
        if p <= self.pressure_minimum {
            return Err(invalid(argument, "inadmissible Euler pressure"));
        }
        Ok(p)
    }
    #[cold]
    fn pressure_extended<const D: usize, const N: usize>(
        &self,
        u: &[R; N],
    ) -> Result<R, SciOxideError> {
        let rho = Scaled::from_value(u[0])?;
        let half = Scaled::from_value(R::HALF)?;
        let mut kinetic = Scaled::ZERO;
        for &m in &u[1..=D] {
            let m = Scaled::from_value(m)?;
            kinetic = kinetic.add(m.div(rho)?.mul(m)?.mul(half)?)?;
        }
        Scaled::from_value(u[N - 1])?
            .sub(kinetic)?
            .mul(Scaled::from_value(self.gamma - R::ONE)?)?
            .round()
    }
    /// Build a conserved state from positive density/pressure and finite velocity.
    /// # Errors
    /// Rejects wrong component count, inadmissibility or range overflow.
    pub fn conserved<const D: usize, const N: usize>(
        &self,
        rho: R,
        velocity: [R; D],
        pressure: R,
    ) -> Result<[R; N], SciOxideError> {
        let invalid_rho = !rho.is_finite() || rho <= self.density_minimum;
        let invalid_pressure = !pressure.is_finite() || pressure <= self.pressure_minimum;
        let invalid_velocity = velocity.iter().any(|v| !v.is_finite());
        if N != D + 2
            || !(1..=3).contains(&D)
            || invalid_rho
            || invalid_pressure
            || invalid_velocity
        {
            return Err(invalid(
                if invalid_rho {
                    scioxide_core::argument!("rho")
                } else if invalid_pressure {
                    scioxide_core::argument!("pressure")
                } else if invalid_velocity {
                    scioxide_core::argument!("velocity")
                } else {
                    scioxide_core::argument!("self")
                },
                "invalid Euler primitive state",
            ));
        }
        let mut u = [R::ZERO; N];
        u[0] = rho;
        for j in 0..D {
            u[j + 1] = finite(
                rho * velocity[j],
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
        u[N - 1] = finite(
            pressure / (self.gamma - R::ONE)
                + velocity
                    .iter()
                    .zip(&u[1..=D])
                    .fold(R::ZERO, |s, (&v, &m)| s + (v * R::HALF) * m),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        self.checked_pressure::<D, N>(&u, scioxide_core::argument!("pressure"))?;
        Ok(u)
    }
    /// Recover a finite admissible state once; retain velocities for Roe and HLLC.
    fn primitive<const D: usize, const N: usize>(
        &self,
        u: &[R; N],
        n: [R; D],
        argument: scioxide_core::Argument,
    ) -> Result<EulerPrimitive<R, D>, SciOxideError> {
        let pressure = self.checked_pressure::<D, N>(u, argument)?;
        let mut velocity = [R::ZERO; D];
        for j in 0..D {
            velocity[j] = finite(
                u[j + 1] / u[0],
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
        let normal_velocity = finite(
            dot(velocity, n),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        let ratio = pressure / u[0];
        let square = self.gamma * ratio;
        let sound_speed = if ratio.is_normal() && square.is_normal() {
            square.sqrt()
        } else {
            Scaled::from_value(self.gamma)?
                .mul(Scaled::from_value(pressure)?)?
                .div(Scaled::from_value(u[0])?)?
                .sqrt()?
                .round()?
        };
        Ok(EulerPrimitive {
            pressure,
            velocity,
            normal_velocity,
            sound_speed,
        })
    }
    fn bounds<const D: usize, const N: usize>(
        &self,
        l: &[R; N],
        r: &[R; N],
        n: [R; D],
        left: EulerPrimitive<R, D>,
        right: EulerPrimitive<R, D>,
    ) -> Result<(R, R), SciOxideError> {
        let root_l = l[0].sqrt();
        let root_r = r[0].sqrt();
        let scale = root_l.max(root_r);
        let wl = root_l / scale;
        let wr = root_r / scale;
        let w = wl / (wl + wr);
        let velocity: [R; D] =
            core::array::from_fn(|j| w * left.velocity[j] + (R::ONE - w) * right.velocity[j]);
        let h_l = (l[N - 1] + left.pressure) / l[0];
        let h_r = (r[N - 1] + right.pressure) / r[0];
        let h = w * h_l + (R::ONE - w) * h_r;
        let c2 = (self.gamma - R::ONE) * (h - dot(velocity, velocity) * R::HALF);
        let c = if h_l.is_normal() && h_r.is_normal() && c2.is_normal() && c2 > R::ZERO {
            c2.sqrt()
        } else {
            self.roe_sound_extended(l, r, left, right, w)?
        };
        let v = dot(velocity, n);
        Ok((
            finite(
                (left.normal_velocity - left.sound_speed)
                    .min(right.normal_velocity - right.sound_speed)
                    .min(v - c),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
            finite(
                (left.normal_velocity + left.sound_speed)
                    .max(right.normal_velocity + right.sound_speed)
                    .max(v + c),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
        ))
    }
    #[cold]
    fn roe_sound_extended<const D: usize, const N: usize>(
        &self,
        l: &[R; N],
        r: &[R; N],
        left: EulerPrimitive<R, D>,
        right: EulerPrimitive<R, D>,
        w: R,
    ) -> Result<R, SciOxideError> {
        // Roe internal enthalpy equals the weighted endpoint internal enthalpies
        // plus w(1-w)|v_l-v_r|²/2. This avoids subtracting large kinetic energies.
        let w = Scaled::from_value(w)?;
        let other = Scaled::from_value(R::ONE)?.sub(w)?;
        let gamma = Scaled::from_value(self.gamma)?;
        let mut c2 = w
            .mul(gamma)?
            .mul(Scaled::from_value(left.pressure)?)?
            .div(Scaled::from_value(l[0])?)?
            .add(
                other
                    .mul(gamma)?
                    .mul(Scaled::from_value(right.pressure)?)?
                    .div(Scaled::from_value(r[0])?)?,
            )?;
        let mut jump2 = Scaled::ZERO;
        for j in 0..D {
            let jump = Scaled::from_value(left.velocity[j])?
                .sub(Scaled::from_value(right.velocity[j])?)?;
            jump2 = jump2.add(jump.square()?)?;
        }
        c2 = c2.add(
            w.mul(other)?
                .mul(jump2)?
                .mul(Scaled::from_value((self.gamma - R::ONE) * R::HALF)?)?,
        )?;
        c2.sqrt()?.round()
    }
}
impl<R: RealScalar, const D: usize, const N: usize> ConservationLaw<R, D, N> for IdealGasEuler<R> {
    fn validate(&self) -> Result<(), SciOxideError> {
        if !(1..=3).contains(&D) || N != D + 2 {
            Err(invalid(
                scioxide_core::argument!("self"),
                "Euler requires D=1..3 and N=D+2",
            ))
        } else {
            Ok(())
        }
    }
    fn admissible(&self, u: &[R; N]) -> bool {
        self.pressure::<D, N>(u).is_ok()
    }
    fn normal_flux(&self, u: &[R; N], n: [R; D]) -> Result<[R; N], SciOxideError> {
        self.primitive::<D, N>(u, n, scioxide_core::argument!("state"))?
            .flux(u, n)
    }
    fn wave_bounds(&self, l: &[R; N], r: &[R; N], n: [R; D]) -> Result<(R, R), SciOxideError> {
        let left = self.primitive::<D, N>(l, n, scioxide_core::argument!("left"))?;
        let right = self.primitive::<D, N>(r, n, scioxide_core::argument!("right"))?;
        self.bounds(l, r, n, left, right)
    }
    fn riemann_data(
        &self,
        l: &[R; N],
        r: &[R; N],
        n: [R; D],
    ) -> Result<RiemannData<R, N>, SciOxideError> {
        check_normal(n)?;
        let left = self.primitive::<D, N>(l, n, scioxide_core::argument!("left"))?;
        if l == r {
            let flux = left.flux(l, n)?;
            return Ok(RiemannData {
                left_flux: flux,
                right_flux: flux,
                wave_bounds: left.signals()?,
            });
        }
        let right = self.primitive::<D, N>(r, n, scioxide_core::argument!("right"))?;
        Ok(RiemannData {
            left_flux: left.flux(l, n)?,
            right_flux: right.flux(r, n)?,
            wave_bounds: self.bounds(l, r, n, left, right)?,
        })
    }
    fn reflect(&self, u: &[R; N], n: [R; D]) -> Result<[R; N], SciOxideError> {
        const STATE: scioxide_core::Argument = scioxide_core::argument!("state");
        self.checked_pressure::<D, N>(u, STATE)?;
        let m = (0..D).fold(R::ZERO, |s, j| s + u[j + 1] * n[j]);
        let mut reflected = *u;
        for j in 0..D {
            reflected[j + 1] = finite(
                u[j + 1] - (m + m) * n[j],
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
        self.checked_pressure::<D, N>(&reflected, STATE)?;
        Ok(reflected)
    }
}

/// Euler HLL flux with enclosing Einfeldt/Roe and endpoint wave bounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hlle;
fn hlle<R: RealScalar, const N: usize>(
    l: &[R; N],
    r: &[R; N],
    fl: [R; N],
    fr: [R; N],
    sl: R,
    sr: R,
    fallback: bool,
) -> Result<RiemannReport<R, N>, SciOxideError> {
    let speed = sl.abs().max(sr.abs());
    if l == r || sl >= R::ZERO {
        return report(fl, speed, fallback);
    }
    if sr <= R::ZERO {
        return report(fr, speed, fallback);
    }
    let width = sr - sl;
    if width <= R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("equation"),
            "degenerate HLL wave fan",
        ));
    }
    let left_weight = sr / width;
    let right_weight = sl / width;
    let flux = core::array::from_fn(|i| {
        left_weight * fl[i] - right_weight * fr[i] + (sl * left_weight) * (r[i] - l[i])
    });
    if width.is_finite()
        && left_weight.is_normal()
        && right_weight.is_normal()
        && (sl * left_weight).is_normal()
        && flux.iter().all(|x| x.is_finite())
    {
        return report(flux, speed, fallback);
    }
    hlle_extended(l, r, fl, fr, sl, sr, fallback)
}
#[cold]
fn hlle_extended<R: RealScalar, const N: usize>(
    l: &[R; N],
    r: &[R; N],
    fl: [R; N],
    fr: [R; N],
    sl: R,
    sr: R,
    fallback: bool,
) -> Result<RiemannReport<R, N>, SciOxideError> {
    let speed = sl.abs().max(sr.abs());
    let sl = Scaled::from_value(sl)?;
    let sr = Scaled::from_value(sr)?;
    let width = sr.sub(sl)?;
    let mut out = [R::ZERO; N];
    for i in 0..N {
        out[i] = sr
            .div(width)?
            .mul(Scaled::from_value(fl[i])?)?
            .sub(sl.div(width)?.mul(Scaled::from_value(fr[i])?)?)?
            .add(
                sl.mul(sr)?
                    .div(width)?
                    .mul(Scaled::from_value(r[i])?.sub(Scaled::from_value(l[i])?)?)?,
            )?
            .round()?;
    }
    report(out, speed, fallback)
}
impl<R: RealScalar, const D: usize, const N: usize> NumericalFlux<R, IdealGasEuler<R>, D, N>
    for Hlle
{
    fn evaluate(
        &self,
        e: &IdealGasEuler<R>,
        l: &[R; N],
        r: &[R; N],
        n: [R; D],
    ) -> Result<RiemannReport<R, N>, SciOxideError> {
        let data = e.riemann_data(l, r, n)?;
        hlle(
            l,
            r,
            data.left_flux,
            data.right_flux,
            data.wave_bounds.0,
            data.wave_bounds.1,
            false,
        )
    }
}
/// Euler contact-restoring HLLC, with explicit HLLE fallback for inadmissible star states.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hllc;
impl<R: RealScalar, const D: usize, const N: usize> NumericalFlux<R, IdealGasEuler<R>, D, N>
    for Hllc
{
    fn evaluate(
        &self,
        e: &IdealGasEuler<R>,
        l: &[R; N],
        r: &[R; N],
        n: [R; D],
    ) -> Result<RiemannReport<R, N>, SciOxideError> {
        const EQUATION: scioxide_core::Argument = scioxide_core::argument!("equation");
        check_normal(n)?;
        let left = e.primitive::<D, N>(l, n, scioxide_core::argument!("left"))?;
        // Consistency: identical states have exactly their physical flux and
        // endpoint eigenvalues, so no Roe average or star construction is needed.
        if l == r {
            let (sl, sr) = left.signals()?;
            return report(left.flux(l, n)?, sl.abs().max(sr.abs()), false);
        }
        let right = e.primitive::<D, N>(r, n, scioxide_core::argument!("right"))?;
        let (sl, sr) = e.bounds(l, r, n, left, right)?;
        let fl = left.flux(l, n)?;
        let fr = right.flux(r, n)?;
        if sl >= R::ZERO || sr <= R::ZERO {
            return hlle(l, r, fl, fr, sl, sr, false);
        }
        let attempt = || -> Result<[R; N], SciOxideError> {
            let (pl, vl) = (left.pressure, left.normal_velocity);
            let (pr, vr) = (right.pressure, right.normal_velocity);
            let dl = finite(
                l[0] * (sl - vl),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            let dr = finite(
                r[0] * (sr - vr),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            let denominator = finite(dl - dr, scioxide_core::NonFiniteOrigin::Intermediate)?;
            if denominator == R::ZERO {
                return Err(invalid(EQUATION, "degenerate HLLC contact"));
            }
            // These products precede division by the contact denominator. A tiny
            // rounded product may otherwise lose a representable contact velocity.
            if !dl.is_normal()
                || !dr.is_normal()
                || !denominator.is_normal()
                || (!pl.is_normal() || !pr.is_normal())
                || (!(dl * vl).is_normal() && vl != R::ZERO)
                || (!(dr * vr).is_normal() && vr != R::ZERO)
            {
                return Err(invalid(EQUATION, "HLLC contact requires extended range"));
            }
            let sm = finite(
                (pr - pl + dl * vl - dr * vr) / denominator,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            if sm <= sl || sm >= sr {
                return Err(invalid(EQUATION, "HLLC contact outside signal bounds"));
            }
            let contact_pressure_left = finite(
                pl + dl * (sm - vl),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            let contact_pressure_right = finite(
                pr + dr * (sm - vr),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            if contact_pressure_left <= e.pressure_minimum
                || contact_pressure_right <= e.pressure_minimum
            {
                return Err(invalid(EQUATION, "inadmissible HLLC contact pressure"));
            }
            let star = |u: &[R; N], p: R, v: R, s: R| -> Result<[R; N], SciOxideError> {
                let d = s - sm;
                let ds = s - v;
                if d == R::ZERO || ds == R::ZERO {
                    return Err(invalid(EQUATION, "degenerate HLLC star"));
                }
                let rho = finite(
                    u[0] * (ds / d),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                let mut us = [R::ZERO; N];
                us[0] = rho;
                for j in 0..D {
                    us[j + 1] = finite(
                        rho * (u[j + 1] / u[0] + (sm - v) * n[j]),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
                us[N - 1] = finite(
                    rho * (u[N - 1] / u[0] + (sm - v) * (sm + p / (u[0] * ds))),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                e.checked_pressure::<D, N>(&us, EQUATION)?;
                Ok(us)
            };
            let ls = star(l, pl, vl, sl)?;
            let rs = star(r, pr, vr, sr)?;
            let (u, us, f, s) = if sm >= R::ZERO {
                (l, ls, fl, sl)
            } else {
                (r, rs, fr, sr)
            };
            let out = core::array::from_fn(|i| f[i] + s * (us[i] - u[i]));
            for &x in &out {
                finite(x, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
            Ok(out)
        };
        match attempt().or_else(|_| hllc_extended(e, l, r, n, left, right, fl, fr, sl, sr)) {
            Ok(f) => report(f, sl.abs().max(sr.abs()), false),
            Err(_) => hlle(l, r, fl, fr, sl, sr, true),
        }
    }
}

/// Recompute the HLLC contact and both star states without intermediate range loss.
#[cold]
#[expect(
    clippy::too_many_arguments,
    reason = "the face operands remain explicit"
)]
fn hllc_extended<R: RealScalar, const D: usize, const N: usize>(
    e: &IdealGasEuler<R>,
    l: &[R; N],
    r: &[R; N],
    n: [R; D],
    left: EulerPrimitive<R, D>,
    right: EulerPrimitive<R, D>,
    fl: [R; N],
    fr: [R; N],
    sl: R,
    sr: R,
) -> Result<[R; N], SciOxideError> {
    let value = Scaled::from_value;
    let pl = value(left.pressure)?;
    let pr = value(right.pressure)?;
    let vl = value(left.normal_velocity)?;
    let vr = value(right.normal_velocity)?;
    let lower = value(sl)?;
    let upper = value(sr)?;
    let dl = value(l[0])?.mul(lower.sub(vl)?)?;
    let dr = value(r[0])?.mul(upper.sub(vr)?)?;
    let denominator = dl.sub(dr)?;
    if denominator.is_zero() {
        return Err(invalid(
            scioxide_core::argument!("equation"),
            "degenerate HLLC contact",
        ));
    }
    let sm = pr
        .sub(pl)?
        .add(dl.mul(vl)?)?
        .sub(dr.mul(vr)?)?
        .div(denominator)?;
    if sm.cmp_value(lower).is_le() || sm.cmp_value(upper).is_ge() {
        return Err(invalid(
            scioxide_core::argument!("equation"),
            "HLLC contact outside signal bounds",
        ));
    }
    let minimum = value(e.pressure_minimum)?;
    if pl.add(dl.mul(sm.sub(vl)?)?)?.cmp_value(minimum).is_le()
        || pr.add(dr.mul(sm.sub(vr)?)?)?.cmp_value(minimum).is_le()
    {
        return Err(invalid(
            scioxide_core::argument!("equation"),
            "inadmissible HLLC contact pressure",
        ));
    }
    let star =
        |u: &[R; N], p: Scaled<R>, v: Scaled<R>, s: Scaled<R>| -> Result<[R; N], SciOxideError> {
            let ds = s.sub(v)?;
            let rho = value(u[0])?.mul(ds)?.div(s.sub(sm)?)?;
            let mut state = [R::ZERO; N];
            state[0] = rho.round()?;
            for j in 0..D {
                state[j + 1] = rho
                    .mul(
                        value(u[j + 1])?
                            .div(value(u[0])?)?
                            .add(sm.sub(v)?.mul(value(n[j])?)?)?,
                    )?
                    .round()?;
            }
            state[N - 1] = rho
                .mul(
                    value(u[N - 1])?
                        .div(value(u[0])?)?
                        .add(sm.sub(v)?.mul(sm.add(p.div(value(u[0])?.mul(ds)?)?)?)?)?,
                )?
                .round()?;
            e.checked_pressure::<D, N>(&state, scioxide_core::argument!("equation"))?;
            Ok(state)
        };
    let ls = star(l, pl, vl, lower)?;
    let rs = star(r, pr, vr, upper)?;
    let (u, us, flux, speed) = if sm.is_sign_negative() {
        (r, rs, fr, upper)
    } else {
        (l, ls, fl, lower)
    };
    let mut out = [R::ZERO; N];
    for i in 0..N {
        out[i] = value(flux[i])?
            .add(speed.mul(value(us[i])?.sub(value(u[i])?)?)?)?
            .round()?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check_width<R: RealScalar>() {
        let cv = |x| R::try_from_f64_lossy(x).unwrap();
        let adv = LinearAdvection { velocity: [cv(2.)] };
        let f = Godunov
            .evaluate(&adv, &[cv(3.)], &[cv(-1.)], [R::ONE])
            .unwrap();
        assert_eq!(f.flux, [cv(6.)]);
        let burgers = Burgers {
            direction: [R::ONE],
        };
        assert_eq!(
            Godunov
                .evaluate(&burgers, &[cv(-1.)], &[R::ONE], [R::ONE])
                .unwrap()
                .flux,
            [R::ZERO]
        );
        assert_eq!(
            Godunov
                .evaluate(&burgers, &[cv(2.)], &[cv(-1.)], [R::ONE])
                .unwrap()
                .flux,
            [cv(2.)]
        );
        let gas = IdealGasEuler::new(cv(1.4)).unwrap();
        let l = gas
            .conserved::<2, 4>(R::ONE, [R::ZERO, cv(0.7)], R::ONE)
            .unwrap();
        let r = gas
            .conserved::<2, 4>(cv(0.125), [R::ZERO, cv(-0.4)], R::ONE)
            .unwrap();
        let expected = [R::ZERO, R::ONE, R::ZERO, R::ZERO];
        let f = Hllc.evaluate(&gas, &l, &r, [R::ONE, R::ZERO]).unwrap();
        let tol = cv(128.) * R::EPSILON;
        assert!(!f.used_hlle_fallback);
        for (&a, b) in f.flux.iter().zip(expected) {
            assert!((a - b).abs() <= tol);
        }
        for flux in [
            &Rusanov as &dyn NumericalFlux<R, IdealGasEuler<R>, 2, 4>,
            &Hlle,
            &Hllc,
        ] {
            let same = flux.evaluate(&gas, &l, &l, [R::ONE, R::ZERO]).unwrap();
            let exact = gas.normal_flux(&l, [R::ONE, R::ZERO]).unwrap();
            let forward = flux.evaluate(&gas, &l, &r, [R::ONE, R::ZERO]).unwrap();
            let reverse = flux.evaluate(&gas, &r, &l, [-R::ONE, R::ZERO]).unwrap();
            for (i, &target) in exact.iter().enumerate() {
                assert!((same.flux[i] - target).abs() <= tol);
                assert!((forward.flux[i] + reverse.flux[i]).abs() <= tol);
            }
        }
        let l = gas.conserved::<1, 3>(R::ONE, [cv(-10.)], R::ONE).unwrap();
        let r = gas.conserved::<1, 3>(R::ONE, [cv(10.)], R::ONE).unwrap();
        let f = Hllc.evaluate(&gas, &l, &r, [R::ONE]).unwrap();
        assert!(f.used_hlle_fallback);
        assert_eq!(f.flux, Hlle.evaluate(&gas, &l, &r, [R::ONE]).unwrap().flux);
        assert!(Hllc.evaluate(&gas, &[R::ZERO; 3], &r, [R::ONE]).is_err());
        assert!(Hllc.evaluate(&gas, &l, &r, [cv(2.)]).is_err());
        assert!(gas.conserved::<3, 4>(R::ONE, [R::ZERO; 3], R::ONE).is_err());
    }
    #[test]
    fn extreme_euler_scales_recover_representable_fluxes_and_speeds() {
        let gas = IdealGasEuler::new(1.4_f64).unwrap();
        for (rho, p) in [(1e200, 1e-200), (1e-200, 1e200)] {
            let u = [rho, 0., p / 0.4];
            let expected_speed = (1.4 * p.sqrt()) / rho.sqrt() / 1.4_f64.sqrt();
            for flux in [
                &Hlle as &dyn NumericalFlux<f64, IdealGasEuler<f64>, 1, 3>,
                &Hllc,
                &Rusanov,
            ] {
                let result = flux.evaluate(&gas, &u, &u, [1.]).unwrap();
                assert!((result.maximum_speed() / expected_speed - 1.).abs() < 2e-14);
                assert!((result.flux()[1] / p - 1.).abs() < 2e-14);
                assert_eq!(result.flux()[0], 0.);
                assert_eq!(result.flux()[2], 0.);
            }
            assert!(
                !Hllc
                    .evaluate(&gas, &u, &u, [1.])
                    .unwrap()
                    .used_hlle_fallback()
            );
        }
        let law = LinearAdvection { velocity: [1e-300] };
        let result = Rusanov
            .evaluate(&law, &[f64::MAX], &[-f64::MAX], [1.])
            .unwrap();
        assert!((result.flux()[0] / (f64::MAX * 1e-300) - 1.).abs() < 1e-14);
    }
    #[test]
    fn subnormal_contact_terms_and_dissipation_replay_before_amplification() {
        let law = LinearAdvection {
            velocity: [f64::from_bits(1)],
        };
        let result = Rusanov
            .evaluate(&law, &[f64::MAX], &[-f64::MAX], [1.])
            .unwrap();
        let expected = f64::MAX * f64::from_bits(1);
        assert!((result.flux()[0] / expected - 1.).abs() < 1e-14);
        let gas = IdealGasEuler::new(1.4_f64).unwrap();
        let u = [1e-300, 1e-320, 1e-323];
        let right = [u[0], 2e-320, u[2]];
        let result = Hllc.evaluate(&gas, &u, &right, [1.]).unwrap();
        assert!((result.flux()[0] / 1.5e-320 - 1.).abs() < 1e-3);
        assert!(!result.used_hlle_fallback());
    }
    #[test]
    fn mass_flux_retains_momentum_when_velocity_underflows() {
        let gas = IdealGasEuler::new(1.4_f64).unwrap();
        let u = [1e300, 1e-200, 2.5];
        for flux in [
            &Hlle as &dyn NumericalFlux<f64, IdealGasEuler<f64>, 1, 3>,
            &Hllc,
            &Rusanov,
        ] {
            let result = flux.evaluate(&gas, &u, &u, [1.]).unwrap();
            assert!((result.flux()[0] / u[1] - 1.).abs() < 1e-14);
        }
    }
    #[test]
    fn invalid_states_and_final_overflow_have_distinct_error_kinds() {
        let gas = IdealGasEuler::new(1.4_f64).unwrap();
        let invalid = gas.pressure::<1, 3>(&[1., 0., -1.]).unwrap_err();
        assert_eq!(
            invalid.kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
        let u = [1e-200, 1., 6e199];
        let overflow = gas.normal_flux(&u, [1.]).unwrap_err();
        assert!(matches!(
            overflow.kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
                | scioxide_core::SciOxideErrorKind::NumericalInstability
        ));
    }
    #[test]
    fn flux_f32() {
        check_width::<f32>();
    }
    #[test]
    fn flux_f64() {
        check_width::<f64>();
    }
}
