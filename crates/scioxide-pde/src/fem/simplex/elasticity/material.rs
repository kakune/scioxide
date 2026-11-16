use super::*;
use scioxide_core::{Interval, ResidualTolerance, StepTolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    BracketExpansion, BracketSearchDirection, BracketSearchInput, BracketSearchOptions,
    SafeguardedInitialGuess, SafeguardedNewton, ScalarFn, ScalarRootOptions, ScalarRootSolver,
    find_bracket_scoped,
};

/// Isotropic shear and finite bulk moduli, measured in stress units.
#[derive(Clone, Copy, Debug)]
pub struct IsotropicElasticMaterial<R: RealScalar> {
    shear: R,
    bulk: R,
}
impl<R: RealScalar> IsotropicElasticMaterial<R> {
    /// Validate positive finite moduli and representable positive bulk compliance.
    pub fn new(shear: R, bulk: R) -> Result<Self, SciOxideError> {
        if !shear.is_finite()
            || !bulk.is_finite()
            || shear <= R::ZERO
            || bulk <= R::ZERO
            || !(R::ONE / bulk).is_finite()
            || R::ONE / bulk == R::ZERO
        {
            return Err(invalid(
                if !shear.is_finite() || shear <= R::ZERO {
                    scioxide_core::argument!("shear")
                } else {
                    scioxide_core::argument!("bulk")
                },
                "elastic moduli must be positive, finite and representable",
            ));
        }
        Ok(Self { shear, bulk })
    }
    /// Convert Young's modulus and Poisson ratio; requires E>0 and -1<nu<1/2.
    pub fn from_young_poisson(young: R, poisson: R) -> Result<Self, SciOxideError> {
        if !young.is_finite()
            || young <= R::ZERO
            || !poisson.is_finite()
            || poisson <= -R::ONE
            || poisson >= R::HALF
        {
            return Err(invalid(
                if !young.is_finite() || young <= R::ZERO {
                    scioxide_core::argument!("young")
                } else {
                    scioxide_core::argument!("poisson")
                },
                "invalid Young modulus or Poisson ratio",
            ));
        }
        let two = R::ONE.scalbn(1);
        Self::new(
            young / (two * (R::ONE + poisson)),
            young / (three::<R>() * (R::ONE - two * poisson)),
        )
    }
    /// Shear modulus.
    pub const fn shear(self) -> R {
        self.shear
    }
    /// Bulk modulus (exact incompressibility is unsupported).
    pub const fn bulk(self) -> R {
        self.bulk
    }
}

fn three<R: RealScalar>() -> R {
    R::ONE + R::TWO
}
fn delta<R: RealScalar>(i: usize, j: usize) -> R {
    if i == j { R::ONE } else { R::ZERO }
}

pub(super) struct Response<R: RealScalar> {
    pub energy: R,
    pub gradient: [R; 10],
    pub tangent: [[R; 10]; 10],
    pub deformation: [[R; 3]; 3],
    pub jacobian: R,
}

/// Full tensorial-shear Hooke Hessian; variable 9 is tensile pressure.
pub(super) fn hooke<R: RealScalar>(
    m: IsotropicElasticMaterial<R>,
    mixed: bool,
    plane: ElasticityKinematics,
) -> Result<[[R; 10]; 10], SciOxideError> {
    let mut c = [[R::ZERO; 10]; 10];
    let lambda = if mixed { R::ZERO } else { m.bulk } - m.shear.scalbn(1) / three::<R>();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    c[3 * i + j][3 * k + l] = finite(
                        m.shear
                            * (delta::<R>(i, k) * delta::<R>(j, l)
                                + delta::<R>(i, l) * delta::<R>(j, k))
                            + lambda * delta::<R>(i, j) * delta::<R>(k, l),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
    }
    if mixed {
        for i in 0..3 {
            c[3 * i + i][9] = R::ONE;
            c[9][3 * i + i] = R::ONE;
        }
        c[9][9] = -R::ONE / m.bulk;
    }
    if plane == ElasticityKinematics::PlaneStress {
        condense(&mut c)?;
    }
    Ok(c)
}

fn condense<R: RealScalar>(c: &mut [[R; 10]; 10]) -> Result<(), SciOxideError> {
    let pivot = c[8][8];
    if !pivot.is_finite() || pivot <= R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("material"),
            "unusable plane-stress tangent",
        ));
    }
    let row = c[8];
    for i in 0..10 {
        for j in 0..10 {
            if i != 8 && j != 8 {
                c[i][j] = finite(
                    c[i][j] - row[i] * (row[j] / pivot),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
    }
    for row in c.iter_mut() {
        row[8] = R::ZERO;
    }
    c[8].fill(R::ZERO);
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NeoEvaluation {
    Full,
    PlaneStress,
}

fn neo<R: RealScalar>(
    m: IsotropicElasticMaterial<R>,
    f: [[R; 3]; 3],
    pressure: Option<R>,
    evaluation: NeoEvaluation,
) -> Result<Response<R>, SciOxideError> {
    if f.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid(
            scioxide_core::argument!("x"),
            "nonfinite deformation",
        ));
    }
    let cofactor: [[R; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            f[(i + 1) % 3][(j + 1) % 3] * f[(i + 2) % 3][(j + 2) % 3]
                - f[(i + 1) % 3][(j + 2) % 3] * f[(i + 2) % 3][(j + 1) % 3]
        })
    });
    let jac = finite(
        (0..3).fold(R::ZERO, |s, j| s + f[0][j] * cofactor[0][j]),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    if jac <= R::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "deformation determinant must be positive",
        ));
    }
    let q: [[R; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| cofactor[i][j] / jac));
    let invariant = finite(
        f.iter().flatten().fold(R::ZERO, |s, &x| s + x * x),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    let a = finite(
        R::ONE / jac.cbrt().powi(2),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    let theta = jac - R::ONE;
    let p = pressure.unwrap_or(m.bulk * theta);
    let mut result = Response {
        energy: finite(
            m.shear * R::HALF * (a * invariant - three::<R>())
                + match pressure {
                    Some(p) => p * theta - R::HALF * p * (p / m.bulk),
                    None => R::HALF * m.bulk * theta * theta,
                },
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?,
        gradient: [R::ZERO; 10],
        tangent: [[R::ZERO; 10]; 10],
        deformation: f,
        jacobian: jac,
    };
    for i in 0..3 {
        for j in 0..3 {
            if evaluation == NeoEvaluation::PlaneStress && (i != 2 || j != 2) {
                continue;
            }
            result.gradient[3 * i + j] = finite(
                m.shear * a * (f[i][j] - invariant * q[i][j] / three::<R>()) + p * cofactor[i][j],
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            for k in 0..3 {
                for l in 0..3 {
                    if evaluation == NeoEvaluation::PlaneStress && (k != 2 || l != 2) {
                        continue;
                    }
                    let qq = q[i][j] * q[k][l];
                    let cross = q[i][l] * q[k][j];
                    let iso = m.shear
                        * a
                        * (delta::<R>(i, k) * delta::<R>(j, l)
                            - R::ONE.scalbn(1) / three::<R>()
                                * (f[i][j] * q[k][l] + f[k][l] * q[i][j])
                            + invariant
                                * (R::ONE.scalbn(1) / three::<R>().powi(2) * qq
                                    + cross / three::<R>()));
                    let vol = p * jac * (qq - cross)
                        + if pressure.is_none() {
                            m.bulk * cofactor[i][j] * cofactor[k][l]
                        } else {
                            R::ZERO
                        };
                    result.tangent[3 * i + j][3 * k + l] =
                        finite(iso + vol, scioxide_core::NonFiniteOrigin::Intermediate)?;
                }
            }
            if pressure.is_some() {
                result.tangent[3 * i + j][9] = cofactor[i][j];
                result.tangent[9][3 * i + j] = cofactor[i][j];
            }
        }
    }
    if let Some(p) = pressure {
        result.gradient[9] = finite(
            theta - p / m.bulk,
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        result.tangent[9][9] = -R::ONE / m.bulk;
    }
    Ok(result)
}

pub(super) fn neo_response<R: RealScalar>(
    m: IsotropicElasticMaterial<R>,
    mut f: [[R; 3]; 3],
    pressure: Option<R>,
    plane: ElasticityKinematics,
    options: &PlaneStressOptions<R>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Response<R>, SciOxideError> {
    if plane != ElasticityKinematics::PlaneStress {
        return neo(m, f, pressure, NeoEvaluation::Full);
    }
    // P33 is increasing for fixed positive in-plane determinant.
    let evaluate = |stretch: R| {
        let mut f = f;
        f[2][2] = stretch;
        neo(m, f, pressure, NeoEvaluation::PlaneStress)
    };
    let stress = ScalarFn::new(|x| Ok(evaluate(x)?.gradient[8]));
    let search = BracketSearchOptions::default()
        .with_expansion(BracketExpansion::Multiplicative)
        .with_domain(
            core::ops::Bound::Excluded(R::ZERO),
            core::ops::Bound::Unbounded,
        )
        .with_max_expansions(options.max_iterations);
    let bracket = find_bracket_scoped(
        &stress,
        BracketSearchInput::Guess(R::ONE),
        BracketSearchDirection::Increasing,
        &search,
        &mut *context,
    )?;
    let lo = bracket.lower();
    let hi = bracket.upper();
    let root = if bracket.lower_value() == R::ZERO {
        lo
    } else if bracket.upper_value() == R::ZERO {
        hi
    } else {
        let scale = m.shear.max(pressure.map_or(m.bulk, |p| p.abs()));
        let function = ScalarFn::new(|x| Ok(evaluate(x)?.gradient[8] / scale))
            .with_derivative(|x| Ok(evaluate(x)?.tangent[8][8] / scale));
        let solver = SafeguardedNewton::new(DerivativeStrategy::Analytic);
        let settings = ScalarRootOptions::default()
            .with_residual_scale(R::ONE)
            .with_max_iterations(options.max_iterations)
            .with_residual_tolerance(ResidualTolerance::new(
                options.tolerance,
                options.tolerance,
            )?)
            .with_step_tolerance(StepTolerance::new(options.tolerance, options.tolerance)?);
        let guess = SafeguardedInitialGuess::new(
            Interval::new(lo, hi)?,
            crate::spatial::midpoint(lo, hi)?,
        )?;
        let result = solver.solve_scoped(&function, guess, &settings, &mut *context)?;
        if !result.is_success() || result.residual().abs() > options.tolerance.scalbn(2) {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::NestedSolver,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                "plane-stress stress residual did not converge",
            ));
        }
        result.point()
    };
    f[2][2] = root;
    let mut result = neo(m, f, pressure, NeoEvaluation::Full)?;
    condense(&mut result.tangent)?;
    Ok(result)
}
