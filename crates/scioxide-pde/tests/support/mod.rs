#![allow(dead_code, unreachable_pub)]
//! One-dimensional problems solved through the unified FDM, FEM and FVM plans.
use std::vec::Vec;

use crate::mesh_fixture::nodes;
use scioxide_spatial::{
    AxisSide, BoundarySet, IntervalMesh, Point, SegmentMesh, SimplexBoundaryTags,
};

use scioxide_core::{
    Budget, NumericScalar, RealScalar, ResidualTolerance, SciOxideError, Tolerance,
};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryKind, EllipticOptions, FieldData, FvmCoefficients,
    PdeSolveReport, PdeTermination, SimplexFemCoefficients, SimplexFemLimits, SimplexFemOptions,
    SimplexFemOrder, SimplexFemPlan, SpatialDiscretization, SpatialLimits, StructuredBoundaries,
    StructuredFdmCoefficients, StructuredFdmPlan, StructuredFvmPlan, TaggedBoundaries,
};

/// Steady problem `-d/dx(k du/dx) + c u = f` described pointwise.
pub trait EllipticPde<S> {
    fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError>;
    fn reaction(&self, x: &[S]) -> Result<S, SciOxideError>;
    fn source(&self, x: &[S]) -> Result<S, SciOxideError>;
}

/// Heat problem `u_t = d/dx(k du/dx) + f`; diffusion is sampled at `t = 0`.
pub trait ParabolicPde<S> {
    fn diffusion(&self, x: &[S], t: S) -> Result<S, SciOxideError>;
    fn source(&self, x: &[S], t: S) -> Result<S, SciOxideError>;
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError>;
}

/// Wave problem `u_tt = d/dx(c² du/dx) + f`.
pub trait HyperbolicPde<S> {
    fn wave_speed(&self, x: &[S]) -> Result<S, SciOxideError>;
    fn source(&self, x: &[S], t: S) -> Result<S, SciOxideError>;
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError>;
    fn initial_velocity(&self, x: &[S]) -> Result<S, SciOxideError>;
}

/// Problem data shared by the discretization adapters.
pub trait Problem<S: RealScalar>: Send + Sync + Clone + 'static {
    fn diffusion(&self, x: S) -> Result<S, SciOxideError>;
    fn reaction(&self, _x: S) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn source(&self, x: S, t: S) -> Result<S, SciOxideError>;
}

#[derive(Clone)]
pub struct Steady<P>(pub P);
impl<S: RealScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static> Problem<S> for Steady<P> {
    fn diffusion(&self, x: S) -> Result<S, SciOxideError> {
        self.0.diffusion(&[x])
    }
    fn reaction(&self, x: S) -> Result<S, SciOxideError> {
        self.0.reaction(&[x])
    }
    fn source(&self, x: S, _: S) -> Result<S, SciOxideError> {
        self.0.source(&[x])
    }
}

#[derive(Clone)]
pub struct Heat<P>(pub P);
impl<S: RealScalar, P: ParabolicPde<S> + Send + Sync + Clone + 'static> Problem<S> for Heat<P> {
    fn diffusion(&self, x: S) -> Result<S, SciOxideError> {
        self.0.diffusion(&[x], S::ZERO)
    }
    fn source(&self, x: S, t: S) -> Result<S, SciOxideError> {
        self.0.source(&[x], t)
    }
}

#[derive(Clone)]
pub struct Wave<P>(pub P);
impl<S: RealScalar, P: HyperbolicPde<S> + Send + Sync + Clone + 'static> Problem<S> for Wave<P> {
    fn diffusion(&self, x: S) -> Result<S, SciOxideError> {
        let c = self.0.wave_speed(&[x])?;
        scioxide_core::checked_float_product(c, c)
    }
    fn source(&self, x: S, t: S) -> Result<S, SciOxideError> {
        self.0.source(&[x], t)
    }
}

struct Coefficients<'a, P>(&'a P);
impl<S: RealScalar, P: Problem<S>> StructuredFdmCoefficients<S, S, 1> for Coefficients<'_, P> {
    fn diffusion(&self, _: usize, _: usize, point: [S; 1]) -> Result<S, SciOxideError> {
        self.0.diffusion(point[0])
    }
    fn reaction(&self, _: usize, _: usize, point: [S; 1]) -> Result<S, SciOxideError> {
        self.0.reaction(point[0])
    }
}
impl<S: RealScalar, P: Problem<S>> FvmCoefficients<S, S, 1> for Coefficients<'_, P> {
    fn diffusion(&self, point: [S; 1]) -> Result<[[S; 1]; 1], SciOxideError> {
        Ok([[self.0.diffusion(point[0])?]])
    }
    fn reaction(&self, point: [S; 1]) -> Result<S, SciOxideError> {
        self.0.reaction(point[0])
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, P: Problem<S>> SimplexFemCoefficients<S, S, 1>
    for Coefficients<'_, P>
{
    fn evaluate(
        &self,
        point: [S; 1],
        diffusion: &mut [S],
        reaction: &mut [S],
    ) -> Result<(), SciOxideError> {
        diffusion[0] = self.0.diffusion(point[0])?;
        reaction[0] = self.0.reaction(point[0])?;
        Ok(())
    }
}

/// Time-invariant boundary data `g(x)`.
pub fn value<S: TestScalar>(
    f: impl Fn(S) -> Result<S, SciOxideError> + Send + Sync + 'static,
) -> BoundaryData<S, S, 1> {
    BoundaryData::time_invariant(move |p| f(p.point[0]))
}

/// Time-dependent boundary data `g(x, t)`.
pub fn moving<S: TestScalar>(
    f: impl Fn(S, S) -> Result<S, SciOxideError> + Send + Sync + 'static,
) -> BoundaryData<S, S, 1> {
    BoundaryData::time_dependent(move |p, t| f(p.point[0], t))
}

/// One endpoint condition; natural data is the outward conormal flux `k du/dn`.
#[derive(Clone)]
pub struct End<S: TestScalar> {
    pub kind: BoundaryKind<S>,
    pub data: BoundaryData<S, S, 1>,
}
impl<S: TestScalar> End<S> {
    pub fn dirichlet(data: BoundaryData<S, S, 1>) -> Self {
        Self {
            kind: BoundaryKind::Dirichlet,
            data,
        }
    }
    pub fn natural(data: BoundaryData<S, S, 1>) -> Self {
        Self {
            kind: BoundaryKind::Natural,
            data,
        }
    }
    pub fn robin(alpha: S, data: BoundaryData<S, S, 1>) -> Self {
        Self {
            kind: BoundaryKind::Robin { alpha },
            data,
        }
    }
    fn condition<Region>(&self, region: Region) -> BoundaryCondition<S, S, 1, Region> {
        match self.kind {
            BoundaryKind::Dirichlet => BoundaryCondition::dirichlet(region, self.data.clone()),
            BoundaryKind::Natural => BoundaryCondition::natural(region, self.data.clone()),
            BoundaryKind::Robin { alpha } => {
                BoundaryCondition::robin(region, alpha, self.data.clone())
            }
        }
    }
}

/// Both endpoint conditions, or none for a periodic mesh.
#[derive(Clone)]
pub struct Ends<S: TestScalar> {
    pub left: Option<End<S>>,
    pub right: Option<End<S>>,
}
impl<S: TestScalar> Ends<S> {
    pub fn new(left: End<S>, right: End<S>) -> Self {
        Self {
            left: Some(left),
            right: Some(right),
        }
    }
    pub fn periodic() -> Self {
        Self {
            left: None,
            right: None,
        }
    }
    pub fn structured(
        &self,
        mesh: &SegmentMesh<S>,
    ) -> Result<StructuredBoundaries<S, S, 1>, SciOxideError> {
        let mut conditions = Vec::new();
        for (side, end) in [(AxisSide::Start, &self.left), (AxisSide::End, &self.right)] {
            if let Some(end) = end {
                conditions.push(end.condition(BoundarySet::axis_side(mesh, 0, side)?));
            }
        }
        StructuredBoundaries::new(mesh, 1, conditions)
    }
    pub fn tagged(
        &self,
        mesh: &IntervalMesh<S>,
    ) -> Result<TaggedBoundaries<S, S, 1>, SciOxideError> {
        let last = mesh.vertex_count() - 1;
        let entries = mesh
            .boundary_faces()
            .map(|face| {
                (
                    face,
                    u32::from(mesh.face_vertices(face).unwrap()[0] == last),
                )
            })
            .collect();
        let tags = SimplexBoundaryTags::new(mesh, entries)?;
        let mut conditions = Vec::new();
        for (tag, end) in [(0, &self.left), (1, &self.right)] {
            if let Some(end) = end {
                conditions.push(end.condition(tag));
            }
        }
        TaggedBoundaries::new(mesh, &tags, 1, conditions)
    }
}

/// Interval simplices with the same nodes as a segment mesh.
pub fn interval_mesh<S: TestScalar>(
    mesh: &SegmentMesh<S>,
) -> Result<IntervalMesh<S>, SciOxideError> {
    let nodes = nodes(mesh);
    IntervalMesh::new(
        nodes
            .iter()
            .map(|&x| Point::new([x]))
            .collect::<Result<_, _>>()?,
        (0..nodes.len() - 1).map(|i| [i, i + 1]).collect(),
    )
}

pub fn limits() -> SpatialLimits {
    SpatialLimits::new(1 << 26, 1 << 28, usize::MAX)
}

pub fn fem_options<S: TestScalar>() -> SimplexFemOptions<S> {
    SimplexFemOptions::new().with_limits(SimplexFemLimits::new(1 << 26, 1 << 28, 64, usize::MAX))
}

/// A prepared one-dimensional plan of one family.
pub enum Plan1d<S: TestScalar> {
    Fdm(StructuredFdmPlan<S, S, 1>),
    Fem(SimplexFemPlan<S, S, 1>),
    Fvm(StructuredFvmPlan<S, S, 1>),
}

/// Run a generic expression on the plan of any family.
#[macro_export]
macro_rules! with_plan {
    ($plan:expr, |$p:ident| $body:expr) => {
        match $plan {
            $crate::support::Plan1d::Fdm($p) => $body,
            $crate::support::Plan1d::Fem($p) => $body,
            $crate::support::Plan1d::Fvm($p) => $body,
        }
    };
}

impl<S: TestScalar> Plan1d<S> {
    /// Full physical field with periodic seams restored, in coordinate order.
    pub fn full(&self, values: &[S]) -> Result<Vec<S>, SciOxideError> {
        Ok(match self {
            Self::Fdm(plan) => plan.field(values)?[0].values().to_vec(),
            Self::Fem(plan) => plan.field(values)?.vertices()[0].values().to_vec(),
            Self::Fvm(plan) => plan.field(values)?.values().to_vec(),
        })
    }
}

/// Discretization family of a one-dimensional test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretization {
    Fdm,
    Fem,
    Fvm,
}

pub const DISCRETIZATIONS: [Discretization; 3] = [
    Discretization::Fdm,
    Discretization::Fem,
    Discretization::Fvm,
];

impl Discretization {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fdm => "FDM",
            Self::Fem => "FEM",
            Self::Fvm => "FVM",
        }
    }

    pub fn plan<S: TestScalar, P: Problem<S>>(
        self,
        mesh: &SegmentMesh<S>,
        problem: &P,
        ends: &Ends<S>,
    ) -> Result<Plan1d<S>, SciOxideError> {
        let coefficients = Coefficients(problem);
        Ok(match self {
            Self::Fdm => Plan1d::Fdm(StructuredFdmPlan::with_options(
                mesh.clone(),
                &coefficients,
                &ends.structured(mesh)?,
                limits(),
            )?),
            Self::Fvm => Plan1d::Fvm(StructuredFvmPlan::with_options(
                mesh.clone(),
                &coefficients,
                &ends.structured(mesh)?,
                limits(),
            )?),
            Self::Fem => {
                let intervals = interval_mesh(mesh)?;
                let boundaries = ends.tagged(&intervals)?;
                Plan1d::Fem(SimplexFemPlan::with_options(
                    intervals,
                    scioxide_pde::SimplexFemGeometry::Affine,
                    SimplexFemOrder::P1,
                    &coefficients,
                    &boundaries,
                    fem_options(),
                )?)
            }
        })
    }

    /// Unknown sites: nodes for FDM/FEM, cell midpoints for FVM.
    pub fn coordinates<S: TestScalar>(self, mesh: &SegmentMesh<S>) -> Vec<S> {
        let nodes = nodes(mesh);
        match self {
            Self::Fvm => nodes
                .windows(2)
                .map(|p| (p[0] + p[1]) / S::try_from_f64_exact(2.0).unwrap())
                .collect(),
            _ => nodes,
        }
    }
}

/// Time-dependent source data of a problem.
pub fn source<S: TestScalar, P: Problem<S>>(problem: &P) -> FieldData<S, S, 1> {
    let problem = problem.clone();
    FieldData::time_dependent(move |p: scioxide_pde::FieldPoint<S, 1>, t| {
        problem.source(p.point[0], t)
    })
}

/// Initial data `u(x, 0)` of a heat problem.
pub fn initial<S: TestScalar, P: ParabolicPde<S> + Send + Sync + Clone + 'static>(
    problem: &P,
) -> FieldData<S, S, 1> {
    let problem = problem.clone();
    FieldData::time_invariant(move |p: scioxide_pde::FieldPoint<S, 1>| {
        problem.initial_condition(&p.point)
    })
}

/// Initial displacement and velocity of a wave problem.
pub fn wave_initial<S: TestScalar, P: HyperbolicPde<S> + Send + Sync + Clone + 'static>(
    problem: &P,
) -> (FieldData<S, S, 1>, FieldData<S, S, 1>) {
    let (a, b) = (problem.clone(), problem.clone());
    (
        FieldData::time_invariant(move |p: scioxide_pde::FieldPoint<S, 1>| {
            a.initial_condition(&p.point)
        }),
        FieldData::time_invariant(move |p: scioxide_pde::FieldPoint<S, 1>| {
            b.initial_velocity(&p.point)
        }),
    )
}

pub trait TestScalar:
    RealScalar
    + NumericScalar<Real = Self>
    + scioxide_core::NumericScalar<Real = Self>
    + core::fmt::Debug
    + core::fmt::Display
    + 'static
{
    const NAME: &str;
    const SOLVE_TOLERANCE: f64;
    const ELLIPTIC_REFINEMENT_RATIO: f64;
    const PARABOLIC_REFINEMENT_RATIO: f64;
    const BACKWARD_EULER_RATIO: f64;
    const CRANK_NICOLSON_RATIO: f64;
    const ADAPTIVE_LOOSE_TOLERANCE: f64;
    const ADAPTIVE_FINE_TOLERANCE: f64;

    fn scalar_from_f64(value: f64) -> Self {
        Self::try_from_f64_lossy(value).unwrap()
    }

    fn zero_value() -> Self {
        <Self as scioxide_core::AlgebraicScalar>::ZERO
    }

    fn one_value() -> Self {
        <Self as scioxide_core::AlgebraicScalar>::ONE
    }

    fn non_finite() -> Self;
}

impl TestScalar for f64 {
    const NAME: &str = "f64";
    const SOLVE_TOLERANCE: f64 = 1.0e-10;
    const ELLIPTIC_REFINEMENT_RATIO: f64 = 3.2;
    const PARABOLIC_REFINEMENT_RATIO: f64 = 3.5;
    const BACKWARD_EULER_RATIO: f64 = 1.8;
    const CRANK_NICOLSON_RATIO: f64 = 3.5;
    const ADAPTIVE_LOOSE_TOLERANCE: f64 = 1.0e-4;
    const ADAPTIVE_FINE_TOLERANCE: f64 = 1.0e-11;

    fn non_finite() -> Self {
        f64::NAN
    }
}

impl TestScalar for f32 {
    const NAME: &str = "f32";
    const SOLVE_TOLERANCE: f64 = 1.0e-3;
    const ELLIPTIC_REFINEMENT_RATIO: f64 = 2.5;
    const PARABOLIC_REFINEMENT_RATIO: f64 = 2.5;
    const BACKWARD_EULER_RATIO: f64 = 1.15;
    const CRANK_NICOLSON_RATIO: f64 = 2.30;
    const ADAPTIVE_LOOSE_TOLERANCE: f64 = 1.0e-2;
    const ADAPTIVE_FINE_TOLERANCE: f64 = 1.0e-7;

    fn non_finite() -> Self {
        f32::NAN
    }
}

pub fn scalar<S: TestScalar>(value: f64) -> S {
    S::scalar_from_f64(value)
}

pub fn as_f64<S: TestScalar>(value: S) -> f64 {
    value.try_to_f64_exact().unwrap()
}

pub fn solve_tolerance<S: TestScalar>() -> Tolerance<S> {
    let threshold = scalar::<S>(S::SOLVE_TOLERANCE);
    Tolerance::new(threshold, threshold).unwrap()
}

/// Elliptic options with the scalar's test tolerance for residual and
/// compatibility checks and a 4,000-iteration cap for injected solvers.
pub fn elliptic_options<S: TestScalar>() -> EllipticOptions<S> {
    let tolerance = ResidualTolerance::from_tolerance(solve_tolerance::<S>());
    EllipticOptions::new()
        .with_compatibility(
            scioxide_linalg::CompatibilityPolicy::RequireCompatible,
            tolerance,
        )
        .with_residual_tolerance(tolerance)
        .with_max_linear_iterations(Budget::try_from(4_000).unwrap())
}

pub fn homogeneous_dirichlet<S: TestScalar>() -> Ends<S> {
    Ends::new(
        End::dirichlet(value(|_| Ok(S::zero_value()))),
        End::dirichlet(value(|_| Ok(S::zero_value()))),
    )
}

/// Check a successful report and finite fields of the expected length, with
/// increasing sample times for transient output.
pub fn assert_valid_solution<S: TestScalar>(
    name: &str,
    report: &PdeSolveReport<S>,
    times: &[S],
    fields: &[&[S]],
    unknowns: usize,
) {
    assert_eq!(
        report.termination(),
        PdeTermination::Completed,
        "{name}/{} did not complete: {report}",
        S::NAME
    );
    assert!(report.is_success(), "{name}/{} was unsuccessful", S::NAME);
    assert!(!fields.is_empty(), "{name}/{}", S::NAME);
    assert!(
        times.iter().all(|t| t.is_finite()) && times.windows(2).all(|t| t[1] > t[0]),
        "{name}/{} times",
        S::NAME
    );
    for values in fields {
        assert_eq!(values.len(), unknowns);
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name}/{} contains non-finite values",
            S::NAME
        );
    }
    if let Some(residual) = report.diagnostics().residual_norm() {
        assert!(
            residual.is_finite() && residual >= S::zero_value(),
            "{name}/{}",
            S::NAME
        );
    }
}

pub fn max_abs_error<S: TestScalar>(
    actual: &[S],
    coordinates: &[S],
    exact: impl Fn(S) -> S,
) -> f64 {
    assert_eq!(actual.len(), coordinates.len());
    actual
        .iter()
        .zip(coordinates)
        .map(|(&value, &coordinate)| as_f64(value - exact(coordinate)).abs())
        .fold(0.0_f64, f64::max)
}

pub fn assert_refinement(
    name: &str,
    scalar_name: &str,
    coarse_error: f64,
    fine_error: f64,
    minimum_ratio: f64,
) {
    let ratio = coarse_error / fine_error;
    assert!(
        coarse_error.is_finite()
            && fine_error.is_finite()
            && fine_error < coarse_error
            && ratio >= minimum_ratio,
        "{name}/{scalar_name}: coarse={coarse_error:e}, fine={fine_error:e}, ratio={ratio:e}, required={minimum_ratio:e}"
    );
}

pub fn assert_scaled_residual<S: TestScalar>(
    context: &str,
    residual: f64,
    scale: f64,
    truncation_bound: f64,
) {
    let tolerance = S::SOLVE_TOLERANCE * (1.0 + scale.abs()) + truncation_bound;
    let normalized = residual / tolerance.max(f64::MIN_POSITIVE);
    assert!(
        residual.is_finite() && normalized <= 1.0,
        "{context}/{}: residual={residual:e}, scale={scale:e}, truncation={truncation_bound:e}, normalized={normalized:e}",
        S::NAME
    );
}
