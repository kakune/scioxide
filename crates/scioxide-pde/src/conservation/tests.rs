use super::*;
use alloc::vec::Vec;
use scioxide_core::{
    Budget, ExecutionContext, PartialFailure, ProblemToken, RealScalar, SciOxideErrorView,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    ForwardEuler, Heun, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeProgress,
    OdeSolution, OdeState, OdeSystem,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, Cell, CellField, CellMesh, CoordinateAxis, FieldView,
    Point, SimplexBoundaryTags, SimplexMesh, StructuredGrid, TetrahedronMesh, TriangleMesh,
};

const TEST_ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");

/// Fixed-step built-in Heun from time zero; admissibility is enforced by the system.
#[allow(clippy::result_large_err)]
fn heun<R: RealScalar, Sys: OdeSystem<R>>(
    system: &Sys,
    step: R,
    end: R,
    initial: Vector<R>,
) -> Result<OdeSolution<R>, PartialFailure<OdeProgress<R>>> {
    Heun::new().solve(
        &OdeProblem::new(system, end),
        &(OdeOptions::new().with_initial_step(step)),
        OdeState::new(R::ZERO, initial),
    )
}

fn limits() -> ConservationLimits {
    ConservationLimits::new(2_000_000, 3_000_000, 1_000_000_000)
}
/// Owned structured residual through the caller-buffer form with a fresh workspace.
fn structured_residual<R, E, F, const D: usize, const N: usize>(
    p: &StructuredConservationPlan<R, D, N>,
    e: &E,
    f: &F,
    time: R,
    state: FieldView<'_, [R; N], Cell, D>,
) -> Result<(Vec<[R; N]>, ConservationReport<R, N>), SciOxideError>
where
    R: RealScalar,
    E: ConservationLaw<R, D, N>,
    F: NumericalFlux<R, E, D, N>,
{
    let mut out = vec![[R::ZERO; N]; p.volumes().len()];
    let report = p.residual_into(
        e,
        f,
        time,
        state,
        &mut out,
        &mut p.workspace()?,
        ExecutionContext::sequential(),
    )?;
    Ok((out, report))
}
/// Owned simplex residual through the caller-buffer form with a fresh workspace.
fn simplex_residual<R, E, F, const D: usize, const N: usize>(
    p: &SimplexConservationPlan<R, D, N>,
    e: &E,
    f: &F,
    time: R,
    state: FieldView<'_, [R; N], Cell, D>,
) -> Result<(Vec<[R; N]>, ConservationReport<R, N>), SciOxideError>
where
    R: RealScalar,
    E: ConservationLaw<R, D, N>,
    F: NumericalFlux<R, E, D, N>,
{
    let mut out = vec![[R::ZERO; N]; p.volumes().len()];
    let report = p.residual_into(
        e,
        f,
        time,
        state,
        &mut out,
        &mut p.workspace()?,
        ExecutionContext::sequential(),
    )?;
    Ok((out, report))
}
/// Structured face fluxes normal to `axis`: canonical faces are axis-major.
fn axis_face_fluxes<R, E, F, const D: usize, const N: usize>(
    p: &StructuredConservationPlan<R, D, N>,
    e: &E,
    f: &F,
    time: R,
    state: FieldView<'_, [R; N], Cell, D>,
    axis: usize,
) -> Result<Vec<[R; N]>, SciOxideError>
where
    R: RealScalar,
    E: ConservationLaw<R, D, N>,
    F: NumericalFlux<R, E, D, N>,
{
    let sizes = p.mesh().topology_sizes();
    let mut all = vec![[R::ZERO; N]; sizes.faces()];
    p.face_fluxes_into(
        e,
        f,
        time,
        state,
        &mut all,
        &mut p.workspace()?,
        ExecutionContext::sequential(),
    )?;
    let offset = (0..axis).try_fold(0, |sum, a| {
        Ok::<_, SciOxideError>(sum + sizes.faces_normal_to(a)?)
    })?;
    Ok(all[offset..offset + sizes.faces_normal_to(axis)?].to_vec())
}
fn mesh<R: RealScalar, const D: usize>(cells: usize, periodic: bool) -> CellMesh<R, D> {
    CellMesh::new(
        StructuredGrid::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(R::ZERO, R::ONE, cells + 1).unwrap()),
            [if periodic {
                AxisTopology::Periodic
            } else {
                AxisTopology::Bounded
            }; D],
        )
        .unwrap(),
    )
    .unwrap()
}
fn boundaries<R: RealScalar, const D: usize, const N: usize>(
    mesh: &CellMesh<R, D>,
) -> Vec<(BoundarySet<D>, ConservationBoundary<R, D, N>)> {
    (0..D)
        .flat_map(|axis| {
            [AxisSide::Start, AxisSide::End].map(move |side| {
                (
                    BoundarySet::axis_side(mesh, axis, side).unwrap(),
                    ConservationBoundary::Transmissive,
                )
            })
        })
        .collect()
}
fn plan<R: RealScalar, const D: usize, const N: usize>(
    cells: usize,
    periodic: bool,
    reconstruction: ConservationReconstruction,
) -> StructuredConservationPlan<R, D, N> {
    let mesh = mesh(cells, periodic);
    let boundaries = if periodic { vec![] } else { boundaries(&mesh) };
    StructuredConservationPlan::with_options(
        mesh,
        boundaries,
        ConservationOptions::new()
            .with_reconstruction(reconstruction)
            .with_limits(limits()),
    )
    .unwrap()
}
fn scalar_balance<R: RealScalar, const D: usize>() {
    let cv = |v| R::try_from_f64_lossy(v).unwrap();
    for periodic in [false, true] {
        for count in [1, 4] {
            let p = plan::<R, D, 1>(
                count,
                periodic,
                ConservationReconstruction::MonotonizedCentral,
            );
            let law = LinearAdvection {
                velocity: [R::ONE; D],
            };
            let field = CellField::from_fn(p.mesh(), |i| {
                [cv(1.) + p.centers()[i].iter().copied().sum::<R>()]
            })
            .unwrap();
            let mut work = p.workspace().unwrap();
            let mut residual = vec![[R::ZERO]; p.volumes().len()];
            let report = p
                .residual_into(
                    &law,
                    &Godunov,
                    R::ZERO,
                    field.view(),
                    &mut residual,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            let sum = residual
                .iter()
                .zip(p.volumes())
                .fold(R::ZERO, |s, (r, &v)| s + r[0] * v);
            assert!(
                (sum + report.boundary_flux[0]).abs()
                    < cv(256.) * R::EPSILON * cv(count as f64).max(R::ONE)
            );
            if periodic {
                assert_eq!(report.boundary_flux, [R::ZERO]);
            }
            assert!(report.maximum_step.unwrap() > R::ZERO);
            let constant = CellField::from_fn(p.mesh(), |_| [cv(2.)]).unwrap();
            p.residual_into(
                &law,
                &Godunov,
                R::ZERO,
                constant.view(),
                &mut residual,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert!(residual.iter().all(|r| r[0].abs() <= cv(64.) * R::EPSILON));
            let (owned, _) =
                structured_residual(&p, &law, &Godunov, R::ZERO, constant.view()).unwrap();
            assert_eq!(owned, residual);
            let faces = axis_face_fluxes(&p, &law, &Godunov, R::ZERO, constant.view(), 0).unwrap();
            assert_eq!(
                faces.len(),
                p.mesh().topology_sizes().faces_normal_to(0).unwrap()
            );
            assert!(faces.iter().all(|u| u[0] > R::ZERO));
        }
    }
}
#[test]
fn structured_balance_both_widths_all_dimensions() {
    scalar_balance::<f32, 1>();
    scalar_balance::<f32, 2>();
    scalar_balance::<f32, 3>();
    scalar_balance::<f64, 1>();
    scalar_balance::<f64, 2>();
    scalar_balance::<f64, 3>();
}

fn run_scalar<R: RealScalar>(cells: usize) -> f64 {
    let p = plan::<R, 1, 1>(cells, true, ConservationReconstruction::MonotonizedCentral);
    let law = LinearAdvection { velocity: [R::ONE] };
    let h = R::ONE / R::try_from_usize_exact(cells).unwrap();
    let pi = R::try_from_f64_lossy(core::f64::consts::PI).unwrap();
    let two = R::TWO;
    let initial = CellField::from_fn(p.mesh(), |i| {
        [(two * pi * p.centers()[i][0]).sin() * (pi * h).sin() / (pi * h)]
    })
    .unwrap();
    let mut workspace = p.workspace().unwrap();
    let sys = p.bind(&law, &Godunov, &mut workspace).unwrap();
    let dt = R::try_from_f64_lossy(0.15 / cells as f64).unwrap();
    let end = R::try_from_f64_lossy(0.2).unwrap();
    let solution = heun(
        &sys,
        dt,
        end,
        p.initial_state(&law, initial.view()).unwrap(),
    )
    .unwrap();
    assert!((solution.report().final_time() - end).abs() <= R::EPSILON);
    let errors = solution.state().values().iter().enumerate().map(|(i, &u)| {
        let exact = (two * pi * (p.centers()[i][0] - end)).sin() * (pi * h).sin() / (pi * h);
        (u - exact).abs().into_f64()
    });
    errors.sum::<f64>() / cells as f64
}
#[test]
fn smooth_scalar_second_order_both_widths() {
    for errors in [
        [
            run_scalar::<f64>(40),
            run_scalar::<f64>(80),
            run_scalar::<f64>(160),
        ],
        [
            run_scalar::<f32>(40),
            run_scalar::<f32>(80),
            run_scalar::<f32>(160),
        ],
    ] {
        std::eprintln!("smooth advection L1 errors {errors:?}");
        for pair in errors.windows(2) {
            assert!(pair[0] / pair[1] > 2_f64.powf(1.7), "{errors:?}");
        }
    }
}

#[test]
fn periodic_burgers_tvd_entropy_and_conservation() {
    for reconstruction in [
        ConservationReconstruction::Constant,
        ConservationReconstruction::Minmod,
        ConservationReconstruction::MonotonizedCentral,
    ] {
        let p = plan::<f64, 1, 1>(80, true, reconstruction);
        let law = Burgers { direction: [1.] };
        let initial = CellField::from_fn(p.mesh(), |i| {
            [if (20..40).contains(&i) { 1. } else { -0.25 }]
        })
        .unwrap();
        let mass = initial.values().iter().map(|u| u[0]).sum::<f64>();
        let mut entropy = initial
            .values()
            .iter()
            .map(|u| u[0] * u[0] * 0.5)
            .sum::<f64>();
        let mut tv = 2.5;
        let mut work = p.workspace().unwrap();
        let sys = p.bind(&law, &Godunov, &mut work).unwrap();
        let solve = (
            Heun::new(),
            OdeOptions::new()
                .with_initial_step(0.001)
                .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
        );
        let problem = OdeProblem::new(&sys, 0.1);
        let mut state = OdeState::new(0., p.initial_state(&law, initial.view()).unwrap());
        let _ = solve
            .0
            .advance(
                &problem,
                &solve.1,
                &mut state,
                &mut |_, u: &[f64]| {
                    let next_tv = u
                        .iter()
                        .zip(u.iter().cycle().skip(1))
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>();
                    let next_entropy = u.iter().map(|v| v * v * 0.5).sum::<f64>();
                    assert!(next_tv <= tv + 1e-11);
                    assert!(next_entropy <= entropy + 1e-11);
                    assert!((u.iter().sum::<f64>() - mass).abs() < 1e-11);
                    tv = next_tv;
                    entropy = next_entropy;
                    Ok(())
                },
                &mut solve.0.workspace(&problem, &solve.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
    }
}

fn triangles(cells: usize) -> TriangleMesh<f64> {
    let points = (0..=cells)
        .flat_map(|j| {
            (0..=cells).map(move |i| {
                Point::new([i as f64 / cells as f64, j as f64 / cells as f64]).unwrap()
            })
        })
        .collect();
    let faces = (0..cells)
        .flat_map(|j| {
            (0..cells).flat_map(move |i| {
                let a = j * (cells + 1) + i;
                let b = a + 1;
                let d = a + cells + 1;
                let c = d + 1;
                [[a, b, c], [a, c, d]]
            })
        })
        .collect();
    TriangleMesh::new(points, faces).unwrap()
}
fn tetrahedra(cells: usize) -> TetrahedronMesh<f64> {
    let points = (0..=cells)
        .flat_map(|k| {
            (0..=cells).flat_map(move |j| {
                (0..=cells).map(move |i| {
                    Point::new([
                        i as f64 / cells as f64,
                        j as f64 / cells as f64,
                        k as f64 / cells as f64,
                    ])
                    .unwrap()
                })
            })
        })
        .collect();
    let index = |p: [usize; 3]| (p[2] * (cells + 1) + p[1]) * (cells + 1) + p[0];
    let mut elements = Vec::new();
    for k in 0..cells {
        for j in 0..cells {
            for i in 0..cells {
                for (id, order) in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ]
                .into_iter()
                .enumerate()
                {
                    let mut p = [i, j, k];
                    let a = index(p);
                    p[order[0]] += 1;
                    let b = index(p);
                    p[order[1]] += 1;
                    let c = index(p);
                    p[order[2]] += 1;
                    let d = index(p);
                    elements.push(if [0, 3, 4].contains(&id) {
                        [a, b, c, d]
                    } else {
                        [a, c, b, d]
                    });
                }
            }
        }
    }
    TetrahedronMesh::new(points, elements).unwrap()
}
fn simplex_plan<const D: usize, const N: usize>(
    mesh: SimplexMesh<f64, D>,
    reconstruction: ConservationReconstruction,
) -> SimplexConservationPlan<f64, D, N> {
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    SimplexConservationPlan::with_options(
        mesh,
        &tags,
        vec![(0, ConservationBoundary::Transmissive)],
        ConservationOptions::new()
            .with_reconstruction(reconstruction)
            .with_limits(limits()),
    )
    .unwrap()
}
fn simplex_balance<const D: usize>(mesh: SimplexMesh<f64, D>) {
    for reconstruction in [
        ConservationReconstruction::Constant,
        ConservationReconstruction::BarthJespersen,
    ] {
        let p = simplex_plan::<D, 1>(mesh.clone(), reconstruction);
        let law = LinearAdvection { velocity: [1.; D] };
        let field =
            CellField::from_fn(p.mesh(), |i| [1. + p.centers()[i].iter().sum::<f64>()]).unwrap();
        let (result, report) = simplex_residual(&p, &law, &Godunov, 0., field.view()).unwrap();
        let sum = result
            .iter()
            .zip(p.volumes())
            .map(|(u, v)| u[0] * v)
            .sum::<f64>();
        assert!((sum + report.boundary_flux[0]).abs() < 1e-12);
        let field = CellField::from_fn(p.mesh(), |_| [1.]).unwrap();
        let (result, _) = simplex_residual(&p, &law, &Godunov, 0., field.view()).unwrap();
        assert!(result.iter().all(|u| u[0].abs() < 1e-12));
        let mut faces = vec![[f64::NAN]; p.mesh().face_count()];
        p.face_fluxes_into(
            &law,
            &Godunov,
            0.,
            field.view(),
            &mut faces,
            &mut p.workspace().unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(faces.iter().all(|u| u[0].is_finite()));
    }
}
#[test]
fn simplex_balance_and_constant_preservation() {
    simplex_balance(triangles(4));
    simplex_balance(tetrahedra(2));
}

#[test]
fn euler_slip_walls_and_both_component_widths() {
    fn check<R: RealScalar, const D: usize, const N: usize>() {
        let gas = IdealGasEuler::new(R::try_from_f64_lossy(1.4).unwrap()).unwrap();
        let mesh = mesh::<R, D>(2, false);
        let boundaries = (0..D)
            .flat_map(|a| [AxisSide::Start, AxisSide::End].map(move |s| (a, s)))
            .map(|(a, s)| {
                (
                    BoundarySet::axis_side(&mesh, a, s).unwrap(),
                    ConservationBoundary::SlipWall,
                )
            })
            .collect();
        let p = StructuredConservationPlan::<R, D, N>::with_options(
            mesh,
            boundaries,
            ConservationOptions::new()
                .with_reconstruction(ConservationReconstruction::Minmod)
                .with_limits(limits()),
        )
        .unwrap();
        let state = gas.conserved(R::ONE, [R::ZERO; D], R::ONE).unwrap();
        let field = CellField::from_fn(p.mesh(), |_| state).unwrap();
        let (residual, report) =
            structured_residual(&p, &gas, &Hllc, R::ZERO, field.view()).unwrap();
        assert!(
            residual
                .iter()
                .flatten()
                .all(|x| x.abs() <= R::EPSILON * R::try_from_f64_lossy(256.).unwrap())
        );
        assert_eq!(report.boundary_flux[0], R::ZERO);
        assert_eq!(report.boundary_flux[N - 1], R::ZERO);
    }
    check::<f32, 1, 3>();
    check::<f32, 2, 4>();
    check::<f32, 3, 5>();
    check::<f64, 1, 3>();
    check::<f64, 2, 4>();
    check::<f64, 3, 5>();
}
/// Bounded scalar advection `u_t + u_x = 0` with admissible states in `[0, 1]`.
struct Unit;
impl ConservationLaw<f64, 1, 1> for Unit {
    fn validate(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn admissible(&self, u: &[f64; 1]) -> bool {
        (0. ..=1.).contains(&u[0])
    }
    fn normal_flux(&self, u: &[f64; 1], n: [f64; 1]) -> Result<[f64; 1], SciOxideError> {
        Ok([n[0] * u[0]])
    }
    fn wave_bounds(
        &self,
        _: &[f64; 1],
        _: &[f64; 1],
        n: [f64; 1],
    ) -> Result<(f64, f64), SciOxideError> {
        Ok((n[0], n[0]))
    }
}
/// Exact upwind flux reporting a tenth of the wave speed, so CFL admits too large steps.
struct Underreported;
impl NumericalFlux<f64, Unit, 1, 1> for Underreported {
    fn evaluate(
        &self,
        _: &Unit,
        left: &[f64; 1],
        right: &[f64; 1],
        n: [f64; 1],
    ) -> Result<RiemannReport<f64, 1>, SciOxideError> {
        Ok(RiemannReport {
            flux: [n[0] * if n[0] > 0. { left[0] } else { right[0] }],
            maximum_speed: 0.1 * n[0].abs(),
            used_hlle_fallback: false,
        })
    }
}
fn argument_of(error: &SciOxideError) -> Option<scioxide_core::Argument> {
    match error.view() {
        SciOxideErrorView::InvalidInput { argument, .. } => Some(argument),
        _ => None,
    }
}
#[test]
fn checked_time_failure_preserves_initial_state_and_budgets() {
    // dt/h = 2 drives the upwind update of [1, 0, 0, 0] outside [0, 1]: forward Euler's
    // candidate and Heun's second stage input are both inadmissible. Rejections retry
    // with half the step; a minimum step that allows no retry stops at the initial state.
    let p = plan::<f64, 1, 1>(4, true, ConservationReconstruction::Constant);
    let field = CellField::from_fn(p.mesh(), |i| [if i == 0 { 1. } else { 0. }]).unwrap();
    let initial = p.initial_state(&Unit, field.view()).unwrap();
    let mut work = p.workspace().unwrap();
    let sys = p.bind(&Unit, &Underreported, &mut work).unwrap();
    let problem = OdeProblem::new(&sys, 0.5);
    for method in [
        scioxide_ode::BuiltinOdeMethod::from(ForwardEuler::new()),
        Heun::new().into(),
    ] {
        let bounded = OdeOptions::new()
            .with_initial_step(0.5)
            .with_minimum_step(0.5);
        let solve = (method.clone(), bounded);
        let mut state = OdeState::new(0., initial.clone());
        let report = solve
            .0
            .advance(
                &problem,
                &solve.1,
                &mut state,
                &mut (),
                &mut solve.0.workspace(&problem, &solve.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            Some(scioxide_ode::OdeTermination::StepSizeLimit)
        );
        assert_eq!(report.rejected_steps(), 1);
        assert_eq!(state.time(), 0.);
        assert_eq!(state.values(), initial.as_slice());
        let retried = (method, OdeOptions::new().with_initial_step(0.5));
        let mut state = OdeState::new(0., initial.clone());
        let report = retried
            .0
            .advance(
                &problem,
                &retried.1,
                &mut state,
                &mut (),
                &mut retried.0.workspace(&problem, &retried.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!(report.rejected_steps() >= 1);
        assert!(state.values().iter().all(|&u| (0. ..=1.).contains(&u)));
    }
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(631),
        scioxide_core::algorithm_id!("test", "conservation", 1),
        0,
        0,
    );
    let mut cx = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut out = [7.; 4];
    let error = sys
        .rhs(0., &[2., 0., 0., 0.], &mut out, &mut cx)
        .unwrap_err();
    assert_eq!(argument_of(&error), Some(scioxide_core::argument!("y")));
    assert_eq!(out, [7.; 4]);
    assert!(!sys.admits(0.5, &[-1., 2., 0., 0.], &mut cx).unwrap());
    assert!(sys.admits(0.5, initial.as_slice(), &mut cx).unwrap());
    let p = plan::<f64, 1, 1>(4, true, ConservationReconstruction::Constant);
    let law = LinearAdvection { velocity: [1.] };
    let field = CellField::from_fn(p.mesh(), |_| [1.]).unwrap();
    let mut work = p.workspace().unwrap();
    let sys = p.bind(&law, &Godunov, &mut work).unwrap();
    let solve = (
        Heun::new(),
        OdeOptions::new().with_max_evaluations(Budget::ONE),
    );
    let problem = OdeProblem::new(&sys, 1.);
    let mut state = OdeState::new(0., p.initial_state(&law, field.view()).unwrap());
    // The CFL limit consumes the only evaluation; the denied stage fails the solve.
    let error = solve
        .0
        .advance(
            &problem,
            &solve.1,
            &mut state,
            &mut (),
            &mut solve.0.workspace(&problem, &solve.1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), [1.; 4]);
}

fn boundary_maximum_principle<const D: usize, E: ConservationLaw<f64, D, 1>>(law: E)
where
    Godunov: NumericalFlux<f64, E, D, 1>,
{
    for reconstruction in [
        ConservationReconstruction::Constant,
        ConservationReconstruction::Minmod,
        ConservationReconstruction::MonotonizedCentral,
    ] {
        for prescribed in [false, true] {
            let mesh = mesh::<f64, D>(20, false);
            let mut conditions = boundaries(&mesh);
            if prescribed {
                for (_, boundary) in &mut conditions {
                    *boundary = ConservationBoundary::exterior(|_, _| Ok([0.]));
                }
            }
            let options = ConservationOptions::new()
                .with_reconstruction(reconstruction)
                .with_cfl(0.4);
            let p = StructuredConservationPlan::with_options(
                mesh,
                conditions,
                options.with_limits(limits()),
            )
            .unwrap();
            let mut state = CellField::from_fn(p.mesh(), |i| {
                [if p.centers()[i][0] < 0.05 { 0. } else { 1. }]
            })
            .unwrap();
            let mut work = p.workspace().unwrap();
            let mut rhs = vec![[0.]; state.len()];
            for _ in 0..8 {
                let report = p
                    .residual_into(
                        &law,
                        &Godunov,
                        0.,
                        state.view(),
                        &mut rhs,
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                let dt = report.maximum_step.unwrap();
                let values = state
                    .values()
                    .iter()
                    .zip(&rhs)
                    .map(|(u, f)| [u[0] + dt * f[0]])
                    .collect();
                state = CellField::new(p.mesh(), values).unwrap();
                assert!(
                    state
                        .values()
                        .iter()
                        .all(|u| u[0] >= -1e-14 && u[0] <= 1. + 1e-14),
                    "D={D}, {reconstruction:?}, prescribed={prescribed}"
                );
            }
        }
    }
}

#[test]
fn boundary_muscl_preserves_advection_and_burgers_bounds() {
    for sign in [-1., 1.] {
        boundary_maximum_principle(LinearAdvection { velocity: [sign] });
        boundary_maximum_principle(LinearAdvection {
            velocity: [sign, 0.25],
        });
        boundary_maximum_principle(Burgers { direction: [sign] });
        boundary_maximum_principle(Burgers {
            direction: [sign, 0.25],
        });
    }
}

#[test]
fn simplex_boundary_reconstruction_preserves_scalar_bounds() {
    for reconstruction in [
        ConservationReconstruction::Constant,
        ConservationReconstruction::BarthJespersen,
    ] {
        for prescribed in [false, true] {
            let mesh = triangles(8);
            let tags =
                SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect())
                    .unwrap();
            let condition = if prescribed {
                ConservationBoundary::exterior(|_, _| Ok([0.]))
            } else {
                ConservationBoundary::Transmissive
            };
            let p = SimplexConservationPlan::with_options(
                mesh,
                &tags,
                vec![(0, condition)],
                ConservationOptions::new()
                    .with_reconstruction(reconstruction)
                    .with_limits(limits()),
            )
            .unwrap();
            let state = CellField::from_fn(p.mesh(), |i| {
                [if p.centers()[i][0] < 0.125 { 0. } else { 1. }]
            })
            .unwrap();
            for burgers in [false, true] {
                let mut work = p.workspace().unwrap();
                let mut rhs = vec![[0.]; state.len()];
                let report = if burgers {
                    p.residual_into(
                        &Burgers {
                            direction: [1., 0.25],
                        },
                        &Godunov,
                        0.,
                        state.view(),
                        &mut rhs,
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                } else {
                    p.residual_into(
                        &LinearAdvection {
                            velocity: [1., 0.25],
                        },
                        &Godunov,
                        0.,
                        state.view(),
                        &mut rhs,
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                }
                .unwrap();
                assert!(state.values().iter().zip(&rhs).all(|(u, f)| {
                    let next = u[0] + report.maximum_step.unwrap() * f[0];
                    (-1e-14..=1. + 1e-14).contains(&next)
                }));
            }
        }
    }
}

#[test]
fn admissibility_rejections_retry_without_an_error_estimator() {
    // Fast decay drives large Euler/Heun steps through zero density; the gas guard
    // rejects those trials and the plan retries with half the step.
    struct Decay<'a>(&'a IdealGasEuler<f64>);
    impl OdeSystem<f64> for Decay<'_> {
        fn dimension(&self) -> usize {
            3
        }
        fn rhs(
            &self,
            _: f64,
            state: &[f64],
            output: &mut [f64],
            _: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            for (f, &u) in output.iter_mut().zip(state) {
                *f = -10. * u;
            }
            Ok(())
        }
        fn admits(
            &self,
            _: f64,
            state: &[f64],
            _: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<bool, SciOxideError> {
            Ok(
                <IdealGasEuler<f64> as ConservationLaw<f64, 1, 3>>::admissible(
                    self.0,
                    state.try_into().unwrap(),
                ),
            )
        }
    }
    let gas = IdealGasEuler::new(1.4).unwrap();
    let initial = gas.conserved::<1, 3>(1., [0.], 1.).unwrap();
    let system = Decay(&gas);
    let problem = OdeProblem::new(&system, 0.2);
    for method in [
        scioxide_ode::BuiltinOdeMethod::from(ForwardEuler::new()),
        Heun::new().into(),
    ] {
        let solve = (method, OdeOptions::new().with_initial_step(0.2));
        let mut state = OdeState::from_slice(0., &initial).unwrap();
        let mut sink = |_: f64, state: &[f64]| {
            assert!(
                <IdealGasEuler<f64> as ConservationLaw<f64, 1, 3>>::admissible(
                    &gas,
                    state.try_into().unwrap()
                )
            );
            Ok(())
        };
        let report = solve
            .0
            .advance(
                &problem,
                &solve.1,
                &mut state,
                &mut sink,
                &mut solve.0.workspace(&problem, &solve.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.final_time(), 0.2);
        assert!(report.rejected_steps() >= 2);
        assert!(report.accepted_steps() > 0);
    }
}

#[test]
fn stage_guard_preserves_provider_errors_and_nonfinite_output() {
    struct Broken(bool);
    impl OdeSystem<f64> for Broken {
        fn dimension(&self) -> usize {
            3
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            output: &mut [f64],
            _: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            if self.0 {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "provider breakdown",
                ));
            }
            output.fill(f64::NAN);
            Ok(())
        }
    }
    let gas = IdealGasEuler::new(1.4).unwrap();
    let initial = gas.conserved::<1, 3>(1., [0.], 1.).unwrap();
    let solve = (Heun::new(), OdeOptions::new().with_initial_step(0.01));
    for provider_error in [false, true] {
        let system = Broken(provider_error);
        let problem = OdeProblem::new(&system, 0.01);
        let mut state = OdeState::from_slice(0., &initial).unwrap();
        let error = solve
            .0
            .advance(
                &problem,
                &solve.1,
                &mut state,
                &mut (),
                &mut solve.0.workspace(&problem, &solve.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        if provider_error {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ));
        } else {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Callback,
                    ..
                }
            ));
        }
        assert_eq!(state.values(), initial);
    }
}

fn double_rarefaction<F: NumericalFlux<f64, IdealGasEuler<f64>, 1, 3>>(flux: F) {
    // Expose the same spatial RHS and admissibility guard without a CFL hint as well:
    // an oversized initial proposal must be recovered by stage rejection, rather than
    // depending on the driver's stage-one wave-speed bound for admissibility.
    struct WithoutCfl<'a, S>(&'a S);
    impl<S: OdeSystem<f64>> OdeSystem<f64> for WithoutCfl<'_, S> {
        fn dimension(&self) -> usize {
            self.0.dimension()
        }
        fn rhs(
            &self,
            t: f64,
            y: &[f64],
            out: &mut [f64],
            cx: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.0.rhs(t, y, out, cx)
        }
        fn admits(
            &self,
            t: f64,
            y: &[f64],
            cx: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<bool, SciOxideError> {
            self.0.admits(t, y, cx)
        }
    }
    let solve = (Heun::new(), OdeOptions::new().with_initial_step(0.01));
    for advertise_cfl in [true, false] {
        let gas = IdealGasEuler::new(1.4).unwrap();
        let p = plan::<f64, 1, 3>(100, false, ConservationReconstruction::MonotonizedCentral);
        let initial = CellField::from_fn(p.mesh(), |i| {
            gas.conserved::<1, 3>(1., [if i < 50 { -2. } else { 2. }], 0.4)
                .unwrap()
        })
        .unwrap();
        let mut work = p.workspace().unwrap();
        let bound = p.bind(&gas, &flux, &mut work).unwrap();
        let no_cfl = WithoutCfl(&bound);
        let system: &dyn OdeSystem<f64> = if advertise_cfl { &bound } else { &no_cfl };
        let problem = OdeProblem::new(system, 0.15);
        let mut state = OdeState::new(0., p.initial_state(&gas, initial.view()).unwrap());
        let mut sink = |_: f64, state: &[f64]| {
            assert!(state.as_chunks::<3>().0.iter().all(|u| {
                <IdealGasEuler<f64> as ConservationLaw<f64, 1, 3>>::admissible(&gas, u)
            }));
            Ok(())
        };
        let report = solve
            .0
            .advance(
                &problem,
                &solve.1,
                &mut state,
                &mut sink,
                &mut solve.0.workspace(&problem, &solve.1).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.final_time(), 0.15);
        if !advertise_cfl {
            assert!(report.rejected_steps() > 0);
        }
        assert!(p.field(&gas, state.values()).is_ok());
    }
}

#[test]
fn toro_double_rarefaction_completes_with_hlle_and_hllc() {
    double_rarefaction(Hlle);
    double_rarefaction(Hllc);
}

#[allow(
    unreachable_pub,
    reason = "shared with integration tests, where items are public"
)]
#[path = "../../tests/support/conservation_reference.rs"]
mod reference;
use reference::{ExactRiemann, Primitive};

fn shock_tube<R: RealScalar>(left: Primitive, right: Primitive, end: f64) -> [f64; 3] {
    let cv = |x| R::try_from_f64_lossy(x).unwrap();
    let gas = IdealGasEuler::new(cv(1.4)).unwrap();
    let p = plan::<R, 1, 3>(400, false, ConservationReconstruction::MonotonizedCentral);
    let initial = CellField::from_fn(p.mesh(), |i| {
        let state = if i < 200 { left } else { right };
        gas.conserved(cv(state.rho), [cv(state.velocity)], cv(state.pressure))
            .unwrap()
    })
    .unwrap();
    let mut work = p.workspace().unwrap();
    let sys = p.bind(&gas, &Hllc, &mut work).unwrap();
    let solution = heun(
        &sys,
        cv(0.0001),
        cv(end),
        p.initial_state(&gas, initial.view()).unwrap(),
    )
    .unwrap();
    assert_eq!(solution.report().final_time(), cv(end));
    let field = p.field(&gas, solution.state().values()).unwrap();
    let exact = ExactRiemann::new(left, right);
    let scale = core::array::from_fn::<_, 3, _>(|j| {
        left.conserved()[j]
            .abs()
            .max(right.conserved()[j].abs())
            .max(1.)
    });
    let mut error = [0.; 3];
    for (i, u) in field.values().iter().enumerate() {
        let reference = exact.cell_average(i as f64 / 400., (i + 1) as f64 / 400., end);
        for j in 0..3 {
            error[j] += (u[j].into_f64() - reference[j]).abs() / 400. / scale[j];
        }
    }
    assert!(error.iter().all(|&e| e < 0.05), "shock errors {error:?}");
    error
}
#[test]
fn sod_and_lax_independent_cell_average_accuracy() {
    let sod = (
        Primitive {
            rho: 1.,
            velocity: 0.,
            pressure: 1.,
        },
        Primitive {
            rho: 0.125,
            velocity: 0.,
            pressure: 0.1,
        },
    );
    let exact = ExactRiemann::new(sod.0, sod.1);
    assert!((exact.pressure - 0.30313017805064685).abs() < 1e-14);
    assert!((exact.velocity - 0.9274526200489499).abs() < 1e-14);
    let lax = (
        Primitive {
            rho: 0.445,
            velocity: 0.698,
            pressure: 3.528,
        },
        Primitive {
            rho: 0.5,
            velocity: 0.,
            pressure: 0.571,
        },
    );
    std::eprintln!(
        "Sod f64 {:?}; f32 {:?}",
        shock_tube::<f64>(sod.0, sod.1, 0.2),
        shock_tube::<f32>(sod.0, sod.1, 0.2)
    );
    std::eprintln!(
        "Lax f64 {:?}; f32 {:?}",
        shock_tube::<f64>(lax.0, lax.1, 0.14),
        shock_tube::<f32>(lax.0, lax.1, 0.14)
    );
}

fn simplex_smooth_error<const D: usize>(mesh: SimplexMesh<f64, D>) -> f64 {
    // Smooth compact support stays away from the exterior; centroid quadrature
    // of the initial/final cell averages has O(h²) error independently of the flux.
    let p = simplex_plan::<D, 1>(mesh, ConservationReconstruction::BarthJespersen);
    let law = LinearAdvection { velocity: [0.4; D] };
    let end = 0.1;
    let bump = |point: [f64; D], time: f64| {
        point
            .iter()
            .map(|x| {
                let d = (*x - 0.5 - 0.4 * time) / 0.35;
                if d.abs() < 1. {
                    (1. - d * d).powi(4)
                } else {
                    0.
                }
            })
            .product::<f64>()
    };
    let initial = CellField::from_fn(p.mesh(), |i| [bump(p.centers()[i], 0.)]).unwrap();
    let mut work = p.workspace().unwrap();
    let sys = p.bind(&law, &Godunov, &mut work).unwrap();
    let dt = sys
        .report(
            0.,
            initial.values().as_flattened(),
            ExecutionContext::sequential(),
        )
        .unwrap()
        .maximum_step
        .unwrap()
        * 0.8;
    let solution = heun(
        &sys,
        dt,
        end,
        p.initial_state(&law, initial.view()).unwrap(),
    )
    .unwrap();
    solution
        .state()
        .values()
        .iter()
        .zip(p.centers())
        .zip(p.volumes())
        .map(|((&u, &x), &v)| (u - bump(x, end)).abs() * v)
        .sum()
}
#[test]
fn simplex_smooth_refinement() {
    let triangles = [
        simplex_smooth_error(triangles(12)),
        simplex_smooth_error(triangles(24)),
        simplex_smooth_error(triangles(48)),
    ];
    let tetrahedra = [
        simplex_smooth_error(tetrahedra(4)),
        simplex_smooth_error(tetrahedra(8)),
        simplex_smooth_error(tetrahedra(16)),
    ];
    std::eprintln!("triangle errors {triangles:?}; tetrahedron errors {tetrahedra:?}");
    for errors in [triangles, tetrahedra] {
        assert!(errors[1] / errors[2] >= 2_f64.powf(1.5), "{errors:?}");
    }
}

#[test]
fn near_vacuum_reconstruction_limits_without_changing_averages() {
    fn check<R: RealScalar>() {
        let cv = |x| R::try_from_f64_lossy(x).unwrap();
        let gas = IdealGasEuler::new(cv(1.4)).unwrap();
        let p = plan::<R, 1, 3>(5, false, ConservationReconstruction::MonotonizedCentral);
        let initial = CellField::from_fn(p.mesh(), |i| {
            gas.conserved(cv(1.), [cv(i as f64 - 2.)], cv(1e-6))
                .unwrap()
        })
        .unwrap();
        let original = initial.values().to_vec();
        let mut work = p.workspace().unwrap();
        let mut out = vec![[R::ZERO; 3]; 5];
        let r = p
            .residual_into(
                &gas,
                &Hllc,
                R::ZERO,
                initial.view(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(r.limited_cells > 0);
        assert_eq!(initial.values(), original);
        let sys = p.bind(&gas, &Hllc, &mut work).unwrap();
        let solve = (Heun::new(), OdeOptions::new().with_initial_step(cv(1e-5)));
        let problem = OdeProblem::new(&sys, cv(1e-4));
        let mut state = OdeState::new(R::ZERO, p.initial_state(&gas, initial.view()).unwrap());
        let result = solve.0.advance(
            &problem,
            &solve.1,
            &mut state,
            &mut |_, y: &[R]| {
                assert!(
                    y.as_chunks::<3>()
                        .0
                        .iter()
                        .all(|u| gas.pressure::<1, 3>(u).is_ok())
                );
                Ok(())
            },
            &mut solve.0.workspace(&problem, &solve.1).unwrap(),
            ExecutionContext::sequential(),
        );
        match result {
            Ok(report) => assert_eq!(report.final_time(), cv(1e-4)),
            Err(_) => {
                assert!(
                    state
                        .values()
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .all(|u| gas.pressure::<1, 3>(u).is_ok())
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn simplex_both_precisions_and_rank_fallback() {
    fn check<R: RealScalar, const D: usize, const N: usize>(mesh: SimplexMesh<R, D>) {
        let tags = SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect())
            .unwrap();
        let p = SimplexConservationPlan::<R, D, N>::with_options(
            mesh,
            &tags,
            vec![(0, ConservationBoundary::SlipWall)],
            ConservationOptions::new()
                .with_reconstruction(ConservationReconstruction::BarthJespersen)
                .with_limits(limits()),
        )
        .unwrap();
        let gas = IdealGasEuler::new(R::try_from_f64_lossy(1.4).unwrap()).unwrap();
        let state = gas.conserved::<D, N>(R::ONE, [R::ZERO; D], R::ONE).unwrap();
        let field = CellField::from_fn(p.mesh(), |_| state).unwrap();
        let (out, report) = simplex_residual(&p, &gas, &Hllc, R::ZERO, field.view()).unwrap();
        assert_eq!(report.rank_deficient_cells, 1);
        assert!(
            out.iter()
                .flatten()
                .all(|x| x.abs() < R::try_from_f64_lossy(512.).unwrap() * R::EPSILON)
        );
    }
    check::<f32, 2, 4>(
        TriangleMesh::new(
            vec![
                Point::new([0., 0.]).unwrap(),
                Point::new([1., 0.]).unwrap(),
                Point::new([0., 1.]).unwrap(),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap(),
    );
    check::<f32, 3, 5>(
        TetrahedronMesh::new(
            vec![
                Point::new([0., 0., 0.]).unwrap(),
                Point::new([1., 0., 0.]).unwrap(),
                Point::new([0., 1., 0.]).unwrap(),
                Point::new([0., 0., 1.]).unwrap(),
            ],
            vec![[0, 1, 2, 3]],
        )
        .unwrap(),
    );
    check::<f64, 2, 4>(
        TriangleMesh::new(
            vec![
                Point::new([0., 0.]).unwrap(),
                Point::new([1., 0.]).unwrap(),
                Point::new([0., 1.]).unwrap(),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap(),
    );
    check::<f64, 3, 5>(
        TetrahedronMesh::new(
            vec![
                Point::new([0., 0., 0.]).unwrap(),
                Point::new([1., 0., 0.]).unwrap(),
                Point::new([0., 1., 0.]).unwrap(),
                Point::new([0., 0., 1.]).unwrap(),
            ],
            vec![[0, 1, 2, 3]],
        )
        .unwrap(),
    );
}

#[test]
fn cfl_retains_unrepresentable_intermediate_face_rates() {
    for (length, speed, expected) in [(1e-200_f64, 1e-200, 0.225e100), (1e200, 1e200, 0.225e-200)] {
        let vertical = if length < 1. { 1e-100 } else { 1. };
        let m = CellMesh::new(
            StructuredGrid::new(
                [
                    CoordinateAxis::uniform(0., length, 2).unwrap(),
                    CoordinateAxis::uniform(0., vertical, 2).unwrap(),
                ],
                [AxisTopology::Bounded; 2],
            )
            .unwrap(),
        )
        .unwrap();
        let b = boundaries(&m);
        let p = StructuredConservationPlan::<_, 2, 1>::with_options(
            m,
            b,
            ConservationOptions::new()
                .with_reconstruction(ConservationReconstruction::Constant)
                .with_limits(limits()),
        )
        .unwrap();
        let field = CellField::from_fn(p.mesh(), |_| [0.]).unwrap();
        let (_, r) = structured_residual(
            &p,
            &LinearAdvection {
                velocity: [0., speed],
            },
            &Godunov,
            0.,
            field.view(),
        )
        .unwrap();
        assert!((r.maximum_step.unwrap() / expected - 1.).abs() < 1e-14);
    }
}

#[test]
fn moving_euler_contact_preserves_pressure_velocity_and_totals() {
    fn check<R: RealScalar>() {
        let cv = |x| R::try_from_f64_lossy(x).unwrap();
        let p = plan::<R, 1, 3>(80, true, ConservationReconstruction::MonotonizedCentral);
        let gas = IdealGasEuler::new(cv(1.4)).unwrap();
        let initial = CellField::from_fn(p.mesh(), |i| {
            gas.conserved(
                if (20..40).contains(&i) {
                    cv(2.)
                } else {
                    R::ONE
                },
                [R::ONE],
                R::ONE,
            )
            .unwrap()
        })
        .unwrap();
        let total = core::array::from_fn::<_, 3, _>(|j| {
            initial
                .values()
                .iter()
                .map(|u| u[j].into_f64() / 80.)
                .sum::<f64>()
        });
        let mut work = p.workspace().unwrap();
        let sys = p.bind(&gas, &Hllc, &mut work).unwrap();
        let solution = heun(
            &sys,
            cv(0.0005),
            cv(0.1),
            p.initial_state(&gas, initial.view()).unwrap(),
        )
        .unwrap();
        let result = p.field(&gas, solution.state().values()).unwrap();
        let tolerance = R::EPSILON.into_f64() * 4096.;
        for u in result.values() {
            assert!((gas.pressure::<1, 3>(u).unwrap() - R::ONE).abs().into_f64() < tolerance);
            assert!((u[1] / u[0] - R::ONE).abs().into_f64() < tolerance);
        }
        for j in 0..3 {
            let after = result
                .values()
                .iter()
                .map(|u| u[j].into_f64() / 80.)
                .sum::<f64>();
            assert!((after - total[j]).abs() < tolerance);
        }
        let error = result
            .values()
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let x = p.centers()[i][0].into_f64();
                (u[0].into_f64() - if (0.35..0.6).contains(&x) { 2. } else { 1. }).abs() / 80.
            })
            .sum::<f64>();
        assert!(error < 0.025, "contact L1 {error}");
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn burgers_shock_and_rarefaction_match_entropy_solutions() {
    for rarefaction in [false, true] {
        let p = plan::<f64, 1, 1>(160, false, ConservationReconstruction::Minmod);
        let law = Burgers { direction: [1.] };
        let initial = CellField::from_fn(p.mesh(), |i| {
            [if i < 80 {
                if rarefaction { -1. } else { 1. }
            } else {
                if rarefaction { 1. } else { 0. }
            }]
        })
        .unwrap();
        let mut work = p.workspace().unwrap();
        let sys = p.bind(&law, &Godunov, &mut work).unwrap();
        let solution = heun(
            &sys,
            0.001,
            0.1,
            p.initial_state(&law, initial.view()).unwrap(),
        )
        .unwrap();
        let error = solution
            .state()
            .values()
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let x = p.centers()[i][0];
                let exact = if rarefaction {
                    ((x - 0.5) / 0.1).clamp(-1., 1.)
                } else if x < 0.55 {
                    1.
                } else {
                    0.
                };
                (u - exact).abs() / 160.
            })
            .sum::<f64>();
        assert!(
            error < 0.012,
            "Burgers rarefaction={rarefaction} L1={error}"
        );
    }
}
#[test]
fn boundary_riemann_time_and_callback_failure_are_explicit() {
    let m = mesh::<f64, 1>(4, false);
    let b = vec![
        (
            BoundarySet::axis_side(&m, 0, AxisSide::Start).unwrap(),
            ConservationBoundary::exterior(|_, t| {
                if t > 1. {
                    Err(invalid(TEST_ARGUMENT, "test exterior failure"))
                } else {
                    Ok([1. + t])
                }
            }),
        ),
        (
            BoundarySet::axis_side(&m, 0, AxisSide::End).unwrap(),
            ConservationBoundary::Transmissive,
        ),
    ];
    let p = StructuredConservationPlan::<_, 1, 1>::with_options(
        m,
        b,
        ConservationOptions::new()
            .with_reconstruction(ConservationReconstruction::Constant)
            .with_limits(limits()),
    )
    .unwrap();
    let law = LinearAdvection { velocity: [1.] };
    let initial = CellField::from_fn(p.mesh(), |_| [0.]).unwrap();
    let (r, d) = structured_residual(&p, &law, &Godunov, 0.5, initial.view()).unwrap();
    assert_eq!(d.boundary_flux, [-1.5]);
    assert_eq!(r, [[6.], [0.], [0.], [0.]]);
    let mut work = p.workspace().unwrap();
    let mut output = vec![[42.]; 4];
    assert!(
        p.residual_into(
            &law,
            &Godunov,
            2.,
            initial.view(),
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, vec![[42.]; 4]);
}
