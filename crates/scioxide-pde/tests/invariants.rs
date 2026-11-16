#![allow(clippy::unwrap_used)]

#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use scioxide_spatial::SegmentMesh;

use scioxide_core::{NonFiniteOrigin, SciOxideError};
use scioxide_pde::{
    BoundaryKind, EllipticOptions, EllipticProblem, ParabolicOptions, ParabolicProblem,
    ParabolicScheme, PdeOutputSchedule, PdeSolution, PdeSolveReport, PdeTermination,
};

use support::{
    DISCRETIZATIONS, Discretization, EllipticPde, End, Ends, ParabolicPde, Plan1d, Steady,
    TestScalar, as_f64, assert_refinement, assert_scaled_residual, assert_valid_solution,
    elliptic_options, homogeneous_dirichlet, initial, max_abs_error, moving, scalar,
    solve_tolerance, source, value,
};

fn unit_grid<S: TestScalar>(node_count: usize) -> SegmentMesh<S> {
    mesh_fixture::uniform(S::zero_value(), S::one_value(), node_count)
}

#[derive(Clone)]
struct ManufacturedPoisson;

impl<S: TestScalar> EllipticPde<S> for ManufacturedPoisson {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        let pi = S::PI;
        Ok(pi * pi * (pi * x[0]).sin())
    }
}

/// Steady solve of `problem`: the full field in coordinate order and its report.
fn elliptic<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<S>,
    boundaries: &Ends<S>,
    problem: &P,
    options: EllipticOptions<S>,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let steady = Steady(problem.clone());
    let plan = discretization.plan(grid, &steady, boundaries)?;
    let source = source(&steady);
    let solver = options;
    let (values, report) = with_plan!(&plan, |p| EllipticProblem::new(p, &source)
        .solve_with_options(&solver))?
    .into_parts();
    Ok((plan.full(&values)?, report))
}

fn solve_elliptic<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<S>,
    boundaries: &Ends<S>,
    problem: &P,
) -> (Vec<S>, PdeSolveReport<S>) {
    let pure_natural = [&boundaries.left, &boundaries.right]
        .into_iter()
        .all(|end| {
            end.as_ref()
                .is_some_and(|end| end.kind == BoundaryKind::Natural)
        });
    let options = if pure_natural {
        elliptic_options::<S>()
            .with_gauge(scioxide_pde::EllipticGauge::WeightedMean(S::zero_value()))
    } else {
        elliptic_options::<S>()
    };
    let solution = elliptic(discretization, grid, boundaries, problem, options).unwrap();
    assert_valid_solution(
        discretization.name(),
        &solution.1,
        &[],
        &[&solution.0],
        discretization.coordinates(grid).len(),
    );
    solution
}

fn poisson_error<S: TestScalar>(discretization: Discretization, node_count: usize) -> f64 {
    let grid = unit_grid::<S>(node_count);
    let (values, _) = solve_elliptic(
        discretization,
        &grid,
        &homogeneous_dirichlet(),
        &ManufacturedPoisson,
    );
    let coordinates = discretization.coordinates(&grid);
    if matches!(discretization, Discretization::Fem) {
        // Gaussian P1 Poisson solutions are superconvergent at vertices. Measure
        // the actual piecewise-linear field between vertices, above f32 roundoff.
        return values
            .windows(2)
            .zip(coordinates.windows(2))
            .map(|(u, x)| {
                let midpoint = (as_f64(x[0]) + as_f64(x[1])) * 0.5;
                ((as_f64(u[0]) + as_f64(u[1])) * 0.5 - (core::f64::consts::PI * midpoint).sin())
                    .abs()
            })
            .fold(0.0_f64, f64::max);
    }
    max_abs_error(&values, &coordinates, |x| (S::PI * x).sin())
}

fn assert_elliptic_refinement<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        assert_refinement(
            discretization.name(),
            S::NAME,
            poisson_error::<S>(discretization, 25),
            poisson_error::<S>(discretization, 49),
            S::ELLIPTIC_REFINEMENT_RATIO,
        );
    }
}

#[test]
fn every_elliptic_discretization_refines_for_f64() {
    assert_elliptic_refinement::<f64>();
}

#[test]
fn every_elliptic_discretization_refines_for_f32() {
    assert_elliptic_refinement::<f32>();
}

#[derive(Clone)]
struct LinearHarmonic;

impl<S: TestScalar> EllipticPde<S> for LinearHarmonic {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn source(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

#[derive(Debug, Clone, Copy)]
enum BoundaryFixture {
    Dirichlet,
    Neumann,
    Robin,
}

fn constant<S: TestScalar>(data: f64) -> scioxide_pde::BoundaryData<S, S, 1> {
    value(move |_| Ok(scalar::<S>(data)))
}

/// Left ends satisfied by `u = 1 + 2x` with unit diffusion: `u(0) = 1`,
/// outward flux `-u'(0) = -2`, or `-u'(0) + u(0) = -1`; `u(1) = 3` on the right.
fn linear_boundaries<S: TestScalar>(fixture: BoundaryFixture) -> Ends<S> {
    let left = match fixture {
        BoundaryFixture::Dirichlet => End::dirichlet(constant(1.0)),
        BoundaryFixture::Neumann => End::natural(constant(-2.0)),
        BoundaryFixture::Robin => End::robin(S::one_value(), constant(-1.0)),
    };
    Ends::new(left, End::dirichlet(constant(3.0)))
}

fn endpoint_trace_and_slope<S: TestScalar>(values: &[S], x: &[S]) -> (f64, f64, f64) {
    let x0 = as_f64(x[0]);
    let x1 = as_f64(x[1]);
    let last = x.len() - 1;
    let slope_left = as_f64(values[1] - values[0]) / (x1 - x0);
    let slope_right = as_f64(values[last] - values[last - 1]) / as_f64(x[last] - x[last - 1]);
    let left_trace = as_f64(values[0]) - slope_left * x0;
    let right_trace = as_f64(values[last]) + slope_right * (1.0 - as_f64(x[last]));
    (left_trace, right_trace, slope_left)
}

fn assert_boundary_residuals<S: TestScalar>() {
    let node_count = 33;
    let spacing = 1.0 / (node_count - 1) as f64;
    for discretization in DISCRETIZATIONS {
        for fixture in [
            BoundaryFixture::Dirichlet,
            BoundaryFixture::Neumann,
            BoundaryFixture::Robin,
        ] {
            let grid = unit_grid::<S>(node_count);
            let (values, _) = solve_elliptic(
                discretization,
                &grid,
                &linear_boundaries(fixture),
                &LinearHarmonic,
            );
            let (left_trace, right_trace, slope) =
                endpoint_trace_and_slope(&values, &discretization.coordinates(&grid));

            assert_scaled_residual::<S>(
                &format!("{}/{fixture:?}/right Dirichlet", discretization.name()),
                (right_trace - 3.0).abs(),
                3.0,
                4.0 * spacing * S::SOLVE_TOLERANCE,
            );
            let (left_residual, truncation) = match fixture {
                BoundaryFixture::Dirichlet => ((left_trace - 1.0).abs(), 0.0),
                BoundaryFixture::Neumann => ((-slope + 2.0).abs(), 0.0),
                BoundaryFixture::Robin => ((left_trace - slope + 1.0).abs(), 0.0),
            };
            assert_scaled_residual::<S>(
                &format!("{}/{fixture:?}/left", discretization.name()),
                left_residual,
                3.0,
                truncation + 4.0 * spacing * S::SOLVE_TOLERANCE,
            );
        }
    }
}

#[test]
fn elliptic_boundary_residuals_are_normalized_for_f64() {
    assert_boundary_residuals::<f64>();
}

#[test]
fn elliptic_boundary_residuals_are_normalized_for_f32() {
    assert_boundary_residuals::<f32>();
}

fn assert_fvm_flux_balance<S: TestScalar>() {
    let node_count = 33;
    let grid = unit_grid::<S>(node_count);
    let (values, _) = solve_elliptic(
        Discretization::Fvm,
        &grid,
        &homogeneous_dirichlet(),
        &ManufacturedPoisson,
    );
    let x = Discretization::Fvm.coordinates(&grid);
    let width = 1.0 / (node_count - 1) as f64;
    let mut max_residual = 0.0_f64;
    let mut max_scale = 0.0_f64;
    let mut global_residual = 0.0_f64;
    for index in 0..values.len() {
        let value = as_f64(values[index]);
        let left = if index == 0 {
            value / (width / 2.0)
        } else {
            (value - as_f64(values[index - 1])) / width
        };
        let right = if index + 1 == values.len() {
            value / (width / 2.0)
        } else {
            (value - as_f64(values[index + 1])) / width
        };
        let source = std::f64::consts::PI.powi(2) * (std::f64::consts::PI * as_f64(x[index])).sin();
        let residual = (left + right) / width - source;
        let scale = (left.abs() + right.abs()) / width + source.abs();
        max_residual = max_residual.max(residual.abs());
        max_scale = max_scale.max(scale);
        global_residual += residual * width;
    }
    assert_scaled_residual::<S>("FVM local TPFA balance", max_residual, max_scale, 0.0);
    assert_scaled_residual::<S>(
        "FVM global TPFA balance",
        global_residual.abs(),
        max_scale,
        0.0,
    );
}

#[test]
fn finite_volume_flux_balance_holds_for_f64() {
    assert_fvm_flux_balance::<f64>();
}

#[test]
fn finite_volume_flux_balance_holds_for_f32() {
    assert_fvm_flux_balance::<f32>();
}

#[derive(Clone)]
struct ConstantSource<S> {
    source: S,
}

impl<S: TestScalar> EllipticPde<S> for ConstantSource<S> {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn source(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(self.source)
    }
}

#[derive(Clone)]
struct CompatibleSinNeumann;

impl<S: TestScalar> EllipticPde<S> for CompatibleSinNeumann {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok((S::TAU * x[0]).sin())
    }
}

/// Outward conormal fluxes on both ends (unit diffusion).
fn pure_neumann<S: TestScalar>(left: S, right: S) -> Ends<S> {
    Ends::new(
        End::natural(value(move |_| Ok(left))),
        End::natural(value(move |_| Ok(right))),
    )
}

fn assert_pure_neumann_contract<S: TestScalar>() {
    let node_count = 33;
    for discretization in DISCRETIZATIONS {
        let grid = unit_grid::<S>(node_count);
        let derivative = S::one_value() / S::TAU;
        let boundaries = pure_neumann(-derivative, derivative);
        let (_, report) = solve_elliptic(discretization, &grid, &boundaries, &CompatibleSinNeumann);
        assert!(report.diagnostics().nullspace().is_some());
        let spacing = 1.0 / (node_count - 1) as f64;
        let source_integral = (0..node_count)
            .map(|index| {
                let coordinate = index as f64 * spacing;
                let weight = if index == 0 || index + 1 == node_count {
                    0.5
                } else {
                    1.0
                };
                weight * (2.0 * std::f64::consts::PI * coordinate).sin()
            })
            .sum::<f64>()
            * spacing;
        let boundary_flux = as_f64(-derivative + derivative);
        let compatibility = (source_integral + boundary_flux).abs();
        assert_scaled_residual::<S>(
            &format!("{} pure-Neumann compatibility", discretization.name()),
            compatibility,
            4.0,
            0.0,
        );

        let result = elliptic(
            discretization,
            &grid,
            &pure_neumann(S::zero_value(), S::zero_value()),
            &ConstantSource {
                source: S::one_value(),
            },
            elliptic_options::<S>()
                .with_gauge(scioxide_pde::EllipticGauge::WeightedMean(S::zero_value())),
        );
        assert!(matches!(
            result.map(|_| ()).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn pure_neumann_compatibility_is_normalized_for_f64() {
    assert_pure_neumann_contract::<f64>();
}

#[test]
fn pure_neumann_compatibility_is_normalized_for_f32() {
    assert_pure_neumann_contract::<f32>();
}

#[derive(Clone)]
struct PeriodicSourceFree;

impl<S: TestScalar> EllipticPde<S> for PeriodicSourceFree {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn source(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

fn assert_periodic_source_free_balance<S: TestScalar>() {
    let grid = mesh_fixture::periodic(&unit_grid::<S>(33));
    let (values, report) = elliptic(
        Discretization::Fdm,
        &grid,
        &Ends::periodic(),
        &PeriodicSourceFree,
        elliptic_options::<S>(),
    )
    .unwrap();
    assert_valid_solution(
        "FDM periodic source-free",
        &report,
        &[],
        &[&values],
        Discretization::Fdm.coordinates(&grid).len(),
    );
    let last = values.len() - 1;
    let value_residual = as_f64(values[0] - values[last]).abs();
    let flux_residual = as_f64((values[1] - values[0]) - (values[last] - values[last - 1])).abs();
    let mass = values.iter().copied().map(as_f64).sum::<f64>() / values.len() as f64;
    assert_scaled_residual::<S>("periodic value", value_residual, 1.0, 0.0);
    assert_scaled_residual::<S>("periodic wrapped flux", flux_residual, 1.0, 0.0);
    assert_scaled_residual::<S>("periodic source-free mass", mass.abs(), 1.0, 0.0);
}

#[test]
fn periodic_source_free_balance_holds_for_f64() {
    assert_periodic_source_free_balance::<f64>();
}

#[test]
fn periodic_source_free_balance_holds_for_f32() {
    assert_periodic_source_free_balance::<f32>();
}

// Parabolic refinement and solve-path contracts.

#[derive(Clone)]
struct Heat;

impl<S: TestScalar> ParabolicPde<S> for Heat {
    fn diffusion(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn source(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok((S::PI * x[0]).sin())
    }
}

fn fixed_parabolic_options<S: TestScalar>(
    step: f64,
    scheme: ParabolicScheme,
) -> ParabolicOptions<S> {
    ParabolicOptions::new()
        .with_scheme(scheme)
        .with_step(scalar::<S>(step))
        .with_error_tolerance(scioxide_core::ErrorTolerance::from_tolerance(
            solve_tolerance::<S>(),
            scioxide_core::NormSelection::RootMeanSquare,
        ))
}

fn adaptive_parabolic_options<S: TestScalar>(
    initial_step: f64,
    tolerance: f64,
) -> ParabolicOptions<S> {
    let threshold = scalar::<S>(tolerance);
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(scalar::<S>(initial_step))
        .with_error_tolerance(
            scioxide_core::ErrorTolerance::scalar(
                threshold,
                threshold,
                scioxide_core::NormSelection::RootMeanSquare,
            )
            .unwrap(),
        )
}

/// Heat solve of `problem` to `end` with accepted-step output: the prepared plan
/// (to restore full fields) and the owned solution.
fn solve_parabolic<S: TestScalar, P: ParabolicPde<S> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<S>,
    boundaries: &Ends<S>,
    problem: &P,
    options: &ParabolicOptions<S>,
    end: f64,
) -> Result<(Plan1d<S>, PdeSolution<S>), SciOxideError> {
    let heat = support::Heat(problem.clone());
    let plan = discretization.plan(grid, &heat, boundaries)?;
    let source = source(&heat);
    let initial = initial(problem);
    let solver = options
        .clone()
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = with_plan!(&plan, |p| ParabolicProblem::new(
        p,
        &source,
        scalar::<S>(end)
    )?
    .solve_with_options(&initial, &solver))?;
    Ok((plan, solution))
}

/// Check every emitted sample and return the final full field in coordinate order.
fn final_field<S: TestScalar>(
    name: &str,
    grid: &SegmentMesh<S>,
    discretization: Discretization,
    (plan, solution): &(Plan1d<S>, PdeSolution<S>),
) -> Vec<S> {
    let fields = solution
        .trajectory()
        .states()
        .iter()
        .map(|state| plan.full(state))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let views = fields.iter().map(Vec::as_slice).collect::<Vec<_>>();
    assert_valid_solution(
        name,
        solution.report(),
        solution.trajectory().times(),
        &views,
        discretization.coordinates(grid).len(),
    );
    let last = plan.full(solution.values()).unwrap();
    assert_eq!(fields.last(), Some(&last));
    last
}

fn heat_error<S: TestScalar>(values: &[S], coordinates: &[S], final_time: f64) -> f64 {
    let decay = scalar::<S>((-std::f64::consts::PI.powi(2) * final_time).exp());
    max_abs_error(values, coordinates, |x| decay * (S::PI * x).sin())
}

fn parabolic_spatial_error<S: TestScalar>(
    discretization: Discretization,
    node_count: usize,
) -> f64 {
    let grid = unit_grid::<S>(node_count);
    let options = fixed_parabolic_options::<S>(0.0001, ParabolicScheme::CrankNicolson);
    let solution = solve_parabolic(
        discretization,
        &grid,
        &homogeneous_dirichlet(),
        &Heat,
        &options,
        0.01,
    )
    .unwrap();
    let values = final_field(discretization.name(), &grid, discretization, &solution);
    heat_error(&values, &discretization.coordinates(&grid), 0.01)
}

fn assert_parabolic_spatial_refinement<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        assert_refinement(
            &format!("{} parabolic spatial", discretization.name()),
            S::NAME,
            parabolic_spatial_error::<S>(discretization, 21),
            parabolic_spatial_error::<S>(discretization, 41),
            S::PARABOLIC_REFINEMENT_RATIO,
        );
    }
}

#[test]
fn every_parabolic_discretization_has_spatial_refinement_for_f64() {
    assert_parabolic_spatial_refinement::<f64>();
}

#[test]
fn every_parabolic_discretization_has_spatial_refinement_for_f32() {
    assert_parabolic_spatial_refinement::<f32>();
}

#[derive(Clone)]
struct GrowingExponential;

impl<S: TestScalar> ParabolicPde<S> for GrowingExponential {
    fn diffusion(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn source(&self, _x: &[S], time: S) -> Result<S, SciOxideError> {
        Ok(time.exp())
    }

    fn initial_condition(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
}

fn exponential_boundaries<S: TestScalar>() -> Ends<S> {
    let end = || End::dirichlet(moving(|_, time: S| Ok(time.exp())));
    Ends::new(end(), end())
}

fn exponential_error<S: TestScalar>(values: &[S], final_time: f64) -> f64 {
    let exact = final_time.exp();
    values
        .iter()
        .map(|&value| (as_f64(value) - exact).abs())
        .fold(0.0_f64, f64::max)
}

fn temporal_error<S: TestScalar>(
    discretization: Discretization,
    scheme: ParabolicScheme,
    step: f64,
) -> f64 {
    let grid = unit_grid::<S>(17);
    let options = fixed_parabolic_options::<S>(step, scheme);
    let solution = solve_parabolic(
        discretization,
        &grid,
        &exponential_boundaries(),
        &GrowingExponential,
        &options,
        1.0,
    )
    .unwrap();
    let values = final_field(discretization.name(), &grid, discretization, &solution);

    let (left_trace, right_trace, _) =
        endpoint_trace_and_slope(&values, &discretization.coordinates(&grid));
    let boundary_truncation = if matches!(discretization, Discretization::Fvm) {
        (1.0_f64 / 16.0).powi(2)
    } else {
        0.0
    };
    assert_scaled_residual::<S>(
        &format!("{} time-dependent left boundary", discretization.name()),
        (left_trace - std::f64::consts::E).abs(),
        std::f64::consts::E,
        boundary_truncation,
    );
    assert_scaled_residual::<S>(
        &format!("{} time-dependent right boundary", discretization.name()),
        (right_trace - std::f64::consts::E).abs(),
        std::f64::consts::E,
        boundary_truncation,
    );
    exponential_error(&values, 1.0)
}

fn assert_parabolic_temporal_refinement<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        for (scheme, minimum_ratio) in [
            (ParabolicScheme::BackwardEuler, S::BACKWARD_EULER_RATIO),
            (ParabolicScheme::CrankNicolson, S::CRANK_NICOLSON_RATIO),
        ] {
            assert_refinement(
                &format!("{} {scheme:?} temporal", discretization.name()),
                S::NAME,
                temporal_error::<S>(discretization, scheme, 0.2),
                temporal_error::<S>(discretization, scheme, 0.1),
                minimum_ratio,
            );
        }
    }
}

#[test]
fn implicit_schemes_have_temporal_refinement_and_boundary_residuals_for_f64() {
    assert_parabolic_temporal_refinement::<f64>();
}

#[test]
fn implicit_schemes_have_temporal_refinement_and_boundary_residuals_for_f32() {
    assert_parabolic_temporal_refinement::<f32>();
}

fn maximum_difference<S: TestScalar>(left: &[S], right: &[S]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| as_f64(left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_implicit_repeatability<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        for scheme in [
            ParabolicScheme::BackwardEuler,
            ParabolicScheme::CrankNicolson,
        ] {
            let grid = unit_grid::<S>(17);
            let boundaries = exponential_boundaries();
            let options = fixed_parabolic_options::<S>(0.05, scheme);
            let first = solve_parabolic(
                discretization,
                &grid,
                &boundaries,
                &GrowingExponential,
                &options,
                0.2,
            )
            .unwrap();
            let second = solve_parabolic(
                discretization,
                &grid,
                &boundaries,
                &GrowingExponential,
                &options,
                0.2,
            )
            .unwrap();
            let first_values = final_field(discretization.name(), &grid, discretization, &first);
            let second_values = final_field(discretization.name(), &grid, discretization, &second);
            let value_residual = maximum_difference(&first_values, &second_values);
            assert_scaled_residual::<S>(
                &format!("{} {scheme:?} repeated values", discretization.name()),
                value_residual,
                std::f64::consts::E,
                0.0,
            );
            assert_eq!(
                first.1.report().diagnostics(),
                second.1.report().diagnostics()
            );
        }
    }
}

#[test]
fn implicit_solve_objects_are_repeatable_for_f64() {
    assert_implicit_repeatability::<f64>();
}

#[test]
fn implicit_solve_objects_are_repeatable_for_f32() {
    assert_implicit_repeatability::<f32>();
}

/// `u_t = e^t` at free unknowns. FDM admits zero diffusion; FEM and FVM
/// require positive-definite diffusion, so they use a negligible `1e-20`.
#[derive(Clone)]
struct SourceOnly<S> {
    diffusion: S,
}

impl<S: TestScalar> SourceOnly<S> {
    fn for_discretization(discretization: Discretization) -> Self {
        Self {
            diffusion: if discretization == Discretization::Fdm {
                S::zero_value()
            } else {
                scalar(1e-20)
            },
        }
    }
}

impl<S: TestScalar> ParabolicPde<S> for SourceOnly<S> {
    fn diffusion(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(self.diffusion)
    }

    fn source(&self, _x: &[S], time: S) -> Result<S, SciOxideError> {
        Ok(time.exp())
    }

    fn initial_condition(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

fn source_only_error<S: TestScalar>(
    discretization: Discretization,
    values: &[S],
    final_time: f64,
) -> f64 {
    let exact = final_time.exp() - 1.0;
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let boundary_node = !matches!(discretization, Discretization::Fvm)
                && (index == 0 || index + 1 == values.len());
            (as_f64(value) - if boundary_node { 0.0 } else { exact }).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Final full field and report of an adaptive source-only solve.
fn adaptive_source_solve<S: TestScalar>(
    discretization: Discretization,
    tolerance: f64,
) -> (Vec<S>, PdeSolveReport<S>) {
    let grid = unit_grid::<S>(17);
    let options = adaptive_parabolic_options::<S>(1.0, tolerance);
    let solution = solve_parabolic(
        discretization,
        &grid,
        &homogeneous_dirichlet(),
        &SourceOnly::for_discretization(discretization),
        &options,
        1.0,
    )
    .unwrap();
    let values = final_field(discretization.name(), &grid, discretization, &solution);
    let report = solution.1.report().clone();
    let estimate = report.diagnostics().last_step_error_estimate().unwrap();
    assert!(
        estimate.is_finite() && estimate <= S::one_value(),
        "{}/{} normalized adaptive estimate was {estimate}",
        discretization.name(),
        S::NAME
    );
    (values, report)
}

fn assert_adaptive_tolerance_refinement<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        let loose = adaptive_source_solve::<S>(discretization, S::ADAPTIVE_LOOSE_TOLERANCE);
        let fine = adaptive_source_solve::<S>(discretization, S::ADAPTIVE_FINE_TOLERANCE);
        assert_refinement(
            &format!("{} adaptive tolerance", discretization.name()),
            S::NAME,
            source_only_error(discretization, &loose.0, 1.0),
            source_only_error(discretization, &fine.0, 1.0),
            1.05,
        );
        assert!(
            fine.1.diagnostics().num_time_steps() >= loose.1.diagnostics().num_time_steps(),
            "{}/{} stricter tolerance attempted fewer steps",
            discretization.name(),
            S::NAME
        );
    }
}

#[test]
fn adaptive_explicit_tolerance_refines_for_f64() {
    assert_adaptive_tolerance_refinement::<f64>();
}

#[test]
fn adaptive_explicit_tolerance_refines_for_f32() {
    assert_adaptive_tolerance_refinement::<f32>();
}

fn assert_explicit_entry_point<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        let grid = unit_grid::<S>(17);
        let final_time = scalar::<S>(0.25);
        let initial_step = scalar::<S>(0.1);
        let options = ParabolicOptions::new()
            .with_scheme(ParabolicScheme::ExplicitDopri5)
            .with_step(initial_step)
            .with_error_tolerance(scioxide_core::ErrorTolerance::from_tolerance(
                solve_tolerance::<S>(),
                scioxide_core::NormSelection::RootMeanSquare,
            ));
        let solution = solve_parabolic(
            discretization,
            &grid,
            &homogeneous_dirichlet(),
            &SourceOnly::for_discretization(discretization),
            &options,
            0.25,
        )
        .unwrap();
        final_field(discretization.name(), &grid, discretization, &solution);
        let (_, solution) = solution;
        let stats = solution.report().diagnostics();
        assert_eq!(solution.trajectory().times().last(), Some(&final_time));
        assert_eq!(solution.state().time(), final_time);
        assert_eq!(stats.method(), Some("DormandPrince5"));
        assert_eq!(stats.termination(), PdeTermination::Completed);
        assert_eq!(
            stats.num_time_steps(),
            stats.num_accepted_steps() + stats.num_rejected_steps()
        );
    }
}

#[test]
fn explicit_entry_point_is_options_based_for_f64() {
    assert_explicit_entry_point::<f64>();
}

#[test]
fn explicit_entry_point_is_options_based_for_f32() {
    assert_explicit_entry_point::<f32>();
}

#[derive(Clone)]
struct NegativeDiffusion;

impl<S: TestScalar> ParabolicPde<S> for NegativeDiffusion {
    fn diffusion(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(-S::one_value())
    }

    fn source(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn initial_condition(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

#[derive(Clone)]
struct LateNonFiniteSource;

impl<S: TestScalar> ParabolicPde<S> for LateNonFiniteSource {
    fn diffusion(&self, _x: &[S], _time: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }

    fn source(&self, _x: &[S], time: S) -> Result<S, SciOxideError> {
        Ok(if time == S::zero_value() {
            S::zero_value()
        } else {
            S::non_finite()
        })
    }

    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok((S::PI * x[0]).sin())
    }
}

fn late_non_finite_boundaries<S: TestScalar>() -> Ends<S> {
    let end = || {
        End::dirichlet(moving(|_, time: S| {
            Ok(if time == S::zero_value() {
                S::zero_value()
            } else {
                S::non_finite()
            })
        }))
    };
    Ends::new(end(), end())
}

fn assert_callback_non_finite(context: &str, error: SciOxideError) {
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ),
        "{context}: expected callback NonFinite, got {error}"
    );
}

fn assert_late_failure_transaction<
    S: TestScalar,
    P: ParabolicPde<S> + Send + Sync + Clone + 'static,
>(
    discretization: Discretization,
    problem: &P,
    boundaries: &Ends<S>,
    grid: &SegmentMesh<S>,
    label: &str,
) {
    let options = fixed_parabolic_options::<S>(0.01, ParabolicScheme::BackwardEuler);
    let error = solve_parabolic(discretization, grid, boundaries, problem, &options, 0.02)
        .map(|_| ())
        .unwrap_err();
    assert_callback_non_finite(
        &format!("{} {label} explicit lifecycle", discretization.name()),
        error,
    );
}

fn assert_typed_failures_and_transactionality<S: TestScalar>() {
    for discretization in DISCRETIZATIONS {
        let grid = unit_grid::<S>(17);
        let boundaries = homogeneous_dirichlet();
        let options = fixed_parabolic_options::<S>(0.01, ParabolicScheme::BackwardEuler);
        let one_shot_error = solve_parabolic(
            discretization,
            &grid,
            &boundaries,
            &NegativeDiffusion,
            &options,
            0.02,
        )
        .map(|_| ())
        .unwrap_err();
        // Every family reports inadmissible sampled diffusion as a domain error.
        assert!(
            matches!(
                one_shot_error.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ),
            "{} negative diffusion/{}: {one_shot_error}",
            discretization.name(),
            S::NAME
        );
        assert_late_failure_transaction(
            discretization,
            &LateNonFiniteSource,
            &boundaries,
            &grid,
            "source",
        );
        assert_late_failure_transaction(
            discretization,
            &Heat,
            &late_non_finite_boundaries(),
            &grid,
            "boundary",
        );
    }
}

#[test]
fn parabolic_paths_share_typed_failures_and_transactionality_for_f64() {
    assert_typed_failures_and_transactionality::<f64>();
}

#[test]
fn parabolic_paths_share_typed_failures_and_transactionality_for_f32() {
    assert_typed_failures_and_transactionality::<f32>();
}
