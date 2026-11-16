//! Independent polynomial and exponential tests for each advertised extension.
//!
//! Dense output is observed through requested output times of single accepted steps.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    ErrorTolerance, ExecutionContext, NormSelection, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    Bdf, BdfFormula, DormandPrince5, DormandPrince853, OdeMethod, OdeOptions, OdeOutputSchedule,
    OdeProblem, OdeState, OdeSystem, OdeTrajectory, OdeWorkspace, RadauIia, Rkf45, Verner65,
};

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
struct Polynomial(i32);
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Polynomial {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = scalar::<S>(f64::from(self.0)) * t.powi(self.0 - 1);
        Ok(())
    }
}

/// Requested-time options of one accepted step of magnitude `|h|` under a loose tolerance.
fn single_step<S: RealScalar + NumericScalar<Real = S>>(h: S, times: Vec<S>) -> OdeOptions<S> {
    OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(scalar(1.0), scalar(1.0), NormSelection::RootMeanSquare)
                .unwrap(),
        )
        .with_initial_step(NumericScalar::abs(h))
        .with_output_schedule(OdeOutputSchedule::times(times))
}

/// Dense samples of one accepted step `[0, h]` from `y(0) = initial`, and the endpoint.
fn dense_step<S, M, Sys>(
    method: &M,
    system: &Sys,
    initial: S,
    h: S,
    times: Vec<S>,
    workspace: &mut OdeWorkspace<S, M>,
) -> (Vec<S>, S)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
    Sys: OdeSystem<S>,
{
    let plan = (method.clone(), single_step(h, times));
    let mut state = OdeState::new(scalar(0.0), Vector::from_vec(vec![initial]));
    let mut samples = Vec::new();
    let report = plan
        .0
        .advance(
            &OdeProblem::new(system, h),
            &plan.1,
            &mut state,
            &mut |_: S, y: &[S]| {
                samples.push(y[0]);
                Ok(())
            },
            workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(report.rejected_steps(), 0);
    (samples, state.values()[0])
}

/// `order` is the advertised exact-start continuous-extension order.
fn refinement<S, M>(method: M, order: i32)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    let mut workspace = OdeWorkspace::new();
    let degree = order + 1;
    let thetas = [0.2, 0.4, 0.7, 0.9];
    // A refinement ratio for t^(p+1) alone can hide a lower-order formula:
    // all stages scale homogeneously with h. Reproduction through degree p
    // independently rejects a cubic fallback advertised as quartic/septic.
    for polynomial_degree in 1..degree {
        for h in [-1.0, 1.0] {
            let h = scalar::<S>(h);
            let times: Vec<S> = thetas.iter().map(|&theta| h * scalar::<S>(theta)).collect();
            let (samples, _) = dense_step(
                &method,
                &Polynomial(polynomial_degree),
                scalar(0.0),
                h,
                times.clone(),
                &mut workspace,
            );
            for (t, value) in times.into_iter().zip(samples) {
                let tolerance = scalar::<S>(512.0) * S::EPSILON;
                assert!(
                    NumericScalar::abs(value - t.powi(polynomial_degree)) < tolerance,
                    "order {order} failed polynomial reproduction of degree {polynomial_degree}"
                );
            }
        }
    }
    for direction in [-1.0, 1.0] {
        let mut errors = Vec::new();
        for h in [1.0, 0.5, 0.25] {
            let h = scalar::<S>(direction * h);
            let zero = scalar::<S>(0.0);
            let interior: Vec<S> = thetas.iter().map(|&theta| h * scalar::<S>(theta)).collect();
            let times = [zero].into_iter().chain(interior.clone()).chain([h]);
            let (samples, endpoint) = dense_step(
                &method,
                &Polynomial(degree),
                zero,
                h,
                times.collect(),
                &mut workspace,
            );
            assert_eq!(samples.len(), 6);
            assert_eq!(samples[0], zero);
            assert_eq!(samples[5], endpoint);
            let value_error = interior
                .iter()
                .zip(&samples[1..5])
                .map(|(&t, &value)| {
                    NumericScalar::abs(value - t.powi(degree))
                        .try_to_f64_exact()
                        .unwrap()
                })
                .fold(0.0_f64, f64::max);
            errors.push(value_error);
        }
        for pair in errors.windows(2) {
            let value_ratio = pair[0] / pair[1];
            let expected = 2.0_f64.powi(degree);
            assert!(
                (0.95 * expected..1.05 * expected).contains(&value_ratio),
                "order {order}: {errors:?}"
            );
        }
    }
}

struct Exponential;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Exponential {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        state: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(state);
        Ok(())
    }
}
fn prepared<S, M>(method: M)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    const TIMES: [f64; 5] = [0.0, 0.025, 0.05, 0.075, 0.1];
    let h = scalar::<S>(0.1);
    let plan = (
        method,
        single_step(h, TIMES.iter().map(|&time| scalar(time)).collect()),
    );
    let problem = OdeProblem::new(&Exponential, h);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(scalar(0.0), Vector::from_vec(vec![scalar::<S>(1.0)]));
    let mut samples = 0;
    let mut sink = |time: S, y: &[S]| {
        let exact = scalar::<S>(time.try_to_f64_exact().unwrap().exp());
        assert!(NumericScalar::abs(y[0] - exact) < scalar(2e-6));
        samples += 1;
        Ok(())
    };
    // Warm method-owned tableau preparation, then verify the reusable path.
    let warm = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(warm.is_success());
    {
        state.reset(scalar(0.0), &[scalar(1.0)]).unwrap();
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.accepted_steps(), 1);
    };
    assert_eq!(samples, 2 * TIMES.len());
}

macro_rules! qualify {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            refinement::<$scalar, _>(DormandPrince5::<$scalar>::new(), 4);
            refinement::<$scalar, _>(DormandPrince853::<$scalar>::new(), 7);
            refinement::<$scalar, _>(Rkf45::<$scalar>::new(), 3);
            refinement::<$scalar, _>(Verner65::<$scalar>::new(), 3);
            refinement::<$scalar, _>(RadauIia::<$scalar>::new(), 3);
            prepared::<$scalar, _>(DormandPrince5::<$scalar>::new());
            prepared::<$scalar, _>(DormandPrince853::<$scalar>::new());
            prepared::<$scalar, _>(Rkf45::<$scalar>::new());
            prepared::<$scalar, _>(Verner65::<$scalar>::new());
            prepared::<$scalar, _>(RadauIia::<$scalar>::new());
        }
    };
}
qualify!(extensions_f32, f32);
qualify!(extensions_f64, f64);

/// `y₁'' + y₁ = cos 2t` as a first-order system with a constant Jacobian.
struct Forced;
fn forced(t: f64) -> [f64; 2] {
    [
        4. / 3. * t.cos() - (2. * t).cos() / 3.,
        -4. / 3. * t.sin() + 2. / 3. * (2. * t).sin(),
    ]
}
impl OdeSystem<f64> for Forced {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        f: &mut [f64],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = y[1];
        f[1] = -y[0] + (2. * t).cos();
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (row, col, value) in [(0, 0, 0.), (0, 1, 1.), (1, 0, -1.), (1, 1, 0.)] {
            j.set(row, col, value)?;
        }
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn fixed_order_bdf_startup_dense_output_has_the_step_order() {
    for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
        for q in [4_usize, 5] {
            // Requests inside the q - 1 extrapolated startup steps and the first
            // full-order step, including step 0.
            let error = |n: u32| {
                let h = 2. / f64::from(n);
                let times: Vec<f64> = (0..q).map(|i| (i as f64 + 0.37) * h).collect();
                let options = OdeOptions::new()
                    .with_error_tolerance(
                        ErrorTolerance::scalar(1e-13, 1e-13, NormSelection::Infinity).unwrap(),
                    )
                    .with_initial_step(h)
                    .with_output_schedule(OdeOutputSchedule::times(times));
                let method = Bdf::new(q).unwrap().with_formula(formula);
                let plan = (method, options);
                let problem = OdeProblem::new(&Forced, q as f64 * h);
                let mut output = OdeTrajectory::new();
                let mut state = OdeState::from_slice(0., &forced(0.)).unwrap();
                let report = plan
                    .0
                    .advance(
                        &problem,
                        &plan.1,
                        &mut state,
                        &mut output,
                        &mut OdeWorkspace::new(),
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(report.is_success());
                assert_eq!(output.times().len(), q);
                output
                    .times()
                    .iter()
                    .zip(output.states())
                    .flat_map(|(&t, y)| {
                        let exact = forced(t);
                        [0, 1].map(|i| (y.as_slice()[i] - exact[i]).abs())
                    })
                    .fold(0., f64::max)
            };
            let (coarse, fine) = (error(40), error(80));
            let order = (coarse / fine).log2();
            assert!(
                order >= q as f64 - 0.3,
                "{formula:?}({q}): {coarse} {fine} {order}"
            );
        }
    }
    // A prepared startup segment allocates nothing.
    prepared::<f64, _>(Bdf::<f64>::new(5).unwrap());
}

struct Constant<S>(S);
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Constant<S> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = self.0;
        Ok(())
    }
}
fn extreme<S, M>(method: M, magnitude: S, h: S)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    let mut workspace = OdeWorkspace::new();
    for direction in [-1.0, 1.0] {
        let h = h * scalar::<S>(direction);
        let (samples, _) = dense_step(
            &method,
            &Constant(magnitude),
            scalar(0.0),
            h,
            vec![h * scalar(0.5)],
            &mut workspace,
        );
        assert!(NumericScalar::abs(samples[0] / (magnitude * h) - scalar::<S>(0.5)) < scalar(2e-4));
    }
}
#[test]
fn extreme_stage_products() {
    macro_rules! run {
        ($s:ty, $large:expr, $h:expr) => {
            extreme::<$s, _>(DormandPrince5::<$s>::new(), $large, $h);
            extreme::<$s, _>(DormandPrince853::<$s>::new(), $large, $h);
            extreme::<$s, _>(Rkf45::<$s>::new(), $large, $h);
            extreme::<$s, _>(Verner65::<$s>::new(), $large, $h);
            extreme::<$s, _>(RadauIia::<$s>::new(), $large, $h);
        };
    }
    run!(f64, f64::MAX / 4.0, 1e-300);
    run!(f32, f32::MAX / 4.0, 1e-30);
}
