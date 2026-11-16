//! Native complex methods are checked against analytic and independently packed real systems.
#![allow(clippy::unwrap_used)]
use core::marker::PhantomData;
use scioxide_core::*;
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;

fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
/// Steps of the 50-step grid alternating `0.75/50` and `1.25/50`, snapped back to the grid.
fn alternating_limit<S: RealScalar + NumericScalar<Real = S>>(t: S) -> S {
    // Grid points are at least 0.75 units apart, so rounding of `t` stays well inside.
    let u = t.try_to_f64_exact().unwrap() * 50.0;
    let offset = u - 2.0 * ((u + 0.01) / 2.0).floor();
    let remaining = if offset < 0.375 {
        0.75 - offset
    } else {
        2.0 - offset
    };
    s(remaining / 50.0)
}
/// `alternating` imposes the alternating grid through the system step limit.
struct Linear<S> {
    alternating: bool,
    scalar: PhantomData<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S, Complex<S>> for Linear<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn dimension(&self) -> usize {
        1
    }
    fn step_limit(
        &self,
        t: S,
        _: &[Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok(self.alternating.then(|| alternating_limit(t)))
    }
    fn rhs(
        &self,
        _: S,
        y: &[Complex<S>],
        out: &mut [Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = Complex::new(s::<S>(-0.25), s::<S>(1.)) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[Complex<S>],
        out: &mut MatrixViewMut<'_, Complex<S>>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, Complex::new(s(-0.25), s(1.)))?;
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: S,
        _: &[Complex<S>],
        _: StructuralPatternView<'_>,
        out: &mut [Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = Complex::new(s(-0.25), s(1.));
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        t: S,
        _: &[Complex<S>],
        x: &[Complex<S>],
        out: &mut [Complex<S>],
        _: &mut [Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, x, out, _cx).map(Some)
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
struct RealBlock<S> {
    alternating: bool,
    scalar: PhantomData<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for RealBlock<S> {
    fn dimension(&self) -> usize {
        2
    }
    fn step_limit(
        &self,
        t: S,
        _: &[S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok(self.alternating.then(|| alternating_limit(t)))
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = s::<S>(-0.25) * y[0] - y[1];
        out[1] = y[0] + s::<S>(-0.25) * y[1];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        out: &mut MatrixViewMut<'_, S>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, s(-0.25))?;
        out.set(0, 1, s(-1.))?;
        out.set(1, 0, s(1.))?;
        out.set(1, 1, s(-0.25))?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
/// Integrate over `[0, 1]` on the uniform 50-step grid or the alternating grid.
///
/// Fixed-step methods take `1/50` (`0.75/50` alternating) steps; their systems impose no
/// step limit, which would permanently shorten the re-proposed fixed step. Adaptive
/// methods follow the grid through the maximum step or the system step limit; `loose`
/// methods use a unit tolerance so grid steps are accepted.
fn advance<S, T, M, P>(
    plan_method: &M,
    problem: &P,
    state: &mut OdeState<S, T>,
    w: &mut OdeWorkspace<S, M, T>,
    alternating: bool,
    loose: bool,
) -> OdeSolveReport<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T> + Clone,
    P: OdeSystem<S, T>,
{
    let tol = if loose {
        ErrorTolerance::scalar(s::<S>(1.), s::<S>(1.), NormSelection::Infinity).unwrap()
    } else {
        ErrorTolerance::scalar(s::<S>(2e-6), s::<S>(2e-6), NormSelection::Infinity).unwrap()
    };
    let h = s::<S>(if alternating { 0.75 } else { 1. }) / s::<S>(50.);
    let options = OdeOptions::new()
        .with_error_tolerance(tol)
        .with_initial_step(h);
    let options = if alternating {
        options
    } else {
        options.with_maximum_step(h)
    };
    let plan = (plan_method.clone(), options);
    plan.0
        .advance(
            &OdeProblem::new(problem, s::<S>(1.)),
            &plan.1,
            state,
            &mut (),
            w,
            ExecutionContext::sequential(),
        )
        .unwrap()
}
fn compare<S, M, R>(complex: M, real: R, bound: f64)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    M: OdeMethod<S, Complex<S>> + Clone + Into<BuiltinOdeMethod<S, Complex<S>>>,
    R: OdeMethod<S> + Clone,
{
    let name = complex.clone().into().name();
    // Only adaptive methods accept an explicit controller; the others (including
    // fixed-order multistep methods) always take the configured step.
    let fixed = complex
        .validate(
            &(OdeOptions::new().with_step_control(OdeStepControl::ProportionalIntegral(
                ProportionalIntegralController::new(
                    S::try_from_f64_lossy(0.9).unwrap(),
                    S::try_from_f64_lossy(5.0).unwrap(),
                    S::try_from_f64_lossy(0.2).unwrap(),
                    S::ONE,
                    S::ZERO,
                )
                .unwrap(),
            ))),
            1,
        )
        .is_err();
    // Automatic switching would reject grid steps at the stage tolerance, and restarts
    // its startup sequence after a switch.
    let loose = name == "AutoStiff";
    let follows_grid = !loose;
    let check = |report: OdeSolveReport<S>| {
        assert!(report.is_success(), "{name}: {report:?}");
        if follows_grid {
            assert_eq!(report.rejected_steps(), 0, "{name}: {report:?}");
        }
    };
    let initial = Complex::new(s::<S>(1.), s::<S>(0.4));
    let mut cw = OdeWorkspace::new();
    let mut rw = OdeWorkspace::new();
    for alternating in [false, true] {
        let limited = alternating && !fixed;
        let linear = Linear {
            alternating: limited,
            scalar: PhantomData,
        };
        let block = RealBlock {
            alternating: limited,
            scalar: PhantomData,
        };
        let mut y = OdeState::new(s::<S>(0.), Vector::from_vec(vec![initial]));
        let mut r = OdeState::new(s::<S>(0.), Vector::from_vec(vec![initial.re, initial.im]));
        check(advance(
            &complex,
            &linear,
            &mut y,
            &mut cw,
            alternating,
            loose,
        ));
        check(advance(&real, &block, &mut r, &mut rw, alternating, loose));
        let exact = initial
            * Complex::new(s::<S>(1.).cos(), s::<S>(1.).sin())
            * Complex::new(s::<S>(-0.25).exp(), s(0.));
        assert!(
            (y.values()[0] - exact).abs() < s::<S>(bound),
            "{name} analytic error {:?}",
            (y.values()[0] - exact).abs()
        );
        let difference = (y.values()[0] - Complex::new(r.values()[0], r.values()[1])).abs();
        assert!(
            difference < s::<S>(5e-5),
            "{name} real block difference {difference:?}"
        );
        let mut fresh = OdeWorkspace::new();
        let mut z = OdeState::new(s::<S>(0.), Vector::from_vec(vec![initial]));
        check(advance(
            &complex,
            &linear,
            &mut z,
            &mut fresh,
            alternating,
            loose,
        ));
        assert_eq!(y, z, "reused workspace must reset its epoch");
    }
}
fn methods<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    macro_rules! pair {
        ($method:ident,$bound:expr) => {
            compare::<S, _, _>($method::new(), $method::new(), $bound);
        };
    }
    pair!(ForwardEuler, 0.02);
    pair!(ExplicitMidpoint, 0.0003);
    pair!(Heun, 0.0003);
    pair!(ClassicalRk4, 0.00001);
    pair!(Rkf45, 0.00001);
    pair!(DormandPrince5, 0.00001);
    pair!(DormandPrince853, 0.00001);
    pair!(Verner65, 0.00001);
    pair!(BackwardEuler, 0.02);
    pair!(ImplicitTrapezoid, 0.0003);
    pair!(GaussRk, 0.00001);
    pair!(RadauIia, 0.00001);
    pair!(Sdirk, 0.00002);
    for order in 1..=5 {
        compare::<S, _, _>(
            AdamsBashforth::new(order).unwrap(),
            AdamsBashforth::new(order).unwrap(),
            0.025,
        );
        compare::<S, _, _>(
            AdamsMoulton::new(order).unwrap(),
            AdamsMoulton::new(order).unwrap(),
            0.025,
        );
        for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
            compare::<S, _, _>(
                Bdf::new(order).unwrap().with_formula(formula),
                Bdf::new(order).unwrap().with_formula(formula),
                0.025,
            );
        }
    }
    compare::<S, _, _>(
        AdamsBdf::adams_bdf().unwrap(),
        AdamsBdf::adams_bdf().unwrap(),
        0.025,
    );
}
#[test]
fn all_methods_f32() {
    methods::<f32>();
}
#[test]
fn all_methods_f64() {
    methods::<f64>();
}
fn backends<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let sparse = OdeStageBackend::Sparse {
        pattern: StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
        limits: SparseLuLimits::new(16, 4096),
    };
    let krylov = OdeStageBackend::Krylov {
        iterations: Budget::try_from(30).unwrap(),
        restart: Budget::try_from(4).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    for backend in [sparse, krylov.clone()] {
        compare::<S, _, _>(
            BackwardEuler::new().with_backend(backend.clone()),
            BackwardEuler::new(),
            0.02,
        );
        compare::<S, _, _>(
            ImplicitTrapezoid::new().with_backend(backend.clone()),
            ImplicitTrapezoid::new(),
            0.0003,
        );
        compare::<S, _, _>(
            Bdf::new(3).unwrap().with_backend(backend),
            Bdf::new(3).unwrap(),
            0.001,
        );
    }
    // Coupled stages factor the blocked sparse pattern; SDIRK solves stage by stage.
    let sparse = OdeStageBackend::Sparse {
        pattern: StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
        limits: SparseLuLimits::new(16, 4096),
    };
    for backend in [sparse, krylov] {
        compare::<S, _, _>(
            GaussRk::new().with_backend(backend.clone()),
            GaussRk::new(),
            0.00002,
        );
        compare::<S, _, _>(
            RadauIia::new().with_backend(backend.clone()),
            RadauIia::new(),
            0.00002,
        );
        compare::<S, _, _>(Sdirk::new().with_backend(backend), Sdirk::new(), 0.00003);
    }
}
#[test]
fn native_backends_f32() {
    backends::<f32>();
}
#[test]
fn native_backends_f64() {
    backends::<f64>();
}
