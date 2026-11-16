//! Complex Newton, Halley and secant iterations.
use super::derivative_methods::{Derivatives, NewtonStep};
use super::open::{self, Step};
use super::{Run, Sample, correction};
use crate::{ComplexRootResult, ComplexRootSolver, ComplexScalarFunction, ScalarRootOptions};
use scioxide_core::{Complex, EvaluationRole, RealScalar, SciOxideError};

macro_rules! solver {
    ($name:ident, $init:ty, $array:expr, $method:expr, $doc:literal) => {
        #[doc=$doc]
        ///
        /// The solve stops as described in [`ScalarRootOptions`], with
        /// residuals measured by the modulus `|f(z)|` and steps by `|Δz|`. It
        /// ends with `Stagnation` when the step is undefined or repeats a point.
        /// After `Stagnation` or `IterationLimit`, the result is the evaluated
        /// point with the smallest `|f|`.
        ///
        /// # Errors
        ///
        /// The errors listed on [`ComplexRootSolver::solve`].
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;
        impl<S: RealScalar> ComplexRootSolver<S> for $name {
            type Initialization = $init;
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
                scioxide_core::algorithm_id!("rootfind", stringify!($name), 1)
            }
            fn solve_scoped<'scope, 'capability, F: ComplexScalarFunction<S> + ?Sized>(
                &self,
                f: &F,
                init: $init,
                options: &ScalarRootOptions<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<ComplexRootResult<S>, SciOxideError> {
                crate::context::run(
                    scope.into(),
                    <Self as ComplexRootSolver<S>>::algorithm_id(self),
                    options.budgets(),
                    |context| {
                        open::solve(
                            context,
                            options,
                            ($array)(init),
                            |x, context| f.eval_scoped(x, context),
                            ($method)(f),
                        )
                    },
                )
            }
        }
    };
}
solver!(
    ComplexNewton,
    Complex<S>,
    |x| [x],
    |f| NewtonStep(ComplexDerivatives::<_, false>(f)),
    "Finds a zero of a complex function by Newton's method, \
     `zₖ₊₁ = zₖ − f(zₖ) / f′(zₖ)`, from one starting point.\n\n\
     It requires the holomorphic derivative [`ComplexScalarFunction::derivative`]. \
     Convergence to a simple zero is quadratic when the starting point is close enough. \
     If `f` maps real numbers to real numbers, iterates from a real starting point \
     stay real, so a nonreal zero needs a nonreal starting point."
);
solver!(
    ComplexHalley,
    Complex<S>,
    |x| [x],
    |f| NewtonStep(ComplexDerivatives::<_, true>(f)),
    "Finds a zero of a complex function by Halley's method from one starting point.\n\n\
     The step is that of [`Halley`](crate::Halley), in complex arithmetic. It requires \
     the holomorphic derivatives [`ComplexScalarFunction::derivative`] and \
     [`ComplexScalarFunction::second_derivative`]. Convergence to a simple zero is \
     cubic when the starting point is close enough."
);
solver!(
    ComplexSecant,
    (Complex<S>, Complex<S>),
    |x: (Complex<S>, Complex<S>)| [x.0, x.1],
    |_| open::proposal(|_, [previous, current]: &[Sample<Complex<S>>; 2]| {
        correction::secant(*previous, *current).map(Step::proposed)
    }),
    "Finds a zero of a complex function by the secant method from two starting points.\n\n\
     The step is that of [`Secant`](crate::Secant), in complex arithmetic, and needs no \
     derivative. Convergence to a simple zero has order `(1 + √5)/2 ≈ 1.618` when the \
     starting points are close enough."
);

/// Analytic holomorphic derivatives, each counted and checked.
struct ComplexDerivatives<'f, F: ?Sized, const HALLEY: bool>(&'f F);

impl<S, F, const HALLEY: bool> Derivatives<Complex<S>> for ComplexDerivatives<'_, F, HALLEY>
where
    S: RealScalar,
    F: ComplexScalarFunction<S> + ?Sized,
{
    #[inline(always)]
    fn at(
        &self,
        run: &mut Run<'_, '_, '_, Complex<S>>,
        current: Sample<Complex<S>>,
    ) -> Result<(Complex<S>, Option<Complex<S>>), SciOxideError> {
        let first = run.derivative(EvaluationRole::Jacobian, || self.0.derivative(current.x))?;
        let second = if HALLEY {
            Some(run.derivative(EvaluationRole::Hessian, || {
                self.0.second_derivative(current.x)
            })?)
        } else {
            None
        };
        Ok((first, second))
    }
}
