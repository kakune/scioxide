//! An enum of every built-in method, for choosing the method at run time.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

use crate::result::StepResult;
use crate::stepper::{MethodInfo, StepRequest, Stepper};
use crate::traits::OdeSystem;
use crate::{
    AdamsBashforth, AdamsBdf, AdamsMoulton, BackwardEuler, Bdf, ClassicalRk4, DormandPrince5,
    DormandPrince853, ExplicitMidpoint, ForwardEuler, GaussRk, Heun, ImplicitTrapezoid, RadauIia,
    Rkf45, RosenbrockW, Sdirk, Verner65,
};

fn mismatch() -> SciOxideError {
    crate::safety::invalid(
        scioxide_core::argument!("workspace"),
        "workspace was prepared for another built-in method",
    )
}

macro_rules! builtin_methods {
    ($($(#[$doc:meta])* $variant:ident($method:ident)),* $(,)?) => {
        /// Selects any built-in initial-value method at run time.
        ///
        /// Each method type converts into this enum with [`From`]. All plans over
        /// this enum use one workspace type, so one [`OdeWorkspace`](crate::OdeWorkspace)
        /// can serve several methods; it is reallocated when the method changes.
        #[derive(Debug, Clone)]
        #[non_exhaustive]
        pub enum BuiltinOdeMethod<S, T = S>
        where
            S: RealScalar + NumericScalar<Real = S>,
            T: NumericScalar<Real = S>,
        {
            $($(#[$doc])* $variant($method<S, T>),)*
        }

        /// Storage of the selected method.
        #[derive(Default)]
        pub(crate) enum BuiltinWorkspace<S, T>
        where
            S: RealScalar + NumericScalar<Real = S>,
            T: NumericScalar<Real = S>,
        {
            #[default]
            Empty,
            $($variant(<$method<S, T> as Stepper<S, T>>::Workspace),)*
        }

        $(
            impl<S, T> From<$method<S, T>> for BuiltinOdeMethod<S, T>
            where
                S: RealScalar + NumericScalar<Real = S>,
                T: NumericScalar<Real = S>,
            {
                fn from(method: $method<S, T>) -> Self {
                    Self::$variant(method)
                }
            }
        )*

        impl<S, T> Stepper<S, T> for BuiltinOdeMethod<S, T>
        where
            S: RealScalar + NumericScalar<Real = S>,
            T: NumericScalar<Real = S>,
        {
            type Workspace = BuiltinWorkspace<S, T>;

            fn stage_kernel<'w>(&self, workspace: &'w Self::Workspace) -> Option<&'w crate::stiff::kernel::StageKernel<S, T>> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => method.stage_kernel(workspace),)*
                    _ => None,
                }
            }

            fn info(&self) -> MethodInfo {
                match self {
                    $(Self::$variant(method) => method.info(),)*
                }
            }

            fn stability(&self) -> crate::OdeLinearStability<S> {
                match self {
                    $(Self::$variant(method) => method.stability(),)*
                }
            }

            fn accepts_initial_rhs(&self) -> bool {
                match self {
                    $(Self::$variant(method) => method.accepts_initial_rhs(),)*
                }
            }

            fn fits(&self, dimension: usize, provided: bool, workspace: &Self::Workspace) -> bool {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.fits(dimension, provided, workspace)
                    })*
                    _ => false,
                }
            }

            fn reserve(
                &self,
                dimension: usize,
                provided: bool,
                workspace: &mut Self::Workspace,
            ) -> Result<(), SciOxideError> {
                match self {
                    $(Self::$variant(method) => {
                        if !matches!(workspace, BuiltinWorkspace::$variant(_)) {
                            *workspace = BuiltinWorkspace::$variant(Default::default());
                        }
                        let BuiltinWorkspace::$variant(inner) = workspace else {
                            return Err(mismatch());
                        };
                        method.reserve(dimension, provided, inner)
                    })*
                }
            }

            fn begin(
                &self,
                dimension: usize,
                provided: bool,
                workspace: &mut Self::Workspace,
            ) -> Result<(), SciOxideError> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.begin(dimension, provided, workspace)
                    })*
                    _ => Err(mismatch()),
                }
            }

            fn step<'w, System: OdeSystem<S, T> + ?Sized>(
                &self,
                system: &System,
                t: S,
                y: &mut Vector<T>,
                h: S,
                request: StepRequest<'_, S, T>,
                workspace: &'w mut Self::Workspace,
                cx: &mut SolveContext<'_, '_>,
            ) -> Result<StepResult<'w, S, T>, SciOxideError> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.step(system, t, y, h, request, workspace, cx)
                    })*
                    _ => Err(mismatch()),
                }
            }

            fn complete_dense<'w, System: OdeSystem<S, T> + ?Sized>(
                &self,
                system: &System,
                t: S,
                h: S,
                request: StepRequest<'_, S, T>,
                workspace: &'w mut Self::Workspace,
                cx: &mut SolveContext<'_, '_>,
            ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.complete_dense(system, t, h, request, workspace, cx)
                    })*
                    _ => Err(mismatch()),
                }
            }

            fn accept(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.accept(workspace)
                    })*
                    _ => Err(mismatch()),
                }
            }

            fn discard(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
                match (self, workspace) {
                    $((Self::$variant(method), BuiltinWorkspace::$variant(workspace)) => {
                        method.discard(workspace)
                    })*
                    _ => Err(mismatch()),
                }
            }
        }
    };
}

builtin_methods! {
    /// [`ForwardEuler`](crate::ForwardEuler).
    ForwardEuler(ForwardEuler),
    /// [`ExplicitMidpoint`](crate::ExplicitMidpoint).
    ExplicitMidpoint(ExplicitMidpoint),
    /// [`Heun`](crate::Heun).
    Heun(Heun),
    /// [`ClassicalRk4`](crate::ClassicalRk4).
    ClassicalRk4(ClassicalRk4),
    /// [`Rkf45`](crate::Rkf45).
    Rkf45(Rkf45),
    /// [`DormandPrince5`](crate::DormandPrince5).
    DormandPrince5(DormandPrince5),
    /// [`DormandPrince853`](crate::DormandPrince853).
    DormandPrince853(DormandPrince853),
    /// [`Verner65`](crate::Verner65).
    Verner65(Verner65),
    /// [`BackwardEuler`](crate::BackwardEuler).
    BackwardEuler(BackwardEuler),
    /// [`ImplicitTrapezoid`](crate::ImplicitTrapezoid).
    ImplicitTrapezoid(ImplicitTrapezoid),
    /// [`GaussRk`](crate::GaussRk).
    GaussRk(GaussRk),
    /// [`Sdirk`](crate::Sdirk).
    Sdirk(Sdirk),
    /// [`RadauIia`](crate::RadauIia).
    RadauIia(RadauIia),
    /// [`RosenbrockW`](crate::RosenbrockW).
    RosenbrockW(RosenbrockW),
    /// [`AdamsBashforth`](crate::AdamsBashforth).
    AdamsBashforth(AdamsBashforth),
    /// [`AdamsMoulton`](crate::AdamsMoulton).
    AdamsMoulton(AdamsMoulton),
    /// [`Bdf`](crate::Bdf).
    Bdf(Bdf),
    /// [`AdamsBdf`](crate::AdamsBdf).
    AdamsBdf(AdamsBdf),
}

impl<S, T> BuiltinOdeMethod<S, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    /// Returns the name of the selected method, such as `"DormandPrince5"`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        Stepper::<S, T>::info(self).name
    }
}
