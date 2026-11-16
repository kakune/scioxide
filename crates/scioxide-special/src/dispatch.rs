//! Scalar conversion and the generator of plain, detailed and scoped entry points.
use scioxide_core::{Complex, NumericScalar, RealScalar};

mod private {
    use scioxide_core::{Complex, NumericScalar, RealScalar};

    /// Converts a parameter to the scalar type of the argument it accompanies.
    pub trait Promote<T> {
        fn promote(self) -> T;
    }
    impl<T: NumericScalar> Promote<T> for T {
        #[inline]
        fn promote(self) -> T {
            self
        }
    }
    impl<R: RealScalar> Promote<Complex<R>> for R
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        #[inline]
        fn promote(self) -> Complex<R> {
            Complex::new(self, R::ZERO)
        }
    }
}

/// Marks a value accepted as a parameter beside an argument of scalar type `T`.
///
/// A parameter is an order, shape or modulus whose role differs from the argument's, such as the
/// order `ν` of [`bessel_j`](crate::bessel_j) or the parameters `a`, `b`, `c` of
/// [`hyp2f1`](crate::hyp2f1). The trait is implemented for `T` and, when `T` is `Complex<R>`,
/// for `R`: a real parameter is the complex number with imaginary part `+0`, so the call returns
/// the same bits as the call with that complex value. The argument's type sets `T` and the type
/// of the result. The trait is sealed and has no public methods.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
///
/// // J_2.5(2 + i) with a real order
/// let z = Complex::new(2.0_f64, 1.0);
/// let j = scioxide_special::bessel_j(2.5, z)?;
/// assert_eq!(j, scioxide_special::bessel_j(Complex::new(2.5, 0.0), z)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// A complex parameter needs a complex argument:
///
/// ```compile_fail
/// let _ = scioxide_special::bessel_j(scioxide_core::Complex::new(0.5_f64, 0.25), 1.0_f64);
/// ```
pub trait Parameter<T: NumericScalar>: Copy + private::Promote<T> {}
impl<T: NumericScalar> Parameter<T> for T {}
impl<R: RealScalar> Parameter<Complex<R>> for R where Complex<R>: NumericScalar<Real = R> {}

/// Converts a parameter to the argument's scalar type.
#[inline]
pub(crate) fn promote<T: NumericScalar, P: Parameter<T>>(parameter: P) -> T {
    private::Promote::promote(parameter)
}

#[inline]
pub(crate) fn complex<T: NumericScalar>(value: T) -> Complex<T::Real> {
    Complex::new(value.real_component(), value.imaginary_component())
}

#[inline]
pub(crate) fn convert_scalar<U, T>(value: U) -> T
where
    U: NumericScalar<Real = T::Real>,
    T: NumericScalar,
{
    T::from_components(value.real_component(), value.imaginary_component())
}

/// Generates one operation's plain, detailed and optional scoped forms.
///
/// Families supply scalar bounds, argument types, kernels and real/complex evaluators.
/// The macro writes the dispatch, the execution forms and their shared documentation;
/// the algorithms stay in their private evaluators. A plain form evaluates once and
/// returns the value without forming diagnostics. A scoped form counts its work in one
/// root or child epoch and returns the same result as its unscoped counterpart. The
/// optional `promote [p, …] to T;` clause converts each listed `impl Parameter<T>`
/// argument to `T` before any body runs.
macro_rules! entry {
    (
        $(#[$doc:meta])* $vis:vis $plain:ident [$($generic:tt)*]
        ($($arg:ident: $arg_ty:ty),* $(,)?) -> $value:ty
        where [$($bound:tt)*];
        $(promote [$($param:ident),+ $(,)?] to $promoted:ty;)?
        options $options:ident: $options_ty:ty;
        detailed $detailed:ident -> $report:ty;
        dispatch $scalar:ident;
        value real $real_value:block complex $complex_value:block
        evaluate |$eval_options:ident, $context:ident| real $real:block complex $complex:block
        $(scoped $scope_vis:vis $scoped:ident [$($life:lifetime),+]
          ($scope:ident: $scope_ty:ty);)?
        $(complex scoped $complex_vis:vis $complex_scoped:ident [$($complex_life:lifetime),+]
          ($complex_scope:ident: $complex_scope_ty:ty);)?
    ) => {
        $crate::dispatch::entry! {
            @forms $(#[$doc])* $vis $plain [$($generic)*] ($($arg: $arg_ty),*) -> $value
            where [$($bound)*];
            promote ($($($param,)+)?) = ($($($crate::dispatch::promote::<$promoted, _>($param),)+)?);
            options $options: $options_ty;
            detailed $detailed -> $report;
            value { if $scalar::IS_REAL $real_value else $complex_value }
            report {
                let $eval_options = $options;
                let $context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
                if $scalar::IS_REAL {
                    { $real }?.publish()?.convert()
                } else {
                    { $complex }?.publish()?.convert()
                }
            }
            $(scoped $scope_vis $scoped [$($life),+] ($scope: $scope_ty) {
                $crate::util::scoped($scope, $options.iteration_budget(), |context| {
                    let $eval_options = $options;
                    let $context = Some(context);
                    if $scalar::IS_REAL {
                        { $real }?.publish()?.convert()
                    } else {
                        { $complex }?.publish()?.convert()
                    }
                })
            })?
        }
        $crate::dispatch::entry!(@complex_optional
            [$($arg),*] [$options: $options_ty] [$plain]
            [$eval_options, $context] [$complex]
            $([$complex_vis] [$complex_scoped] [$($complex_life),+] [$complex_scope: $complex_scope_ty])?
        );
    };
    (@complex_optional [$($arg:ident),*] [$options:ident: $options_ty:ty] [$plain:ident]
        [$eval_options:ident, $context:ident] [$complex:block]
        [$vis:vis] [$scoped:ident] [$($life:lifetime),+] [$scope:ident: $scope_ty:ty]
    ) => {
        $crate::dispatch::entry! {
            @scoped $vis $scoped [$($life,)+ S: scioxide_core::RealScalar]
            ($($arg: scioxide_core::Complex<S>,)* $options: $options_ty, $scope: $scope_ty)
            -> $crate::SpecialFunctionResult<scioxide_core::Complex<S>>
            where [scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>];
            operation $plain;
            { $crate::util::scoped($scope, $options.iteration_budget(), |context| {
                let $eval_options = $options;
                let $context = Some(context);
                { $complex }
            })?.publish() }
        }
    };
    (@complex_optional [$($arg:ident),*] [$options:ident: $options_ty:ty] [$plain:ident]
        [$eval_options:ident, $context:ident] [$complex:block]
    ) => {};
    (
        $(#[$doc:meta])* $vis:vis $plain:ident [$($generic:tt)*]
        ($($arg:ident: $arg_ty:ty),* $(,)?) -> $value:ty
        where [$($bound:tt)*];
        $(promote [$($param:ident),+ $(,)?] to $promoted:ty;)?
        options $options:ident: $options_ty:ty;
        detailed $detailed:ident -> $report:ty;
        evaluate |$eval_options:ident, $context:ident| $evaluate:block
        $(scoped $scope_vis:vis $scoped:ident [$($life:lifetime),+]
          ($scope:ident: $scope_ty:ty);)?
    ) => {
        $crate::dispatch::entry! {
            @forms $(#[$doc])* $vis $plain [$($generic)*] ($($arg: $arg_ty),*) -> $value
            where [$($bound)*];
            promote ($($($param,)+)?) = ($($($crate::dispatch::promote::<$promoted, _>($param),)+)?);
            options $options: $options_ty;
            detailed $detailed -> $report;
            value {
                let $eval_options = <$options_ty>::default();
                let $context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
                let _ = &$context;
                $evaluate?.value()
            }
            report {
                let $eval_options = $options;
                let $context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
                let _ = &$context;
                $evaluate?.publish()
            }
            $(scoped $scope_vis $scoped [$($life),+] ($scope: $scope_ty) {
                $crate::util::scoped($scope, $options.iteration_budget(), |context| {
                    let $eval_options = $options;
                    let $context = Some(context);
                    $evaluate
                })?.publish()
            })?
        }
    };
    (
        $(#[$doc:meta])* $vis:vis $plain:ident [$($generic:tt)*]
        ($($arg:ident: $arg_ty:ty),* $(,)?) -> $value:ty
        where [$($bound:tt)*];
        $(promote [$($param:ident),+ $(,)?] to $promoted:ty;)?
        options $options:ident: $options_ty:ty;
        detailed $detailed:ident -> $report:ty;
        value $value_body:block
        report $report_body:block
        $(scoped $scope_vis:vis $scoped:ident [$($life:lifetime),+]
          ($scope:ident: $scope_ty:ty) $scope_body:block)?
    ) => {
        $crate::dispatch::entry! {
            @forms $(#[$doc])* $vis $plain [$($generic)*] ($($arg: $arg_ty),*) -> $value
            where [$($bound)*];
            promote ($($($param,)+)?) = ($($($crate::dispatch::promote::<$promoted, _>($param),)+)?);
            options $options: $options_ty;
            detailed $detailed -> $report;
            value $value_body
            report $report_body
            $(scoped $scope_vis $scoped [$($life),+] ($scope: $scope_ty) $scope_body)?
        }
    };
    (
        @forms $(#[$doc:meta])* $vis:vis $plain:ident [$($generic:tt)*]
        ($($arg:ident: $arg_ty:ty),*) -> $value:ty
        where [$($bound:tt)*];
        promote $promoted:pat = $promotion:expr;
        options $options:ident: $options_ty:ty;
        detailed $detailed:ident -> $report:ty;
        value $value_body:block
        report $report_body:block
        $(scoped $scope_vis:vis $scoped:ident [$($life:lifetime),+]
          ($scope:ident: $scope_ty:ty) $scope_body:block)?
    ) => {
        $(#[$doc])*
        $vis fn $plain<$($generic)*>($($arg: $arg_ty),*)
            -> Result<$value, scioxide_core::SciOxideError>
        where $($bound)*
        {
            #[allow(clippy::let_unit_value)]
            let $promoted = $promotion;
            $value_body
        }

        #[doc = $crate::dispatch::entry!(@doc $plain, "Detailed", "Options bound total work. The result retains termination, counts, estimates, and available scaled, logarithmic and derivative forms.")]
        $vis fn $detailed<$($generic)*>($($arg: $arg_ty,)* $options: $options_ty)
            -> Result<$report, scioxide_core::SciOxideError>
        where $($bound)*
        {
            #[allow(clippy::let_unit_value)]
            let $promoted = $promotion;
            $report_body
        }

        $crate::dispatch::entry!(@optional
            [$vis] [$($generic)*] [$($arg: $arg_ty,)*] [$options: $options_ty]
            [$report] [$($bound)*] [$plain]
            $([$scope_vis] [$scoped] [$($life),+] [$scope: $scope_ty] {
                #[allow(clippy::let_unit_value)]
                let $promoted = $promotion;
                $scope_body
            })?
        );
    };
    (@optional [$vis:vis] [$($generic:tt)*] [$($args:tt)*] [$($options:tt)*]
        [$report:ty] [$($bound:tt)*] [$plain:ident]
        [$scope_vis:vis] [$scoped:ident] [$($life:lifetime),+] [$scope:ident: $scope_ty:ty] $body:block
    ) => {
        $crate::dispatch::entry! {
            @scoped $scope_vis $scoped [$($life,)+ $($generic)*]
            ($($args)* $($options)*, $scope: $scope_ty) -> $report
            where [$($bound)*]; operation $plain; $body
        }
    };
    (@optional [$vis:vis] [$($generic:tt)*] [$($args:tt)*] [$($options:tt)*]
        [$report:ty] [$($bound:tt)*] [$plain:ident]
    ) => {};
    (@scoped $vis:vis $scoped:ident [$($generic:tt)*]
        ($($args:tt)*) -> $report:ty where [$($bound:tt)*];
        operation $plain:ident; $body:block
    ) => {
        #[doc = $crate::dispatch::entry!(@doc $plain, "Scoped", "Runs in one root or child epoch. Options bound local work; ancestors receive nested counts and propagate cancellation and deadlines. Accepts the scalar types declared in this signature; some fixed-cost real operations have complex-only scoped continuation.")]
        $vis fn $scoped<$($generic)*>($($args)*)
            -> Result<$report, scioxide_core::SciOxideError>
        where $($bound)*
        $body
    };

    (@fixed_operation, $definition:expr, $accuracy:expr, $errors:expr) => {
        concat!($definition, "\n\n# Accuracy\n\n", $accuracy, "\n\n# Errors\n\n", $errors)
    };
    (@operation, $definition:expr, $accuracy:expr, $errors:expr) => {
        concat!($definition, "\n\n# Accuracy\n\n", $accuracy,
            "\n\n# Errors\n\n",
            "Non-finite inputs return `NonFinite`. Arguments outside the documented domain or at its poles return `DomainError`; invalid discrete parameters return `InvalidInput`. Numerical breakdown or projection of an unrepresentable value returns `NumericalInstability`. Iterative work exhaustion follows the result contract: a useful incomplete estimate may retain unsuccessful termination, otherwise it returns `ConvergenceFailure`.\n\n",
            $errors)
    };
    (@doc $plain:ident, $form:literal, $execution:literal) => {
        $crate::dispatch::entry!(@operation,
            concat!($form, " evaluation of [`", stringify!($plain), "`].\n\n", $execution),
            "The linked operation's definition, domain, branch conventions, limits and accuracy apply. Inspect termination and representation availability before using a diagnostic estimate.",
            "Propagates the linked operation's errors. Scoped cancellation and deadlines return `ExecutionStopped`; a missing execution capability returns `CapabilityUnavailable`.")
    };
}
pub(crate) use entry;
