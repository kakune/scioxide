//! Validated function families and their prepared numerical representations.
use super::{MatrixFunction, branch, domain};
use crate::Matrix;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Principal {
    Exp,
    Log,
    Sqrt,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Trigonometric {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
}

/// Constructed only by `ValidatedFunction::new`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FiniteExponent<R>(R);
impl<R: Copy> FiniteExponent<R> {
    pub(crate) fn get(self) -> R {
        self.0
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SchurFunction<'f, R: RealScalar> {
    Power(FiniteExponent<R>),
    Sign,
    Trigonometric(Trigonometric),
    Analytic(&'f (dyn super::AnalyticMatrixFunction<R> + Sync)),
}

#[derive(Clone, Copy)]
pub(crate) enum ValidatedFunction<'f, R: RealScalar> {
    Principal(Principal),
    Schur(SchurFunction<'f, R>),
}
impl<R: RealScalar> core::fmt::Debug for ValidatedFunction<'_, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.function().fmt(f)
    }
}
impl<'f, R: RealScalar> ValidatedFunction<'f, R> {
    pub(crate) fn new(function: MatrixFunction<'f, R>) -> Result<Self, SciOxideError> {
        use MatrixFunction as F;
        Ok(match function {
            F::Exp => Self::Principal(Principal::Exp),
            F::Log => Self::Principal(Principal::Log),
            F::Sqrt => Self::Principal(Principal::Sqrt),
            F::Power(p) => {
                if !p.is_finite() {
                    return Err(domain(
                        scioxide_core::argument!("function"),
                        "fractional exponent must be finite",
                    ));
                }
                Self::Schur(SchurFunction::Power(FiniteExponent(p)))
            }
            F::Sign => Self::Schur(SchurFunction::Sign),
            F::Sin => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Sin)),
            F::Cos => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Cos)),
            F::Tan => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Tan)),
            F::Sinh => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Sinh)),
            F::Cosh => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Cosh)),
            F::Tanh => Self::Schur(SchurFunction::Trigonometric(Trigonometric::Tanh)),
            F::Analytic(f) => Self::Schur(SchurFunction::Analytic(f)),
        })
    }
    pub(crate) fn function(self) -> MatrixFunction<'f, R> {
        use MatrixFunction as F;
        match self {
            Self::Principal(Principal::Exp) => F::Exp,
            Self::Principal(Principal::Log) => F::Log,
            Self::Principal(Principal::Sqrt) => F::Sqrt,
            Self::Schur(SchurFunction::Power(p)) => F::Power(p.get()),
            Self::Schur(SchurFunction::Sign) => F::Sign,
            Self::Schur(SchurFunction::Analytic(f)) => F::Analytic(f),
            Self::Schur(SchurFunction::Trigonometric(t)) => match t {
                Trigonometric::Sin => F::Sin,
                Trigonometric::Cos => F::Cos,
                Trigonometric::Tan => F::Tan,
                Trigonometric::Sinh => F::Sinh,
                Trigonometric::Cosh => F::Cosh,
                Trigonometric::Tanh => F::Tanh,
            },
        }
    }
}

/// Dense variants imply exp; real Schur variants accept only principal kernels.
#[derive(Clone, Copy)]
pub(crate) enum PreparedRef<'a, R: RealScalar> {
    DenseReal {
        a: &'a [R],
        n: usize,
    },
    DenseComplex {
        a: &'a [Complex<R>],
        n: usize,
    },
    RealSchur {
        t: &'a [R],
        q: &'a [R],
        n: usize,
        function: Principal,
    },
    ComplexSchur {
        t: &'a [Complex<R>],
        q: &'a [Complex<R>],
        n: usize,
        function: ValidatedFunction<'a, R>,
    },
}

pub(crate) fn real_eigenvalues<R: RealScalar>(
    t: &[R],
    n: usize,
) -> impl Iterator<Item = Complex<R>> + Clone + '_ {
    (0..n).map(move |i| {
        let pair = if i + 1 < n && t[(i + 1) * n + i] != R::ZERO {
            Some((i, R::ONE))
        } else if i > 0 && t[i * n + i - 1] != R::ZERO {
            Some((i - 1, -R::ONE))
        } else {
            None
        };
        match pair {
            Some((j, sign)) => Complex::new(
                t[j * n + j],
                sign * t[j * n + j + 1].abs().sqrt() * t[(j + 1) * n + j].abs().sqrt(),
            ),
            None => Complex::new(t[i * n + i], R::ZERO),
        }
    })
}
impl<'a, R: RealScalar> PreparedRef<'a, R> {
    pub(crate) fn order(self) -> usize {
        match self {
            Self::DenseReal { n, .. }
            | Self::DenseComplex { n, .. }
            | Self::RealSchur { n, .. }
            | Self::ComplexSchur { n, .. } => n,
        }
    }
    pub(crate) fn function(self) -> ValidatedFunction<'a, R> {
        match self {
            Self::DenseReal { .. } | Self::DenseComplex { .. } => {
                ValidatedFunction::Principal(Principal::Exp)
            }
            Self::RealSchur { function, .. } => ValidatedFunction::Principal(function),
            Self::ComplexSchur { function, .. } => function,
        }
    }
    pub(crate) fn branch(
        self,
        derivative: bool,
        real_output: bool,
    ) -> Result<Option<R>, SciOxideError> {
        let function = self.function();
        // No fictitious spectrum for dense exponentials.
        match self {
            Self::DenseReal { .. } | Self::DenseComplex { .. } => Ok(None),
            Self::RealSchur { t, n, .. } => {
                check_branch(function, real_eigenvalues(t, n), derivative, real_output)
            }
            Self::ComplexSchur { t, n, .. } => check_branch(
                function,
                (0..n).map(|i| t[i * n + i]),
                derivative,
                real_output,
            ),
        }
    }
}

fn check_branch<R: RealScalar>(
    function: ValidatedFunction<'_, R>,
    values: impl Iterator<Item = Complex<R>> + Clone,
    derivative: bool,
    real_output: bool,
) -> Result<Option<R>, SciOxideError> {
    if real_output
        && matches!(function, ValidatedFunction::Principal(Principal::Sqrt))
        && values.clone().any(|z| z.im == R::ZERO && z.re < R::ZERO)
    {
        return Err(domain(
            scioxide_core::argument!("output"),
            "principal square root on a negative real eigenvalue requires complex output",
        ));
    }
    branch(function, values, derivative)
}

#[derive(Debug, Clone)]
pub(crate) enum Prepared<'f, R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    DenseReal(Matrix<R>),
    DenseComplex(Matrix<Complex<R>>),
    RealSchur {
        t: Matrix<R>,
        q: Matrix<R>,
        function: Principal,
    },
    ComplexSchur {
        t: Matrix<Complex<R>>,
        q: Matrix<Complex<R>>,
        function: ValidatedFunction<'f, R>,
    },
}
impl<R: RealScalar> Prepared<'_, R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    pub(crate) fn factors(&self) -> PreparedRef<'_, R> {
        match self {
            Self::DenseReal(a) => PreparedRef::DenseReal {
                a: a.as_slice(),
                n: a.rows(),
            },
            Self::DenseComplex(a) => PreparedRef::DenseComplex {
                a: a.as_slice(),
                n: a.rows(),
            },
            Self::RealSchur { t, q, function } => PreparedRef::RealSchur {
                t: t.as_slice(),
                q: q.as_slice(),
                n: t.rows(),
                function: *function,
            },
            Self::ComplexSchur { t, q, function } => PreparedRef::ComplexSchur {
                t: t.as_slice(),
                q: q.as_slice(),
                n: t.rows(),
                function: *function,
            },
        }
    }
}
