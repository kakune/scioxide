//! SciPy `cont2discrete`, `tf2ss`, `zpk2ss` and `tf2zpk` references.
//!
//! Fixture: `fixtures/lti_scipy.tsv`, written by
//! `generate_lti.py`. Discretized matrices are
//! compared directly; non-unique realizations through D and the Markov
//! parameters C A^k B (k < 2n); roots after nearest-neighbour matching. Errors
//! are normwise: the largest entry error divided by `max(max |reference|, 1)`.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind, SciOxideErrorView};
use scioxide_linalg::Matrix;
use scioxide_signal::*;

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
fn complexes(line: &str) -> Vec<Complex<f64>> {
    numbers(line)
        .chunks_exact(2)
        .map(|v| Complex::new(v[0], v[1]))
        .collect()
}
fn complexes_real(line: &str) -> Vec<Complex<f64>> {
    numbers(line)
        .into_iter()
        .map(|x| Complex::new(x, 0.0))
        .collect()
}
fn s<S: RealScalar>(v: f64) -> S {
    S::try_from_f64_lossy(v).unwrap()
}
fn wide<T: NumericScalar>(x: T) -> Complex<f64> {
    Complex::new(
        x.real_component().into_f64(),
        x.imaginary_component().into_f64(),
    )
}
fn matrix<S: RealScalar>(rows: usize, cols: usize, line: &str) -> Matrix<S> {
    Matrix::from_vec(rows, cols, numbers(line).into_iter().map(s).collect()).unwrap()
}
fn normwise(actual: &[Complex<f64>], expected: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.norm()));
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a - b).norm())
        .fold(0.0, f64::max)
        / scale
}
fn dot(a: &[Complex<f64>], b: &[Complex<f64>]) -> Complex<f64> {
    a.iter()
        .zip(b)
        .fold(Complex::new(0.0, 0.0), |sum, (&a, &b)| sum + a * b)
}
/// D followed by C A^k B for k < 2n, evaluated in f64 from the realization.
fn markov<T: NumericScalar>(system: &StateSpace<T>) -> Vec<Complex<f64>> {
    let n = system.states();
    let a: Vec<_> = system.a().as_slice().iter().map(|&x| wide(x)).collect();
    let c: Vec<_> = system.c().as_slice().iter().map(|&x| wide(x)).collect();
    let mut power: Vec<_> = system.b().as_slice().iter().map(|&x| wide(x)).collect();
    let mut terms = vec![wide(system.d().as_slice()[0])];
    for _ in 0..2 * n {
        terms.push(dot(&c, &power));
        power = a.chunks_exact(n).map(|row| dot(row, &power)).collect();
    }
    terms
}
/// Greedy nearest-neighbour matching of simple, well-separated roots.
fn root_error(actual: &[Complex<f64>], expected: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut remaining = expected.to_vec();
    let mut error = 0.0_f64;
    for root in actual {
        let (index, distance) = remaining
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, (*root - e).norm() / e.norm().max(1.0)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        error = error.max(distance);
        remaining.swap_remove(index);
    }
    error
}
fn time<S: RealScalar>(digital: bool) -> LtiTime<S> {
    if digital {
        LtiTime::Discrete {
            sample_period: S::ONE,
        }
    } else {
        LtiTime::Continuous
    }
}
fn domain(digital: bool) -> IirDomain {
    if digital {
        IirDomain::Digital
    } else {
        IirDomain::Analog
    }
}

/// f64 meets 1e-12 normwise (observed worst 2.3e-15, tf2zpk). All fixture
/// inputs are binary fractions exact in f32, so f32 errors are arithmetic
/// rounding alone: observed at most 2.5 epsilon, bounded here by 64 epsilon.
fn reference<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        64.0 * S::EPSILON.into_f64()
    } else {
        1e-12
    };
    let lines: Vec<_> = include_str!("fixtures/lti_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    let mut cursor = 0;
    while cursor < lines.len() {
        let header: Vec<_> = lines[cursor].split_whitespace().collect();
        let digital = header[0] != "discretize" && header[1] == "1";
        let (error, consumed) = match header[0] {
            "discretize" => {
                let [n, m, p]: [usize; 3] =
                    core::array::from_fn(|i| header[i + 1].parse().unwrap());
                let dt: f64 = header[4].parse().unwrap();
                let method = match header[5] {
                    "zoh" => DiscretizationMethod::ZeroOrderHold,
                    _ => DiscretizationMethod::GeneralizedBilinear {
                        alpha: s(header[6].parse().unwrap()),
                    },
                };
                let shapes = [(n, n), (n, m), (p, n), (p, m)];
                let [a, b, c, d]: [Matrix<S>; 4] = core::array::from_fn(|i| {
                    matrix(shapes[i].0, shapes[i].1, lines[cursor + 1 + i])
                });
                let system =
                    StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view())
                        .unwrap()
                        .discretize(s(dt), method)
                        .unwrap();
                assert_eq!(
                    system.time(),
                    LtiTime::Discrete {
                        sample_period: s(dt)
                    }
                );
                let actual = [system.a(), system.b(), system.c(), system.d()];
                let error = actual
                    .iter()
                    .enumerate()
                    .map(|(i, actual)| {
                        let actual: Vec<_> = actual.as_slice().iter().map(|&x| wide(x)).collect();
                        normwise(&actual, &complexes_real(lines[cursor + 5 + i]))
                    })
                    .fold(0.0, f64::max);
                (error, 9)
            }
            "tf2ss" => {
                let b: Vec<S> = numbers(lines[cursor + 1]).into_iter().map(s).collect();
                let a: Vec<S> = numbers(lines[cursor + 2]).into_iter().map(s).collect();
                let system = TransferFunction::new(domain(digital), &b, &a)
                    .unwrap()
                    .to_state_space(time(digital))
                    .unwrap();
                (
                    normwise(&markov(&system), &complexes_real(lines[cursor + 3])),
                    4,
                )
            }
            "zpk2ss" => {
                let convert = |line: &str| -> Vec<Complex<S>> {
                    complexes(line)
                        .into_iter()
                        .map(|z| Complex::new(s(z.re), s(z.im)))
                        .collect()
                };
                let zeros = convert(lines[cursor + 1]);
                let poles = convert(lines[cursor + 2]);
                let gain = Complex::new(s(numbers(lines[cursor + 3])[0]), S::ZERO);
                let system = PoleZeroGain::new(domain(digital), &zeros, &poles, gain)
                    .unwrap()
                    .to_state_space(time(digital))
                    .unwrap();
                (
                    normwise(&markov(&system), &complexes_real(lines[cursor + 4])),
                    5,
                )
            }
            record => {
                assert_eq!(record, "tf2zpk");
                let b: Vec<S> = numbers(lines[cursor + 1]).into_iter().map(s).collect();
                let a: Vec<S> = numbers(lines[cursor + 2]).into_iter().map(s).collect();
                let zpk = TransferFunction::new(domain(digital), &b, &a)
                    .unwrap()
                    .to_zpk()
                    .unwrap();
                let wide_roots = |roots: &[Complex<S>]| -> Vec<Complex<f64>> {
                    roots.iter().map(|&z| wide(z)).collect()
                };
                let zeros = root_error(&wide_roots(zpk.zeros()), &complexes(lines[cursor + 3]));
                let poles = root_error(&wide_roots(zpk.poles()), &complexes(lines[cursor + 4]));
                let gain = normwise(
                    &[wide(zpk.gain())],
                    &[Complex::new(numbers(lines[cursor + 5])[0], 0.0)],
                );
                (zeros.max(poles).max(gain), 6)
            }
        };
        assert!(
            error <= tolerance,
            "{}: normwise error {error:e} > {tolerance:e}",
            lines[cursor]
        );
        cursor += consumed;
    }
}

#[test]
fn scipy_lti_conversions_f64() {
    reference::<f64>();
}
#[test]
fn scipy_lti_conversions_f32() {
    reference::<f32>();
}

#[test]
fn rejected_discretizations_report_their_kind() {
    let one = Matrix::from_vec(1, 1, vec![-1.0_f64]).unwrap();
    let continuous = StateSpace::new(
        LtiTime::Continuous,
        one.view(),
        one.view(),
        one.view(),
        one.view(),
    )
    .unwrap();
    // The generalized bilinear transform requires alpha in [0, 1].
    for (dt, method, kind) in [
        (
            0.0,
            DiscretizationMethod::ZeroOrderHold,
            SciOxideErrorKind::InvalidInput,
        ),
        (
            f64::NAN,
            DiscretizationMethod::ZeroOrderHold,
            SciOxideErrorKind::NonFinite,
        ),
        (
            0.5,
            DiscretizationMethod::GeneralizedBilinear { alpha: 1.5 },
            SciOxideErrorKind::InvalidInput,
        ),
    ] {
        assert_eq!(continuous.discretize(dt, method).unwrap_err().kind(), kind);
    }
    let discrete = continuous
        .discretize(0.5, DiscretizationMethod::ZeroOrderHold)
        .unwrap();
    let error = discrete
        .discretize(0.5, DiscretizationMethod::ZeroOrderHold)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}
