//! Independent SciPy and high-precision references on width-rounded inputs.
#![allow(clippy::unwrap_used)]

mod support;

use std::collections::BTreeMap;

use scioxide_core::{
    Interval, NdView, NumericScalar, RankThreshold, RealScalar, StridedLayout, Tolerance,
};
use scioxide_interp::*;
use scioxide_spatial::{Delaunay, PointSet};
use support::{assert_close, from_f64};

struct Case {
    family: &'static str,
    name: &'static str,
    records: BTreeMap<&'static str, Vec<Vec<f64>>>,
}

impl Case {
    fn row(&self, name: &str) -> &[f64] {
        &self.records[name][0]
    }
    fn scalars<S: RealScalar>(&self, name: &str) -> Vec<S> {
        self.row(name).iter().copied().map(from_f64).collect()
    }
    fn points<S: RealScalar>(&self) -> Vec<[S; 2]> {
        self.row("points")
            .chunks_exact(2)
            .map(|p| [from_f64(p[0]), from_f64(p[1])])
            .collect()
    }
    fn basis<S: RealScalar>(&self, field: &str, degree: usize) -> BSplineBasis<S> {
        BSplineBasis::new(self.scalars(field), degree).unwrap()
    }
    fn expected(&self) -> &[Vec<f64>] {
        &self.records["expect"]
    }
}

fn cases(width: u32, families: &[&str]) -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for line in include_str!("data/reference_battery.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        if fields[1].parse::<u32>().unwrap() != width || !families.contains(&fields[0]) {
            continue;
        }
        if cases
            .last()
            .is_none_or(|case| case.family != fields[0] || case.name != fields[2])
        {
            cases.push(Case {
                family: fields[0],
                name: fields[2],
                records: BTreeMap::new(),
            });
        }
        cases
            .last_mut()
            .unwrap()
            .records
            .entry(fields[3])
            .or_default()
            .push(
                fields[4..]
                    .iter()
                    .map(|field| field.parse().unwrap())
                    .collect(),
            );
    }
    assert!(!cases.is_empty());
    for family in families {
        assert!(
            cases.iter().any(|case| case.family == *family),
            "missing {family} references"
        );
    }
    cases
}

/// Values through the public protocol, rather than another SciOxide implementation.
fn values<S: RealScalar, I: Interpolant<S, D, Value = S>, const D: usize>(
    case: &Case,
    model: &I,
    absolute: f64,
    relative: f64,
) {
    for row in case.expected() {
        let query = core::array::from_fn(|i| from_f64(row[i]));
        let actual = model.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
        assert_close(actual, row[D], absolute, relative);
    }
}

fn one_dimensional<S: RealScalar>(width: u32) {
    let families = [
        "cubic",
        "akima",
        "makima",
        "pchip",
        "hermite",
        "barycentric",
        "newton",
        "neville",
        "fh",
        "bspline",
        "bulirsch",
    ];
    for case in cases(width, &families) {
        eprintln!("{} {}", case.family, case.name);
        let epsilon = S::EPSILON.into_f64();
        // These five-node polynomial/Hermite tables have moderate Lebesgue and
        // divided-difference amplification; the other families are local cubics.
        let factor = if case.family == "hermite" {
            2048.
        } else {
            128.
        };
        let magnitude = if case.family == "bspline" {
            case.row("coefficients")
        } else {
            case.row("y")
        }
        .iter()
        .fold(0_f64, |a, x| a.max(x.abs()));
        let absolute = factor * epsilon * magnitude;
        let relative = factor * epsilon;
        let (x, y) = if case.family == "bspline" {
            (Vec::new(), Vec::new())
        } else {
            (case.scalars::<S>("x"), case.scalars::<S>("y"))
        };
        match case.family {
            "cubic" => {
                let boundary = match case.row("boundary")[0] as usize {
                    0 => SplineBoundary::Natural,
                    1 => SplineBoundary::Clamped(
                        from_f64(case.row("dy")[0]),
                        from_f64(case.row("dy")[1]),
                    ),
                    2 => SplineBoundary::NotAKnot,
                    _ => SplineBoundary::Periodic,
                };
                let lower = x[0];
                let model = CubicSpline::with_boundary(x, y, boundary).unwrap();
                values(&case, &model, absolute, relative);
                for row in case.expected() {
                    let q = from_f64(row[0]);
                    let span = case.row("x").last().unwrap() - case.row("x")[0];
                    assert_close(
                        model.gradient([q], ExtrapolationPolicy::Reject).unwrap()[0],
                        row[2],
                        absolute / span,
                        relative,
                    );
                    assert_close(
                        model
                            .integral([lower], [q], ExtrapolationPolicy::Reject)
                            .unwrap(),
                        row[3],
                        absolute * span,
                        relative,
                    );
                }
            }
            "akima" | "makima" => {
                let method = if case.family == "akima" {
                    AkimaMethod::Akima
                } else {
                    AkimaMethod::Makima
                };
                values(
                    &case,
                    &AkimaInterpolator::with_method(x, y, method).unwrap(),
                    absolute,
                    relative,
                );
            }
            "bulirsch" => values(
                &case,
                &BulirschStoerInterpolator::new(x, y).unwrap(),
                absolute,
                relative,
            ),
            "pchip" => values(
                &case,
                &PchipInterpolator::new(x, y).unwrap(),
                absolute,
                relative,
            ),
            "hermite" => values(
                &case,
                &HermiteInterpolator::new(x, y, case.scalars("dy")).unwrap(),
                absolute,
                relative,
            ),
            "barycentric" => values(
                &case,
                &BarycentricInterpolator::new(x, y).unwrap(),
                absolute,
                relative,
            ),
            "newton" => values(
                &case,
                &NewtonInterpolator::new(x, y).unwrap(),
                absolute,
                relative,
            ),
            "fh" => values(
                &case,
                &FloaterHormannInterpolator::new(x, y, 3).unwrap(),
                absolute,
                relative,
            ),
            "neville" => {
                let model = NevilleInterpolator::new(x, y).unwrap();
                let mut workspace = model.workspace().unwrap();
                for row in case.expected() {
                    let (mut value, mut error) = (S::ZERO, S::ZERO);
                    model
                        .evaluate_with_error_into(
                            [from_f64(row[0])],
                            ExtrapolationPolicy::Reject,
                            &mut value,
                            &mut error,
                            &mut workspace,
                        )
                        .unwrap();
                    assert_close(value, row[1], absolute, relative);
                    assert_close(error, row[2], absolute, relative);
                }
            }
            "bspline" => {
                let lower = from_f64(case.row("knots")[0]);
                let span = case.row("knots").last().unwrap() - case.row("knots")[0];
                let model =
                    BSpline::new([case.basis("knots", 3)], case.scalars::<S>("coefficients"))
                        .unwrap();
                values(&case, &model, absolute, relative);
                for row in case.expected() {
                    let q = from_f64(row[0]);
                    assert_close(
                        model.gradient([q], ExtrapolationPolicy::Reject).unwrap()[0],
                        row[2],
                        absolute / span,
                        relative,
                    );
                    assert_close(
                        model
                            .integral([lower], [q], ExtrapolationPolicy::Reject)
                            .unwrap(),
                        row[3],
                        absolute * span,
                        relative,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

fn spline_fits<S: RealScalar>(width: u32) {
    let tolerance = 256. * S::EPSILON.into_f64();
    for case in cases(width, &["penalized", "tensor", "nurbs", "adaptive"]) {
        eprintln!("{} {}", case.family, case.name);
        match case.family {
            "penalized" => {
                let penalty = case.row("penalty");
                let plan = PenalizedSplinePlan::new(
                    case.basis("knots", 3),
                    case.scalars::<S>("x"),
                    Some(case.scalars("weights")),
                    SplinePenalty::new(from_f64(penalty[0]), penalty[1] as usize).unwrap(),
                )
                .unwrap();
                let fit = plan.fit(&case.scalars::<S>("y")).unwrap();
                assert_eq!(
                    fit.spline().coefficients().len(),
                    case.row("coefficients").len()
                );
                for (&actual, &expected) in fit
                    .spline()
                    .coefficients()
                    .iter()
                    .zip(case.row("coefficients"))
                {
                    assert_close(actual, expected, tolerance, tolerance);
                }
                let objective = case.row("objective");
                assert_close(
                    fit.report().data_fidelity(),
                    objective[0],
                    tolerance,
                    tolerance,
                );
                assert_close(fit.report().penalty(), objective[1], tolerance, tolerance);
                assert!(fit.report().penalty() > from_f64::<S>(0.01));
                values(&case, fit.spline(), tolerance, tolerance);
            }
            "tensor" => {
                let plan = TensorSplineInterpolationPlan::new(
                    [case.basis("knots0", 3), case.basis("knots1", 3)],
                    [case.scalars::<S>("x"), case.scalars("y")],
                )
                .unwrap();
                let model = plan.fit(&case.scalars::<S>("values")).unwrap();
                assert_eq!(model.coefficients().len(), case.row("coefficients").len());
                for (&actual, &expected) in
                    model.coefficients().iter().zip(case.row("coefficients"))
                {
                    assert_close(actual, expected, tolerance, tolerance);
                }
                values(&case, &model, tolerance, tolerance);
            }
            "nurbs" => {
                let controls = case
                    .row("values")
                    .chunks_exact(2)
                    .map(|p| [from_f64::<S>(p[0]), from_f64(p[1])])
                    .collect();
                let model = Nurbs::new([case.basis("knots", 2)], controls, case.scalars("weights"))
                    .unwrap();
                for row in case.expected() {
                    let actual = model
                        .evaluate([from_f64(row[0])], ExtrapolationPolicy::Reject)
                        .unwrap();
                    for (a, e) in actual.into_iter().zip(&row[1..]) {
                        assert_close(a, *e, tolerance, tolerance);
                    }
                }
            }
            "adaptive" => {
                let opts = case.row("options");
                let options = AdaptiveSplineOptions::new()
                    .with_target(SplineFitTarget::Residual(from_f64::<S>(opts[0])))
                    .with_limits(
                        SplineFitLimits::new().with_max_penalty_iterations(opts[2] as usize),
                    );
                if case.name == "curve" {
                    let parameters = case.scalars::<S>("x");
                    let fit = fit_parametric_spline_with_options(
                        &case.scalars::<S>("y"),
                        CurveParameterization::Supplied(&parameters),
                        None,
                        options,
                    )
                    .unwrap();
                    assert_eq!(
                        fit.report().termination(),
                        AdaptiveSplineTermination::PenaltySearchLimit
                    );
                    assert_eq!(fit.report().penalty(), from_f64(opts[1]));
                    assert_eq!(fit.spline().bases()[0].knots(), case.scalars::<S>("knots"));
                    for (&a, &e) in fit
                        .spline()
                        .coefficients()
                        .iter()
                        .zip(case.row("coefficients"))
                    {
                        assert_close(a, e, tolerance, tolerance);
                    }
                    assert_close(
                        fit.report().residual(),
                        case.row("residual")[0],
                        tolerance,
                        tolerance,
                    );
                    values(&case, fit.spline(), tolerance, tolerance);
                } else {
                    let points = case.points::<S>();
                    let fit = fit_spherical_spline_with_options(
                        SphericalSamples::Scattered(&points),
                        &case.scalars::<S>("values"),
                        None,
                        options,
                    )
                    .unwrap();
                    assert_eq!(
                        fit.report().termination(),
                        AdaptiveSplineTermination::PenaltySearchLimit
                    );
                    assert_eq!(fit.report().penalty(), from_f64(opts[1]));
                    assert_eq!(fit.report().refinements(), 1);
                    let [theta, phi] = fit.spline().knots();
                    assert_eq!(theta, case.scalars::<S>("knots0"));
                    assert_eq!(phi, case.scalars::<S>("knots1"));
                    assert_eq!(fit.spline().coefficients().len(), 10);
                    for (&a, &e) in fit
                        .spline()
                        .coefficients()
                        .iter()
                        .zip(case.row("coefficients"))
                    {
                        assert_close(a, e, tolerance, tolerance);
                    }
                    assert_close(
                        fit.report().residual(),
                        case.row("residual")[0],
                        tolerance,
                        tolerance,
                    );
                    values(&case, fit.spline(), tolerance, tolerance);
                }
            }
            _ => unreachable!(),
        }
    }
}

fn scattered<S: RealScalar + NumericScalar<Real = S>>(width: u32) {
    for case in cases(width, &["nearest", "linear", "clough", "rbf"]) {
        eprintln!("{} {}", case.family, case.name);
        let points = case.points::<S>();
        let samples = case.scalars::<S>("values");
        let epsilon = <S as RealScalar>::EPSILON.into_f64();
        if case.family == "rbf" {
            let options = case.row("options");
            let kernel = match options[0] as usize {
                0 => RbfKernel::Gaussian(from_f64(options[1])),
                1 => RbfKernel::Multiquadric(from_f64(options[1])),
                2 => RbfKernel::InverseMultiquadric(from_f64(options[1])),
                _ => RbfKernel::ThinPlateSpline,
            };
            let options = RbfOptions::new()
                .with_regularization(RbfRegularization::Diagonal(from_f64(options[2])))
                .with_rank_threshold(
                    RankThreshold::new(S::ZERO, from_f64::<S>(8.) * <S as RealScalar>::EPSILON)
                        .unwrap(),
                );
            let model = RbfInterpolator::with_options(points, samples, kernel, options).unwrap();
            // Account for amplification by the retained kernel system's measured
            // conditioning rather than assigning an arbitrary decimal tolerance.
            let relative = 32. * epsilon
                / model
                    .conditioning()
                    .reciprocal_condition_number()
                    .into_f64();
            values(&case, &model, relative, relative);
            continue;
        }
        let coordinates: Vec<S> = points.iter().flatten().copied().collect();
        let shape = [points.len(), 2];
        let strides = [2, 1];
        let view =
            NdView::new(&coordinates, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
        let mesh = Delaunay::new(&PointSet::new(view).unwrap()).unwrap();
        let samples = ScatteredValues::from_input_values(&mesh, samples).unwrap();
        let options = ScatteredOptions::new();
        match case.family {
            "nearest" => values(
                &case,
                &NearestNeighborInterpolator::<_, 2>::with_options(&mesh, &samples, options)
                    .unwrap(),
                0.,
                0.,
            ),
            "linear" => values(
                &case,
                &SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &samples, options).unwrap(),
                64. * epsilon,
                64. * epsilon,
            ),
            "clough" => {
                let options = CloughTocherOptions::new()
                    .with_maximum_iterations(1000)
                    .with_gradient_tolerance(
                        Tolerance::new(
                            from_f64::<S>(8.) * <S as RealScalar>::EPSILON,
                            from_f64::<S>(8.) * <S as RealScalar>::EPSILON,
                        )
                        .unwrap(),
                    );
                values(
                    &case,
                    &CloughTocherInterpolator::with_options(&mesh, &samples, options).unwrap(),
                    512. * epsilon,
                    512. * epsilon,
                );
            }
            _ => unreachable!(),
        }
    }
}

fn lawson<S: RealScalar + NumericScalar<Real = S>>(width: u32) {
    for case in cases(width, &["lawson"]) {
        let x = case.scalars::<S>("x");
        let options = case.row("options");
        let options = AaaOptions::new()
            .with_max_terms(options[0] as usize)
            .with_tolerance(Tolerance::new(S::ZERO, S::ZERO).unwrap())
            .with_lawson_iterations(options[1] as usize);
        let fit = aaa_with_options(
            &x,
            &case.scalars::<S>("y"),
            RationalDomain::Real(Interval::new(x[0], x[x.len() - 1]).unwrap()),
            options,
        )
        .unwrap();
        let tolerance = 2048. * <S as RealScalar>::EPSILON.into_f64();
        assert_close(
            fit.report().unwrap().max_residual(),
            case.row("residual")[0],
            tolerance,
            tolerance,
        );
        values(&case, fit.approximant().unwrap(), tolerance, tolerance);
    }
}

macro_rules! reference_tests {
    ($module:ident, $scalar:ty, $width:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn one_dimensional_references() {
                one_dimensional::<$scalar>($width);
            }
            #[test]
            fn spline_and_adaptive_references() {
                spline_fits::<$scalar>($width);
            }
            #[test]
            fn scattered_references() {
                scattered::<$scalar>($width);
            }
            #[test]
            fn lawson_references() {
                lawson::<$scalar>($width);
            }
        }
    };
}
reference_tests!(f32, f32, 32);
reference_tests!(f64, f64, 64);
