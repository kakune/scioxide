//! Independent beta derivatives, CPV identities and Fourier polynomial oracles.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, RealScalar};
use scioxide_integrate::*;

macro_rules! references {
    ($test:ident,$s:ty,$name:literal,$tol:expr) => {
        #[test]
        fn $test() {
            let options = SubdivisionOptions::default()
                .with_tolerance(
                    ErrorTolerance::scalar($tol, $tol, NormSelection::Infinity).unwrap(),
                )
                .with_max_subdivisions(Budget::try_from(256).unwrap());
            for row in include_str!("../data/weighted.csv")
                .lines()
                .filter(|r| r.starts_with($name))
            {
                let fields: Vec<_> = row.split(',').collect();
                let family = fields[1];
                let p: $s = fields[2].parse().unwrap();
                let degree: usize = fields[3].parse().unwrap();
                let expected: $s = fields[4].parse().unwrap();
                let interval = Interval::new(0.0, 1.0).unwrap();
                let result = if family == "tail" {
                    FourierTailPlan::new(
                        FourierDomain::Upper { lower: 0.0 },
                        p,
                        FourierWeight::Sine,
                        scioxide_integrate::FourierTailOptions::new(options.clone())
                            .with_max_cycles(Budget::try_from(128).unwrap())
                            .with_moment_levels(Budget::try_from(20).unwrap()),
                    )
                    .unwrap()
                    .integrate(|x: $s| Ok(1.0 / (1.0 + x)))
                    .unwrap()
                } else {
                    let plan = if family.starts_with("endpoint") {
                        let left = family.ends_with("left") || family.ends_with("both");
                        let right = family.ends_with("right") || family.ends_with("both");
                        WeightedQuadraturePlan::endpoint(
                            interval,
                            &[],
                            EndpointWeight::new(p, 0.25, left, right).unwrap(),
                            scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
                        )
                        .unwrap()
                    } else if family == "cpv" {
                        WeightedQuadraturePlan::principal_value(
                            interval,
                            p,
                            scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
                        )
                        .unwrap()
                    } else {
                        WeightedQuadraturePlan::fourier(
                            interval,
                            &[],
                            p,
                            if family.ends_with("cos") {
                                FourierWeight::Cosine
                            } else {
                                FourierWeight::Sine
                            },
                            scioxide_integrate::WeightedQuadratureOptions::new(options.clone())
                                .with_moment_levels(Budget::try_from(20).unwrap()),
                        )
                        .unwrap()
                    };
                    plan.integrate(|x: $s| {
                        if family.starts_with("fourier") {
                            let x = 2.0 * x - 1.0;
                            let mut previous = 1.0;
                            let mut current = x;
                            if degree == 0 {
                                return Ok(previous);
                            }
                            for _ in 1..degree {
                                let next = 2.0 * x * current - previous;
                                previous = current;
                                current = next;
                            }
                            Ok(current)
                        } else {
                            Ok(x.powi(i32::try_from(degree).unwrap()))
                        }
                    })
                    .unwrap()
                };
                let value = result.0;
                let error = (value - expected).abs();
                let allowance = $tol * expected.abs().max(1.0);
                assert!(
                    error <= allowance,
                    "{row}: {result:?}, actual error {error}"
                );
                assert!(result.1.is_success(), "{row}: {result:?}");
                let estimated = result.1.error_estimate().unwrap();
                assert!(
                    error <= estimated + 100.0 * <$s as RealScalar>::EPSILON * expected.abs(),
                    "underestimated {row}: actual {error}, estimate {estimated}"
                );
            }
        }
    };
}
references!(independent_f64, f64, "f64", 2e-9);
references!(independent_f32, f32, "f32", 3e-4);
