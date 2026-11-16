//! Measured fixture envelopes, independent of the result's own uncertainty.
//!
//! Run the consuming integration test with `-- --nocapture` for maxima in
//! `max(epsilon * reference_scale, least_subnormal)` units. Complex scales
//! are vector norms; scaled representations use their own coordinates.
//! Bessel integrals, parabolic, spherical/Riccati and H3d use mixed absolute/relative
//! envelopes, epsilon * max(1, |reference|), including their zero strips.
//! Kelvin uses independently referenced pair magnitudes and ODE propagation.
//! Limits are frozen powers of two covering twice each measured maximum;
//! reference rounding and subnormal-specific assertions remain independent.
#![allow(dead_code, clippy::panic)]

use scioxide_core::{Complex, NumericScalar, RealScalar};
use std::collections::BTreeMap;

#[derive(Default)]
struct Measurement {
    error: f64,
    reported: f64,
    witness: String,
}

pub struct Checker {
    family: &'static str,
    width: &'static str,
    measurements: BTreeMap<String, Measurement>,
}

// (accuracy exponent, reported-error exponent), in the units above.
// Measurement: cargo test -p scioxide-special --test <consumer> -- --nocapture.
fn limits(family: &str, width: &str, component: &str) -> (i32, i32) {
    match (family, width, component) {
        ("airy", "32", "derivative") => (0, 6),
        ("airy", "32", "value") => (0, 3),
        ("airy", "64", "derivative") => (2, 11),
        ("airy", "64", "value") => (14, 20),
        ("bessel_turning", "32", "value") => (0, 20),
        ("bessel_turning", "64", "value") => (11, 49),
        ("complex_bessel", "32", "derivative") => (0, 6),
        ("complex_bessel", "32", "value") => (0, 3),
        ("complex_bessel", "64", "derivative") => (9, 26),
        ("complex_bessel", "64", "value") => (9, 24),
        ("coulomb_sequence", "32", "component") => (0, 4),
        ("coulomb_sequence", "64", "component") => (11, 30),
        ("h3d", "32", "value") => (5, 112),
        ("h3d", "64", "value") => (5, 112),
        ("kelvin", "32", "derivative") => (0, 6),
        ("kelvin", "32", "prime") => (0, 3),
        ("kelvin", "32", "second") => (1, 6),
        ("kelvin", "32", "value") => (0, 3),
        ("kelvin", "64", "derivative") => (9, 16),
        ("kelvin", "64", "prime") => (9, 16),
        ("kelvin", "64", "second") => (10, 18),
        ("kelvin", "64", "value") => (9, 16),
        ("parabolic", "32", "derivative") => (0, 15),
        ("parabolic", "32", "value") => (0, 16),
        ("parabolic", "64", "derivative") => (11, 44),
        ("parabolic", "64", "value") => (12, 45),
        ("spherical", "32", "derivative") => (8, 152),
        ("spherical", "32", "value") => (11, 152),
        ("spherical", "64", "derivative") => (10, 684),
        ("spherical", "64", "value") => (10, 681),
        ("struve", "32", "derivative") => (0, 2),
        ("struve", "32", "value") => (0, 11),
        ("struve", "64", "derivative") => (9, 11),
        ("struve", "64", "value") => (9, 11),
        ("weighted_bessel", "32", "value") => (0, 2),
        ("weighted_bessel", "64", "value") => (8, 14),
        ("bessel_integrals", "32", "hint:derivative") => (0, 1),
        ("bessel_integrals", "32", "hint:value") => (0, 2),
        ("bessel_integrals", "32", "htail:derivative") => (0, 1),
        ("bessel_integrals", "32", "htail:value") => (0, 2),
        ("bessel_integrals", "32", "iint:derivative") => (0, 2),
        ("bessel_integrals", "32", "iint:value") => (0, 2),
        ("bessel_integrals", "32", "iremainder:derivative") => (0, 2),
        ("bessel_integrals", "32", "iremainder:value") => (0, 2),
        ("bessel_integrals", "32", "jcompl:derivative") => (0, 0),
        ("bessel_integrals", "32", "jcompl:value") => (0, 2),
        ("bessel_integrals", "32", "jint:derivative") => (0, 1),
        ("bessel_integrals", "32", "jint:value") => (0, 2),
        ("bessel_integrals", "32", "kint:derivative") => (0, 2),
        ("bessel_integrals", "32", "kint:value") => (0, 2),
        ("bessel_integrals", "32", "ktail:derivative") => (0, 2),
        ("bessel_integrals", "32", "ktail:value") => (0, 2),
        ("bessel_integrals", "32", "lint:derivative") => (0, 2),
        ("bessel_integrals", "32", "lint:value") => (0, 2),
        ("bessel_integrals", "32", "yint:derivative") => (0, 2),
        ("bessel_integrals", "32", "yint:value") => (0, 1),
        ("bessel_integrals", "32", "ytail:derivative") => (0, 2),
        ("bessel_integrals", "32", "ytail:value") => (0, 2),
        ("bessel_integrals", "64", "hint:derivative") => (11, 21),
        ("bessel_integrals", "64", "hint:value") => (11, 20),
        ("bessel_integrals", "64", "htail:derivative") => (9, 17),
        ("bessel_integrals", "64", "htail:value") => (10, 17),
        ("bessel_integrals", "64", "iint:derivative") => (3, 8),
        ("bessel_integrals", "64", "iint:value") => (8, 14),
        ("bessel_integrals", "64", "iremainder:derivative") => (4, 8),
        ("bessel_integrals", "64", "iremainder:value") => (14, 19),
        ("bessel_integrals", "64", "jcompl:derivative") => (9, 17),
        ("bessel_integrals", "64", "jcompl:value") => (9, 16),
        ("bessel_integrals", "64", "jint:derivative") => (13, 21),
        ("bessel_integrals", "64", "jint:value") => (11, 21),
        ("bessel_integrals", "64", "kint:derivative") => (14, 19),
        ("bessel_integrals", "64", "kint:value") => (14, 18),
        ("bessel_integrals", "64", "ktail:derivative") => (11, 15),
        ("bessel_integrals", "64", "ktail:value") => (11, 15),
        ("bessel_integrals", "64", "lint:derivative") => (3, 8),
        ("bessel_integrals", "64", "lint:value") => (8, 14),
        ("bessel_integrals", "64", "yint:derivative") => (13, 18),
        ("bessel_integrals", "64", "yint:value") => (13, 18),
        ("bessel_integrals", "64", "ytail:derivative") => (9, 14),
        ("bessel_integrals", "64", "ytail:value") => (10, 15),
        _ => panic!("unmeasured fixture family {family}/{width}/{component}"),
    }
}

impl Checker {
    pub fn new(family: &'static str, width: &str) -> Self {
        Self {
            family,
            width: if width == "32" { "32" } else { "64" },
            measurements: BTreeMap::new(),
        }
    }

    pub fn scalar<S: RealScalar>(
        &mut self,
        component: &str,
        actual: S,
        expected: S,
        reported: Option<S>,
        row: &str,
    ) {
        self.sample(
            component,
            (actual - expected).abs().into_f64(),
            expected.abs().into_f64(),
            reported.map(RealScalar::into_f64),
            row,
        );
    }

    pub fn complex<S: RealScalar>(
        &mut self,
        component: &str,
        actual: Complex<S>,
        expected: Complex<S>,
        reported: Option<S>,
        row: &str,
    ) where
        Complex<S>: NumericScalar<Real = S>,
    {
        self.sample(
            component,
            (actual - expected).norm().into_f64(),
            expected.norm().into_f64(),
            reported.map(RealScalar::into_f64),
            row,
        );
    }

    pub fn sample(
        &mut self,
        component: &str,
        difference: f64,
        scale: f64,
        reported: Option<f64>,
        row: &str,
    ) {
        let (epsilon, minimum) = if self.width == "32" {
            (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
        } else {
            (f64::EPSILON, f64::from_bits(1))
        };
        let reference_unit = (epsilon * scale).max(minimum);
        let absolute_floor = matches!(
            self.family,
            "parabolic" | "spherical" | "h3d" | "bessel_integrals"
        );
        let unit = (epsilon
            * if absolute_floor {
                scale.max(1.0)
            } else {
                scale
            })
        .max(minimum);
        assert!(
            difference.is_finite() && scale.is_finite(),
            "{row}: nonfinite fixture comparison"
        );
        let measurement = self.measurements.entry(component.to_owned()).or_default();
        let error = difference / unit;
        if error > measurement.error {
            measurement.error = error;
            measurement.witness = row.to_owned();
        }
        let (accuracy, report_cap) = limits(self.family, self.width, component);
        assert!(
            error <= 2.0_f64.powi(accuracy),
            "{row}: fixture accuracy {error} exceeds 2^{accuracy} units"
        );
        if let Some(reported) = reported {
            assert!(
                reported.is_finite() && reported >= 0.0,
                "{row}: invalid reported error"
            );
            assert!(
                difference <= reported + reference_unit,
                "{row}: error {difference:e} exceeds reported {reported:e}"
            );
            let reported_units = reported / unit;
            assert!(
                reported_units <= 2.0_f64.powi(report_cap),
                "{row}: reported error {reported_units} exceeds 2^{report_cap} units"
            );
            measurement.reported = measurement.reported.max(reported_units);
        }
    }
}

impl Drop for Checker {
    fn drop(&mut self) {
        for (component, m) in &self.measurements {
            eprintln!(
                "fixture {}/{} {component}: error={} reported={} witness={}",
                self.family, self.width, m.error, m.reported, m.witness
            );
        }
    }
}
