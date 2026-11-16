//! Execute every shipped example through the ordinary test harness.

macro_rules! example {
    ($name:ident, $test:ident, $path:literal) => {
        #[path = $path]
        mod $name;

        #[test]
        fn $test() -> Result<(), Box<dyn std::error::Error>> {
            $name::main()?;
            Ok(())
        }
    };
}

example!(angular, angular_runs, "../examples/angular.rs");
example!(
    bessel_extensions,
    bessel_extensions_runs,
    "../examples/bessel_extensions.rs"
);
example!(
    carlson_jacobi,
    carlson_jacobi_runs,
    "../examples/carlson_jacobi.rs"
);
example!(
    combinatorial,
    combinatorial_runs,
    "../examples/combinatorial.rs"
);
example!(
    complex_airy,
    complex_airy_runs,
    "../examples/complex_airy.rs"
);
example!(
    complex_elliptic,
    complex_elliptic_runs,
    "../examples/complex_elliptic.rs"
);
example!(
    complex_error,
    complex_error_runs,
    "../examples/complex_error.rs"
);
example!(
    complex_gamma,
    complex_gamma_runs,
    "../examples/complex_gamma.rs"
);
example!(
    complex_order_bessel,
    complex_order_bessel_runs,
    "../examples/complex_order_bessel.rs"
);
example!(coulomb, coulomb_runs, "../examples/coulomb.rs");
example!(ellipsoidal, ellipsoidal_runs, "../examples/ellipsoidal.rs");
example!(
    hypergeometric,
    hypergeometric_runs,
    "../examples/hypergeometric.rs"
);
example!(integrals, integrals_runs, "../examples/integrals.rs");
example!(
    lambert_polylog,
    lambert_polylog_runs,
    "../examples/lambert_polylog.rs"
);
example!(mathieu, mathieu_runs, "../examples/mathieu.rs");
example!(probability, probability_runs, "../examples/probability.rs");
example!(spherical, spherical_runs, "../examples/spherical.rs");
example!(spheroidal, spheroidal_runs, "../examples/spheroidal.rs");
example!(thermal, thermal_runs, "../examples/thermal.rs");
example!(transport, transport_runs, "../examples/transport.rs");
