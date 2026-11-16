//! Lane kernels against MPFR fixtures for every width and lane position.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
use scioxide_core::RealScalar;
use scioxide_core::backend::{
    lanes_cos, lanes_exp, lanes_exp_m1, lanes_ln, lanes_ln_1p, lanes_powf, lanes_sin, lanes_tanh,
};

const OPERATIONS: [&str; 8] = ["exp", "exp_m1", "ln", "ln_1p", "sin", "cos", "tanh", "powf"];

fn lanes<const W: usize>(operation: &str, x: [f64; W], y: [f64; W]) -> [f64; W] {
    match operation {
        "exp" => lanes_exp(x),
        "exp_m1" => lanes_exp_m1(x),
        "ln" => lanes_ln(x),
        "ln_1p" => lanes_ln_1p(x),
        "sin" => lanes_sin(x),
        "cos" => lanes_cos(x),
        "tanh" => lanes_tanh(x),
        "powf" => lanes_powf(x, y),
        other => panic!("{other}"),
    }
}

fn scalar(operation: &str, x: f64, y: f64) -> f64 {
    match operation {
        "exp" => RealScalar::exp(x),
        "exp_m1" => RealScalar::exp_m1(x),
        "ln" => RealScalar::ln(x),
        "ln_1p" => RealScalar::ln_1p(x),
        "sin" => RealScalar::sin(x),
        "cos" => RealScalar::cos(x),
        "tanh" => RealScalar::tanh(x),
        "powf" => RealScalar::powf(x, y),
        other => panic!("{other}"),
    }
}

/// Evaluates every input with `W` lanes after rotating the inputs by `offset`,
/// so each row lands in a different lane position; the final partial array is
/// padded with NaN, which takes the scalar route.
fn evaluate<const W: usize>(operation: &str, inputs: &[(f64, f64)], offset: usize) -> Vec<f64> {
    let n = inputs.len();
    let mut results = vec![f64::NAN; n];
    for start in (0..n).step_by(W) {
        let mut x = [f64::NAN; W];
        let mut y = [f64::NAN; W];
        for lane in 0..W.min(n - start) {
            (x[lane], y[lane]) = inputs[(start + lane + offset) % n];
        }
        let out = lanes(operation, x, y);
        for lane in 0..W.min(n - start) {
            results[(start + lane + offset) % n] = out[lane];
        }
    }
    results
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
}

fn decode(word: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(word, 16).unwrap())
}

/// Checks every binary64 row of `data` (`width op input0 input1 expected`, hex
/// bits) with the elementary envelope: NaN, exact zeros and infinities, and
/// otherwise the same class and sign within 4 ulps.
fn check_fixture(name: &str, data: &str) {
    let mut failures = Vec::new();
    for operation in OPERATIONS {
        let rows: Vec<_> = data
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split_whitespace().collect::<Vec<_>>())
            .filter(|row| row[0] == "64" && row[1] == operation)
            .map(|row| (decode(row[2]), decode(row[3]), decode(row[4])))
            .collect();
        let inputs: Vec<_> = rows.iter().map(|&(x, y, _)| (x, y)).collect();
        let reference = evaluate::<1>(operation, &inputs, 0);
        let mut maximum = 0;
        for (&(x, y, expected), &actual) in rows.iter().zip(&reference) {
            let distance = actual.to_bits().abs_diff(expected.to_bits());
            let valid = if expected.is_nan() {
                actual.is_nan()
            } else if !expected.is_finite() || expected == 0.0 {
                actual.to_bits() == expected.to_bits()
            } else {
                actual.classify() == expected.classify()
                    && actual.is_sign_negative() == expected.is_sign_negative()
                    && distance <= 4
            };
            if expected.is_finite() && actual.is_finite() {
                maximum = maximum.max(distance);
            }
            if !valid {
                failures.push(format!(
                    "{operation}({x:e}, {y:e}) = {actual:e}, want {expected:e} ({distance} ulp)"
                ));
            }
        }
        for offset in 0..16 {
            let widths = [
                evaluate::<2>(operation, &inputs, offset % 2),
                evaluate::<4>(operation, &inputs, offset % 4),
                evaluate::<8>(operation, &inputs, offset % 8),
                evaluate::<16>(operation, &inputs, offset),
            ];
            for results in &widths {
                for ((&(x, y), &expected), &actual) in inputs.iter().zip(&reference).zip(results) {
                    assert!(
                        same_bits(actual, expected),
                        "{operation}({x:e}, {y:e}): {actual:e} differs from one lane {expected:e}"
                    );
                }
            }
        }
        assert!(!rows.is_empty(), "{name}: no {operation} rows");
        eprintln!("{name} {operation}: {} rows, max {maximum} ulp", rows.len());
    }
    assert!(
        failures.is_empty(),
        "{name}: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(25)].join("\n")
    );
}

#[test]
fn elementary_rows_for_every_width() {
    check_fixture("elementary", include_str!("data/elementary.txt"));
}

#[test]
fn selection_boundary_rows_for_every_width() {
    check_fixture("lane_kernels", include_str!("data/lane_kernels.txt"));
}

#[test]
fn ieee_limits() {
    let zero = 0.0_f64;
    let negative_zero = -0.0_f64;
    let (infinity, nan) = (f64::INFINITY, f64::NAN);
    let bits = |values: [f64; 4]| values.map(f64::to_bits);
    assert_eq!(
        bits(lanes_exp([-infinity, infinity, zero, negative_zero])),
        bits([0.0, infinity, 1.0, 1.0])
    );
    assert_eq!(
        bits(lanes_exp_m1([-infinity, infinity, zero, negative_zero])),
        bits([-1.0, infinity, zero, negative_zero])
    );
    assert_eq!(
        bits(lanes_ln([zero, negative_zero, infinity, 1.0])),
        bits([-infinity, -infinity, infinity, zero])
    );
    assert_eq!(
        bits(lanes_ln_1p([-1.0, negative_zero, infinity, zero])),
        bits([-infinity, negative_zero, infinity, zero])
    );
    assert_eq!(
        bits(lanes_sin([zero, negative_zero, 1e-310, -1e-310])),
        bits([zero, negative_zero, 1e-310, -1e-310])
    );
    assert_eq!(
        bits(lanes_cos([zero, negative_zero, 1e-310, -1e-310])),
        bits([1.0; 4])
    );
    assert_eq!(
        bits(lanes_tanh([-infinity, infinity, zero, negative_zero])),
        bits([-1.0, 1.0, zero, negative_zero])
    );
    assert_eq!(
        bits(lanes_tanh([1e-310, -1e-310, 30.0, -19.0])),
        bits([1e-310, -1e-310, 1.0, -19.0_f64.tanh()])
    );
    for result in [
        lanes_exp([nan]),
        lanes_exp_m1([nan]),
        lanes_ln([nan]),
        lanes_ln([-1.0]),
        lanes_ln([-infinity]),
        lanes_ln_1p([nan]),
        lanes_ln_1p([-2.0]),
        lanes_sin([nan]),
        lanes_sin([infinity]),
        lanes_cos([-infinity]),
        lanes_tanh([nan]),
        lanes_powf([nan], [2.0]),
        lanes_powf([-2.0], [0.5]),
        lanes_powf([2.0], [nan]),
    ] {
        assert!(result[0].is_nan());
    }
}

#[test]
fn power_rules_match_the_scalar_function() {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        3.0,
        -3.0,
        1e-310,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut bases = Vec::new();
    let mut exponents = Vec::new();
    for &a in &specials {
        for &b in &specials {
            bases.push(a);
            exponents.push(b);
        }
    }
    let inputs: Vec<_> = bases.iter().copied().zip(exponents).collect();
    for offset in 0..8 {
        for (&(a, b), &actual) in inputs.iter().zip(&evaluate::<8>("powf", &inputs, offset)) {
            let expected = a.powf(b);
            if expected.is_nan() || expected == 0.0 || expected.is_infinite() || expected == 1.0 {
                assert!(
                    same_bits(actual, expected),
                    "powf({a:e}, {b:e}) = {actual:e}"
                );
            } else {
                assert!(
                    actual.to_bits().abs_diff(expected.to_bits()) <= 4,
                    "powf({a:e}, {b:e}) = {actual:e}, want {expected:e}"
                );
            }
        }
    }
    assert_eq!(
        lanes_powf([2.0; 3], [-1074.0, -1075.0, 1024.0]),
        [f64::from_bits(1), 0.0, f64::INFINITY]
    );
    assert_eq!(lanes_powf([-2.0, 2.0], [3.0, 10.0]), [-8.0, 1024.0]);
}

#[test]
fn lanes_outside_each_domain_take_the_scalar_route() {
    let next_up = |x: f64| f64::from_bits(x.to_bits() + 1);
    let next_down = |x: f64| f64::from_bits(x.to_bits() - 1);
    let limit = 1_048_576.0;
    let cases: [(&str, [f64; 4], [f64; 4]); 7] = [
        (
            "ln",
            [
                f64::MIN_POSITIVE,
                next_down(f64::MIN_POSITIVE),
                1e-320,
                f64::MAX,
            ],
            [0.0; 4],
        ),
        (
            "ln_1p",
            [-next_down(1.0), -1.0, -next_up(1.0), f64::MAX],
            [0.0; 4],
        ),
        (
            "sin",
            [limit, next_up(limit), -next_up(limit), 1e22],
            [0.0; 4],
        ),
        (
            "cos",
            [-limit, -next_up(limit), next_up(limit), 1e22],
            [0.0; 4],
        ),
        ("exp", [745.0, f64::MAX, -f64::MAX, -745.2], [0.0; 4]),
        ("exp_m1", [746.0, f64::MAX, -f64::MAX, -746.0], [0.0; 4]),
        (
            "powf",
            [1.0 + f64::EPSILON, 1.0, 0.5, 0.5],
            [
                2f64.powi(512),
                next_up(2f64.powi(512)),
                -next_up(2f64.powi(512)),
                -2f64.powi(512),
            ],
        ),
    ];
    for (operation, x, y) in cases {
        let actual = lanes(operation, x, y);
        for lane in 0..4 {
            let expected = scalar(operation, x[lane], y[lane]);
            assert!(
                same_bits(actual[lane], expected)
                    || actual[lane].to_bits().abs_diff(expected.to_bits()) <= 4,
                "{operation}({:e}, {:e}) = {:e}, want {expected:e}",
                x[lane],
                y[lane],
                actual[lane]
            );
        }
    }
    // A lane just outside a domain returns exactly the scalar result.
    for x in [next_up(limit), 1e22, f64::INFINITY] {
        assert!(same_bits(lanes_sin([0.5, x])[1], x.sin()));
        assert!(same_bits(lanes_cos([x, 0.5])[0], x.cos()));
    }
    for x in [next_down(f64::MIN_POSITIVE), f64::from_bits(1), -1.0] {
        assert!(same_bits(lanes_ln([2.0, x])[1], x.ln()));
    }
    let huge = next_up(2f64.powi(512));
    assert!(same_bits(lanes_powf([0.5, 1.0], [1.0, huge])[1], 1.0));
    assert!(same_bits(
        lanes_powf([-8.0, 0.5], [1.0 / 3.0, 1.0])[0],
        (-8.0_f64).powf(1.0 / 3.0)
    ));
}
