//! Taylor coefficients against the independent mpmath references in `tests/data/`.
#![allow(clippy::unwrap_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;

use scioxide_ad::{Jet, Real};
use scioxide_core::RealScalar;
use support::{bits, decode, moderate, rows};

/// Coefficients through order 8: nine directional, 45 mixed.
const ORDER: usize = 8;
const WIDTH: usize = ORDER + 1;

/// The operation of a row; `y` is the second operand, or the exponent of `powi`.
fn operation<S: RealScalar, R: Real<S>>(name: &str, x: R, y: R, exponent: i32) -> R {
    let name = name.strip_suffix("_affine").unwrap_or(name);
    match name {
        "exp" => x.exp(),
        "expm1" => x.exp_m1(),
        "ln" => x.ln(),
        "ln1p" => x.ln_1p(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "sqrt" => x.sqrt(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "recip" => x.recip(),
        "tanh" => x.tanh(),
        "atan" => x.atan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        "exp2" => x.exp2(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "cbrt" => x.cbrt(),
        "sigmoid" => x.sigmoid(),
        "softplus" => x.softplus(),
        "log_sigmoid" => x.log_sigmoid(),
        "logit" => x.logit(),
        "sin_pi" => x.sin_pi(),
        "cos_pi" => x.cos_pi(),
        "tan_pi" => x.tan_pi(),
        "sin_degrees" => x.sin_degrees(),
        "cos_degrees" => x.cos_degrees(),
        "tan_degrees" => x.tan_degrees(),
        "powi" => x.powi(exponent),
        "powf" | "powf_constant" => x.powf(y),
        "mul" => x * y,
        "div" => x / y,
        "hypot" => x.hypot(y),
        "logaddexp" | "logaddexp_masked" => x.logaddexp(y),
        _ => panic!("unknown operation {name}"),
    }
}

/// Whether the second operand varies with the expansion variables.
fn binary(name: &str) -> bool {
    matches!(
        name,
        "powf" | "logaddexp" | "hypot" | "mul" | "div" | "logaddexp_masked_affine"
    )
}

/// The input perturbations of the generator, `(i, j) → weight` with `i` the outer
/// (first) and `j` the inner (second) variable; the coefficient is `scale·weight`.
fn perturbation(name: &str, mixed: bool, second: bool) -> &'static [((usize, usize), f64)] {
    if name.ends_with("_affine") {
        return &[((1, 0), 1.0)];
    }
    match (mixed, second) {
        (false, false) => &[((1, 0), 0.5), ((2, 0), 0.125), ((3, 0), -0.0625)],
        (false, true) => &[((1, 0), -0.25), ((2, 0), 0.0625), ((3, 0), 0.03125)],
        (true, false) => &[
            ((1, 0), 0.5),
            ((0, 1), 0.25),
            ((1, 1), 0.125),
            ((2, 0), 0.0625),
        ],
        (true, true) => &[
            ((1, 0), -0.25),
            ((0, 1), 0.5),
            ((1, 1), -0.0625),
            ((0, 2), 0.125),
        ],
    }
}

/// The input polynomial as a `WIDTH × WIDTH` grid of coefficients.
fn polynomial<S: RealScalar>(
    name: &str,
    mixed: bool,
    second: bool,
    value: S,
    scale: S,
) -> [[S; WIDTH]; WIDTH] {
    let mut grid = [[S::ZERO; WIDTH]; WIDTH];
    grid[0][0] = value;
    if !second || binary(name) {
        for &((i, j), weight) in perturbation(name, mixed, second) {
            grid[i][j] = scale * S::from_f64_ieee(weight);
        }
    }
    grid
}

/// The directional coefficients through a jet of `N` coefficients.
fn directional<S: RealScalar, const N: usize>(
    name: &str,
    x: &[[S; WIDTH]; WIDTH],
    y: &[[S; WIDTH]; WIDTH],
    exponent: i32,
) -> [S; N] {
    let jet = |grid: &[[S; WIDTH]; WIDTH]| Jet::<S, N>::new(core::array::from_fn(|k| grid[k][0]));
    operation(name, jet(x), jet(y), exponent).coefficients()
}

/// Asserts that a shorter jet reproduces the leading coefficients bit for bit.
fn prefix<S: RealScalar, const N: usize>(
    name: &str,
    x: &[[S; WIDTH]; WIDTH],
    y: &[[S; WIDTH]; WIDTH],
    exponent: i32,
    full: &[S; WIDTH],
    row: &str,
) {
    let short = directional::<S, N>(name, x, y, exponent);
    for (k, (&a, &b)) in short.iter().zip(full).enumerate() {
        assert!(
            bits(a) == bits(b) || (a.is_nan() && b.is_nan()),
            "{row}: N = {N}, coefficient {k}: {a:?} != {b:?}"
        );
    }
}

/// The mixed coefficients `c[i][j]` through nested jets.
fn mixed<S: RealScalar>(
    name: &str,
    x: &[[S; WIDTH]; WIDTH],
    y: &[[S; WIDTH]; WIDTH],
    exponent: i32,
) -> [[S; WIDTH]; WIDTH] {
    let jet = |grid: &[[S; WIDTH]; WIDTH]| {
        Jet::<Jet<S, WIDTH>, WIDTH>::new(core::array::from_fn(|i| Jet::new(grid[i])))
    };
    operation(name, jet(x), jet(y), exponent)
        .coefficients()
        .map(Jet::coefficients)
}

/// Error relative to `64 ε (k + 1) g + MIN_SUBNORMAL`; at most 1 passes.
fn normalized<S: RealScalar>(actual: S, expected: S, growth: S, order: usize) -> f64 {
    let bound = 64.0 * S::EPSILON.into_f64() * (order + 1) as f64 * growth.into_f64()
        + S::MIN_SUBNORMAL.into_f64();
    (actual.into_f64() - expected.into_f64()).abs() / bound
}

/// Checks one coefficient: non-finite expectations exactly, finite ones within the
/// bound when the row is `bounded`. Returns the normalized error.
fn coefficient<S: RealScalar>(
    actual: S,
    (expected, growth): (S, S),
    order: usize,
    bounded: bool,
    row: &str,
) -> f64 {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{row}: order {order}: {actual:?}, expected NaN"
        );
        return 0.0;
    }
    if !expected.is_finite() {
        assert_eq!(actual, expected, "{row}: order {order}");
        return 0.0;
    }
    if !bounded {
        return 0.0;
    }
    let error = normalized(actual, expected, growth, order);
    assert!(
        error <= 1.0,
        "{row}: order {order}: {actual:?} != {expected:?}, growth {growth:?}, {error} bounds"
    );
    error
}

/// Asserts every row of `width` bits; returns the worst normalized error per
/// operation and the number of bounded rows per operation and mode.
///
/// Every row checks `c₀` against the primal bit for bit. Rows whose arguments are
/// moderate check non-finite expectations exactly; rows whose arguments and
/// expected coefficients are all moderate check the bound.
#[allow(clippy::type_complexity)]
fn references<S: RealScalar>(
    width: usize,
) -> (BTreeMap<String, f64>, BTreeMap<(String, bool), usize>) {
    let mut worst = BTreeMap::<String, f64>::new();
    let mut coverage = BTreeMap::<(String, bool), usize>::new();
    for (row, fields) in rows(include_str!("data/taylor_coefficients.txt")) {
        if fields[0].parse::<usize>().unwrap() != width {
            continue;
        }
        let name = fields[1];
        let is_mixed = fields[2] == "M";
        let (x0, y0, scale) = (
            decode::<S>(fields[3], width),
            decode::<S>(fields[4], width),
            decode::<S>(fields[5], width),
        );
        let exponent = y0.into_f64() as i32;
        let references: Vec<(S, S)> = fields[6..]
            .iter()
            .map(|reference| {
                let (expected, growth) = reference.split_once(':').unwrap();
                (decode(expected, width), decode(growth, width))
            })
            .collect();
        let x = polynomial(name, is_mixed, false, x0, scale);
        let y = polynomial(name, is_mixed, true, y0, scale);
        let primal = bits(operation(name, x0, y0, exponent));
        let checked = moderate(&[x0, y0, scale]);
        let expected: Vec<S> = references.iter().map(|&(e, _)| e).collect();
        let bounded = checked && moderate(&expected);
        let mut row_worst = 0.0_f64;
        if is_mixed {
            assert_eq!(references.len(), WIDTH * (WIDTH + 1) / 2, "{row}");
            let c = mixed(name, &x, &y, exponent);
            assert_eq!(bits(c[0][0]), primal, "{row}");
            let indices = (0..=ORDER).flat_map(|n| (0..=n).map(move |i| (i, n - i)));
            for ((i, j), &reference) in indices.zip(&references).skip(1) {
                if checked {
                    let error = coefficient(c[i][j], reference, i + j, bounded, row);
                    row_worst = row_worst.max(error);
                }
            }
        } else {
            assert_eq!(references.len(), WIDTH, "{row}");
            let c = directional::<S, WIDTH>(name, &x, &y, exponent);
            assert_eq!(bits(c[0]), primal, "{row}");
            prefix::<S, 1>(name, &x, &y, exponent, &c, row);
            prefix::<S, 2>(name, &x, &y, exponent, &c, row);
            prefix::<S, 3>(name, &x, &y, exponent, &c, row);
            prefix::<S, 4>(name, &x, &y, exponent, &c, row);
            prefix::<S, 5>(name, &x, &y, exponent, &c, row);
            prefix::<S, 6>(name, &x, &y, exponent, &c, row);
            prefix::<S, 7>(name, &x, &y, exponent, &c, row);
            prefix::<S, 8>(name, &x, &y, exponent, &c, row);
            for (k, (&actual, &reference)) in c.iter().zip(&references).enumerate().skip(1) {
                if checked {
                    row_worst = row_worst.max(coefficient(actual, reference, k, bounded, row));
                }
            }
        }
        if bounded {
            *coverage.entry((name.to_owned(), is_mixed)).or_default() += 1;
            let entry = worst.entry(name.to_owned()).or_default();
            *entry = entry.max(row_worst);
        }
    }
    (worst, coverage)
}

const OPERATIONS: [&str; 39] = [
    "exp",
    "expm1",
    "ln",
    "ln1p",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "sinh",
    "cosh",
    "recip",
    "powi",
    "mul",
    "div",
    "hypot",
    "tanh",
    "atan",
    "asin",
    "acos",
    "asinh",
    "acosh",
    "atanh",
    "exp2",
    "log2",
    "log10",
    "cbrt",
    "sigmoid",
    "softplus",
    "log_sigmoid",
    "logit",
    "sin_pi",
    "cos_pi",
    "tan_pi",
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
    "powf_constant",
    "powf",
    "logaddexp",
];

/// Every operation has bounded rows in both modes: at least three in `f64`, two in
/// `f32`, whose small range excludes more rows with tiny coefficients.
fn check<S: RealScalar>(width: usize) {
    let (worst, coverage) = references::<S>(width);
    let minimum = if width == 64 { 3 } else { 2 };
    for name in OPERATIONS {
        for is_mixed in [false, true] {
            let rows = coverage
                .get(&(name.to_owned(), is_mixed))
                .copied()
                .unwrap_or(0);
            assert!(
                rows >= minimum,
                "binary{width} {name} mixed={is_mixed}: {rows} rows"
            );
        }
    }
    assert!(worst.len() >= OPERATIONS.len());
    assert!(worst.values().all(|&error| error <= 1.0));
}

#[test]
fn directional_and_mixed_coefficients_match_references_binary64() {
    check::<f64>(64);
}

#[test]
fn directional_and_mixed_coefficients_match_references_binary32() {
    check::<f32>(32);
}
