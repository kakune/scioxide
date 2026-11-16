//! Plain functions return the ordinary value of their detailed counterparts
//! bit for bit on the checked-in oracle inputs, and fail with the same
//! typed error.
//!
//! Every registered pair must see at least one input. Derived functions
//! without their own oracle rows use the rows of the function they derive
//! from. Rows are evaluated at their own width.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::collections::BTreeMap;

use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_special::*;

const TIERED: [&str; 13] = [
    include_str!("data/special_oracle.tsv"),
    include_str!("data/airy_reference.tsv"),
    include_str!("data/bessel_reference.tsv"),
    include_str!("data/coulomb_dense.tsv"),
    include_str!("data/large_argument_bands.tsv"),
    include_str!("data/gamma_beta_reference.tsv"),
    include_str!("data/zeta_reference.tsv"),
    include_str!("data/complex_zeta_reference.tsv"),
    include_str!("data/legendre_reference.tsv"),
    include_str!("data/erfc_reference.tsv"),
    include_str!("data/hypergeometric_reference.tsv"),
    include_str!("data/gamma_regions.tsv"),
    include_str!("data/elliptic_measured.tsv"),
];
const CONDITIONS: [&str; 7] = [
    include_str!("data/airy_conditions.tsv"),
    include_str!("data/bessel_conditions.tsv"),
    include_str!("data/coulomb_conditions.tsv"),
    include_str!("data/gamma_beta_conditions.tsv"),
    include_str!("data/hypergeometric_conditions.tsv"),
    include_str!("data/legendre_conditions.tsv"),
    include_str!("data/special_conditions.tsv"),
];
const SCALAR_FUNCTION: [&str; 6] = [
    include_str!("data/probability_conditioned.tsv"),
    include_str!("data/probability_inverse.tsv"),
    include_str!("data/probability_normal.tsv"),
    include_str!("data/probability_normal_boundaries.tsv"),
    include_str!("data/zeta_oracle.tsv"),
    include_str!("data/transport_oracle.tsv"),
];
/// `width kind decimal...` files and their argument counts.
const DECIMAL: [(&str, &str, usize); 10] = [
    ("airy", include_str!("data/airy_real.tsv"), 1),
    ("airy_complex", include_str!("data/airy_complex.tsv"), 2),
    (
        "complex_carlson",
        include_str!("data/complex_carlson.tsv"),
        8,
    ),
    (
        "complex_elliptic",
        include_str!("data/complex_elliptic.tsv"),
        4,
    ),
    ("complex_error", include_str!("data/complex_error.tsv"), 2),
    ("complex_gamma", include_str!("data/complex_gamma.tsv"), 4),
    ("complex_jacobi", include_str!("data/complex_jacobi.tsv"), 4),
    (
        "complex_order_bessel",
        include_str!("data/complex_order_bessel.tsv"),
        4,
    ),
    ("probability", include_str!("data/probability.tsv"), 3),
    (
        "lambert_polylog",
        include_str!("data/lambert_polylog.tsv"),
        7,
    ),
];
/// `kind width bits...` files and their argument counts.
const KIND_BITS: [(&str, &str, usize); 6] = [
    (
        "bessel_integrals",
        include_str!("data/bessel_integrals.tsv"),
        1,
    ),
    ("complex_bessel", include_str!("data/complex_bessel.tsv"), 3),
    ("parabolic", include_str!("data/parabolic.tsv"), 2),
    ("struve", include_str!("data/struve.tsv"), 2),
    ("struve", include_str!("data/struve_reference.tsv"), 2),
    ("bessel_turning", include_str!("data/bessel_turning.tsv"), 3),
];
/// `kind width order bits` files.
const KIND_ORDER_BITS: [(&str, &str); 3] = [
    ("riccati", include_str!("data/riccati.tsv")),
    ("spherical", include_str!("data/spherical.tsv")),
    ("integrals", include_str!("data/integrals.tsv")),
];

fn lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
}

fn width(token: &str) -> u32 {
    match token {
        "32" | "f32" => 32,
        "64" | "f64" => 64,
        other => panic!("unknown width {other}"),
    }
}

fn decode(width: u32, hex: &str) -> f64 {
    let bits = u64::from_str_radix(hex, 16).unwrap();
    if width == 32 {
        f64::from(f32::from_bits(u32::try_from(bits).unwrap()))
    } else {
        f64::from_bits(bits)
    }
}

fn prefixed(width: u32, argument: &str) -> f64 {
    let (kind, value) = argument.split_once(':').unwrap();
    match kind {
        "f" => decode(width, value),
        "u" | "i" => value.parse().unwrap(),
        _ => panic!("unknown argument kind {kind}"),
    }
}

fn complex_list(text: &str) -> Vec<f64> {
    text.split(';')
        .filter(|item| !item.is_empty() && *item != "-")
        .flat_map(|item| item.split(',').map(|v| v.parse::<f64>().unwrap()))
        .collect()
}

/// Splits interleaved complex coordinates into a real row when every
/// imaginary part vanishes.
fn split(key: String, width: u32, pairs: Vec<f64>, prefix: Vec<f64>, inputs: &mut Inputs) {
    if pairs.iter().skip(1).step_by(2).all(|&im| im == 0.0) {
        let mut row = prefix;
        row.extend(pairs.iter().step_by(2));
        inputs.add(key, width, row);
    } else {
        let mut row = prefix;
        row.extend(pairs);
        inputs.add(format!("{key}:complex"), width, row);
    }
}

/// Oracle inputs by family key, as exact binary64 arguments.
#[derive(Default)]
struct Inputs(BTreeMap<String, Vec<(u32, Vec<f64>)>>);

impl Inputs {
    fn add(&mut self, key: impl Into<String>, width: u32, row: Vec<f64>) {
        self.0.entry(key.into()).or_default().push((width, row));
    }

    fn rows<S: RealScalar>(&self, keys: &[&str]) -> Vec<&[f64]> {
        let width = if S::MANTISSA_DIGITS == 24 { 32 } else { 64 };
        keys.iter()
            .flat_map(|key| {
                assert!(self.0.contains_key(*key), "no oracle rows for {key}");
                &self.0[*key]
            })
            .filter(|(w, _)| *w == width)
            .map(|(_, row)| row.as_slice())
            .collect()
    }

    fn load() -> Self {
        let mut inputs = Self::default();
        for f in TIERED.into_iter().flat_map(lines) {
            let w = width(f[1]);
            inputs.add(f[2], w, f[3].split(',').map(|a| prefixed(w, a)).collect());
        }
        for f in CONDITIONS.into_iter().flat_map(lines) {
            let w = width(f[0]);
            inputs.add(f[1], w, f[2].split(',').map(|a| prefixed(w, a)).collect());
        }
        for f in SCALAR_FUNCTION.into_iter().flat_map(lines) {
            let w = width(f[0]);
            inputs.add(f[1], w, f[2].split(',').map(|a| decode(w, a)).collect());
        }
        for (name, text, count) in DECIMAL {
            for f in lines(text) {
                let row = f[2..2 + count].iter().map(|v| v.parse().unwrap()).collect();
                inputs.add(format!("{name}:{}", f[1]), width(f[0]), row);
            }
        }
        for (name, text, count) in KIND_BITS {
            for f in lines(text) {
                let w = width(f[1]);
                let row = f[2..2 + count].iter().map(|v| decode(w, v)).collect();
                inputs.add(format!("{name}:{}", f[0]), w, row);
            }
        }
        for (name, text) in KIND_ORDER_BITS {
            for f in lines(text) {
                let w = width(f[1]);
                let row = vec![f[2].parse().unwrap(), decode(w, f[3])];
                inputs.add(format!("{name}:{}", f[0]), w, row);
            }
        }
        for f in lines(include_str!("data/h3d.tsv")) {
            let w = width(f[0]);
            let row = vec![f[1].parse().unwrap(), decode(w, f[2]), decode(w, f[3])];
            inputs.add("h3d", w, row);
        }
        for f in lines(include_str!("data/kelvin.tsv")) {
            let w = width(f[0]);
            inputs.add(format!("kelvin:{}", f[1]), w, vec![decode(w, f[2])]);
        }
        for f in lines(include_str!("data/weighted_bessel.tsv")) {
            let w = width(f[0]);
            let row = f[1..4].iter().map(|v| decode(w, v)).collect();
            inputs.add("weighted_bessel", w, row);
        }
        for f in lines(include_str!("data/thermal_oracle.tsv")) {
            let w = width(f[0]);
            let row = vec![f[2].parse().unwrap(), decode(w, f[3]), decode(w, f[4])];
            inputs.add(format!("thermal:{}:{}", f[1], f[2]), w, row);
        }
        for f in lines(include_str!("data/combinatorial.tsv")) {
            let w = width(f[0]);
            let prefix = vec![f[2].parse().unwrap(), f[3].parse().unwrap()];
            let pairs = f[4..8].iter().map(|v| decode(w, v)).collect();
            split(
                format!("combinatorial:{}", f[1]),
                w,
                pairs,
                prefix,
                &mut inputs,
            );
        }
        for f in lines(include_str!("data/lambert_branch_point.tsv")) {
            let w = width(f[0]);
            let row = vec![f[2].parse().unwrap(), decode(w, f[3]), decode(w, f[4])];
            inputs.add(format!("lambert_branch:{}", f[1]), w, row);
        }
        for f in lines(include_str!("data/ellipsoidal.tsv")) {
            let row = f[1..6].iter().map(|v| v.parse().unwrap()).collect();
            inputs.add("ellipsoidal", width(f[0]), row);
        }
        for f in lines(include_str!("data/ellipsoidal_complex.tsv")) {
            let row = f[1..7].iter().map(|v| v.parse().unwrap()).collect();
            inputs.add("ellipsoidal_complex", width(f[0]), row);
        }
        for f in lines(include_str!("data/angular.tsv")) {
            let row = f[2].split(',').map(|v| v.parse().unwrap()).collect();
            inputs.add(format!("angular:{}", f[1]), width(f[0]), row);
        }
        for f in lines(include_str!("data/elliptic.tsv")) {
            let pairs = f[3..7].iter().flat_map(|v| complex_list(v)).collect();
            split(
                format!("elliptic:{}", f[2]),
                width(f[0]),
                pairs,
                Vec::new(),
                &mut inputs,
            );
        }
        for f in lines(include_str!("data/complex_hypergeometric.tsv")) {
            if f[3] != "none" || f[1] == "11" && f[2] == "1" {
                continue;
            }
            let key = format!("hypergeometric:pfq:{}:{}", f[1], f[2]);
            let pairs = [complex_list(f[4]), complex_list(f[5])].concat();
            split(key, width(f[0]), pairs, Vec::new(), &mut inputs);
        }
        for f in lines(include_str!("data/hypergeometric.tsv")) {
            if f[3] != "none" {
                continue;
            }
            let (a, b) = (complex_list(f[4]), complex_list(f[5]));
            let key = format!(
                "hypergeometric:{}:{}{}:{}",
                f[1],
                a.len() / 2,
                b.len() / 2,
                f[2]
            );
            let pairs = [a, b, complex_list(f[6])].concat();
            split(key, width(f[0]), pairs, Vec::new(), &mut inputs);
        }
        for f in lines(include_str!("data/coulomb.tsv")) {
            let w = width(f[1]);
            let n: Vec<f64> = f[2..6].iter().map(|v| v.parse().unwrap()).collect();
            match f[0] {
                "h" | "hp" => inputs.add(
                    "coulomb:h",
                    w,
                    vec![n[0], n[1], f[9].parse().unwrap(), n[2]],
                ),
                "c" => inputs.add("coulomb:c", w, vec![n[0], n[1]]),
                kind => split(
                    format!("coulomb:{kind}"),
                    w,
                    n[2..].to_vec(),
                    n[..2].to_vec(),
                    &mut inputs,
                ),
            }
        }
        for f in lines(include_str!("data/mathieu.tsv")) {
            let w = width(f[1]);
            let n: Vec<f64> = f[3..7].iter().map(|v| v.parse().unwrap()).collect();
            let key = format!("mathieu:{}", f[2]);
            match f[7] {
                "char" => inputs.add(format!("{key}:char"), w, n[..2].to_vec()),
                "angular" => split(
                    format!("{key}:angular"),
                    w,
                    n[2..].to_vec(),
                    n[..2].to_vec(),
                    &mut inputs,
                ),
                radial => split(
                    format!("{key}:radial"),
                    w,
                    n[2..].to_vec(),
                    vec![n[0], n[1], radial_index(radial)],
                    &mut inputs,
                ),
            }
        }
        // Spheroidal rows hold exact decimal inputs; each serves both widths.
        let spheroidal = [
            include_str!("data/spheroidal_mpmath.tsv"),
            include_str!("data/spheroidal_large.tsv"),
        ];
        for f in spheroidal.into_iter().flat_map(lines) {
            let n: Vec<f64> = f[1..6].iter().map(|v| v.parse().unwrap()).collect();
            let prefix = vec![f64::from(f[0] != "pro"), n[0], n[1], n[2]];
            for w in [32, 64] {
                match f[6] {
                    "angular" => split(
                        "spheroidal:angular".into(),
                        w,
                        n[3..].to_vec(),
                        prefix.clone(),
                        &mut inputs,
                    ),
                    radial => split(
                        "spheroidal:radial".into(),
                        w,
                        n[3..].to_vec(),
                        [prefix.as_slice(), &[radial_index(radial)]].concat(),
                        &mut inputs,
                    ),
                }
            }
        }
        for f in lines(include_str!("data/spheroidal_scipy.tsv")) {
            let n = f[1..4].iter().map(|v| v.parse().unwrap());
            let row: Vec<f64> = core::iter::once(f64::from(f[0] != "pro"))
                .chain(n)
                .collect();
            for w in [32, 64] {
                inputs.add("spheroidal:characteristic", w, row.clone());
            }
        }
        inputs
    }
}

/// Maps the radial kinds `r1` to `r4` of the spectral fixtures to `1` to `4`.
fn radial_index(kind: &str) -> f64 {
    kind.strip_prefix('r').unwrap().parse().unwrap()
}

fn bits<V: NumericScalar>(value: V) -> (u64, u64) {
    (
        value.real_component().into_f64().to_bits(),
        value.imaginary_component().into_f64().to_bits(),
    )
}

/// Checks that `plain` is the ordinary projection of `detailed`.
fn agree<V: NumericScalar>(
    row: &[f64],
    plain: Result<V, SciOxideError>,
    detailed: Result<SpecialFunctionResult<V>, SciOxideError>,
) -> Result<(), String> {
    match (plain, detailed.map(|r| r.try_value())) {
        (Ok(plain), Ok(Ok(value))) if bits(plain) == bits(value) => Ok(()),
        (Err(plain), Ok(Err(value))) if format!("{plain:?}") == format!("{value:?}") => Ok(()),
        (Err(plain), Err(detailed)) if plain.kind() == detailed.kind() => Ok(()),
        (plain, detailed) => Err(format!("{row:?}: {plain:?} vs {detailed:?}")),
    }
}

/// Checks that a one-shot returns the bits of its prepared counterpart.
fn same<V: NumericScalar>(
    row: &[f64],
    plain: Result<V, SciOxideError>,
    prepared: Result<V, SciOxideError>,
) -> Result<(), String> {
    match (plain, prepared) {
        (Ok(plain), Ok(prepared)) if bits(plain) == bits(prepared) => Ok(()),
        (Err(plain), Err(prepared)) if plain.kind() == prepared.kind() => Ok(()),
        (plain, prepared) => Err(format!("{row:?}: {plain:?} vs {prepared:?}")),
    }
}

/// Rows checked per pair and the first disagreement of each pair.
#[derive(Default)]
struct Tally {
    rows: BTreeMap<&'static str, usize>,
    failures: BTreeMap<&'static str, (usize, String)>,
}

impl Tally {
    fn register(&mut self, name: &'static str) {
        self.rows.entry(name).or_insert(0);
    }

    fn record(&mut self, name: &'static str, outcome: Result<(), String>) {
        *self.rows.entry(name).or_insert(0) += 1;
        if let Err(message) = outcome {
            self.failures.entry(name).or_insert((0, message)).0 += 1;
        }
    }
}

/// Registers a pair and checks it on every row of `keys` that passes `filter`.
macro_rules! pair {
    ($inputs:ident, $tally:ident, [$($key:literal),+], |$a:ident| $plain:ident $(::<$t:ty>)? ($($arg:expr),*), $detailed:ident) => {
        pair!($inputs, $tally, [$($key),+], |$a| true, $plain $(::<$t>)? ($($arg),*), $detailed)
    };
    ($inputs:ident, $tally:ident, [$($key:literal),+], |$a:ident| $filter:expr, $plain:ident $(::<$t:ty>)? ($($arg:expr),*), $detailed:ident) => {{
        $tally.register(stringify!($plain));
        for $a in $inputs.rows::<S>(&[$($key),+]) {
            if $filter {
                $tally.record(
                    stringify!($plain),
                    agree(
                        $a,
                        $plain $(::<$t>)? ($($arg),*),
                        $detailed $(::<$t>)? ($($arg,)* Default::default()),
                    ),
                );
            }
        }
    }};
}

/// Checks `hyp_pfq` on every `ₚF_q` row whose key ends in `suffix`; `convert` reads one value
/// from the start of a slice of the row.
fn pfq_pair<S: RealScalar, V: NumericScalar>(
    inputs: &Inputs,
    counts: &mut Tally,
    suffix: &str,
    convert: impl Fn(&[f64]) -> V,
) {
    let stride = if suffix.is_empty() { 1 } else { 2 };
    counts.register("hyp_pfq");
    let keys: Vec<_> = inputs
        .0
        .keys()
        .filter(|key| {
            key.strip_prefix("hypergeometric:pfq:")
                .and_then(|rest| rest.strip_suffix(suffix))
                .is_some_and(|rest| rest.len() == 4 && rest.ends_with(":0"))
        })
        .collect();
    for key in keys {
        let (p, q) = key.as_bytes()[19..21]
            .iter()
            .map(|digit| usize::from(digit - b'0'))
            .collect::<Vec<_>>()
            .try_into()
            .map(|[p, q]: [usize; 2]| (p, q))
            .unwrap();
        for row in inputs.rows::<S>(&[key]) {
            assert_eq!(row.len(), (p + q + 1) * stride, "{key}");
            let values: Vec<V> = row.chunks(stride).map(&convert).collect();
            let (z, parameters) = values.split_last().unwrap();
            let (a, b) = parameters.split_at(p);
            counts.record(
                "hyp_pfq",
                agree(
                    row,
                    hyp_pfq(a, b, *z),
                    hyp_pfq_with_options(a, b, *z, Default::default()),
                ),
            );
        }
    }
}

#[allow(clippy::too_many_lines)]
fn check<S: RealScalar>(inputs: &Inputs, counts: &mut Tally)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let r = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let c = |re: f64, im: f64| Complex::new(r(re), r(im));
    let u = |x: f64| x as u32;
    let n = |x: f64| x as usize;
    let i = |x: f64| x as i32;
    let integer = |x: f64| x >= 0.0 && x.fract() == 0.0 && x <= 1e6;

    // Real arguments.
    pair!(
        inputs,
        counts,
        ["airy_ai", "airy:ai"],
        |a| airy_ai(r(a[0])),
        airy_ai_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_ai_scaled", "airy:ai_scaled"],
        |a| airy_ai_scaled(r(a[0])),
        airy_ai_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_bi", "airy:bi"],
        |a| airy_bi(r(a[0])),
        airy_bi_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_bi_scaled", "airy:bi_scaled"],
        |a| airy_bi_scaled(r(a[0])),
        airy_bi_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:0"],
        |a| ber(r(a[0])),
        ber_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:1"],
        |a| bei(r(a[0])),
        bei_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:2"],
        |a| ker(r(a[0])),
        ker_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:3"],
        |a| kei(r(a[0])),
        kei_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:0"],
        |a| berp(r(a[0])),
        berp_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:1"],
        |a| beip(r(a[0])),
        beip_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:2"],
        |a| kerp(r(a[0])),
        kerp_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:3"],
        |a| keip(r(a[0])),
        keip_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:iint"],
        |a| bessel_i0_integral(r(a[0])),
        bessel_i0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:iremainder"],
        |a| bessel_i0_remainder_integral(r(a[0])),
        bessel_i0_remainder_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:jcompl"],
        |a| bessel_j0_complement_integral(r(a[0])),
        bessel_j0_complement_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:jint"],
        |a| bessel_j0_integral(r(a[0])),
        bessel_j0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:kint"],
        |a| bessel_k0_integral(r(a[0])),
        bessel_k0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:ktail"],
        |a| bessel_k0_tail_integral(r(a[0])),
        bessel_k0_tail_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:yint"],
        |a| bessel_y0_integral(r(a[0])),
        bessel_y0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:ytail"],
        |a| bessel_y0_tail_integral(r(a[0])),
        bessel_y0_tail_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:hint"],
        |a| struve_h0_integral(r(a[0])),
        struve_h0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:htail"],
        |a| struve_h0_tail_integral(r(a[0])),
        struve_h0_tail_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_integrals:lint"],
        |a| struve_l0_integral(r(a[0])),
        struve_l0_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:atanint"],
        |a| atanint(r(a[1])),
        atanint_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:chi"],
        |a| chi(r(a[1])),
        chi_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:chi_scaled"],
        |a| chi_scaled(r(a[1])),
        chi_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:ci"],
        |a| ci(r(a[1])),
        ci_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:dawson"],
        |a| dawson(r(a[1])),
        dawson_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_3"],
        |a| expint_3(r(a[1])),
        expint_3_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_ei"],
        |a| expint_ei(r(a[1])),
        expint_ei_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_ei_scaled"],
        |a| expint_ei_scaled(r(a[1])),
        expint_ei_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en"],
        |a| expint_e1(r(a[1])),
        expint_e1_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en_scaled"],
        |a| expint_e1_scaled(r(a[1])),
        expint_e1_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en"],
        |a| expint_e2(r(a[1])),
        expint_e2_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en_scaled"],
        |a| expint_e2_scaled(r(a[1])),
        expint_e2_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:fresnel_c"],
        |a| fresnel_c(r(a[1])),
        fresnel_c_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:fresnel_s"],
        |a| fresnel_s(r(a[1])),
        fresnel_s_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:logarithmic_integral"],
        |a| logarithmic_integral(r(a[1])),
        logarithmic_integral_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:logarithmic_integral_offset"],
        |a| logarithmic_integral_offset(r(a[1])),
        logarithmic_integral_offset_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:shi"],
        |a| shi(r(a[1])),
        shi_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:shi_scaled"],
        |a| shi_scaled(r(a[1])),
        shi_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:si"],
        |a| si(r(a[1])),
        si_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en"],
        |a| expint_en(u(a[0]), r(a[1])),
        expint_en_with_options
    );
    pair!(
        inputs,
        counts,
        ["integrals:expint_en_scaled"],
        |a| expint_en_scaled(u(a[0]), r(a[1])),
        expint_en_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["clausen"],
        |a| clausen(r(a[0])),
        clausen_with_options
    );
    pair!(
        inputs,
        counts,
        ["synchrotron_1"],
        |a| synchrotron_1(r(a[0])),
        synchrotron_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["synchrotron_1_scaled"],
        |a| synchrotron_1_scaled(r(a[0])),
        synchrotron_1_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["synchrotron_2"],
        |a| synchrotron_2(r(a[0])),
        synchrotron_2_with_options
    );
    pair!(
        inputs,
        counts,
        ["synchrotron_2_scaled"],
        |a| synchrotron_2_scaled(r(a[0])),
        synchrotron_2_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["transport_2"],
        |a| transport_2(r(a[0])),
        transport_2_with_options
    );
    pair!(
        inputs,
        counts,
        ["transport_3"],
        |a| transport_3(r(a[0])),
        transport_3_with_options
    );
    pair!(
        inputs,
        counts,
        ["transport_4"],
        |a| transport_4(r(a[0])),
        transport_4_with_options
    );
    pair!(
        inputs,
        counts,
        ["transport_5"],
        |a| transport_5(r(a[0])),
        transport_5_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:1"],
        |a| debye_1(r(a[1])),
        debye_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:2"],
        |a| debye_2(r(a[1])),
        debye_2_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:3"],
        |a| debye_3(r(a[1])),
        debye_3_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:4"],
        |a| debye_4(r(a[1])),
        debye_4_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:5"],
        |a| debye_5(r(a[1])),
        debye_5_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:0:6"],
        |a| debye_6(r(a[1])),
        debye_6_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:-1"],
        |a| fermi_dirac_m1(r(a[1])),
        fermi_dirac_m1_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:0"],
        |a| fermi_dirac_0(r(a[1])),
        fermi_dirac_0_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:1"],
        |a| fermi_dirac_1(r(a[1])),
        fermi_dirac_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:2"],
        |a| fermi_dirac_2(r(a[1])),
        fermi_dirac_2_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:-0.5"],
        |a| fermi_dirac_mhalf(r(a[1])),
        fermi_dirac_mhalf_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:0.5"],
        |a| fermi_dirac_half(r(a[1])),
        fermi_dirac_half_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:1:1.5"],
        |a| fermi_dirac_3half(r(a[1])),
        fermi_dirac_3half_with_options
    );
    pair!(
        inputs,
        counts,
        [
            "thermal:1:-101",
            "thermal:1:-31",
            "thermal:1:-10",
            "thermal:1:-4",
            "thermal:1:-3",
            "thermal:1:-2",
            "thermal:1:-1",
            "thermal:1:0",
            "thermal:1:1",
            "thermal:1:2",
            "thermal:1:3",
            "thermal:1:4",
            "thermal:1:10",
            "thermal:1:31",
            "thermal:1:100"
        ],
        |a| fermi_dirac_int(i(a[0]), r(a[1])),
        fermi_dirac_int_with_options
    );
    pair!(
        inputs,
        counts,
        ["thermal:2:0"],
        |a| fermi_dirac_inc_0(r(a[1]), r(a[2])),
        fermi_dirac_inc_0_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma"],
        |a| gamma(r(a[0])),
        gamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["ln_gamma"],
        |a| ln_gamma(r(a[0])),
        ln_gamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["digamma"],
        |a| digamma(r(a[0])),
        digamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["polygamma"],
        |a| polygamma(n(a[0]), r(a[1])),
        polygamma_with_options
    );
    pair!(inputs, counts, ["erf"], |a| erf(r(a[0])), erf_with_options);
    pair!(
        inputs,
        counts,
        ["erfc"],
        |a| erfc(r(a[0])),
        erfc_with_options
    );
    pair!(
        inputs,
        counts,
        ["erfinv"],
        |a| erfinv(r(a[0])),
        erfinv_with_options
    );
    pair!(
        inputs,
        counts,
        ["erfcinv"],
        |a| erfcinv(r(a[0])),
        erfcinv_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:erfcx"],
        |a| erfcx(r(a[0])),
        erfcx_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:erfi"],
        |a| erfi(r(a[0])),
        erfi_with_options
    );

    pair!(
        inputs,
        counts,
        ["probability:log_erfc"],
        |a| log_erfc(r(a[0])),
        log_erfc_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_cdf"],
        |a| normal_cdf(r(a[0])),
        normal_cdf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_sf"],
        |a| normal_sf(r(a[0])),
        normal_sf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_log_cdf", "probability:normal_log_cdf"],
        |a| normal_log_cdf(r(a[0])),
        normal_log_cdf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_log_sf"],
        |a| normal_log_sf(r(a[0])),
        normal_log_sf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_quantile"],
        |a| normal_quantile(r(a[0])),
        normal_quantile_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_isf"],
        |a| normal_isf(r(a[0])),
        normal_isf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_inv_log_cdf", "probability:normal_inv_log_cdf"],
        |a| normal_inv_log_cdf(r(a[0])),
        normal_inv_log_cdf_with_options
    );
    pair!(
        inputs,
        counts,
        ["normal_inv_log_sf"],
        |a| normal_inv_log_sf(r(a[0])),
        normal_inv_log_sf_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:owens_t"],
        |a| owens_t(r(a[0]), r(a[1])),
        owens_t_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:voigt_profile"],
        |a| voigt_profile(r(a[0]), r(a[1]), r(a[2])),
        voigt_profile_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:voigt_profile"],
        |a| voigt_complex(r(a[0]), r(a[1]), r(a[2])),
        voigt_complex_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_lower"],
        |a| gamma_lower(r(a[0]), r(a[1])),
        gamma_lower_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_upper"],
        |a| gamma_upper(r(a[0]), r(a[1])),
        gamma_upper_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_p"],
        |a| gamma_p(r(a[0]), r(a[1])),
        gamma_p_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_q"],
        |a| gamma_q(r(a[0]), r(a[1])),
        gamma_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:gamma_log_p"],
        |a| gamma_log_p(r(a[0]), r(a[1])),
        gamma_log_p_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:gamma_log_q"],
        |a| gamma_log_q(r(a[0]), r(a[1])),
        gamma_log_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_p_inv"],
        |a| gamma_p_inv(r(a[0]), r(a[1])),
        gamma_p_inv_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_q_inv"],
        |a| gamma_q_inv(r(a[0]), r(a[1])),
        gamma_q_inv_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_p_inv_log", "probability:gamma_p_inv_log"],
        |a| gamma_p_inv_log(r(a[0]), r(a[1])),
        gamma_p_inv_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["gamma_q_inv_log", "probability:gamma_q_inv_log"],
        |a| gamma_q_inv_log(r(a[0]), r(a[1])),
        gamma_q_inv_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta"],
        |a| beta(r(a[0]), r(a[1])),
        beta_with_options
    );
    pair!(
        inputs,
        counts,
        ["ln_beta"],
        |a| ln_beta(r(a[0]), r(a[1])),
        ln_beta_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc"],
        |a| beta_inc(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc"],
        |a| beta_inc_q(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:beta_inc_log"],
        |a| beta_inc_log(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:beta_inc_log_q"],
        |a| beta_inc_log_q(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_log_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc_inv"],
        |a| beta_inc_inv(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_inv_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc_q_inv"],
        |a| beta_inc_q_inv(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_q_inv_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc_inv_log", "probability:beta_inc_inv_log"],
        |a| beta_inc_inv_log(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_inv_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["beta_inc_q_inv_log"],
        |a| beta_inc_q_inv_log(r(a[0]), r(a[1]), r(a[2])),
        beta_inc_q_inv_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["riemann_zeta"],
        |a| riemann_zeta(r(a[0])),
        riemann_zeta_with_options
    );
    pair!(
        inputs,
        counts,
        ["dirichlet_eta"],
        |a| dirichlet_eta(r(a[0])),
        dirichlet_eta_with_options
    );
    pair!(
        inputs,
        counts,
        ["hurwitz_zeta"],
        |a| hurwitz_zeta(r(a[0]), r(a[1])),
        hurwitz_zeta_with_options
    );
    pair!(
        inputs,
        counts,
        ["hurwitz_zeta"],
        |a| hurwitz_zeta_log(r(a[0]), r(a[1])),
        hurwitz_zeta_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:dilog"],
        |a| dilog(r(a[1])),
        dilog_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:spence"],
        |a| spence(r(a[1])),
        spence_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_branch:w0"],
        |a| lambert_w0(r(a[1])),
        lambert_w0_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_branch:wm1"],
        |a| lambert_wm1(r(a[1])),
        lambert_wm1_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_branch:omega"],
        |a| wright_omega(r(a[1])),
        wright_omega_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:polylog"],
        |a| polylog(r(a[3]), r(a[1])),
        polylog_with_options
    );
    pair!(
        inputs,
        counts,
        ["lerch_phi_complex"],
        |a| lerch_phi(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        lerch_phi_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:lerch"],
        |a| lerch_phi(r(a[1]), r(a[3]), r(a[5])),
        lerch_phi_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic_k"],
        |a| elliptic_k(r(a[0])),
        elliptic_k_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic_e"],
        |a| elliptic_e(r(a[0])),
        elliptic_e_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic_f"],
        |a| elliptic_f(r(a[0]), r(a[1])),
        elliptic_f_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic_e_inc"],
        |a| elliptic_e_inc(r(a[0]), r(a[1])),
        elliptic_e_inc_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic_pi"],
        |a| elliptic_pi(r(a[0]), r(a[1])),
        elliptic_pi_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:rf"],
        |a| carlson_rf(r(a[0]), r(a[1]), r(a[2])),
        carlson_rf_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:rd"],
        |a| carlson_rd(r(a[0]), r(a[1]), r(a[2])),
        carlson_rd_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:rg"],
        |a| carlson_rg(r(a[0]), r(a[1]), r(a[2])),
        carlson_rg_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:rc"],
        |a| carlson_rc(r(a[0]), r(a[1])),
        carlson_rc_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:rj"],
        |a| carlson_rj(r(a[0]), r(a[1]), r(a[2]), r(a[3])),
        carlson_rj_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:sn"],
        |a| jacobi_sn(r(a[0]), r(a[1])),
        jacobi_sn_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:cn"],
        |a| jacobi_cn(r(a[0]), r(a[1])),
        jacobi_cn_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:dn"],
        |a| jacobi_dn(r(a[0]), r(a[1])),
        jacobi_dn_with_options
    );
    pair!(
        inputs,
        counts,
        ["elliptic:am"],
        |a| jacobi_amplitude(r(a[0]), r(a[1])),
        jacobi_amplitude_with_options
    );
    pair!(
        inputs,
        counts,
        ["hyp0f1", "hypergeometric:pfq:01:0"],
        |a| hyp0f1(r(a[0]), r(a[1])),
        hyp0f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hyp1f1", "hypergeometric:pfq:11:0"],
        |a| hyp1f1(r(a[0]), r(a[1]), r(a[2])),
        hyp1f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:pfq:20:0"],
        |a| hyp2f0(r(a[0]), r(a[1]), r(a[2])),
        hyp2f0_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:u:11:0"],
        |a| hyperu(r(a[0]), r(a[1]), r(a[2])),
        hyperu_with_options
    );
    pair!(
        inputs,
        counts,
        ["hyp2f1", "hypergeometric:pfq:21:0"],
        |a| hyp2f1(r(a[0]), r(a[1]), r(a[2]), r(a[3])),
        hyp2f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hyp2f1", "hypergeometric:pfq:21:1"],
        |a| hyp2f1_regularized(r(a[0]), r(a[1]), r(a[2]), r(a[3])),
        hyp2f1_regularized_with_options
    );
    pfq_pair::<S, S>(inputs, counts, "", |a| r(a[0]));
    pair!(
        inputs,
        counts,
        ["legendre_p"],
        |a| legendre_p(n(a[0]), r(a[1])),
        legendre_p_with_options
    );
    pair!(
        inputs,
        counts,
        ["legendre_q"],
        |a| legendre_q(n(a[0]), r(a[1])),
        legendre_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["legendre_p_assoc", "angular:associated"],
        |a| legendre_p_assoc(n(a[0]), i(a[1]), r(a[2])),
        legendre_p_assoc_with_options
    );
    pair!(
        inputs,
        counts,
        ["h3d"],
        |a| legendre_h3d(u(a[0]), r(a[1]), r(a[2])),
        legendre_h3d_with_options
    );
    pair!(
        inputs,
        counts,
        ["weighted_bessel"],
        |a| bessel_weighted_integral(r(a[0]), r(a[1]), r(a[2])),
        bessel_weighted_integral_with_options
    );

    // Bessel, Struve and parabolic cylinder functions of real order.
    pair!(
        inputs,
        counts,
        ["bessel_j", "bessel_turning:j"],
        |a| bessel_j(r(a[0]), r(a[1])),
        bessel_j_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_y", "bessel_turning:y"],
        |a| bessel_y(r(a[0]), r(a[1])),
        bessel_y_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_i"],
        |a| bessel_i(r(a[0]), r(a[1])),
        bessel_i_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_i_scaled"],
        |a| bessel_i_scaled(r(a[0]), r(a[1])),
        bessel_i_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_k"],
        |a| bessel_k(r(a[0]), r(a[1])),
        bessel_k_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_k"],
        |a| bessel_k_log(r(a[0]), r(a[1])),
        bessel_k_log_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_k_scaled"],
        |a| bessel_k_scaled(r(a[0]), r(a[1])),
        bessel_k_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["bessel_j"],
        |a| bessel_lambda(r(a[0]), r(a[1])),
        bessel_lambda_with_options
    );
    pair!(
        inputs,
        counts,
        ["hankel_1"],
        |a| hankel_1(r(a[0]), r(a[1])),
        hankel_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hankel_2"],
        |a| hankel_2(r(a[0]), r(a[1])),
        hankel_2_with_options
    );
    for degree in [1, 2] {
        pair!(
            inputs,
            counts,
            ["bessel_j"],
            |a| bessel_j_derivative(r(a[0]), r(a[1]), degree),
            bessel_j_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["bessel_y"],
            |a| bessel_y_derivative(r(a[0]), r(a[1]), degree),
            bessel_y_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["bessel_i"],
            |a| bessel_i_derivative(r(a[0]), r(a[1]), degree),
            bessel_i_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["bessel_i_scaled"],
            |a| bessel_i_scaled_derivative(r(a[0]), r(a[1]), degree),
            bessel_i_scaled_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["bessel_k"],
            |a| bessel_k_derivative(r(a[0]), r(a[1]), degree),
            bessel_k_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["bessel_k_scaled"],
            |a| bessel_k_scaled_derivative(r(a[0]), r(a[1]), degree),
            bessel_k_scaled_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["hankel_1"],
            |a| hankel_1_derivative(r(a[0]), r(a[1]), degree),
            hankel_1_derivative_with_options
        );
        pair!(
            inputs,
            counts,
            ["hankel_2"],
            |a| hankel_2_derivative(r(a[0]), r(a[1]), degree),
            hankel_2_derivative_with_options
        );
    }
    pair!(
        inputs,
        counts,
        ["struve:h"],
        |a| struve(r(a[0]), r(a[1])),
        struve_with_options
    );
    pair!(
        inputs,
        counts,
        ["struve:l"],
        |a| modstruve(r(a[0]), r(a[1])),
        modstruve_with_options
    );
    pair!(
        inputs,
        counts,
        ["parabolic:d"],
        |a| parabolic_d(r(a[0]), r(a[1])),
        parabolic_d_with_options
    );
    pair!(
        inputs,
        counts,
        ["parabolic:v"],
        |a| parabolic_v(r(a[0]), r(a[1])),
        parabolic_v_with_options
    );
    pair!(
        inputs,
        counts,
        ["parabolic:w"],
        |a| parabolic_w(r(a[0]), r(a[1])),
        parabolic_w_with_options
    );
    pair!(
        inputs,
        counts,
        ["riccati:rj"],
        |a| riccati_jn(u(a[0]), r(a[1])),
        riccati_jn_with_options
    );
    pair!(
        inputs,
        counts,
        ["riccati:ry"],
        |a| riccati_yn(u(a[0]), r(a[1])),
        riccati_yn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_jn(u(a[0]), r(a[1])),
        spherical_jn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:y"],
        |a| spherical_yn(u(a[0]), r(a[1])),
        spherical_yn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:i"],
        |a| spherical_in(u(a[0]), r(a[1])),
        spherical_in_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:is"],
        |a| spherical_in_scaled(u(a[0]), r(a[1])),
        spherical_in_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:k"],
        |a| spherical_kn(u(a[0]), r(a[1])),
        spherical_kn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:ks"],
        |a| spherical_kn_scaled(u(a[0]), r(a[1])),
        spherical_kn_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_1(u(a[0]), r(a[1])),
        spherical_hankel_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_2(u(a[0]), r(a[1])),
        spherical_hankel_2_with_options
    );

    // Integer and combinatorial arguments.
    pair!(
        inputs,
        counts,
        ["factorial", "combinatorial:0"],
        |a| factorial::<S>(n(a[0])),
        factorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["ln_factorial", "combinatorial:0"],
        |a| ln_factorial::<S>(n(a[0])),
        ln_factorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:1"],
        |a| double_factorial::<S>(n(a[0])),
        double_factorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:1"],
        |a| ln_double_factorial::<S>(n(a[0])),
        ln_double_factorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:1"],
        |a| a[1] > 0.0,
        multifactorial::<S>(n(a[0]), n(a[1])),
        multifactorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:1"],
        |a| a[1] > 0.0,
        ln_multifactorial::<S>(n(a[0]), n(a[1])),
        ln_multifactorial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:3"],
        |a| bernoulli::<S>(n(a[0])),
        bernoulli_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:4"],
        |a| euler::<S>(n(a[0])),
        euler_with_options
    );
    pair!(
        inputs,
        counts,
        ["binomial", "combinatorial:2"],
        |a| binomial::<S>(n(a[0]), n(a[1])),
        binomial_with_options
    );
    pair!(
        inputs,
        counts,
        ["binomial", "combinatorial:2"],
        |a| ln_binomial::<S>(n(a[0]), n(a[1])),
        ln_binomial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:5"],
        |a| stirling_first_kind::<S>(n(a[0]), n(a[1])),
        stirling_first_kind_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:6"],
        |a| stirling_second_kind::<S>(n(a[0]), n(a[1])),
        stirling_second_kind_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:7"],
        |a| pochhammer(r(a[2]), r(a[3])),
        pochhammer_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:7:complex"],
        |a| pochhammer(c(a[2], a[3]), c(a[4], a[5])),
        pochhammer_with_options
    );

    pair!(
        inputs,
        counts,
        ["combinatorial:8"],
        |a| pochhammer_relative(r(a[2]), r(a[3])),
        pochhammer_relative_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:8:complex"],
        |a| pochhammer_relative(c(a[2], a[3]), c(a[4], a[5])),
        pochhammer_relative_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:9"],
        |a| generalized_binomial(r(a[2]), n(a[1])),
        generalized_binomial_with_options
    );
    pair!(
        inputs,
        counts,
        ["combinatorial:9:complex"],
        |a| generalized_binomial(c(a[2], a[3]), n(a[1])),
        generalized_binomial_with_options
    );

    pair!(
        inputs,
        counts,
        ["angular:3j"],
        |a| wigner_3j::<S>([i(a[0]), i(a[1]), i(a[2])], [i(a[3]), i(a[4]), i(a[5])]),
        prepared_wigner_3j
    );
    pair!(
        inputs,
        counts,
        ["angular:cg"],
        |a| clebsch_gordan::<S>([i(a[0]), i(a[1]), i(a[2])], [i(a[3]), i(a[4]), i(a[5])]),
        prepared_clebsch_gordan
    );
    pair!(
        inputs,
        counts,
        ["angular:6j"],
        |a| wigner_6j::<S>(core::array::from_fn(|k| i(a[k]))),
        prepared_wigner_6j
    );
    pair!(
        inputs,
        counts,
        ["angular:racah"],
        |a| racah_w::<S>(core::array::from_fn(|k| i(a[k]))),
        prepared_racah_w
    );
    pair!(
        inputs,
        counts,
        ["angular:9j"],
        |a| wigner_9j::<S>(core::array::from_fn(|k| i(a[k]))),
        prepared_wigner_9j
    );
    pair!(
        inputs,
        counts,
        ["angular:gaunt"],
        |a| gaunt::<S>([n(a[0]), n(a[1]), n(a[2])], [i(a[3]), i(a[4]), i(a[5])]),
        prepared_gaunt
    );
    pair!(
        inputs,
        counts,
        ["angular:harmonic"],
        |a| spherical_harmonic(n(a[0]), i(a[1]), r(a[2]), r(a[3])),
        spherical_harmonic_with_options
    );
    pair!(
        inputs,
        counts,
        ["angular:harmonic"],
        |a| spherical_harmonic_real(n(a[0]), i(a[1]), r(a[2]), r(a[3])),
        spherical_harmonic_real_with_options
    );

    // Coulomb and ellipsoidal families.
    pair!(
        inputs,
        counts,
        ["coulomb_f", "coulomb:f"],
        |a| coulomb_f(r(a[0]), r(a[1]), r(a[2])),
        coulomb_f_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb_g", "coulomb:g"],
        |a| coulomb_g(r(a[0]), r(a[1]), r(a[2])),
        coulomb_g_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb_fp", "coulomb:fp"],
        |a| coulomb_f_prime(r(a[0]), r(a[1]), r(a[2])),
        coulomb_f_prime_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb_gp", "coulomb:gp"],
        |a| coulomb_g_prime(r(a[0]), r(a[1]), r(a[2])),
        coulomb_g_prime_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb_f", "coulomb:f"],
        |a| coulomb_spherical_f(r(a[0]), r(a[1]), r(a[2])),
        coulomb_spherical_f_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb:f:complex"],
        |a| coulomb_f(r(a[0]), r(a[1]), c(a[2], a[3])),
        coulomb_f_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb:g:complex"],
        |a| coulomb_g(r(a[0]), r(a[1]), c(a[2], a[3])),
        coulomb_g_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb:c"],
        |a| coulomb_normalization(r(a[0]), r(a[1])),
        coulomb_normalization_with_options
    );
    pair!(
        inputs,
        counts,
        ["coulomb:h"],
        |a| hydrogenic_radial(u(a[0]), u(a[1]), r(a[2]), r(a[3])),
        hydrogenic_radial_with_options
    );
    pair!(
        inputs,
        counts,
        ["ellipsoidal"],
        |a| ellipsoidal_harmonic(
            r(a[0]),
            r(a[1]),
            n(a[2]),
            n(a[3]),
            r(a[4]),
            scioxide_special::EllipsoidalSigns::default()
        ),
        prepared_ellipsoidal_harmonic
    );
    pair!(
        inputs,
        counts,
        ["ellipsoidal_complex"],
        |a| ellipsoidal_harmonic(
            r(a[0]),
            r(a[1]),
            n(a[2]),
            n(a[3]),
            c(a[4], a[5]),
            scioxide_special::EllipsoidalSigns::default()
        ),
        prepared_ellipsoidal_harmonic
    );
    pair!(
        inputs,
        counts,
        ["ellipsoidal"],
        |a| ellipsoidal_harmonic_second_kind(r(a[0]), r(a[1]), n(a[2]), n(a[3]), r(a[4])),
        prepared_ellipsoidal_harmonic_second_kind
    );
    pair!(
        inputs,
        counts,
        ["ellipsoidal_complex"],
        |a| ellipsoidal_harmonic_second_kind(r(a[0]), r(a[1]), n(a[2]), n(a[3]), c(a[4], a[5])),
        prepared_ellipsoidal_harmonic_second_kind
    );
    pair!(
        inputs,
        counts,
        ["ellipsoidal"],
        |a| ellipsoidal_normalization(r(a[0]), r(a[1]), n(a[2]), n(a[3])),
        prepared_ellipsoidal_normalization
    );

    // Mathieu and spheroidal families.
    let radial = |x: f64| match x as u32 {
        1 => RadialKind::First,
        2 => RadialKind::Second,
        3 => RadialKind::Third,
        _ => RadialKind::Fourth,
    };
    let kind = |x: f64| {
        if x == 0.0 {
            SpheroidalKind::Prolate
        } else {
            SpheroidalKind::Oblate
        }
    };
    type Characteristic<S> = fn(u32, S) -> Result<S, SciOxideError>;
    for (name, key, parity, one_shot) in [
        (
            "mathieu_a",
            "mathieu:ce:char",
            MathieuParity::Even,
            mathieu_a::<S> as Characteristic<S>,
        ),
        (
            "mathieu_b",
            "mathieu:se:char",
            MathieuParity::Odd,
            mathieu_b::<S>,
        ),
    ] {
        counts.register(name);
        for a in inputs.rows::<S>(&[key]) {
            let prepared = MathieuPlan::new(
                u(a[0]),
                parity,
                r(a[1]),
                SpectralNormalization::Standard,
                Default::default(),
            )
            .and_then(|plan| plan.characteristic_value());
            counts.record(name, same(a, one_shot(u(a[0]), r(a[1])), prepared));
        }
    }
    pair!(
        inputs,
        counts,
        ["mathieu:ce:angular"],
        |a| mathieu_ce(u(a[0]), r(a[1]), r(a[2])),
        prepared_mathieu_ce
    );
    pair!(
        inputs,
        counts,
        ["mathieu:ce:angular:complex"],
        |a| mathieu_ce(u(a[0]), r(a[1]), c(a[2], a[3])),
        prepared_mathieu_ce
    );
    pair!(
        inputs,
        counts,
        ["mathieu:se:angular"],
        |a| mathieu_se(u(a[0]), r(a[1]), r(a[2])),
        prepared_mathieu_se
    );
    pair!(
        inputs,
        counts,
        ["mathieu:se:angular:complex"],
        |a| mathieu_se(u(a[0]), r(a[1]), c(a[2], a[3])),
        prepared_mathieu_se
    );
    pair!(
        inputs,
        counts,
        ["mathieu:ce:radial"],
        |a| mathieu_mc(u(a[0]), r(a[1]), r(a[3]), radial(a[2])),
        prepared_mathieu_mc
    );
    pair!(
        inputs,
        counts,
        ["mathieu:ce:radial:complex"],
        |a| mathieu_mc(u(a[0]), r(a[1]), c(a[3], a[4]), radial(a[2])),
        prepared_mathieu_mc
    );
    pair!(
        inputs,
        counts,
        ["mathieu:se:radial"],
        |a| mathieu_ms(u(a[0]), r(a[1]), r(a[3]), radial(a[2])),
        prepared_mathieu_ms
    );
    pair!(
        inputs,
        counts,
        ["mathieu:se:radial:complex"],
        |a| mathieu_ms(u(a[0]), r(a[1]), c(a[3], a[4]), radial(a[2])),
        prepared_mathieu_ms
    );
    counts.register("spheroidal_characteristic_value");
    for a in inputs.rows::<S>(&["spheroidal:characteristic"]) {
        let (kind, m, n, size) = (kind(a[0]), u(a[1]), u(a[2]), r(a[3]));
        let prepared = SpheroidalPlan::new(
            kind,
            m,
            n,
            size,
            SpectralNormalization::Standard,
            Default::default(),
        )
        .and_then(|plan| plan.characteristic_value());
        counts.record(
            "spheroidal_characteristic_value",
            same(
                a,
                spheroidal_characteristic_value(kind, m, n, size),
                prepared,
            ),
        );
    }
    pair!(
        inputs,
        counts,
        ["spheroidal:angular"],
        |a| spheroidal_angular(kind(a[0]), u(a[1]), u(a[2]), r(a[3]), r(a[4])),
        prepared_spheroidal_angular
    );
    pair!(
        inputs,
        counts,
        ["spheroidal:angular:complex"],
        |a| spheroidal_angular(kind(a[0]), u(a[1]), u(a[2]), r(a[3]), c(a[4], a[5])),
        prepared_spheroidal_angular
    );
    pair!(
        inputs,
        counts,
        ["spheroidal:radial"],
        |a| spheroidal_radial(kind(a[0]), u(a[1]), u(a[2]), r(a[3]), r(a[5]), radial(a[4])),
        prepared_spheroidal_radial
    );
    pair!(
        inputs,
        counts,
        ["spheroidal:radial:complex"],
        |a| spheroidal_radial(
            kind(a[0]),
            u(a[1]),
            u(a[2]),
            r(a[3]),
            c(a[5], a[6]),
            radial(a[4])
        ),
        prepared_spheroidal_radial
    );

    // Complex arguments.
    pair!(
        inputs,
        counts,
        ["airy_complex:ai"],
        |a| airy_ai(c(a[0], a[1])),
        airy_ai_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_complex:ai_scaled"],
        |a| airy_ai_scaled(c(a[0], a[1])),
        airy_ai_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_complex:bi"],
        |a| airy_bi(c(a[0], a[1])),
        airy_bi_with_options
    );
    pair!(
        inputs,
        counts,
        ["airy_complex:bi_scaled"],
        |a| airy_bi_scaled(c(a[0], a[1])),
        airy_bi_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erf"],
        |a| erf(c(a[0], a[1])),
        erf_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erfc"],
        |a| erfc(c(a[0], a[1])),
        erfc_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erfcx"],
        |a| erfcx(c(a[0], a[1])),
        erfcx_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erfi"],
        |a| erfi(c(a[0], a[1])),
        erfi_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erfi_scaled"],
        |a| dawson(c(a[0], a[1])),
        dawson_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:log_erfc"],
        |a| log_erfc(c(a[0], a[1])),
        log_erfc_with_options
    );
    pair!(
        inputs,
        counts,
        ["probability:faddeeva"],
        |a| faddeeva(c(a[0], a[1])),
        faddeeva_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erf"],
        |a| fresnel_c(c(a[0], a[1])),
        fresnel_c_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_error:erf"],
        |a| fresnel_s(c(a[0], a[1])),
        fresnel_s_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:gamma"],
        |a| gamma(c(a[0], a[1])),
        gamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:loggamma"],
        |a| ln_gamma(c(a[0], a[1])),
        ln_gamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:digamma"],
        |a| digamma(c(a[0], a[1])),
        digamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:polygamma"],
        |a| polygamma(2, c(a[0], a[1])),
        polygamma_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:lower"],
        |a| gamma_lower(c(a[0], a[1]), c(a[2], a[3])),
        gamma_lower_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:upper"],
        |a| gamma_upper(c(a[0], a[1]), c(a[2], a[3])),
        gamma_upper_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:p"],
        |a| gamma_p(c(a[0], a[1]), c(a[2], a[3])),
        gamma_p_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:q"],
        |a| gamma_q(c(a[0], a[1]), c(a[2], a[3])),
        gamma_q_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:beta"],
        |a| beta(c(a[0], a[1]), c(a[2], a[3])),
        beta_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:beta"],
        |a| ln_beta(c(a[0], a[1]), c(a[2], a[3])),
        ln_beta_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:beta_inc"],
        |a| beta_inc(c(a[0], a[1]), c(a[2], a[3]), c(0.25, 0.125)),
        beta_inc_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:zeta", "riemann_zeta_complex"],
        |a| riemann_zeta(c(a[0], a[1])),
        riemann_zeta_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:eta", "dirichlet_eta_complex"],
        |a| dirichlet_eta(c(a[0], a[1])),
        dirichlet_eta_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_gamma:hurwitz", "hurwitz_zeta_complex"],
        |a| hurwitz_zeta(c(a[0], a[1]), c(a[2], a[3])),
        hurwitz_zeta_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:dilog"],
        |a| dilog(c(a[1], a[2])),
        dilog_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:spence"],
        |a| spence(c(a[1], a[2])),
        spence_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:omega"],
        |a| wright_omega(c(a[1], a[2])),
        wright_omega_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:w", "lambert_branch:complex"],
        |a| lambert_w_complex(c(a[1], a[2]), i(a[0])),
        lambert_w_complex_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:polylog"],
        |a| polylog(c(a[3], a[4]), c(a[1], a[2])),
        polylog_with_options
    );
    pair!(
        inputs,
        counts,
        ["lambert_polylog:lerch"],
        |a| lerch_phi(c(a[1], a[2]), c(a[3], a[4]), c(a[5], a[6])),
        lerch_phi_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_elliptic:k"],
        |a| elliptic_k(c(a[2], a[3])),
        elliptic_k_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_elliptic:e"],
        |a| elliptic_e(c(a[2], a[3])),
        elliptic_e_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_elliptic:f"],
        |a| elliptic_f(c(a[0], a[1]), c(a[2], a[3])),
        elliptic_f_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_elliptic:ei"],
        |a| elliptic_e_inc(c(a[0], a[1]), c(a[2], a[3])),
        elliptic_e_inc_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_elliptic:pi"],
        |a| elliptic_pi(c(a[0], a[1]), c(a[2], a[3])),
        elliptic_pi_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_carlson:rf", "elliptic:rf:complex"],
        |a| carlson_rf(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        carlson_rf_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_carlson:rd", "elliptic:rd:complex"],
        |a| carlson_rd(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        carlson_rd_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_carlson:rg", "elliptic:rg:complex"],
        |a| carlson_rg(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        carlson_rg_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_carlson:rc", "elliptic:rc:complex"],
        |a| carlson_rc(c(a[0], a[1]), c(a[2], a[3])),
        carlson_rc_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_carlson:rj", "elliptic:rj:complex"],
        |a| carlson_rj(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5]), c(a[6], a[7])),
        carlson_rj_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_jacobi:sn"],
        |a| jacobi_sn(c(a[0], a[1]), c(a[2], a[3])),
        jacobi_sn_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_jacobi:cn"],
        |a| jacobi_cn(c(a[0], a[1]), c(a[2], a[3])),
        jacobi_cn_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_jacobi:dn"],
        |a| jacobi_dn(c(a[0], a[1]), c(a[2], a[3])),
        jacobi_dn_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_jacobi:am"],
        |a| jacobi_amplitude(c(a[0], a[1]), c(a[2], a[3])),
        jacobi_amplitude_with_options
    );
    pfq_pair::<S, Complex<S>>(inputs, counts, ":complex", |a| c(a[0], a[1]));
    pair!(
        inputs,
        counts,
        ["hypergeometric:pfq:01:0:complex"],
        |a| hyp0f1(c(a[0], a[1]), c(a[2], a[3])),
        hyp0f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:pfq:11:0:complex"],
        |a| hyp1f1(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        hyp1f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:pfq:21:0:complex"],
        |a| hyp2f1(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5]), c(a[6], a[7])),
        hyp2f1_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:pfq:21:1:complex"],
        |a| hyp2f1_regularized(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5]), c(a[6], a[7])),
        hyp2f1_regularized_with_options
    );
    pair!(
        inputs,
        counts,
        ["hypergeometric:u:11:0:complex"],
        |a| hyperu(c(a[0], a[1]), c(a[2], a[3]), c(a[4], a[5])),
        hyperu_with_options
    );
    pair!(
        inputs,
        counts,
        ["parabolic:d"],
        |a| integer(a[0]) && a[0] <= 64.0,
        parabolic_d_complex(u(a[0]), c(a[1], 0.0)),
        parabolic_d_complex_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_jn(u(a[0]), c(a[1], 0.0)),
        spherical_jn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:y"],
        |a| spherical_yn(u(a[0]), c(a[1], 0.0)),
        spherical_yn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:i"],
        |a| spherical_in(u(a[0]), c(a[1], 0.0)),
        spherical_in_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:is"],
        |a| spherical_in_scaled(u(a[0]), c(a[1], 0.0)),
        spherical_in_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:k"],
        |a| spherical_kn(u(a[0]), c(a[1], 0.0)),
        spherical_kn_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:ks"],
        |a| spherical_kn_scaled(u(a[0]), c(a[1], 0.0)),
        spherical_kn_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_1(u(a[0]), c(a[1], 0.0)),
        spherical_hankel_1_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_1_scaled(u(a[0]), c(a[1], 0.0)),
        spherical_hankel_1_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_2(u(a[0]), c(a[1], 0.0)),
        spherical_hankel_2_with_options
    );
    pair!(
        inputs,
        counts,
        ["spherical:j"],
        |a| spherical_hankel_2_scaled(u(a[0]), c(a[1], 0.0)),
        spherical_hankel_2_scaled_with_options
    );

    // Complex Bessel functions of real and complex order.
    macro_rules! complex_bessel {
        ($([$real:literal, $complex:literal] $plain:ident $detailed:ident $derivative:ident $derivative_detailed:ident;)*) => {$(
            pair!(inputs, counts, [$real], |a| $plain(c(a[0], 0.0), c(a[1], a[2])), $detailed);
            pair!(inputs, counts, [$complex], |a| $plain(c(a[0], a[1]), c(a[2], a[3])), $detailed);
            for degree in [1, 2] {
                pair!(inputs, counts, [$real], |a| $derivative(c(a[0], 0.0), c(a[1], a[2]), degree), $derivative_detailed);
                pair!(inputs, counts, [$complex], |a| $derivative(c(a[0], a[1]), c(a[2], a[3]), degree), $derivative_detailed);
            }
        )*};
    }
    complex_bessel! {
        ["complex_bessel:j", "complex_order_bessel:j"] bessel_j bessel_j_with_options bessel_j_derivative bessel_j_derivative_with_options;
        ["complex_bessel:y", "complex_order_bessel:y"] bessel_y bessel_y_with_options bessel_y_derivative bessel_y_derivative_with_options;
        ["complex_bessel:i", "complex_order_bessel:i"] bessel_i bessel_i_with_options bessel_i_derivative bessel_i_derivative_with_options;
        ["complex_bessel:is", "complex_order_bessel:i_scaled"] bessel_i_scaled bessel_i_scaled_with_options bessel_i_scaled_derivative bessel_i_scaled_derivative_with_options;
        ["complex_bessel:k", "complex_order_bessel:k"] bessel_k bessel_k_with_options bessel_k_derivative bessel_k_derivative_with_options;
        ["complex_bessel:ks", "complex_order_bessel:k_scaled"] bessel_k_scaled bessel_k_scaled_with_options bessel_k_scaled_derivative bessel_k_scaled_derivative_with_options;
        ["complex_bessel:h1", "complex_order_bessel:h1"] hankel_1 hankel_1_with_options hankel_1_derivative hankel_1_derivative_with_options;
        ["complex_bessel:h2", "complex_order_bessel:h2"] hankel_2 hankel_2_with_options hankel_2_derivative hankel_2_derivative_with_options;
    }
    pair!(
        inputs,
        counts,
        ["complex_bessel:js"],
        |a| bessel_j_scaled(c(a[0], 0.0), c(a[1], a[2])),
        bessel_j_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_order_bessel:j_scaled"],
        |a| bessel_j_scaled(c(a[0], a[1]), c(a[2], a[3])),
        bessel_j_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_bessel:ys"],
        |a| bessel_y_scaled(c(a[0], 0.0), c(a[1], a[2])),
        bessel_y_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_order_bessel:y_scaled"],
        |a| bessel_y_scaled(c(a[0], a[1]), c(a[2], a[3])),
        bessel_y_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_bessel:h1s"],
        |a| hankel_1_scaled(c(a[0], 0.0), c(a[1], a[2])),
        hankel_1_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_order_bessel:h1_scaled"],
        |a| hankel_1_scaled(c(a[0], a[1]), c(a[2], a[3])),
        hankel_1_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_bessel:h2s"],
        |a| hankel_2_scaled(c(a[0], 0.0), c(a[1], a[2])),
        hankel_2_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["complex_order_bessel:h2_scaled"],
        |a| hankel_2_scaled(c(a[0], a[1]), c(a[2], a[3])),
        hankel_2_scaled_with_options
    );

    pair!(
        inputs,
        counts,
        ["kelvin:0"],
        |a| ber_scaled(r(a[0])),
        ber_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:1"],
        |a| bei_scaled(r(a[0])),
        bei_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:2"],
        |a| ker_scaled(r(a[0])),
        ker_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:3"],
        |a| kei_scaled(r(a[0])),
        kei_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:0"],
        |a| berp_scaled(r(a[0])),
        berp_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:1"],
        |a| beip_scaled(r(a[0])),
        beip_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:2"],
        |a| kerp_scaled(r(a[0])),
        kerp_scaled_with_options
    );
    pair!(
        inputs,
        counts,
        ["kelvin:3"],
        |a| keip_scaled(r(a[0])),
        keip_scaled_with_options
    );

    // Multi-valued plain results.
    counts.register("kelvin");
    for a in inputs.rows::<S>(&["kelvin:0"]) {
        let outcome = match (
            kelvin(r(a[0])),
            kelvin_with_options(r(a[0]), KelvinOptions::default()),
        ) {
            (Ok(plain), Ok([ber, bei, ker, kei])) => {
                let detailed = [ber, bei, ker, kei].map(|r| r.try_value().unwrap());
                if plain.map(bits) == detailed.map(bits) {
                    Ok(())
                } else {
                    Err(format!("{a:?}: {plain:?} vs {detailed:?}"))
                }
            }
            (Err(_), _) => Ok(()),
            (plain, detailed) => Err(format!("{a:?}: {plain:?} vs {detailed:?}")),
        };
        counts.record("kelvin", outcome);
    }
    counts.register("coulomb_wave");
    for a in inputs.rows::<S>(&["coulomb_f", "coulomb:f"]) {
        let outcome = match (
            coulomb_wave(r(a[0]), r(a[1]), r(a[2])),
            coulomb_wave_with_options(r(a[0]), r(a[1]), r(a[2]), CoulombOptions::default())
                .and_then(|report| {
                    [report.f(), report.g(), report.f_prime(), report.g_prime()]
                        .into_iter()
                        .map(|component| component.try_value())
                        .collect::<Result<Vec<_>, _>>()
                }),
        ) {
            (Ok(plain), Ok(detailed))
                if plain
                    .iter()
                    .copied()
                    .map(bits)
                    .eq(detailed.iter().copied().map(bits)) =>
            {
                Ok(())
            }
            (Err(plain), Err(detailed)) if plain.kind() == detailed.kind() => Ok(()),
            (plain, detailed) => Err(format!("{a:?}: {plain:?} vs {detailed:?}")),
        };
        counts.record("coulomb_wave", outcome);
    }
}

#[test]
fn plain_values_equal_detailed_values_on_oracle_inputs() {
    let inputs = Inputs::load();
    let mut tally = Tally::default();
    check::<f64>(&inputs, &mut tally);
    check::<f32>(&inputs, &mut tally);
    let unexercised: Vec<_> = tally
        .rows
        .iter()
        .filter(|(_, rows)| **rows == 0)
        .map(|(name, _)| *name)
        .collect();
    assert!(
        unexercised.is_empty(),
        "pairs without oracle inputs: {unexercised:?}"
    );
    let failures: Vec<_> = tally
        .failures
        .iter()
        .map(|(name, (count, first))| {
            format!(
                "{name}: {count} of {} rows; first {first}",
                tally.rows[name]
            )
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(tally.rows.len(), 241, "registered operations");
}

/// Builds the plan for the largest doubled momentum in `twice_j`.
fn angular_plan(twice_j: &[i32], options: AngularOptions) -> Result<AngularPlan, SciOxideError> {
    AngularPlan::new(
        AngularMomentum::new(twice_j.iter().copied().max().unwrap())?,
        options,
    )
}

fn angular_states(
    twice_j: [i32; 3],
    twice_m: [i32; 3],
) -> Result<[AngularState; 3], SciOxideError> {
    let [a, b, c] = core::array::from_fn(|k| AngularState::new(twice_j[k], twice_m[k]));
    Ok([a?, b?, c?])
}

fn prepared_wigner_3j<S: RealScalar>(
    twice_j: [i32; 3],
    twice_m: [i32; 3],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    angular_plan(&twice_j, options)?.wigner_3j(angular_states(twice_j, twice_m)?)
}

fn prepared_wigner_6j<S: RealScalar>(
    twice_j: [i32; 6],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    angular_plan(&twice_j, options)?.wigner_6j(twice_j.map(|j| AngularMomentum::new(j).unwrap()))
}

fn prepared_wigner_9j<S: RealScalar>(
    twice_j: [i32; 9],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    angular_plan(&twice_j, options)?.wigner_9j(twice_j.map(|j| AngularMomentum::new(j).unwrap()))
}

fn prepared_racah_w<S: RealScalar>(
    twice_j: [i32; 6],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    angular_plan(&twice_j, options)?.racah_w(twice_j.map(|j| AngularMomentum::new(j).unwrap()))
}

fn prepared_clebsch_gordan<S: RealScalar>(
    twice_j: [i32; 3],
    twice_m: [i32; 3],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    let [a, b, c] = angular_states(twice_j, twice_m)?;
    angular_plan(&twice_j, options)?.clebsch_gordan(a, b, c)
}

fn prepared_gaunt<S: RealScalar>(
    degree: [usize; 3],
    order: [i32; 3],
    options: AngularOptions,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    let twice_l = degree.map(|l| i32::try_from(2 * l).unwrap());
    angular_plan(&twice_l, options)?.gaunt(angular_states(twice_l, order.map(|m| 2 * m))?)
}

fn prepared_ellipsoidal_harmonic<T: NumericScalar>(
    h2: T::Real,
    k2: T::Real,
    n: usize,
    p: usize,
    s: T,
    signs: EllipsoidalSigns,
    options: EllipsoidalOptions<T::Real>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    EllipsoidalHarmonicPlan::new(h2, k2, n, p, options)?.first_kind(s, signs)
}
fn prepared_ellipsoidal_harmonic_second_kind<T: NumericScalar>(
    h2: T::Real,
    k2: T::Real,
    n: usize,
    p: usize,
    s: T,
    options: EllipsoidalOptions<T::Real>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    EllipsoidalHarmonicPlan::new(h2, k2, n, p, options)?.second_kind(s)
}
fn prepared_ellipsoidal_normalization<S: RealScalar>(
    h2: S,
    k2: S,
    n: usize,
    p: usize,
    options: EllipsoidalOptions<S>,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    EllipsoidalHarmonicPlan::new(h2, k2, n, p, options)?.normalization()
}
fn mathieu_plan<S: RealScalar>(
    order: u32,
    parity: MathieuParity,
    q: S,
    options: SpectralOptions<S>,
) -> Result<MathieuPlan<S>, SciOxideError> {
    MathieuPlan::new(order, parity, q, SpectralNormalization::Standard, options)
}
fn prepared_mathieu_ce<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    mathieu_plan(order, MathieuParity::Even, q, options)?.angular(z)
}
fn prepared_mathieu_se<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    mathieu_plan(order, MathieuParity::Odd, q, options)?.angular(z)
}
fn prepared_mathieu_mc<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    kind: RadialKind,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<Complex<T::Real>>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    mathieu_plan(order, MathieuParity::Even, q, options)?.radial(z, kind)
}
fn prepared_mathieu_ms<T: NumericScalar>(
    order: u32,
    q: T::Real,
    z: T,
    kind: RadialKind,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<Complex<T::Real>>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    mathieu_plan(order, MathieuParity::Odd, q, options)?.radial(z, kind)
}
fn prepared_spheroidal_angular<T: NumericScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: T::Real,
    x: T,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<T>, SciOxideError> {
    SpheroidalPlan::new(kind, m, n, c, SpectralNormalization::Standard, options)?.angular(x)
}
fn prepared_spheroidal_radial<T: NumericScalar>(
    kind: SpheroidalKind,
    m: u32,
    n: u32,
    c: T::Real,
    x: T,
    radial_kind: RadialKind,
    options: SpectralOptions<T::Real>,
) -> Result<SpecialFunctionResult<Complex<T::Real>>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    SpheroidalPlan::new(kind, m, n, c, SpectralNormalization::Standard, options)?
        .radial(x, radial_kind)
}

#[test]
fn narrowed_reports_keep_optional_estimates_on_shared_oracle_inputs() {
    fn compare(narrow: SpecialFunctionResult<f32>, wide: SpecialFunctionResult<f64>) {
        if narrow.scalar_representation() != ScalarRepresentation::Finite
            || wide.scalar_representation() != ScalarRepresentation::Finite
        {
            return;
        }
        let (Some(n), Some(w)) = (narrow.value(), wide.value()) else {
            return;
        };
        // Relative and log estimates are intentionally absent at rounded zeros and
        // once their uncertainty interval reaches zero; those boundaries have separate tests.
        if *n.value() == 0.0
            || *w.value() == 0.0
            || n.relative_error().is_some_and(|error| error >= 0.25)
            || w.relative_error().is_some_and(|error| error >= 0.25)
        {
            return;
        }
        assert_eq!(
            n.relative_error().is_some(),
            w.relative_error().is_some(),
            "{narrow:?} vs {wide:?}"
        );
        assert_eq!(narrow.scaled().is_some(), wide.scaled().is_some());
        if let (Some(n), Some(w)) = (narrow.scaled(), wide.scaled()) {
            assert_eq!(n.relative_error().is_some(), w.relative_error().is_some());
        }
        assert_eq!(narrow.logarithmic().is_some(), wide.logarithmic().is_some());
        if let Some(w) = wide.derivative()
            && w.value().abs() >= f64::from(f32::MIN_POSITIVE)
            && w.value().abs() <= f64::from(f32::MAX)
        {
            let n = narrow.derivative().unwrap();
            assert_eq!(n.relative_error().is_some(), w.relative_error().is_some());
        }
    }
    let inputs = Inputs::load();
    macro_rules! unary {
        ($key:literal, $function:ident) => {
            for row in inputs.rows::<f32>(&[$key]) {
                let x = row[0] as f32;
                if let (Ok(n), Ok(w)) = (
                    $function(x, Default::default()),
                    $function(f64::from(x), Default::default()),
                ) {
                    compare(n, w);
                }
            }
        };
    }
    macro_rules! binary {
        ($key:literal, $function:ident) => {
            for row in inputs.rows::<f32>(&[$key]) {
                let (a, x) = (row[0] as f32, row[1] as f32);
                if let (Ok(n), Ok(w)) = (
                    $function(a, x, Default::default()),
                    $function(f64::from(a), f64::from(x), Default::default()),
                ) {
                    compare(n, w);
                }
            }
        };
    }
    unary!("gamma", gamma_with_options);
    unary!("ln_gamma", ln_gamma_with_options);
    unary!("digamma", digamma_with_options);
    binary!("gamma_p", gamma_p_with_options);
    binary!("gamma_q", gamma_q_with_options);
    binary!("gamma_lower", gamma_lower_with_options);
    binary!("gamma_upper", gamma_upper_with_options);
    binary!("beta", beta_with_options);
}
