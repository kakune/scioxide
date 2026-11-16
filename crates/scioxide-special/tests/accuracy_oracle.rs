//! Differential accuracy checks against generated mpmath references.

#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_special::*;

const ORACLE: &str = include_str!("data/special_oracle.tsv");
const AIRY: &str = include_str!("data/airy_reference.tsv");
const BESSEL: &str = include_str!("data/bessel_reference.tsv");
const COULOMB: &str = include_str!("data/coulomb_dense.tsv");
const GAMMA_BETA: &str = include_str!("data/gamma_beta_reference.tsv");
const GAMMA_BETA_MEASURED: &str = include_str!("data/gamma_beta_measured.tsv");
const COMPLEX_ZETA: &str = include_str!("data/complex_zeta_reference.tsv");
const ZETA: &str = include_str!("data/zeta_reference.tsv");
const ELLIPTIC_MEASURED: &str = include_str!("data/elliptic_measured.tsv");
const LEGENDRE: &str = include_str!("data/legendre_reference.tsv");
const ERFC: &str = include_str!("data/erfc_reference.tsv");
const HYPERGEOMETRIC: &str = include_str!("data/hypergeometric_reference.tsv");
const REAL_LAMBERT_POLYLOG: &str = include_str!("data/real_lambert_polylog.tsv");
const REAL_FADDEEVA: &str = include_str!("data/real_faddeeva_reference.tsv");

#[derive(Clone, Copy, Debug)]
enum Argument {
    Float(u64),
    Unsigned(usize),
    Signed(i32),
}

#[derive(Debug)]
struct Case<'a> {
    tier: &'a str,
    scalar: &'a str,
    function: &'a str,
    arguments: Vec<Argument>,
    encoded_arguments: &'a str,
    expected_re: u64,
    expected_im: Option<u64>,
    reference_offset: f64,
}

fn cases() -> impl Iterator<Item = Case<'static>> {
    [
        ORACLE,
        AIRY,
        BESSEL,
        COULOMB,
        include_str!("data/large_argument_bands.tsv"),
        GAMMA_BETA,
        GAMMA_BETA_MEASURED,
        ELLIPTIC_MEASURED,
        ZETA,
        COMPLEX_ZETA,
        LEGENDRE,
        include_str!("data/legendre_ownership.tsv"),
        ERFC,
        HYPERGEOMETRIC,
        REAL_FADDEEVA,
        REAL_LAMBERT_POLYLOG,
    ]
    .into_iter()
    .flat_map(cases_in)
}

fn cases_in(data: &'static str) -> impl Iterator<Item = Case<'static>> {
    data.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split('\t');
            let tier = fields.next().unwrap();
            let scalar = fields.next().unwrap();
            let function = fields.next().unwrap();
            let encoded_arguments = fields.next().unwrap();
            let arguments = encoded_arguments
                .split(',')
                .map(|encoded| {
                    let (kind, value) = encoded.split_once(':').unwrap();
                    match kind {
                        "f" => Argument::Float(u64::from_str_radix(value, 16).unwrap()),
                        "u" => Argument::Unsigned(value.parse().unwrap()),
                        "i" => Argument::Signed(value.parse().unwrap()),
                        _ => panic!("unknown oracle argument kind: {kind}"),
                    }
                })
                .collect();
            let expected_re = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let expected_im = match fields.next().unwrap() {
                "-" => None,
                encoded => Some(u64::from_str_radix(encoded, 16).unwrap()),
            };
            Case {
                tier,
                scalar,
                function,
                arguments,
                encoded_arguments,
                expected_re,
                expected_im,
                reference_offset: fields.next().map_or(0.0, |value| value.parse().unwrap()),
            }
        })
}

// The integral corpus retains derivative and logarithmic references in its
// native format; reuse the ordinary rows for the common error-cover contract.
fn integral_cases() -> impl Iterator<Item = Case<'static>> {
    include_str!("data/integrals.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let p: Vec<_> = line.split_whitespace().collect();
            if !matches!(
                p[0],
                "expint_e1"
                    | "expint_ei"
                    | "expint_ei_scaled"
                    | "expint_en"
                    | "expint_en_scaled"
                    | "expint_3"
                    | "si"
                    | "ci"
            ) {
                return None;
            }
            let mut arguments = Vec::new();
            if p[0].starts_with("expint_en") {
                arguments.push(Argument::Unsigned(p[2].parse().unwrap()));
            }
            arguments.push(Argument::Float(u64::from_str_radix(p[3], 16).unwrap()));
            Some(Case {
                tier: "extended",
                scalar: if p[1] == "64" { "f64" } else { "f32" },
                function: p[0],
                arguments,
                encoded_arguments: p[3],
                expected_re: u64::from_str_radix(p[4], 16).unwrap(),
                expected_im: None,
                reference_offset: p.get(7).map_or(0.0, |v| v.parse().unwrap()),
            })
        })
}

fn f64_arg(arguments: &[Argument], index: usize) -> f64 {
    let Argument::Float(bits) = arguments[index] else {
        panic!("expected floating-point oracle argument")
    };
    f64::from_bits(bits)
}

fn f32_arg(arguments: &[Argument], index: usize) -> f32 {
    let Argument::Float(bits) = arguments[index] else {
        panic!("expected floating-point oracle argument")
    };
    f32::from_bits(u32::try_from(bits).unwrap())
}

fn usize_arg(arguments: &[Argument], index: usize) -> usize {
    let Argument::Unsigned(value) = arguments[index] else {
        panic!("expected unsigned oracle argument")
    };
    value
}

fn i32_arg(arguments: &[Argument], index: usize) -> i32 {
    match arguments[index] {
        Argument::Signed(value) => value,
        Argument::Unsigned(value) => i32::try_from(value).unwrap(),
        Argument::Float(_) => panic!("expected signed oracle argument"),
    }
}

#[allow(clippy::too_many_lines)]
fn invoke_f64(case: &Case<'_>) -> Result<(f64, Option<f64>), scioxide_core::SciOxideError> {
    let a = &case.arguments;
    let value = match case.function {
        "coulomb_f" | "coulomb_g" | "coulomb_fp" | "coulomb_gp" => {
            let result = coulomb_wave_with_options(
                f64_arg(a, 0),
                f64_arg(a, 1),
                f64_arg(a, 2),
                scioxide_special::CoulombOptions::default(),
            )?;
            let component = match case.function {
                "coulomb_f" => result.f(),
                "coulomb_g" => result.g(),
                "coulomb_fp" => result.f_prime(),
                _ => result.g_prime(),
            };
            (component.try_value()?, None)
        }
        "lambert_w0" => (lambert_w0(f64_arg(a, 0))?, None),
        "lambert_wm1" => (lambert_wm1(f64_arg(a, 0))?, None),
        "wright_omega" => (wright_omega(f64_arg(a, 0))?, None),
        "dilog" => (dilog(f64_arg(a, 0))?, None),
        "spence" => (spence(f64_arg(a, 0))?, None),
        "ln_gamma" => (ln_gamma(f64_arg(a, 0))?, None),
        "gamma" => (gamma(f64_arg(a, 0))?, None),
        "gamma_log" => {
            let report = gamma_with_options(f64_arg(a, 0), GammaOptions::default())?;
            let value = report.logarithmic().unwrap().value();
            (
                value.log_magnitude().unwrap(),
                Some(value.scaled_phase().round().unwrap()),
            )
        }
        "digamma" => (digamma(f64_arg(a, 0))?, None),
        "polygamma" => (polygamma(usize_arg(a, 0), f64_arg(a, 1))?, None),
        "factorial" => (factorial::<f64>(usize_arg(a, 0))?, None),
        "ln_factorial" => (ln_factorial::<f64>(usize_arg(a, 0))?, None),
        "binomial" => (binomial::<f64>(usize_arg(a, 0), usize_arg(a, 1))?, None),
        "gamma_p" => (gamma_p(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "gamma_q" => (gamma_q(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "gamma_lower" => (gamma_lower(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "gamma_upper" => (gamma_upper(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "gamma_p_inv" => (gamma_p_inv(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "ln_beta" => (ln_beta(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "beta" => (beta(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "beta_inc" => (beta_inc(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?, None),
        "beta_inc_q" => (
            beta_inc_q(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?,
            None,
        ),
        "beta_inc_inv" => (
            beta_inc_inv(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?,
            None,
        ),
        "erf" => (erf(f64_arg(a, 0))?, None),
        "erfc" => (erfc(f64_arg(a, 0))?, None),
        "erfcx" => (erfcx(f64_arg(a, 0))?, None),
        "log_erfc" => (log_erfc(f64_arg(a, 0))?, None),
        "dawson" => (dawson(f64_arg(a, 0))?, None),
        "erfinv" => (erfinv(f64_arg(a, 0))?, None),
        "erfcinv" => (erfcinv(f64_arg(a, 0))?, None),
        "airy_ai" => (airy_ai(f64_arg(a, 0))?, None),
        "airy_bi" => (airy_bi(f64_arg(a, 0))?, None),
        "airy_ai_scaled" => (airy_ai_scaled(f64_arg(a, 0))?, None),
        "airy_bi_scaled" => (airy_bi_scaled(f64_arg(a, 0))?, None),
        "bessel_j" => (bessel_j(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "bessel_y" => (bessel_y(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "bessel_i" => (bessel_i(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "bessel_i_scaled" => (bessel_i_scaled(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "bessel_k" => (bessel_k(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "bessel_k_scaled" => (bessel_k_scaled(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "hankel_1" => {
            let value = hankel_1(f64_arg(a, 0), f64_arg(a, 1))?;
            (value.re, Some(value.im))
        }
        "hankel_2" => {
            let value = hankel_2(f64_arg(a, 0), f64_arg(a, 1))?;
            (value.re, Some(value.im))
        }
        "carlson_rf" => (
            carlson_rf(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?,
            None,
        ),
        "carlson_rd" => (
            carlson_rd(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?,
            None,
        ),
        "carlson_rj" => (
            carlson_rj(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2), f64_arg(a, 3))?,
            None,
        ),
        "jacobi_sn" => (jacobi_sn(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "elliptic_k" => (elliptic_k(f64_arg(a, 0))?, None),
        "elliptic_e" => (elliptic_e(f64_arg(a, 0))?, None),
        "elliptic_f" => (elliptic_f(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "elliptic_e_inc" => (elliptic_e_inc(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "elliptic_pi" => (elliptic_pi(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "hyp0f1" => (hyp0f1(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "hyp1f1" => (hyp1f1(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?, None),
        "hyp2f1" => (
            hyp2f1(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2), f64_arg(a, 3))?,
            None,
        ),
        "hyperu" => (hyperu(f64_arg(a, 0), f64_arg(a, 1), f64_arg(a, 2))?, None),
        "legendre_p" => (legendre_p(usize_arg(a, 0), f64_arg(a, 1))?, None),
        "legendre_p_assoc" => (
            legendre_p_assoc(usize_arg(a, 0), i32_arg(a, 1), f64_arg(a, 2))?,
            None,
        ),
        "legendre_q" => (legendre_q(usize_arg(a, 0), f64_arg(a, 1))?, None),
        "riemann_zeta" => (riemann_zeta(f64_arg(a, 0))?, None),
        "dirichlet_eta" => (dirichlet_eta(f64_arg(a, 0))?, None),
        "hurwitz_zeta" => (hurwitz_zeta(f64_arg(a, 0), f64_arg(a, 1))?, None),
        "riemann_zeta_complex" => {
            let value = riemann_zeta(scioxide_core::Complex::new(f64_arg(a, 0), f64_arg(a, 1)))?;
            (value.re, Some(value.im))
        }
        "dirichlet_eta_complex" => {
            let value = dirichlet_eta(scioxide_core::Complex::new(f64_arg(a, 0), f64_arg(a, 1)))?;
            (value.re, Some(value.im))
        }
        "lerch_phi_complex" => {
            let value = lerch_phi(
                scioxide_core::Complex::new(f64_arg(a, 0), f64_arg(a, 1)),
                scioxide_core::Complex::new(f64_arg(a, 2), f64_arg(a, 3)),
                scioxide_core::Complex::new(f64_arg(a, 4), f64_arg(a, 5)),
            )?;
            (value.re, Some(value.im))
        }
        "hurwitz_zeta_complex" => {
            let value = hurwitz_zeta(
                scioxide_core::Complex::new(f64_arg(a, 0), f64_arg(a, 1)),
                scioxide_core::Complex::new(f64_arg(a, 2), f64_arg(a, 3)),
            )?;
            (value.re, Some(value.im))
        }
        "hurwitz_zeta_log" => (hurwitz_zeta_log(f64_arg(a, 0), f64_arg(a, 1))?, None),
        name => panic!("unknown oracle function {name}"),
    };
    Ok(value)
}

#[allow(clippy::too_many_lines)]
fn invoke_f32(case: &Case<'_>) -> Result<(f32, Option<f32>), scioxide_core::SciOxideError> {
    let a = &case.arguments;
    let value = match case.function {
        "coulomb_f" | "coulomb_g" | "coulomb_fp" | "coulomb_gp" => {
            let result = coulomb_wave_with_options(
                f32_arg(a, 0),
                f32_arg(a, 1),
                f32_arg(a, 2),
                scioxide_special::CoulombOptions::default(),
            )?;
            let component = match case.function {
                "coulomb_f" => result.f(),
                "coulomb_g" => result.g(),
                "coulomb_fp" => result.f_prime(),
                _ => result.g_prime(),
            };
            (component.try_value()?, None)
        }
        "lambert_w0" => (lambert_w0(f32_arg(a, 0))?, None),
        "lambert_wm1" => (lambert_wm1(f32_arg(a, 0))?, None),
        "wright_omega" => (wright_omega(f32_arg(a, 0))?, None),
        "dilog" => (dilog(f32_arg(a, 0))?, None),
        "spence" => (spence(f32_arg(a, 0))?, None),
        "ln_gamma" => (ln_gamma(f32_arg(a, 0))?, None),
        "gamma" => (gamma(f32_arg(a, 0))?, None),
        "gamma_log" => {
            let report = gamma_with_options(f32_arg(a, 0), GammaOptions::default())?;
            let value = report.logarithmic().unwrap().value();
            (
                value.log_magnitude().unwrap(),
                Some(value.scaled_phase().round().unwrap()),
            )
        }
        "digamma" => (digamma(f32_arg(a, 0))?, None),
        "polygamma" => (polygamma(usize_arg(a, 0), f32_arg(a, 1))?, None),
        "factorial" => (factorial::<f32>(usize_arg(a, 0))?, None),
        "ln_factorial" => (ln_factorial::<f32>(usize_arg(a, 0))?, None),
        "binomial" => (binomial::<f32>(usize_arg(a, 0), usize_arg(a, 1))?, None),
        "gamma_p" => (gamma_p(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "gamma_q" => (gamma_q(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "gamma_lower" => (gamma_lower(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "gamma_upper" => (gamma_upper(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "gamma_p_inv" => (gamma_p_inv(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "ln_beta" => (ln_beta(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "beta" => (beta(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "beta_inc" => (beta_inc(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?, None),
        "beta_inc_q" => (
            beta_inc_q(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?,
            None,
        ),
        "beta_inc_inv" => (
            beta_inc_inv(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?,
            None,
        ),
        "erf" => (erf(f32_arg(a, 0))?, None),
        "erfc" => (erfc(f32_arg(a, 0))?, None),
        "erfcx" => (erfcx(f32_arg(a, 0))?, None),
        "log_erfc" => (log_erfc(f32_arg(a, 0))?, None),
        "dawson" => (dawson(f32_arg(a, 0))?, None),
        "erfinv" => (erfinv(f32_arg(a, 0))?, None),
        "erfcinv" => (erfcinv(f32_arg(a, 0))?, None),
        "airy_ai" => (airy_ai(f32_arg(a, 0))?, None),
        "airy_bi" => (airy_bi(f32_arg(a, 0))?, None),
        "airy_ai_scaled" => (airy_ai_scaled(f32_arg(a, 0))?, None),
        "airy_bi_scaled" => (airy_bi_scaled(f32_arg(a, 0))?, None),
        "bessel_j" => (bessel_j(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "bessel_y" => (bessel_y(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "bessel_i" => (bessel_i(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "bessel_i_scaled" => (bessel_i_scaled(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "bessel_k" => (bessel_k(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "bessel_k_scaled" => (bessel_k_scaled(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "hankel_1" => {
            let value = hankel_1(f32_arg(a, 0), f32_arg(a, 1))?;
            (value.re, Some(value.im))
        }
        "hankel_2" => {
            let value = hankel_2(f32_arg(a, 0), f32_arg(a, 1))?;
            (value.re, Some(value.im))
        }
        "carlson_rf" => (
            carlson_rf(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?,
            None,
        ),
        "carlson_rd" => (
            carlson_rd(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?,
            None,
        ),
        "carlson_rj" => (
            carlson_rj(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2), f32_arg(a, 3))?,
            None,
        ),
        "jacobi_sn" => (jacobi_sn(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "elliptic_k" => (elliptic_k(f32_arg(a, 0))?, None),
        "elliptic_e" => (elliptic_e(f32_arg(a, 0))?, None),
        "elliptic_f" => (elliptic_f(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "elliptic_e_inc" => (elliptic_e_inc(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "elliptic_pi" => (elliptic_pi(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "hyp0f1" => (hyp0f1(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "hyp1f1" => (hyp1f1(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?, None),
        "hyp2f1" => (
            hyp2f1(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2), f32_arg(a, 3))?,
            None,
        ),
        "hyperu" => (hyperu(f32_arg(a, 0), f32_arg(a, 1), f32_arg(a, 2))?, None),
        "legendre_p" => (legendre_p(usize_arg(a, 0), f32_arg(a, 1))?, None),
        "legendre_p_assoc" => (
            legendre_p_assoc(usize_arg(a, 0), i32_arg(a, 1), f32_arg(a, 2))?,
            None,
        ),
        "legendre_q" => (legendre_q(usize_arg(a, 0), f32_arg(a, 1))?, None),
        "riemann_zeta" => (riemann_zeta(f32_arg(a, 0))?, None),
        "dirichlet_eta" => (dirichlet_eta(f32_arg(a, 0))?, None),
        "hurwitz_zeta" => (hurwitz_zeta(f32_arg(a, 0), f32_arg(a, 1))?, None),
        "riemann_zeta_complex" => {
            let value = riemann_zeta(scioxide_core::Complex::new(f32_arg(a, 0), f32_arg(a, 1)))?;
            (value.re, Some(value.im))
        }
        "dirichlet_eta_complex" => {
            let value = dirichlet_eta(scioxide_core::Complex::new(f32_arg(a, 0), f32_arg(a, 1)))?;
            (value.re, Some(value.im))
        }
        "lerch_phi_complex" => {
            let value = lerch_phi(
                scioxide_core::Complex::new(f32_arg(a, 0), f32_arg(a, 1)),
                scioxide_core::Complex::new(f32_arg(a, 2), f32_arg(a, 3)),
                scioxide_core::Complex::new(f32_arg(a, 4), f32_arg(a, 5)),
            )?;
            (value.re, Some(value.im))
        }
        "hurwitz_zeta_complex" => {
            let value = hurwitz_zeta(
                scioxide_core::Complex::new(f32_arg(a, 0), f32_arg(a, 1)),
                scioxide_core::Complex::new(f32_arg(a, 2), f32_arg(a, 3)),
            )?;
            (value.re, Some(value.im))
        }
        "hurwitz_zeta_log" => (hurwitz_zeta_log(f32_arg(a, 0), f32_arg(a, 1))?, None),
        name => panic!("unknown oracle function {name}"),
    };
    Ok(value)
}

fn target_ulp(function: &str, scalar: &str) -> u64 {
    let incomplete = matches!(
        function,
        "gamma_p"
            | "gamma_q"
            | "gamma_lower"
            | "gamma_upper"
            | "gamma_p_inv"
            | "beta_inc"
            | "beta_inc_q"
            | "beta_inc_inv"
    );
    let elementary = matches!(
        function,
        "gamma"
            | "gamma_log"
            | "ln_gamma"
            | "ln_beta"
            | "beta"
            | "digamma"
            | "polygamma"
            | "factorial"
            | "ln_factorial"
            | "binomial"
            | "erf"
            | "erfc"
            | "erfcx"
            | "log_erfc"
            | "dawson"
            | "erfinv"
            | "erfcinv"
    );
    match (scalar, elementary, incomplete) {
        ("f32", true, _) | ("f32", _, true) => 1,
        (_, true, _) => 4,
        (_, _, true) => 32,
        _ => 64,
    }
}

#[derive(Default)]
struct Measurement {
    cases: usize,
    ulps: Vec<u64>,
    errors: usize,
    misses: usize,
    conditioned: usize,
    ranges: usize,
    worst: String,
    max_unqualified: u64,
    worst_unqualified: String,
}
type Measurements = std::collections::BTreeMap<(&'static str, &'static str), Measurement>;

fn ordered(bits: u64, scalar: &str) -> u64 {
    let sign = if scalar == "f64" {
        1_u64 << 63
    } else {
        1_u64 << 31
    };
    if bits & sign != 0 {
        sign - (bits & (sign - 1))
    } else {
        sign + bits
    }
}
fn value(bits: u64, scalar: &str) -> f64 {
    if scalar == "f64" {
        f64::from_bits(bits)
    } else {
        f64::from(f32::from_bits(bits as u32))
    }
}
fn conditioned_bound(case: &Case<'_>, imaginary: bool) -> Option<f64> {
    include_str!("data/special_conditions.tsv")
        .lines()
        .chain(include_str!("data/airy_conditions.tsv").lines())
        .chain(include_str!("data/bessel_conditions.tsv").lines())
        .chain(include_str!("data/coulomb_conditions.tsv").lines())
        .chain(include_str!("data/gamma_beta_conditions.tsv").lines())
        .chain(include_str!("data/legendre_conditions.tsv").lines())
        .chain(include_str!("data/legendre_ownership_conditions.tsv").lines())
        .chain(include_str!("data/hypergeometric_conditions.tsv").lines())
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut fields = line.split('\t');
            if fields.next()? != case.scalar
                || fields.next()? != case.function
                || fields.next()? != case.encoded_arguments
            {
                return None;
            }
            let real = fields.next()?;
            let imag = fields.next()?;
            (if imaginary { imag } else { real }).parse().ok()
        })
}

fn measure(cases: impl Iterator<Item = Case<'static>>) -> Measurements {
    measure_with(cases, |case| {
        if case.scalar == "f64" {
            invoke_f64(case).map(|(re, im)| (re.to_bits(), im.map(f64::to_bits)))
        } else {
            invoke_f32(case)
                .map(|(re, im)| (u64::from(re.to_bits()), im.map(|v| u64::from(v.to_bits()))))
        }
    })
}

fn measure_with(
    cases: impl Iterator<Item = Case<'static>>,
    invoke: impl Fn(&Case<'_>) -> Result<(u64, Option<u64>), scioxide_core::SciOxideError>,
) -> Measurements {
    let mut measurements = Measurements::new();
    for case in cases {
        let entry = measurements
            .entry((case.function, case.scalar))
            .or_default();
        entry.cases += 1;
        let result = invoke(&case);
        if case.tier == "underflow" {
            // A nonzero value below the range is its correctly rounded zero.
            let Ok((re, im)) = &result else {
                panic!("{case:?}: expected a rounded zero, got {result:?}");
            };
            for bits in core::iter::once(*re).chain(*im) {
                assert_eq!(
                    value(bits, case.scalar),
                    0.0,
                    "{case:?}: expected a rounded zero, got {result:?}"
                );
            }
            entry.ranges += 1;
            continue;
        }
        let expected = value(case.expected_re, case.scalar);
        if expected.is_infinite()
            || case
                .expected_im
                .is_some_and(|bits| value(bits, case.scalar).is_infinite())
        {
            let bessel =
                case.function.starts_with("bessel_") || case.function.starts_with("hankel_");
            let correct = match &result {
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: scioxide_core::NumericalIssue::Overflow,
                            ..
                        }
                    ) =>
                {
                    bessel
                        || matches!(case.function, "hyp0f1" | "hyp1f1" | "hyp2f1")
                        || case.function.starts_with("coulomb_")
                        || matches!(
                            case.function,
                            "factorial"
                                | "gamma"
                                | "airy_bi"
                                | "riemann_zeta"
                                | "dirichlet_eta"
                                | "hurwitz_zeta"
                                | "legendre_p"
                                | "legendre_p_assoc"
                        )
                }
                Ok((bits, _)) => {
                    value(*bits, case.scalar) == expected
                        && !matches!(case.function, "factorial" | "gamma" | "airy_bi")
                }
                _ => false,
            };
            assert!(
                correct,
                "{} {} {}: expected range classification, got {result:?}",
                case.function, case.scalar, case.encoded_arguments
            );
            entry.ranges += 1;
            continue;
        }
        let (real, imaginary) = match result {
            Ok(value) => value,
            Err(error) => {
                entry.errors += 1;
                entry.misses += 1;
                if entry.errors == 1 {
                    eprintln!(
                        "oracle_error\t{}\t{}\t{}\t{error}",
                        case.function, case.scalar, case.encoded_arguments
                    );
                }
                continue;
            }
        };
        assert_eq!(
            imaginary.is_some(),
            case.expected_im.is_some(),
            "oracle component layout changed"
        );
        for (component, (actual, expected)) in core::iter::once((real, case.expected_re))
            .chain(imaginary.zip(case.expected_im))
            .enumerate()
        {
            assert!(
                value(actual, case.scalar).is_finite(),
                "unexpected nonfinite oracle result"
            );
            let ulps = ordered(actual, case.scalar).abs_diff(ordered(expected, case.scalar));
            if entry
                .ulps
                .iter()
                .max()
                .is_none_or(|maximum| ulps > *maximum)
            {
                entry.worst = case.encoded_arguments.to_owned();
            }
            entry.ulps.push(ulps);
            if ulps > target_ulp(case.function, case.scalar) {
                if conditioned_bound(&case, component != 0).is_some_and(|bound| {
                    (value(actual, case.scalar) - value(expected, case.scalar)).abs() <= bound
                }) {
                    entry.conditioned += 1;
                } else {
                    entry.misses += 1;
                    if ulps > entry.max_unqualified {
                        entry.max_unqualified = ulps;
                        entry.worst_unqualified = case.encoded_arguments.to_owned();
                    }
                }
            }
        }
    }
    measurements
}

fn assert_within_targets(measurements: &Measurements) {
    for (&(function, scalar), measurement) in measurements {
        assert_eq!(
            measurement.misses,
            0,
            "{function} {scalar}: observations exceed the {}-ULP target; worst {}",
            target_ulp(function, scalar),
            measurement.worst_unqualified
        );
    }
}

#[test]
fn quick_high_precision_oracle() {
    assert_within_targets(&measure(cases().filter(|case| case.tier == "quick")));
}

fn erfc_case(case: &Case<'_>) -> bool {
    case.function == "erfc"
}

#[test]
fn erfc_targets_qualified() {
    let measurements = measure(cases().filter(erfc_case));
    assert_eq!(
        measurements.len(),
        2,
        "missing erfc function/precision pairs"
    );
    for ((function, scalar), measurement) in measurements {
        assert!(measurement.cases >= 1000, "missing dense coverage");
        assert_eq!(
            measurement.errors, 0,
            "{function} {scalar}: numerical failures"
        );
        assert_eq!(
            measurement.conditioned, 0,
            "erfc needs no conditioning exceptions"
        );
        assert_eq!(
            measurement.misses, 0,
            "{function} {scalar}: worst {} ({} ULP)",
            measurement.worst_unqualified, measurement.max_unqualified
        );
    }
}

/// The shared erfcx and Dawson kernels and log erfc: dense seeded rows, every
/// bin edge and cutover, both precisions, and no conditioning exceptions.
#[test]
fn real_faddeeva_targets_qualified() {
    let measurements =
        measure(cases().filter(|case| matches!(case.function, "erfcx" | "log_erfc" | "dawson")));
    assert_eq!(measurements.len(), 6, "missing function/precision pairs");
    for ((function, scalar), measurement) in measurements {
        assert!(measurement.cases >= 700, "missing dense coverage");
        assert_eq!(measurement.errors, 0, "{function} {scalar}: failures");
        assert_eq!(measurement.conditioned, 0, "{function} {scalar}");
        assert_eq!(
            measurement.misses, 0,
            "{function} {scalar}: worst {} ({} ULP)",
            measurement.worst_unqualified, measurement.max_unqualified
        );
    }
}

#[test]
fn extended_bessel_high_precision_oracle() {
    assert_within_targets(&measure(cases().filter(|case| {
        case.tier == "extended"
            && (case.function.starts_with("bessel_") || case.function.starts_with("hankel_"))
    })));
}

#[test]
fn bessel_hankel_targets_qualified() {
    let measurements = measure(cases().filter(|case| {
        case.function.starts_with("bessel_") || case.function.starts_with("hankel_")
    }));
    assert_eq!(measurements.len(), 16, "missing Bessel/Hankel oracle pairs");
    for ((function, scalar), measurement) in measurements {
        assert!(
            measurement.cases >= 1000 && measurement.misses == 0 && measurement.errors == 0,
            "{function} {scalar}: {} cases, {} target misses, {} numerical failures; worst unqualified {}",
            measurement.cases,
            measurement.misses,
            measurement.errors,
            measurement.worst_unqualified
        );
    }
}

#[test]
fn airy_f64_targets_qualified() {
    let measurements = measure(cases().filter(|case| {
        case.scalar == "f64"
            && matches!(
                case.function,
                "airy_ai" | "airy_ai_scaled" | "airy_bi" | "airy_bi_scaled"
            )
    }));
    assert_eq!(measurements.len(), 4, "missing Airy f64 oracle functions");
    for ((function, scalar), measurement) in measurements {
        assert!(
            measurement.cases >= 1000 && measurement.misses == 0 && measurement.errors == 0,
            "{function} {scalar}: {} cases, {} target misses, {} numerical failures; worst unqualified {}",
            measurement.cases,
            measurement.misses,
            measurement.errors,
            measurement.worst_unqualified
        );
    }
}

fn coulomb_cases() -> impl Iterator<Item = Case<'static>> {
    cases().filter(|case| case.function.starts_with("coulomb_"))
}

fn coulomb_scalar(case: &Case<'_>) -> Result<(u64, Option<u64>), scioxide_core::SciOxideError> {
    macro_rules! evaluate {
        ($arg:ident) => {{
            let a = &case.arguments;
            let (l, eta, x) = ($arg(a, 0), $arg(a, 1), $arg(a, 2));
            match case.function {
                "coulomb_f" => coulomb_f(l, eta, x),
                "coulomb_fp" => coulomb_f_prime(l, eta, x),
                "coulomb_g" => coulomb_g(l, eta, x),
                "coulomb_gp" => coulomb_g_prime(l, eta, x),
                _ => unreachable!(),
            }
        }};
    }
    let bits = if case.scalar == "f64" {
        evaluate!(f64_arg)?.to_bits()
    } else {
        u64::from(evaluate!(f32_arg)?.to_bits())
    };
    Ok((bits, None))
}

#[test]
fn coulomb_targets_qualified() {
    for (surface, measurements) in [
        ("paired", measure(coulomb_cases())),
        ("scalar", measure_with(coulomb_cases(), coulomb_scalar)),
    ] {
        assert_eq!(measurements.len(), 8, "missing Coulomb oracle pairs");
        for ((function, scalar), measurement) in measurements {
            assert!(
                measurement.cases >= 1000 && measurement.misses == 0 && measurement.errors == 0,
                "{surface} {function} {scalar}: {} cases, {} target misses, {} numerical failures; worst unqualified {} ({} ULP)",
                measurement.cases,
                measurement.misses,
                measurement.errors,
                measurement.worst_unqualified,
                measurement.max_unqualified
            );
        }
    }
}

#[test]
fn coulomb_sequences_qualified() {
    let oracle: Vec<_> = coulomb_cases().collect();
    macro_rules! check {
        ($scalar:ty, $name:literal, $width:literal) => {
            for eta in [-2.0 as $scalar, 2.0] {
                for x in [0.125 as $scalar, 2.0, 32.0, 1000.0] {
                    let seed = coulomb_wave_with_options(
                        0.25,
                        eta,
                        x,
                        scioxide_special::CoulombOptions::default(),
                    )
                    .unwrap();
                    let mut output = [seed; 16];
                    let mut scratch = output;
                    coulomb_wave_sequence_into(
                        0.25,
                        eta,
                        x,
                        CoulombOptions::default(),
                        &mut output,
                        &mut scratch,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                    for (index, report) in output.into_iter().enumerate() {
                        let l = 0.25 + index as $scalar;
                        let encoded = format!(
                            "f:{:0width$x},f:{:0width$x},f:{:0width$x}",
                            l.to_bits(),
                            eta.to_bits(),
                            x.to_bits(),
                            width = $width
                        );
                        for (name, component) in [
                            ("coulomb_f", report.f()),
                            ("coulomb_fp", report.f_prime()),
                            ("coulomb_g", report.g()),
                            ("coulomb_gp", report.g_prime()),
                        ] {
                            let case = oracle
                                .iter()
                                .find(|case| {
                                    case.scalar == $name
                                        && case.function == name
                                        && case.encoded_arguments == encoded
                                })
                                .unwrap();
                            let actual = component.try_value().unwrap().to_bits() as u64;
                            let ulps =
                                ordered(actual, $name).abs_diff(ordered(case.expected_re, $name));
                            assert!(
                                ulps <= 64
                                    || conditioned_bound(case, false).is_some_and(|bound| (value(
                                        actual, $name
                                    )
                                        - value(case.expected_re, $name))
                                    .abs()
                                        <= bound),
                                "sequence {name} {} {encoded}: {ulps} ULP",
                                $name
                            );
                        }
                    }
                }
            }
        };
    }
    check!(f64, "f64", 16);
    check!(f32, "f32", 8);
}

#[test]
fn gamma_beta_targets_qualified() {
    let measurements = measure(cases().filter(|case| {
        matches!(
            case.function,
            "beta"
                | "beta_inc"
                | "beta_inc_inv"
                | "binomial"
                | "gamma_log"
                | "gamma"
                | "digamma"
                | "gamma_p"
                | "gamma_q"
                | "gamma_upper"
                | "ln_beta"
                | "ln_gamma"
                | "polygamma"
        )
    }));
    assert_eq!(measurements.len(), 26, "missing gamma/beta oracle pairs");
    for ((function, scalar), measurement) in measurements {
        if matches!(function, "ln_gamma" | "ln_beta") {
            assert!(
                measurement
                    .ulps
                    .iter()
                    .all(|&ulps| ulps <= target_ulp(function, scalar)),
                "{function} {scalar}: dense maximum {:?} exceeds target: {}",
                measurement.ulps.iter().max(),
                measurement.worst
            );
        }
        assert!(
            measurement.cases >= 1000 && measurement.misses == 0 && measurement.errors == 0,
            "{function} {scalar}: {} cases, {} target misses, {} numerical failures; worst unqualified {}",
            measurement.cases,
            measurement.misses,
            measurement.errors,
            measurement.worst_unqualified
        );
    }
}

#[test]
fn gamma_beta_measured_envelopes() {
    let mut maxima = std::collections::BTreeMap::<&str, (f64, String, usize)>::new();
    for case in cases().filter(|case| {
        case.tier == "measured"
            && matches!(
                case.function,
                "gamma"
                    | "ln_gamma"
                    | "digamma"
                    | "polygamma"
                    | "beta"
                    | "ln_beta"
                    | "gamma_p"
                    | "gamma_q"
                    | "beta_inc"
            )
    }) {
        let actual = invoke_f64(&case).unwrap().0;
        let expected = f64::from_bits(case.expected_re);
        let unit = expected.abs().next_up() - expected.abs();
        let ulps = ((actual - expected) / unit - case.reference_offset).abs();
        let report = detailed::<f64>(&case).unwrap().unwrap();
        assert_eq!(actual.to_bits(), report.value.0.to_bits(), "{case:?}");
        assert!(
            ulps * unit <= report.absolute_error,
            "{case:?}: error estimate"
        );
        let entry = maxima.entry(case.function).or_default();
        entry.2 += 1;
        if ulps > entry.0 {
            entry.0 = ulps;
            entry.1 = case.encoded_arguments.to_owned();
        }
    }
    let mut misses = 0;
    for (name, (maximum, worst, count)) in &maxima {
        let target = match *name {
            "gamma" => 3.7,
            "ln_gamma" => 1.9,
            "digamma" => 2.1,
            "polygamma" => 0.7,
            "beta" => 1.0,
            "ln_beta" => 0.5,
            "gamma_p" | "gamma_q" => 90.0,
            "beta_inc" => 12.0,
            _ => unreachable!(),
        };
        eprintln!("{name}: {maximum:.6} ULP ({count} cases); worst {worst}");
        assert_eq!(*count, 4160);
        misses += usize::from(*maximum > target);
    }
    assert_eq!(maxima.len(), 9);
    assert_eq!(misses, 0, "measured envelopes regressed");
}

#[test]
fn complex_zeta_continuation_references_retain_accurate_components() {
    for case in cases_in(COMPLEX_ZETA) {
        let actual = if case.scalar == "f64" {
            invoke_f64(&case).unwrap()
        } else {
            let (re, im) = invoke_f32(&case).unwrap();
            (f64::from(re), im.map(f64::from))
        };
        let relative = if case.scalar == "f64" {
            5e-11
        } else {
            32.0 * f64::from(f32::EPSILON)
        };
        let scale = if matches!(case.function, "hurwitz_zeta_complex" | "lerch_phi_complex") {
            value(case.expected_re, case.scalar)
                .hypot(value(case.expected_im.unwrap(), case.scalar))
        } else {
            0.0
        };
        let zero_bound = if scale == 0.0
            && matches!(case.function, "hurwitz_zeta_complex" | "lerch_phi_complex")
        {
            if case.scalar == "f64" {
                detailed::<f64>(&case).unwrap().unwrap().absolute_error
            } else {
                detailed::<f32>(&case).unwrap().unwrap().absolute_error
            }
        } else {
            0.0
        };
        for (actual, expected) in [
            (actual.0, case.expected_re),
            (actual.1.unwrap(), case.expected_im.unwrap()),
        ] {
            let expected = value(expected, case.scalar);
            let floor = if case.scalar == "f64" {
                f64::from_bits(1)
            } else {
                f64::from(f32::from_bits(1))
            };
            assert!(
                (actual - expected).abs()
                    <= relative * expected.abs().max(scale) + zero_bound + 8.0 * floor,
                "{case:?}: actual={actual:e}, expected={expected:e}"
            );
        }
    }
}

#[test]
fn zeta_eta_targets_qualified() {
    for detailed in [false, true] {
        let measurements = measure_with(
            cases().filter(|case| matches!(case.function, "riemann_zeta" | "dirichlet_eta")),
            |case| {
                if !detailed {
                    return if case.scalar == "f64" {
                        invoke_f64(case).map(|(value, _)| (value.to_bits(), None))
                    } else {
                        invoke_f32(case).map(|(value, _)| (u64::from(value.to_bits()), None))
                    };
                }
                macro_rules! evaluate {
                    ($scalar:ty, $argument:expr) => {{
                        let result = if case.function == "riemann_zeta" {
                            riemann_zeta_with_options($argument, ZetaOptions::default())?
                        } else {
                            dirichlet_eta_with_options($argument, ZetaOptions::default())?
                        };
                        assert!(result.termination().is_success());
                        let value: $scalar = *result.value().unwrap().value();
                        Ok((u64::from(value.to_bits()), None))
                    }};
                }
                if case.scalar == "f64" {
                    evaluate!(f64, f64_arg(&case.arguments, 0))
                } else {
                    evaluate!(f32, f32_arg(&case.arguments, 0))
                }
            },
        );
        assert_eq!(measurements.len(), 4);
        for ((function, scalar), measured) in measurements {
            assert!(
                measured.cases >= 1000 && measured.misses == 0 && measured.errors == 0,
                "{function} {scalar} detailed={detailed}: {} misses, {} errors; worst {} ({} ULP)",
                measured.misses,
                measured.errors,
                measured.worst_unqualified,
                measured.max_unqualified
            );
        }
    }
}

fn legendre_case(case: &Case<'_>) -> bool {
    matches!(
        case.function,
        "legendre_p" | "legendre_p_assoc" | "legendre_q"
    )
}

#[test]
fn legendre_targets_qualified() {
    for detailed in [false, true] {
        let measurements = measure_with(cases().filter(legendre_case), |case| {
            if !detailed {
                return if case.scalar == "f64" {
                    invoke_f64(case).map(|(v, _)| (v.to_bits(), None))
                } else {
                    invoke_f32(case).map(|(v, _)| (u64::from(v.to_bits()), None))
                };
            }
            macro_rules! evaluate {
                ($scalar:ty, $x:expr) => {{
                    let degree = usize_arg(&case.arguments, 0);
                    let report = match case.function {
                        "legendre_p" => {
                            legendre_p_with_options(degree, $x, LegendreOptions::default())
                        }
                        "legendre_q" => {
                            legendre_q_with_options(degree, $x, LegendreOptions::default())
                        }
                        _ => legendre_p_assoc_with_options(
                            degree,
                            i32_arg(&case.arguments, 1),
                            $x,
                            LegendreOptions::default(),
                        ),
                    }?;
                    assert!(
                        report.termination().is_success(),
                        "{case:?}: {:?}",
                        report.termination()
                    );
                    let value: $scalar = report.try_value()?;
                    Ok((u64::from(value.to_bits()), None))
                }};
            }
            let last = case.arguments.len() - 1;
            if case.scalar == "f64" {
                evaluate!(f64, f64_arg(&case.arguments, last))
            } else {
                evaluate!(f32, f32_arg(&case.arguments, last))
            }
        });
        assert_eq!(measurements.len(), 6);
        for ((function, scalar), measured) in measurements {
            assert!(
                measured.cases >= 1000 && measured.misses == 0 && measured.errors == 0,
                "{function} {scalar} detailed={detailed}: {} misses, {} errors; worst {} ({} ULP)",
                measured.misses,
                measured.errors,
                measured.worst_unqualified,
                measured.max_unqualified
            );
        }
    }
}

fn hypergeometric_case(case: &Case<'_>) -> bool {
    matches!(case.function, "hyp0f1" | "hyp1f1" | "hyp2f1")
}

fn hypergeometric_measurements(surface: usize) -> Measurements {
    measure_with(cases().filter(hypergeometric_case), |case| {
        macro_rules! evaluate {
            ($scalar:ty, $arg:ident) => {{
                let args: Vec<$scalar> = (0..case.arguments.len())
                    .map(|i| $arg(&case.arguments, i))
                    .collect();
                let n = args.len();
                let value = if surface == 0 {
                    match case.function {
                        "hyp0f1" => hyp0f1(args[0], args[1]),
                        "hyp1f1" => hyp1f1(args[0], args[1], args[2]),
                        _ => hyp2f1(args[0], args[1], args[2], args[3]),
                    }
                } else if surface == 1 {
                    match case.function {
                        "hyp0f1" => {
                            hyp0f1_with_options(args[0], args[1], HypergeometricOptions::default())
                        }
                        "hyp1f1" => hyp1f1_with_options(
                            args[0],
                            args[1],
                            args[2],
                            HypergeometricOptions::default(),
                        ),
                        _ => hyp2f1_with_options(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            HypergeometricOptions::default(),
                        ),
                    }?
                    .try_value()
                } else {
                    hyp_pfq_with_options(
                        &args[..n - 2],
                        &args[n - 2..n - 1],
                        args[n - 1],
                        HypergeometricOptions::default(),
                    )?
                    .try_value()
                }?;
                Ok((u64::from(value.to_bits()), None))
            }};
        }
        if case.scalar == "f64" {
            evaluate!(f64, f64_arg)
        } else {
            evaluate!(f32, f32_arg)
        }
    })
}

#[test]
fn hypergeometric_targets_qualified() {
    for surface in 0..3 {
        let measured = hypergeometric_measurements(surface);
        assert_eq!(measured.len(), 6);
        for ((function, scalar), m) in measured {
            assert!(
                m.cases >= 1000 && m.misses == 0 && m.errors == 0,
                "{function} {scalar} surface={surface}: {} misses, {} errors; worst {} ({} ULP)",
                m.misses,
                m.errors,
                m.worst_unqualified,
                m.max_unqualified
            );
        }
    }
}

/// These condition classifications were independently reproduced at 160 and
/// 240 digits. They never replace the ordinary oracle or error-cover gates.
#[test]
fn well_conditioned_kummer_retains_twelve_digits() {
    let mut count = [0; 2];
    for case in cases().filter(|case| case.tier == "well_conditioned") {
        let expected = value(case.expected_re, case.scalar);
        if !expected.is_finite() || expected == 0.0 {
            continue;
        }
        let (actual, target, index) = if case.scalar == "f64" {
            (detailed::<f64>(&case).unwrap().unwrap().value.0, 1e-12, 0)
        } else {
            (
                detailed::<f32>(&case).unwrap().unwrap().value.0,
                64.0 * f64::from(f32::EPSILON),
                1,
            )
        };
        count[index] += 1;
        assert!(
            (actual - expected).abs() <= target * expected.abs(),
            "{case:?}: {actual:e} vs {expected:e}"
        );
    }
    assert!(count[0] >= 100 && count[1] >= 50, "{count:?}");
}

#[test]
fn hypergeometric_reported_errors_cover_every_reference() {
    for case in cases_in(HYPERGEOMETRIC) {
        let expected = value(case.expected_re, case.scalar);
        if !expected.is_finite() || case.tier == "underflow" {
            continue;
        }
        let reported = if case.scalar == "f64" {
            detailed::<f64>(&case)
        } else {
            detailed::<f32>(&case)
        }
        .unwrap()
        .unwrap_or_else(|error| panic!("{case:?}: {error}"));
        let actual = reported.value.0;
        assert!(
            (actual - expected).abs()
                <= reported.absolute_error + half_spacing(case.expected_re, case.scalar),
            "{case:?}: actual={actual:e} expected={expected:e} error={:e}",
            reported.absolute_error
        );
    }
}

/// Tricomi U meets the default 64-ULP target on its seeded comparison domain
/// and on every region edge of the real kernel, in both precisions.
#[test]
fn hyperu_targets_qualified() {
    let measured = measure(cases().filter(|case| case.function == "hyperu"));
    assert_eq!(measured.len(), 2);
    for ((function, scalar), m) in measured {
        assert!(
            m.cases >= 150 && m.misses == 0 && m.errors == 0,
            "{function} {scalar}: {} misses, {} errors; worst {} ({} ULP)",
            m.misses,
            m.errors,
            m.worst_unqualified,
            m.max_unqualified
        );
    }
}

/// A detailed evaluation projected for comparison with an oracle row.
struct Reported {
    value: (f64, Option<f64>),
    absolute_error: f64,
    method: &'static str,
}

fn reported_real<S: scioxide_core::RealScalar>(
    result: SpecialFunctionResult<S>,
) -> Result<Reported, scioxide_core::SciOxideError> {
    let absolute_error = result
        .value()
        .map(|value| value.absolute_error().into_f64());
    let method = result.algorithm().method();
    let value = result.try_value()?.into_f64();
    Ok(Reported {
        value: (value, None),
        absolute_error: absolute_error.unwrap(),
        method,
    })
}

fn reported_complex<S>(
    result: SpecialFunctionResult<scioxide_core::Complex<S>>,
) -> Result<Reported, scioxide_core::SciOxideError>
where
    S: scioxide_core::RealScalar,
    scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let absolute_error = result
        .value()
        .map(|value| value.absolute_error().into_f64());
    let method = result.algorithm().method();
    let value = result.try_value()?;
    Ok(Reported {
        value: (value.re.into_f64(), Some(value.im.into_f64())),
        absolute_error: absolute_error.unwrap(),
        method,
    })
}

/// Detailed evaluation of an oracle row with default options; `None` when the
/// function publishes no error estimate (the IEEE-total error functions).
#[allow(clippy::too_many_lines)]
fn detailed<S>(case: &Case<'_>) -> Option<Result<Reported, scioxide_core::SciOxideError>>
where
    S: scioxide_core::RealScalar,
    scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let a = &case.arguments;
    let x = |index: usize| {
        if case.scalar == "f64" {
            S::try_from_f64_lossy(f64_arg(a, index)).unwrap()
        } else {
            S::try_from_f64_lossy(f64::from(f32_arg(a, index))).unwrap()
        }
    };
    let n = |index: usize| usize_arg(a, index);
    Some(match case.function {
        "lambert_w0" => {
            lambert_w0_with_options(x(0), LambertOptions::default()).and_then(reported_real)
        }
        "lambert_wm1" => {
            lambert_wm1_with_options(x(0), LambertOptions::default()).and_then(reported_real)
        }
        "wright_omega" => {
            wright_omega_with_options(x(0), LambertOptions::default()).and_then(reported_real)
        }
        "dilog" => dilog_with_options(x(0), PolylogOptions::default()).and_then(reported_real),
        "spence" => spence_with_options(x(0), PolylogOptions::default()).and_then(reported_real),
        "coulomb_f" | "coulomb_g" | "coulomb_fp" | "coulomb_gp" => {
            coulomb_wave_with_options(x(0), x(1), x(2), CoulombOptions::default()).and_then(
                |wave| {
                    reported_real(match case.function {
                        "coulomb_f" => wave.f(),
                        "coulomb_g" => wave.g(),
                        "coulomb_fp" => wave.f_prime(),
                        _ => wave.g_prime(),
                    })
                },
            )
        }
        "gamma_log" => gamma_with_options(x(0), GammaOptions::default()).map(|result| {
            let logarithm = result.logarithmic().unwrap();
            Reported {
                value: (
                    logarithm.value().log_magnitude().unwrap().into_f64(),
                    Some(logarithm.value().scaled_phase().round().unwrap().into_f64()),
                ),
                absolute_error: logarithm.absolute_error().into_f64(),
                method: result.algorithm().method(),
            }
        }),
        "ln_gamma" => ln_gamma_with_options(x(0), GammaOptions::default()).and_then(reported_real),
        "gamma" => gamma_with_options(x(0), GammaOptions::default()).and_then(reported_real),
        "digamma" => digamma_with_options(x(0), GammaOptions::default()).and_then(reported_real),
        "polygamma" => {
            polygamma_with_options(n(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "factorial" => factorial_with_options::<S>(n(0), CombinatorialOptions::default())
            .and_then(reported_real),
        "ln_factorial" => ln_factorial_with_options::<S>(n(0), CombinatorialOptions::default())
            .and_then(reported_real),
        "binomial" => binomial_with_options::<S>(n(0), n(1), CombinatorialOptions::default())
            .and_then(reported_real),
        "gamma_p" => {
            gamma_p_with_options(x(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "gamma_q" => {
            gamma_q_with_options(x(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "gamma_lower" => {
            gamma_lower_with_options(x(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "gamma_upper" => {
            gamma_upper_with_options(x(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "gamma_p_inv" => {
            gamma_p_inv_with_options(x(0), x(1), GammaOptions::default()).and_then(reported_real)
        }
        "ln_beta" => {
            ln_beta_with_options(x(0), x(1), BetaOptions::default()).and_then(reported_real)
        }
        "beta" => beta_with_options(x(0), x(1), BetaOptions::default()).and_then(reported_real),
        "beta_inc" => {
            beta_inc_with_options(x(0), x(1), x(2), BetaOptions::default()).and_then(reported_real)
        }
        "beta_inc_q" => beta_inc_q_with_options(x(0), x(1), x(2), BetaOptions::default())
            .and_then(reported_real),
        "beta_inc_inv" => beta_inc_inv_with_options(x(0), x(1), x(2), BetaOptions::default())
            .and_then(reported_real),
        "erf" => erf_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real),
        "erfc" => erfc_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real),
        "erfcx" => {
            erfcx_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real)
        }
        "log_erfc" => {
            log_erfc_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real)
        }
        "dawson" => {
            dawson_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real)
        }
        "erfinv" => {
            erfinv_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real)
        }
        "erfcinv" => {
            erfcinv_with_options(x(0), ErrorFunctionOptions::default()).and_then(reported_real)
        }
        "airy_ai" => airy_ai_with_options(x(0), AiryOptions::default()).and_then(reported_real),
        "airy_bi" => airy_bi_with_options(x(0), AiryOptions::default()).and_then(reported_real),
        "airy_ai_scaled" => {
            airy_ai_scaled_with_options(x(0), AiryOptions::default()).and_then(reported_real)
        }
        "airy_bi_scaled" => {
            airy_bi_scaled_with_options(x(0), AiryOptions::default()).and_then(reported_real)
        }
        "bessel_j" => {
            bessel_j_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_real)
        }
        "bessel_y" => {
            bessel_y_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_real)
        }
        "bessel_i" => {
            bessel_i_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_real)
        }
        "bessel_i_scaled" => bessel_i_scaled_with_options(x(0), x(1), BesselOptions::default())
            .and_then(reported_real),
        "bessel_k" => {
            bessel_k_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_real)
        }
        "bessel_k_scaled" => bessel_k_scaled_with_options(x(0), x(1), BesselOptions::default())
            .and_then(reported_real),
        "hankel_1" => {
            hankel_1_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_complex)
        }
        "hankel_2" => {
            hankel_2_with_options(x(0), x(1), BesselOptions::default()).and_then(reported_complex)
        }
        "carlson_rf" => carlson_rf_with_options(x(0), x(1), x(2), EllipticOptions::default())
            .and_then(reported_real),
        "carlson_rd" => carlson_rd_with_options(x(0), x(1), x(2), EllipticOptions::default())
            .and_then(reported_real),
        "carlson_rj" => carlson_rj_with_options(x(0), x(1), x(2), x(3), EllipticOptions::default())
            .and_then(reported_real),
        "jacobi_sn" => {
            jacobi_sn_with_options(x(0), x(1), EllipticOptions::default()).and_then(reported_real)
        }
        "elliptic_k" => {
            elliptic_k_with_options(x(0), EllipticOptions::default()).and_then(reported_real)
        }
        "elliptic_e" => {
            elliptic_e_with_options(x(0), EllipticOptions::default()).and_then(reported_real)
        }
        "elliptic_f" => {
            elliptic_f_with_options(x(0), x(1), EllipticOptions::default()).and_then(reported_real)
        }
        "elliptic_e_inc" => elliptic_e_inc_with_options(x(0), x(1), EllipticOptions::default())
            .and_then(reported_real),
        "elliptic_pi" => {
            elliptic_pi_with_options(x(0), x(1), EllipticOptions::default()).and_then(reported_real)
        }
        "hyp0f1" => hyp0f1_with_options(x(0), x(1), HypergeometricOptions::default())
            .and_then(reported_real),
        "hyp1f1" => hyp1f1_with_options(x(0), x(1), x(2), HypergeometricOptions::default())
            .and_then(reported_real),
        "hyp2f1" => hyp2f1_with_options(x(0), x(1), x(2), x(3), HypergeometricOptions::default())
            .and_then(reported_real),
        "hyperu" => hyperu_with_options(x(0), x(1), x(2), HypergeometricOptions::default())
            .and_then(reported_real),
        "legendre_p" => {
            legendre_p_with_options(n(0), x(1), LegendreOptions::default()).and_then(reported_real)
        }
        "legendre_p_assoc" => {
            legendre_p_assoc_with_options(n(0), i32_arg(a, 1), x(2), LegendreOptions::default())
                .and_then(reported_real)
        }
        "legendre_q" => {
            legendre_q_with_options(n(0), x(1), LegendreOptions::default()).and_then(reported_real)
        }
        "riemann_zeta" => {
            riemann_zeta_with_options(x(0), ZetaOptions::default()).and_then(reported_real)
        }
        "dirichlet_eta" => {
            dirichlet_eta_with_options(x(0), ZetaOptions::default()).and_then(reported_real)
        }
        "expint_e1" => {
            expint_e1_with_options(x(0), IntegralOptions::default()).and_then(reported_real)
        }
        "expint_ei" => {
            expint_ei_with_options(x(0), IntegralOptions::default()).and_then(reported_real)
        }
        "expint_ei_scaled" => {
            expint_ei_scaled_with_options(x(0), IntegralOptions::default()).and_then(reported_real)
        }
        "expint_en" => expint_en_with_options(n(0) as u32, x(1), IntegralOptions::default())
            .and_then(reported_real),
        "expint_en_scaled" => {
            expint_en_scaled_with_options(n(0) as u32, x(1), IntegralOptions::default())
                .and_then(reported_real)
        }
        "si" => si_with_options(x(0), IntegralOptions::default()).and_then(reported_real),
        "expint_3" => {
            expint_3_with_options(x(0), IntegralOptions::default()).and_then(reported_real)
        }
        "ci" => ci_with_options(x(0), IntegralOptions::default()).and_then(reported_real),
        "hurwitz_zeta_log" => hurwitz_zeta_log_with_options(x(0), x(1), ZetaOptions::default())
            .and_then(reported_real),
        "riemann_zeta_complex" => riemann_zeta_with_options(
            scioxide_core::Complex::new(x(0), x(1)),
            ZetaOptions::default(),
        )
        .and_then(reported_complex),
        "dirichlet_eta_complex" => dirichlet_eta_with_options(
            scioxide_core::Complex::new(x(0), x(1)),
            ZetaOptions::default(),
        )
        .and_then(reported_complex),
        "lerch_phi_complex" => lerch_phi_with_options(
            scioxide_core::Complex::new(x(0), x(1)),
            scioxide_core::Complex::new(x(2), x(3)),
            scioxide_core::Complex::new(x(4), x(5)),
            LerchOptions::default(),
        )
        .and_then(reported_complex),
        "hurwitz_zeta_complex" => hurwitz_zeta_with_options(
            scioxide_core::Complex::new(x(0), x(1)),
            scioxide_core::Complex::new(x(2), x(3)),
            ZetaOptions::default(),
        )
        .and_then(reported_complex),
        "hurwitz_zeta" => {
            hurwitz_zeta_with_options(x(0), x(1), ZetaOptions::default()).and_then(reported_real)
        }
        name => panic!("unknown oracle function {name}"),
    })
}

/// Half the spacing of the destination scalars at a reference value: the
/// rounding of the reference itself.
fn half_spacing(bits: u64, scalar: &str) -> f64 {
    if scalar == "f64" {
        let value = f64::from_bits(bits).abs();
        0.5 * (f64::from_bits(value.to_bits() + 1) - value)
    } else {
        let value = f32::from_bits(bits as u32).abs();
        0.5 * f64::from(f32::from_bits(value.to_bits() + 1) - value)
    }
}

#[derive(Default)]
struct Coverage {
    cases: usize,
    misses: usize,
    worst_ratio: f64,
    worst: String,
}

/// Every reported absolute error covers the observed error against the
/// checked-in references, allowing the half-unit rounding of the reference.
/// Condition-scaled rows instead satisfy their documented bound
/// `target ULP · Σ|∂f/∂xᵢ|·ulp(xᵢ)` (`tests/data/*_conditions.tsv`).
#[test]
fn reported_errors_cover_oracle_references() {
    let mut coverage = std::collections::BTreeMap::<(&str, &str), Coverage>::new();
    for case in cases().chain(integral_cases()) {
        let expected = [Some(case.expected_re), case.expected_im];
        if case.tier == "underflow"
            || expected
                .iter()
                .flatten()
                .any(|&bits| value(bits, case.scalar).is_infinite())
        {
            continue;
        }
        let result = if case.scalar == "f64" {
            detailed::<f64>(&case)
        } else {
            detailed::<f32>(&case)
        };
        let Some(result) = result else {
            continue;
        };
        let entry = coverage.entry((case.function, case.scalar)).or_default();
        entry.cases += 1;
        let reported = match result {
            Ok(reported) => reported,
            Err(error) => {
                entry.misses += 1;
                entry.worst_ratio = f64::INFINITY;
                entry.worst = format!("{} -> {error}", case.encoded_arguments);
                continue;
            }
        };
        let actual = [Some(reported.value.0), reported.value.1];
        for (component, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let (Some(actual), Some(expected)) = (actual, expected) else {
                continue;
            };
            let expected_value = value(expected, case.scalar);
            if case.function == "gamma_log" && component == 1 {
                // The phase of a real gamma value is an exact sign.
                assert_eq!(*actual, expected_value, "{case:?}: gamma sign");
                continue;
            }
            let observed = (actual - expected_value).abs();
            let allowed = reported.absolute_error + half_spacing(expected, case.scalar);
            let ratio = observed / allowed;
            if ratio > 1.0
                && !conditioned_bound(&case, component != 0).is_some_and(|bound| observed <= bound)
            {
                entry.misses += 1;
            }
            if ratio > entry.worst_ratio {
                entry.worst_ratio = ratio;
                entry.worst = format!(
                    "{} [{}] actual {actual:e} expected {expected_value:e} reported {:e}",
                    case.encoded_arguments, reported.method, reported.absolute_error
                );
            }
        }
    }
    let failures: Vec<_> = coverage
        .iter()
        .filter(|(_, entry)| entry.misses > 0)
        .map(|((function, scalar), entry)| {
            format!(
                "{function} {scalar}: {} of {} rows exceed the reported error; worst ratio {:.3e} at {}",
                entry.misses, entry.cases, entry.worst_ratio, entry.worst
            )
        })
        .collect();
    for ((function, scalar), entry) in &coverage {
        eprintln!(
            "coverage\t{function}\t{scalar}\t{}\t{}\t{:.3e}\t{}",
            entry.cases, entry.misses, entry.worst_ratio, entry.worst
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The Issue #1501 and #1502 corpora, qualified without condition-scaled
/// exceptions. Incomplete gamma beyond `a = 20`, through the fixed Temme band
/// edges and both tails up to `a = 1e10`, and the forward upper beta tails
/// complete under default options within their 32-ULP target and reported
/// error. Digamma's reported error covers rows beside its negative zeros.
#[test]
fn large_shapes_and_deep_tails_qualified_without_conditions() {
    let mut counts = std::collections::BTreeMap::<(&str, &str), usize>::new();
    let mut failures = Vec::new();
    for case in cases_in(GAMMA_BETA) {
        let argument = |index| {
            if case.scalar == "f64" {
                f64_arg(&case.arguments, index)
            } else {
                f64::from(f32_arg(&case.arguments, index))
            }
        };
        let incomplete = case.function == "beta_inc_q"
            || (matches!(
                case.function,
                "gamma_p" | "gamma_q" | "gamma_lower" | "gamma_upper" | "gamma_p_inv"
            ) && argument(0) > 20.0);
        let reflected_digamma = case.function == "digamma" && argument(0) < 0.0;
        if !(incomplete || reflected_digamma) {
            continue;
        }
        *counts.entry((case.function, case.scalar)).or_default() += 1;
        let result = if case.scalar == "f64" {
            detailed::<f64>(&case)
        } else {
            detailed::<f32>(&case)
        };
        let reported = match result.expect("published error estimate") {
            Ok(reported) => reported,
            Err(error) => {
                failures.push(format!(
                    "{} {} {}: {error}",
                    case.function, case.scalar, case.encoded_arguments
                ));
                continue;
            }
        };
        let expected = value(case.expected_re, case.scalar);
        let observed = (reported.value.0 - expected).abs();
        if observed > reported.absolute_error + half_spacing(case.expected_re, case.scalar) {
            failures.push(format!(
                "{} {} {}: error {observed:e} exceeds reported {:e} [{}]",
                case.function,
                case.scalar,
                case.encoded_arguments,
                reported.absolute_error,
                reported.method
            ));
        }
        let actual = if case.scalar == "f64" {
            reported.value.0.to_bits()
        } else {
            u64::from((reported.value.0 as f32).to_bits())
        };
        let ulps = ordered(actual, case.scalar).abs_diff(ordered(case.expected_re, case.scalar));
        if incomplete && ulps > target_ulp(case.function, case.scalar) {
            failures.push(format!(
                "{} {} {}: {ulps} ULP [{}]",
                case.function, case.scalar, case.encoded_arguments, reported.method
            ));
        }
    }
    for function in ["gamma_p", "gamma_q", "gamma_p_inv", "beta_inc_q", "digamma"] {
        assert!(
            counts
                .get(&(function, "f64"))
                .is_some_and(|&count| count >= 40),
            "{function}: missing large-shape or deep-tail rows: {counts:?}"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn real_lambert_polylog_targets_qualified() {
    let measurements = measure(cases().filter(|case| {
        matches!(
            case.function,
            "lambert_w0" | "lambert_wm1" | "wright_omega" | "dilog" | "spence"
        )
    }));
    assert_eq!(measurements.len(), 10);
    assert_within_targets(&measurements);
}

/// Qualify both precisions at interior and endpoint-neighbour inputs using
/// fractional ULPs against the high-precision value, not its rounded neighbour.
#[test]
fn complete_elliptic_boundary_envelopes() {
    let mut counts = std::collections::BTreeMap::new();
    for case in cases().filter(|case| {
        case.tier == "boundary" && matches!(case.function, "elliptic_k" | "elliptic_e")
    }) {
        let (actual, expected, unit, report, limit) = if case.scalar == "f64" {
            let expected = f64::from_bits(case.expected_re);
            (
                invoke_f64(&case).unwrap().0,
                expected,
                expected.next_up() - expected,
                detailed::<f64>(&case).unwrap().unwrap(),
                2.0,
            )
        } else {
            let expected = f32::from_bits(case.expected_re as u32);
            (
                f64::from(invoke_f32(&case).unwrap().0),
                f64::from(expected),
                f64::from(expected.next_up() - expected),
                detailed::<f32>(&case).unwrap().unwrap(),
                1.0,
            )
        };
        let ulps = ((actual - expected) / unit - case.reference_offset).abs();
        assert!(ulps <= limit, "{case:?}: {ulps} ULP");
        assert_eq!(actual.to_bits(), report.value.0.to_bits(), "{case:?}");
        assert!(ulps * unit <= report.absolute_error, "{case:?}");
        *counts.entry((case.function, case.scalar)).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 4);
    assert!(counts.values().all(|&count| count >= 3000));
}

/// Exact destination inputs cover both orders of every near-one Pi corner,
/// and fixed-cost K/E throughout the negative real scalar ranges.
#[test]
fn elliptic_corners_and_negative_parameter_error_coverage() {
    let mut counts = std::collections::BTreeMap::new();
    for case in cases_in(include_str!("data/elliptic_corners.tsv")) {
        let (actual, expected, unit, report) = if case.scalar == "f64" {
            let expected = f64::from_bits(case.expected_re);
            (
                invoke_f64(&case).unwrap().0,
                expected,
                expected.next_up() - expected,
                detailed::<f64>(&case).unwrap().unwrap(),
            )
        } else {
            let expected = f32::from_bits(case.expected_re as u32);
            (
                f64::from(invoke_f32(&case).unwrap().0),
                f64::from(expected),
                f64::from(expected.next_up() - expected),
                detailed::<f32>(&case).unwrap().unwrap(),
            )
        };
        let ulps = ((actual - expected) / unit - case.reference_offset).abs();
        let limit = if case.function == "elliptic_pi" {
            64.0
        } else if case.scalar == "f64" {
            4.0
        } else {
            1.0
        };
        assert!(ulps <= limit, "{case:?}: {ulps} ULP");
        assert_eq!(actual.to_bits(), report.value.0.to_bits(), "{case:?}");
        assert!(
            ulps * unit <= report.absolute_error,
            "{case:?}: {ulps} ULP exceeds error {}",
            report.absolute_error
        );
        *counts.entry((case.function, case.scalar)).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 6);
    assert_eq!(counts[&("elliptic_pi", "f64")], 44 * 44);
    assert_eq!(counts[&("elliptic_pi", "f32")], 20 * 20);
}

/// Preserve the measured envelopes for complete and incomplete elliptic forms.
#[test]
fn elliptic_measured_envelopes() {
    let mut errors = std::collections::BTreeMap::<&str, Vec<f64>>::new();
    for case in cases().filter(|case| {
        case.tier == "measured"
            && matches!(
                case.function,
                "carlson_rf"
                    | "carlson_rd"
                    | "carlson_rj"
                    | "elliptic_k"
                    | "elliptic_e"
                    | "elliptic_f"
                    | "elliptic_e_inc"
                    | "jacobi_sn"
                    | "legendre_p"
            )
    }) {
        let actual = invoke_f64(&case).unwrap().0;
        let expected = f64::from_bits(case.expected_re);
        let unit = expected.abs().next_up() - expected.abs();
        let ulps = ((actual - expected) / unit - case.reference_offset).abs();
        let report = detailed::<f64>(&case).unwrap().unwrap();
        assert_eq!(actual.to_bits(), report.value.0.to_bits(), "{case:?}");
        assert!(
            ulps * unit <= report.absolute_error,
            "{case:?}: {ulps} ULP exceeds error estimate"
        );
        errors.entry(case.function).or_default().push(ulps);
    }
    let mut misses = 0;
    for (name, errors) in &mut errors {
        errors.sort_by(f64::total_cmp);
        assert_eq!(errors.len(), 4160);
        let maximum = *errors.last().unwrap();
        let median = (errors[2079] + errors[2080]) * 0.5;
        eprintln!("{name}: max={maximum:.6}, median={median:.6} ULP");
        misses += usize::from(match *name {
            "carlson_rf" | "carlson_rd" | "carlson_rj" => maximum > 4.8,
            "elliptic_k" | "elliptic_e" => maximum > 2.0,
            "elliptic_f" | "elliptic_e_inc" => maximum > 5.7,
            "legendre_p" => maximum > 0.5,
            "jacobi_sn" => median > 1.2,
            _ => unreachable!(),
        });
    }
    assert_eq!(errors.len(), 9);
    assert_eq!(misses, 0, "measured accuracy envelopes regressed");
}
