//! Deterministic documented-domain classification sweep.
//!
//! Each entry receives 2,048 samples in each precision, from a name-derived
//! SplitMix64 seed. Linear, logarithmic, integer and boundary strata exercise
//! distinct parts of the real domain; explicit landmarks cover machine-range
//! transitions and mathematical limits. This complements, rather than replaces,
//! the accuracy oracles. No timing assertion, environment switch or ignored test
//! controls coverage. A failure prints the function, precision, index, seed and
//! exact argument bits, so a single row can be reproduced without a generator.
//!
//! f32 samples are also evaluated as exactly widened f64 inputs. A classification
//! difference is allowed only for a range failure supported by the wider result.
//! Precision-specific branch endpoints are checked separately.

#![allow(clippy::unwrap_used, clippy::panic)]

#[path = "support/domain_inventory.rs"]
mod inventory;

use scioxide_core::{NumericalIssue, RealScalar, SciOxideError, SciOxideErrorView};
use scioxide_special as sf;
use std::collections::BTreeSet;

const COUNT: usize = 2048;
const SEED: u64 = 1055;
type Args = [f64; 6];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Finite,
    Limit(bool),
    Overflow,
}

fn classify(
    result: &Result<f64, SciOxideError>,
    limit: Option<f64>,
    range: bool,
) -> Result<Class, String> {
    match result {
        Ok(v) if v.is_finite() => Ok(Class::Finite),
        Ok(v) if limit == Some(*v) && v.is_infinite() => Ok(Class::Limit(v.is_sign_negative())),
        Ok(v) => Err(format!("undocumented nonfinite success {v}")),
        Err(e) => match e.view() {
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            } if range => Ok(Class::Overflow),
            _ => Err(format!("unexpected {:?}: {e}", e.kind())),
        },
    }
}

fn consistent(a: Class, b: Class, wider: &Result<f64, SciOxideError>) -> bool {
    if a == b {
        return true;
    }
    match (a, b, wider) {
        (Class::Overflow, Class::Finite, Ok(v)) => (*v as f32).is_infinite(),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct Point {
    args: Args,
    range: bool,
}
fn point(values: &[f64]) -> Point {
    let mut args = [0.0; 6];
    args[..values.len()].copy_from_slice(values);
    Point { args, range: false }
}
fn ranged(values: &[f64]) -> Point {
    Point {
        range: true,
        ..point(values)
    }
}
fn r<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

struct Samples {
    state: u64,
    index: usize,
}
impl Samples {
    fn unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 * 2.0_f64.powi(-53)
    }
    fn linear(&mut self, lo: f64, hi: f64) -> f64 {
        match self.index {
            0 => lo,
            1 => hi,
            _ => lo + (hi - lo) * self.unit(),
        }
    }
    fn log(&mut self, lo: f64, hi: f64) -> f64 {
        self.linear(lo.ln(), hi.ln()).exp()
    }
    fn integer(&mut self, max: u32) -> f64 {
        (self.unit() * f64::from(max + 1)).floor()
    }
    fn order(&mut self) -> f64 {
        if self.index.is_multiple_of(4) {
            self.integer(12) - 6.0
        } else {
            self.linear(-6.0, 6.0)
        }
    }
    fn shape(&mut self) -> f64 {
        self.log(1e-4, 100.0)
    }
    fn probability(&mut self) -> f64 {
        if self.index.is_multiple_of(3) {
            self.log(1e-7, 0.5)
        } else {
            self.linear(1e-7, 1.0 - 1e-7)
        }
    }
}
fn angular_sample(s: &mut Samples) -> Point {
    let j1 = s.integer(16);
    let j2 = s.integer(16);
    let m1 = 2.0 * s.integer(j1 as u32) - j1;
    let m2 = 2.0 * s.integer(j2 as u32) - j2;
    let m3 = m1 + m2;
    let j3 = m3.abs() + 2.0 * s.integer(8);
    point(&[j1, m1, j2, m2, j3, m3])
}
fn spheroidal_kind(selector: f64) -> sf::SpheroidalKind {
    if selector == 0.0 {
        sf::SpheroidalKind::Prolate
    } else {
        sf::SpheroidalKind::Oblate
    }
}
fn seed(name: &str) -> u64 {
    name.bytes().fold(SEED, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

struct Entry {
    name: &'static str,
    domain: &'static str,
    sample: fn(&mut Samples) -> Point,
    f32: fn(Args) -> Result<f64, SciOxideError>,
    f64: fn(Args) -> Result<f64, SciOxideError>,
}

fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    macro_rules! add {
        ($p:ident, $S:ident; $domain:literal; $sample:expr; $($name:ident($($arg:expr),*));+ $(;)?) => {
            $(entries.push(Entry {
                name: stringify!($name), domain: $domain, sample: $sample,
                f32: |$p| { type $S = f32; sf::$name::<$S>($($arg),*).map(f64::from) },
                f64: |$p| { type $S = f64; sf::$name::<$S>($($arg),*) },
            });)+
        };
    }
    add!(p, S; "finite real x"; |s| ranged(&[s.linear(-100.0, 100.0)]);
        airy_ai(r::<S>(p[0])); airy_ai_scaled(r::<S>(p[0]));
        airy_bi(r::<S>(p[0])); airy_bi_scaled(r::<S>(p[0]));
        erf(r::<S>(p[0])); erfc(r::<S>(p[0])); erfcx(r::<S>(p[0]));
        erfi(r::<S>(p[0])); log_erfc(r::<S>(p[0]));
        shi(r::<S>(p[0])); shi_scaled(r::<S>(p[0]));
        bessel_i0_integral(r::<S>(p[0])); bessel_i0_remainder_integral(r::<S>(p[0]));
        struve_l0_integral(r::<S>(p[0]));
        fermi_dirac_m1(r::<S>(p[0])); fermi_dirac_0(r::<S>(p[0]));
        fermi_dirac_1(r::<S>(p[0])); fermi_dirac_2(r::<S>(p[0]));
        fermi_dirac_mhalf(r::<S>(p[0])); fermi_dirac_half(r::<S>(p[0])); fermi_dirac_3half(r::<S>(p[0])));
    add!(p, S; "finite real x"; |s| point(&[s.linear(-100.0, 100.0)]);
        dawson(r::<S>(p[0])); fresnel_c(r::<S>(p[0])); fresnel_s(r::<S>(p[0]));
        si(r::<S>(p[0])); atanint(r::<S>(p[0])); clausen(r::<S>(p[0]));
        dilog(r::<S>(p[0])); wright_omega(r::<S>(p[0]));
        normal_cdf(r::<S>(p[0])); normal_sf(r::<S>(p[0]));
        normal_log_cdf(r::<S>(p[0])); normal_log_sf(r::<S>(p[0]));
        bessel_j0_integral(r::<S>(p[0])); bessel_j0_complement_integral(r::<S>(p[0]));
        struve_h0_integral(r::<S>(p[0])); struve_h0_tail_integral(r::<S>(p[0])));
    add!(p, S; "x >= 0"; |s| point(&[s.linear(0.0,100.0)]);
        debye_1(r::<S>(p[0])); debye_2(r::<S>(p[0])); debye_3(r::<S>(p[0]));
        debye_4(r::<S>(p[0])); debye_5(r::<S>(p[0])); debye_6(r::<S>(p[0]));
        expint_3(r::<S>(p[0])); spence(r::<S>(p[0]));
        transport_2(r::<S>(p[0])); transport_3(r::<S>(p[0])); transport_4(r::<S>(p[0])); transport_5(r::<S>(p[0]));
        bessel_y0_integral(r::<S>(p[0])); bessel_k0_integral(r::<S>(p[0]));
        expint_e2(r::<S>(p[0])); expint_e2_scaled(r::<S>(p[0]));
        synchrotron_1(r::<S>(p[0])); synchrotron_1_scaled(r::<S>(p[0])); synchrotron_2(r::<S>(p[0])); synchrotron_2_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| ranged(&[s.log(1e-7,100.0)]);
        ci(r::<S>(p[0])); chi(r::<S>(p[0])); chi_scaled(r::<S>(p[0]));
        ker(r::<S>(p[0])); kei(r::<S>(p[0])); kerp(r::<S>(p[0])); keip(r::<S>(p[0]));
        bessel_y0_tail_integral(r::<S>(p[0])); bessel_k0_tail_integral(r::<S>(p[0])));
    add!(p, S; "finite real x"; |s| ranged(&[s.linear(-100.0,100.0)]);
        ber(r::<S>(p[0])); bei(r::<S>(p[0])); berp(r::<S>(p[0])); beip(r::<S>(p[0])));
    add!(p, S; "x != 0"; |s| { let x=s.log(1e-7,100.0); ranged(&[if s.index.is_multiple_of(2) {-x} else {x}]) };
        expint_ei(r::<S>(p[0])); expint_ei_scaled(r::<S>(p[0])); expint_e1(r::<S>(p[0])); expint_e1_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0, x != 1"; |s| point(&[s.log(1.0001,100.0)]);
        logarithmic_integral(r::<S>(p[0])); logarithmic_integral_offset(r::<S>(p[0])));
    add!(p, S; "x is not a nonpositive integer"; |s| { let x=if s.index.is_multiple_of(2) {s.linear(0.01,180.0)} else {-s.integer(30)-0.25}; ranged(&[x]) };
        gamma(r::<S>(p[0])); ln_gamma(r::<S>(p[0])); digamma(r::<S>(p[0])));
    add!(p, S; "order >= 0; x is not a nonpositive integer"; |s| ranged(&[s.integer(8), if s.index.is_multiple_of(2) {s.log(0.01,100.0)} else {-s.integer(12)-0.25}]);
        polygamma(p[0] as usize,r::<S>(p[1])));
    add!(p, S; "a,b > 0"; |s| ranged(&[s.shape(),s.shape()]); beta(r::<S>(p[0]),r::<S>(p[1])); ln_beta(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "a > 0; x >= 0"; |s| ranged(&[s.shape(),s.log(1e-7,1000.0)]);
        gamma_p(r::<S>(p[0]),r::<S>(p[1])); gamma_q(r::<S>(p[0]),r::<S>(p[1]));
        gamma_lower(r::<S>(p[0]),r::<S>(p[1])); gamma_upper(r::<S>(p[0]),r::<S>(p[1]));
        gamma_log_p(r::<S>(p[0]),r::<S>(p[1])); gamma_log_q(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "a,b > 0; 0 <= x <= 1 (logarithmic zero limits checked separately)"; |s| point(&[s.shape(),s.shape(),s.probability()]);
        beta_inc(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); beta_inc_q(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2]));
        beta_inc_log(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); beta_inc_log_q(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "a > 0; 0 <= p <= 1 (infinite quantiles checked separately)"; |s| ranged(&[s.shape(),s.probability()]);
        gamma_p_inv(r::<S>(p[0]),r::<S>(p[1])); gamma_q_inv(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "a > 0; finite log(p) <= 0"; |s| ranged(&[s.shape(),-s.log(1e-7,1000.0)]);
        gamma_p_inv_log(r::<S>(p[0]),r::<S>(p[1])); gamma_q_inv_log(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "a,b > 0; 0 <= p <= 1"; |s| ranged(&[s.shape(),s.shape(),s.probability()]);
        beta_inc_inv(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); beta_inc_q_inv(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "a,b > 0; finite log(p) <= 0"; |s| ranged(&[s.shape(),s.shape(),-s.log(1e-7,1000.0)]);
        beta_inc_inv_log(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); beta_inc_q_inv_log(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "-1 <= x <= 1; signed infinite endpoints checked separately"; |s| point(&[s.linear(-0.999999,0.999999)]); erfinv(r::<S>(p[0])));
    add!(p, S; "0 <= x <= 2; signed infinite endpoints checked separately"; |s| point(&[s.linear(1e-7,2.0-1e-7)]); erfcinv(r::<S>(p[0])));
    add!(p, S; "0 <= p <= 1; signed infinite endpoints checked separately"; |s| point(&[s.probability()]); normal_quantile(r::<S>(p[0])); normal_isf(r::<S>(p[0])));
    add!(p, S; "finite log(p) <= 0"; |s| point(&[-s.log(1e-7,1000.0)]); normal_inv_log_cdf(r::<S>(p[0])); normal_inv_log_sf(r::<S>(p[0])));
    add!(p, S; "finite h,a"; |s| point(&[s.linear(-10.0,10.0),s.linear(-10.0,10.0)]); owens_t(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "finite x; sigma,gamma >= 0, not both zero"; |s| point(&[s.linear(-10.0,10.0),s.log(0.01,10.0),s.log(0.01,10.0)]); voigt_profile(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "finite real order; x > 0"; |s| ranged(&[s.order(),s.log(1e-5,100.0)]);
        bessel_y(r::<S>(p[0]),r::<S>(p[1])); bessel_y_scaled(r::<S>(p[0]),r::<S>(p[1]));
        bessel_k(r::<S>(p[0]),r::<S>(p[1])); bessel_k_scaled(r::<S>(p[0]),r::<S>(p[1])); bessel_k_log(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "finite order and x > 0; integer order for negative x; nonnegative integer order at zero"; |s| { if s.index.is_multiple_of(2) {ranged(&[s.integer(12),s.linear(-100.0,0.0)])} else {ranged(&[s.order(),s.log(1e-5,100.0)])} };
        bessel_j(r::<S>(p[0]),r::<S>(p[1])); bessel_j_scaled(r::<S>(p[0]),r::<S>(p[1])); bessel_i(r::<S>(p[0]),r::<S>(p[1])); bessel_i_scaled(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "finite order and x > 0; nonnegative derivative degree"; |s| ranged(&[s.order(),s.log(0.01,100.0),s.integer(4)]);
        bessel_j_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32); bessel_y_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32);
        bessel_i_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32); bessel_k_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32);
        bessel_i_scaled_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32); bessel_k_scaled_derivative(r::<S>(p[0]),r::<S>(p[1]),p[2] as u32));
    add!(p, S; "order >= 0; finite real x (entire even continuation)"; |s| point(&[s.linear(0.0,10.0),s.linear(0.0,30.0)]); bessel_lambda(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "negative a requires integer order; exponent + effective order > -1"; |s| point(&[s.linear(0.0,5.0),s.linear(0.0,3.0),s.linear(0.0,3.0)]); bessel_weighted_integral(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "integer n >= 0; finite x (Y excludes zero; K requires x > 0)"; |s| ranged(&[s.integer(32),s.log(1e-5,100.0)]);
        spherical_jn(p[0] as u32,r::<S>(p[1])); spherical_yn(p[0] as u32,r::<S>(p[1]));
        spherical_in(p[0] as u32,r::<S>(p[1])); spherical_kn(p[0] as u32,r::<S>(p[1]));
        spherical_in_scaled(p[0] as u32,r::<S>(p[1])); spherical_kn_scaled(p[0] as u32,r::<S>(p[1]));
        riccati_jn(p[0] as u32,r::<S>(p[1])); riccati_yn(p[0] as u32,r::<S>(p[1])));
    add!(p, S; "real order; negative x requires integer order"; |s| ranged(&[s.order(),s.linear(0.01,30.0)]); struve(r::<S>(p[0]),r::<S>(p[1])); modstruve(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "real order and argument"; |s| ranged(&[s.linear(-5.0,5.0),s.linear(-8.0,8.0)]); parabolic_d(r::<S>(p[0]),r::<S>(p[1])); parabolic_v(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "|a| <= 5, |x| <= 5"; |s| point(&[s.linear(-5.0,5.0),s.linear(-5.0,5.0)]); parabolic_w(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "m <= 1; K(1) is infinite; E(1)=1"; |s| point(&[s.linear(-10.0,0.999999)]); elliptic_k(r::<S>(p[0])); elliptic_e(r::<S>(p[0])));
    add!(p, S; "finite phi; m <= 1 (endpoint singularities excluded for F)"; |s| point(&[s.linear(-10.0,10.0),s.linear(-10.0,0.99)]); elliptic_f(r::<S>(p[0]),r::<S>(p[1])); elliptic_e_inc(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "n < 1, m < 1; documented n=1 or m=1 limits"; |s| point(&[s.linear(-10.0,0.99),s.linear(-10.0,0.99)]); elliptic_pi(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "finite u; real m <= 1"; |s| point(&[s.linear(-100.0,100.0),s.linear(-10.0,1.0)]);
        jacobi_sn(r::<S>(p[0]),r::<S>(p[1])); jacobi_cn(r::<S>(p[0]),r::<S>(p[1])); jacobi_dn(r::<S>(p[0]),r::<S>(p[1])); jacobi_amplitude(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "x,y,z >= 0; at most one zero for RF; z > 0 and x+y > 0 for RD"; |s| ranged(&[s.log(1e-6,1e6),s.log(1e-6,1e6),s.log(1e-6,1e6)]);
        carlson_rf(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); carlson_rd(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); carlson_rg(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "x >= 0; y != 0 (negative y uses principal value)"; |s| point(&[s.log(1e-6,1e6),s.log(1e-6,1e6)]); carlson_rc(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "x,y,z >= 0, at most one zero; p != 0 (negative p uses principal value)"; |s| ranged(&[s.log(1e-6,1e6),s.log(1e-6,1e6),s.log(1e-6,1e6),s.log(1e-6,1e6)]); carlson_rj(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2]),r::<S>(p[3])));
    add!(p, S; "integer n >= 0; finite x"; |s| ranged(&[s.integer(64),s.linear(-2.0,2.0)]); legendre_p(p[0] as usize,r::<S>(p[1])));
    add!(p, S; "integer n >= 0, |m| <= n; |x| <= 1"; |s| { let n=s.integer(32); ranged(&[n,s.integer(2*n as u32)-n,s.linear(-1.0,1.0)]) }; legendre_p_assoc(p[0] as usize,p[1] as i32,r::<S>(p[2])));
    add!(p, S; "integer n >= 0; -1 <= x <= 1; signed endpoint limits"; |s| point(&[s.integer(32),s.linear(-0.999999,0.999999)]); legendre_q(p[0] as usize,r::<S>(p[1])));
    add!(p, S; "integer n >= 0; lambda,eta >= 0"; |s| ranged(&[s.integer(8),s.linear(0.01,5.0),s.linear(0.01,5.0)]); legendre_h3d(p[0] as u32,r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "finite s != 1"; |s| ranged(&[if s.index.is_multiple_of(2) {-s.integer(30)-0.25} else {s.linear(1.0001,50.0)}]); riemann_zeta(r::<S>(p[0])));
    add!(p, S; "finite s"; |s| ranged(&[s.linear(-30.0,50.0)]); dirichlet_eta(r::<S>(p[0])));
    add!(p, S; "s != 1, q > 0"; |s| ranged(&[if s.index.is_multiple_of(2) { s.linear(-60.0,0.9999) } else { s.linear(1.0001,30.0) },s.log(1e-12,1000.0)]); hurwitz_zeta(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "s > 1, q > 0"; |s| ranged(&[s.linear(1.0001,30.0),s.log(0.01,100.0)]); hurwitz_zeta_log(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "real-valued principal continuation; z > 1 requires nonpositive integer order; pole at (s,z)=(1,1)"; |s| point(&[s.linear(-3.0,5.0),s.linear(-1.0,0.95)]); polylog(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "real-valued principal continuation; nonpositive a requires integer s; exclude nonremovable poles"; |s| point(&[s.linear(-0.9,0.9),s.linear(-3.0,5.0),s.log(0.1,10.0)]); lerch_phi(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "b not a nonpositive integer; real z"; |s| ranged(&[s.linear(0.1,10.0),s.linear(-30.0,30.0)]); hyp0f1(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "b not a nonpositive integer, unless numerator terminates first; real z"; |s| ranged(&[s.linear(-3.0,3.0),s.linear(0.1,10.0),s.linear(-20.0,20.0)]); hyp1f1(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "c not a nonpositive integer unless numerator terminates first; z < 1 with finite z=1 cases"; |s| ranged(&[s.linear(-3.0,3.0),s.linear(-3.0,3.0),s.linear(0.1,10.0),s.linear(-10.0,0.99)]);
        hyp2f1(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2]),r::<S>(p[3])); hyp2f1_regularized(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2]),r::<S>(p[3])));
    add!(p, S; "real continuation for z < 0, or terminating numerator"; |s| ranged(&[s.linear(0.1,5.0),s.linear(0.1,5.0),-s.log(0.01,10.0)]); hyp2f0(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "real parameters; z > 0"; |s| ranged(&[s.linear(0.1,5.0),s.linear(-3.0,5.0),s.log(0.01,20.0)]); hyperu(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "pFq convergent series domain, including terminating numerators"; |s| {
        let n=s.integer(2) as usize;
        let d=s.integer(2) as usize;
        let first=if n>d+1 {-s.integer(5)} else {s.linear(0.1,3.0)};
        point(&[s.linear(-0.9,0.9),first,s.linear(0.1,3.0),(n+3*d) as f64,s.linear(0.1,3.0),s.linear(0.1,3.0)])
    }; hyp_pfq(&[r::<S>(p[1]),r::<S>(p[5])][..p[3] as usize%3],&[r::<S>(p[2]),r::<S>(p[4])][..p[3] as usize/3],r::<S>(p[0])));
    add!(p, S; "rounded -1/e <= x"; |s| point(&[s.linear(-0.36,100.0)]); lambert_w0(r::<S>(p[0])));
    add!(p, S; "rounded -1/e <= x < 0"; |s| point(&[-s.log(1e-30,0.36)]); lambert_wm1(r::<S>(p[0])));
    add!(p, S; "n >= 0"; |s| ranged(&[s.integer(200)]); factorial(p[0] as usize); ln_factorial(p[0] as usize); double_factorial(p[0] as usize); ln_double_factorial(p[0] as usize));
    add!(p, S; "n,k >= 0, k <= n"; |s| {let n=s.integer(200); ranged(&[n,s.integer(n as u32)])}; binomial(p[0] as usize,p[1] as usize); ln_binomial(p[0] as usize,p[1] as usize));
    add!(p, S; "n >= 0, step > 0"; |s| ranged(&[s.integer(200),1.0+s.integer(10)]); multifactorial(p[0] as usize,p[1] as usize); ln_multifactorial(p[0] as usize,p[1] as usize));
    add!(p, S; "integer n >= 0; odd orders above one return zero"; |s| ranged(&[s.integer(400)]); bernoulli(p[0] as usize); euler(p[0] as usize));
    add!(p, S; "n,k >= 0; entries outside triangle are zero"; |s| ranged(&[s.integer(32),s.integer(32)]); stirling_first_kind(p[0] as usize,p[1] as usize); stirling_second_kind(p[0] as usize,p[1] as usize));
    add!(p, S; "finite a, integer n or k >= 0"; |s| ranged(&[s.linear(-10.0,10.0),s.integer(32)]); generalized_binomial(r::<S>(p[0]),p[1] as usize));
    add!(p, S; "Gamma(a+x)/Gamma(a), with integer-shift polynomial continuation; exclude unresolved poles"; |s| ranged(&[s.log(0.01,10.0),s.linear(0.0,10.0)]); pochhammer(r::<S>(p[0]),r::<S>(p[1])); pochhammer_relative(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "integer order j >= -101; finite real x"; |s| ranged(&[s.integer(12)-4.0,s.linear(-30.0,30.0)]); fermi_dirac_int(p[0] as i32,r::<S>(p[1])));
    add!(p, S; "finite x, b >= 0"; |s| point(&[s.linear(-30.0,30.0),s.linear(0.0,30.0)]); fermi_dirac_inc_0(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "integer n >= 0, x > 0; x=0 finite for n>1"; |s| ranged(&[s.integer(20),s.log(1e-7,100.0)]); expint_en(p[0] as u32,r::<S>(p[1])); expint_en_scaled(p[0] as u32,r::<S>(p[1])));
    add!(p, S; "l > -1/2, finite eta, real radius > 0; regular origin limits"; |s| ranged(&[s.linear(-0.4,5.0),s.linear(-3.0,3.0),s.log(0.1,20.0)]);
        coulomb_f(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); coulomb_g(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2]));
        coulomb_f_prime(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); coulomb_g_prime(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])); coulomb_spherical_f(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "l > -1; finite eta"; |s| point(&[s.linear(-0.9,5.0),s.linear(-5.0,5.0)]); coulomb_normalization(r::<S>(p[0]),r::<S>(p[1])));
    add!(p, S; "n >= 1, l < n, Z > 0, real r >= 0"; |s| {let n=1.0+s.integer(8); ranged(&[n,s.integer(n as u32-1),s.log(0.1,5.0),s.linear(0.0,30.0)])}; hydrogenic_radial(p[0] as u32,p[1] as u32,r::<S>(p[2]),r::<S>(p[3])));
    add!(p, S; "0 < h² < k², n >= 0, 1 <= p <= 2n+1; second-kind s > k; first-kind real continuation"; |s| {let n=s.integer(4); point(&[s.linear(0.2,0.8),1.0,n,1.0+s.integer(2*n as u32),s.linear(1.01,3.0)])};
        ellipsoidal_harmonic(r::<S>(p[0]), r::<S>(p[1]), p[2] as usize, p[3] as usize, r::<S>(p[4]), scioxide_special::EllipsoidalSigns::default());
        ellipsoidal_harmonic_second_kind(r::<S>(p[0]),r::<S>(p[1]),p[2] as usize,p[3] as usize,r::<S>(p[4]));
        ellipsoidal_normalization(r::<S>(p[0]),r::<S>(p[1]),p[2] as usize,p[3] as usize));
    add!(p, S; "integer order n >= 0; real q; finite real z"; |s| point(&[s.integer(8),s.linear(-25.0,25.0),s.linear(-10.0,10.0)]);
        mathieu_a(p[0] as u32,r::<S>(p[1])); mathieu_ce(p[0] as u32,r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "integer order n >= 1; real q; finite real z"; |s| point(&[1.0+s.integer(7),s.linear(-25.0,25.0),s.linear(-10.0,10.0)]);
        mathieu_b(p[0] as u32,r::<S>(p[1])); mathieu_se(p[0] as u32,r::<S>(p[1]),r::<S>(p[2])));
    add!(p, S; "prolate or oblate; integer 0 <= m <= n; real c; real |x| <= 1"; |s| {let m=s.integer(4); point(&[s.integer(1),m,m+s.integer(6),s.linear(-8.0,8.0),s.linear(-1.0,1.0)])};
        spheroidal_characteristic_value(spheroidal_kind(p[0]),p[1] as u32,p[2] as u32,r::<S>(p[3]));
        spheroidal_angular(spheroidal_kind(p[0]),p[1] as u32,p[2] as u32,r::<S>(p[3]),r::<S>(p[4])));
    add!(p, S; "integer l >= 0, |m| <= l; theta in [0,pi], finite phi"; |s| {let n=s.integer(16); point(&[n,s.integer(2*n as u32)-n,s.linear(0.0,f64::from(std::f32::consts::PI.next_down())),s.linear(-10.0,10.0)])}; spherical_harmonic_real(p[0] as usize,p[1] as i32,r::<S>(p[2]),r::<S>(p[3])));
    add!(p, S; "nonnegative twice-j, |twice-m| <= twice-j and equal parity; forbidden couplings return zero"; angular_sample;
        wigner_3j([p[0] as i32,p[2] as i32,p[4] as i32],[p[1] as i32,p[3] as i32,-p[5] as i32]));
    add!(p, S; "nonnegative twice-angular momenta; forbidden triangles return zero"; |s| point(&[s.integer(16),s.integer(16),s.integer(16),s.integer(16),s.integer(16),s.integer(16)]);
        wigner_6j(p.map(|x|x as i32)); racah_w(p.map(|x|x as i32)));
    add!(p, S; "nonnegative twice-angular momenta; forbidden triangles return zero"; |s| point(&[s.integer(8),s.integer(8),s.integer(8)]); wigner_9j([p[0] as i32,p[1] as i32,p[2] as i32,p[1] as i32,p[2] as i32,p[0] as i32,p[2] as i32,p[0] as i32,p[1] as i32]));
    add!(p, S; "valid angular momentum pairs; forbidden couplings return zero"; angular_sample;
        clebsch_gordan([p[0] as i32,p[2] as i32,p[4] as i32],[p[1] as i32,p[3] as i32,p[5] as i32]));
    add!(p, S; "integer degrees >= 0, |orders| <= degrees; forbidden couplings return zero"; |s| point(&[s.integer(16),s.integer(16),s.integer(16)]); gaunt([p[0] as usize,p[1] as usize,p[2] as usize],[0,0,0]));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); ber_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); bei_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); ker_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); kei_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); berp_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); beip_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); kerp_scaled(r::<S>(p[0])));
    add!(p, S; "x > 0"; |s| point(&[s.log(1e-7,100.0)]); keip_scaled(r::<S>(p[0])));
    entries
}

#[test]
fn every_public_real_function_has_a_domain_entry() {
    let entries = entries();
    let table: BTreeSet<_> = entries.iter().map(|e| e.name.to_owned()).collect();
    assert_eq!(table.len(), entries.len(), "duplicate function table entry");
    assert!(entries.iter().all(|e| !e.domain.is_empty()));
    let public = inventory::functions(false);
    assert_eq!(
        table,
        public,
        "missing: {:?}; stale: {:?}",
        public.difference(&table).collect::<Vec<_>>(),
        table.difference(&public).collect::<Vec<_>>()
    );
}

fn sweep(single: bool) {
    let mut failures = Vec::new();
    for entry in entries() {
        let mut samples = Samples {
            state: seed(entry.name),
            index: 0,
        };
        let mut finite = 0;
        for index in 0..COUNT {
            samples.index = index;
            let mut sample = (entry.sample)(&mut samples);
            if single {
                sample.args = sample.args.map(|v| f64::from(v as f32));
            }
            let result = if single {
                (entry.f32)(sample.args)
            } else {
                (entry.f64)(sample.args)
            };
            let classification = classify(&result, None, sample.range);
            let failure = match classification {
                Err(message) => Some(message),
                Ok(class) if single => {
                    finite += usize::from(class == Class::Finite);
                    let wider = (entry.f64)(sample.args);
                    match classify(&wider, None, sample.range) {
                        Err(message) => Some(format!("widened input: {message}")),
                        Ok(wide) if !consistent(class, wide, &wider) => Some(format!(
                            "precision mismatch {class:?} / {wide:?}: {result:?} / {wider:?}"
                        )),
                        _ => None,
                    }
                }
                Ok(Class::Finite) => {
                    finite += 1;
                    None
                }
                _ => None,
            };
            if let Some(message) = failure {
                failures.push(format!(
                    "{} f{} index={index} seed={:#x} args={:?} bits={:x?}; {}: {message}",
                    entry.name,
                    if single { 32 } else { 64 },
                    seed(entry.name),
                    sample.args,
                    sample.args.map(f64::to_bits),
                    entry.domain
                ));
                break;
            }
        }
        if finite == 0 {
            failures.push(format!("{} has no finite sampled successes", entry.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn documented_domains_f64() {
    sweep(false);
}
#[test]
fn documented_domains_f32_and_classification_agreement() {
    sweep(true);
}

#[test]
fn classification_rejects_unexpected_errors_and_unjustified_range_differences() {
    assert!(classify(&Ok(f64::NAN), None, true).is_err());
    assert!(classify(&Ok(f64::INFINITY), None, true).is_err());
    assert_eq!(
        classify(&Ok(f64::INFINITY), Some(f64::INFINITY), false),
        Ok(Class::Limit(false))
    );
    for issue in [
        NumericalIssue::Overflow,
        NumericalIssue::Underflow,
        NumericalIssue::LossOfSignificance,
    ] {
        let error = Err(SciOxideError::numerical_instability(
            issue,
            "synthetic failure",
        ));
        assert!(classify(&error, None, false).is_err());
        // A rounded zero is a value, so an underflow error is never accepted.
        assert_eq!(
            classify(&error, None, true).is_ok(),
            issue == NumericalIssue::Overflow
        );
    }
    let domain = Err(SciOxideError::domain_error(
        scioxide_core::argument!("x"),
        scioxide_core::DomainReason::OutsideDomain,
        "synthetic",
    ));
    assert!(classify(&domain, None, true).is_err());
    for error in [
        SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::Stagnation,
            scioxide_core::ConvergenceDiagnostics::new(),
            "synthetic",
        ),
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Intermediate, "synthetic"),
        <f32 as RealScalar>::try_from_f64_lossy(f64::MAX).unwrap_err(),
    ] {
        assert!(classify(&Err(error), None, true).is_err());
    }

    assert!(!consistent(Class::Overflow, Class::Finite, &Ok(1.0)));
    assert!(consistent(Class::Overflow, Class::Finite, &Ok(f64::MAX)));
}

/// Native-precision landmarks are independent of the common-input comparison:
/// rounding -1/e, pi, or a tail endpoint can change the other precision's domain.
fn landmarks(single: bool) {
    let tiny = if single {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    let maximum = if single {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let endpoint = if single {
        f64::from((-1.0 / std::f64::consts::E) as f32)
    } else {
        -1.0 / std::f64::consts::E
    };
    let above = if single {
        f64::from((endpoint as f32).next_up())
    } else {
        endpoint.next_up()
    };
    let below = if single {
        f64::from((endpoint as f32).next_down())
    } else {
        endpoint.next_down()
    };
    let entries = entries();
    let evaluate = |name: &str, args: &[f64], expected: Class| {
        let entry = entries.iter().find(|e| e.name == name).unwrap();
        let result = if single {
            (entry.f32)(point(args).args)
        } else {
            (entry.f64)(point(args).args)
        };
        let limit = match expected {
            Class::Limit(true) => Some(f64::NEG_INFINITY),
            Class::Limit(false) => Some(f64::INFINITY),
            _ => None,
        };
        let class = classify(&result, limit, expected == Class::Overflow);
        assert_eq!(
            class,
            Ok(expected),
            "{name} f{} {args:?}: {result:?}",
            if single { 32 } else { 64 }
        );
        result
    };
    for name in ["lambert_w0", "lambert_wm1"] {
        for x in [endpoint, above, above.next_up()] {
            let _ = evaluate(name, &[x], Class::Finite);
        }
        let result = evaluate(name, &[endpoint], Class::Finite).unwrap();
        assert_eq!(result, -1.0);
        let entry = entries.iter().find(|e| e.name == name).unwrap();
        let invalid = if single {
            (entry.f32)(point(&[below]).args)
        } else {
            (entry.f64)(point(&[below]).args)
        };
        assert_eq!(
            invalid.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
    for (name, args, negative) in [
        ("erfinv", vec![-1.0], true),
        ("erfinv", vec![1.0], false),
        ("erfcinv", vec![0.0], false),
        ("erfcinv", vec![2.0], true),
        ("normal_quantile", vec![0.0], true),
        ("normal_quantile", vec![1.0], false),
        ("normal_isf", vec![0.0], false),
        ("normal_isf", vec![1.0], true),
        ("normal_inv_log_cdf", vec![0.0], false),
        ("normal_inv_log_sf", vec![0.0], true),
        ("gamma_p_inv", vec![2.0, 1.0], false),
        ("gamma_q_inv", vec![2.0, 0.0], false),
        ("gamma_p_inv_log", vec![2.0, 0.0], false),
        ("gamma_log_p", vec![2.0, 0.0], true),
        ("beta_inc_log", vec![2.0, 3.0, 0.0], true),
        ("beta_inc_log_q", vec![2.0, 3.0, 1.0], true),
        ("elliptic_k", vec![1.0], false),
        ("elliptic_pi", vec![1.0, 0.5], false),
        ("legendre_q", vec![0.0, -1.0], true),
        ("legendre_q", vec![0.0, 1.0], false),
        ("bessel_k", vec![0.0, 0.0], false),
        ("bessel_k_scaled", vec![0.0, 0.0], false),
    ] {
        let _ = evaluate(name, &args, Class::Limit(negative));
    }
    for (name, args) in [
        ("gamma_p", vec![0.5, tiny]),
        ("gamma_lower", vec![2.0, (4.0 * tiny).sqrt()]),
        ("beta_inc", vec![2.0, 1.0, (4.0 * tiny).sqrt()]),
        ("erf", vec![tiny]),
        ("erfinv", vec![tiny]),
        ("erfcinv", vec![tiny]),
        ("jacobi_sn", vec![tiny, 0.5]),
        ("legendre_p", vec![1.0, tiny]),
        ("wright_omega", vec![-maximum]),
        ("airy_ai", vec![1000.0]),
        ("airy_bi_scaled", vec![1000.0]),
        ("bessel_k", vec![0.0, 1000.0]),
        ("bessel_i_scaled", vec![0.0, 1000.0]),
        (
            "gamma_p_inv",
            vec![0.0014109960028295145, 0.8670461346108965],
        ),
        (
            "beta_inc_inv",
            vec![0.021813515576359375, 0.4295632308260261, 0.4212025283717187],
        ),
        ("beta_inc", vec![2.0, 3.0, 0.0]),
        ("beta_inc", vec![2.0, 3.0, 1.0]),
        ("beta_inc_inv", vec![2.0, 3.0, 0.0]),
        ("beta_inc_inv", vec![2.0, 3.0, 1.0]),
        ("gamma_p", vec![2.0, 0.0]),
        ("gamma_q", vec![2.0, 0.0]),
        ("carlson_rf", vec![0.0, 1.0, 2.0]),
        ("carlson_rd", vec![0.0, 1.0, 2.0]),
        ("carlson_rc", vec![1.0, -1.0]),
        ("carlson_rj", vec![1.0, 2.0, 3.0, -1.0]),
        ("hyp1f1", vec![-3.0, -4.0, 2.0]),
        ("hyp2f1", vec![-3.0, 0.5, -4.0, 2.0]),
        ("hyp2f0", vec![-3.0, 0.5, 2.0]),
        ("hurwitz_zeta", vec![0.5, 1.0]),
        ("logarithmic_integral", vec![0.5]),
        ("bessel_lambda", vec![2.0, -3.0]),
        ("polylog", vec![0.0, 2.0]),
        ("polylog", vec![0.0, 1.0]),
        ("lerch_phi", vec![0.5, -2.0, 0.0]),
        (
            "spherical_harmonic_real",
            vec![
                0.0,
                0.0,
                if single {
                    f64::from(std::f32::consts::PI)
                } else {
                    std::f64::consts::PI
                },
                0.0,
            ],
        ),
        ("spherical_jn", vec![3.0, -2.0]),
        ("spherical_in", vec![3.0, -2.0]),
        ("struve", vec![2.0, -3.0]),
        ("modstruve", vec![2.0, -3.0]),
        ("legendre_p_assoc", vec![3.0, -2.0, 0.5]),
    ] {
        let _ = evaluate(name, &args, Class::Finite);
    }
    let gamma_edge = if single { 36.0 } else { 172.0 };
    let factorial_edge = if single { 35.0 } else { 171.0 };
    let _ = evaluate("gamma", &[gamma_edge - 1.0], Class::Finite);
    let _ = evaluate("gamma", &[gamma_edge], Class::Overflow);
    let _ = evaluate("factorial", &[factorial_edge - 1.0], Class::Finite);
    let _ = evaluate("factorial", &[factorial_edge], Class::Overflow);
    let _ = evaluate("bessel_y", &[200.0, tiny], Class::Overflow);
    let tail = if single { 14.0 } else { 38.0 };
    let value = evaluate("normal_sf", &[tail], Class::Finite).unwrap();
    let normal = if single {
        f64::from(f32::MIN_POSITIVE)
    } else {
        f64::MIN_POSITIVE
    };
    assert!(
        value > 0.0 && value < normal,
        "tail must remain a representable subnormal: {value}"
    );
}
#[test]
fn landmarks_f64() {
    landmarks(false);
}
#[test]
fn landmarks_f32() {
    landmarks(true);
}
