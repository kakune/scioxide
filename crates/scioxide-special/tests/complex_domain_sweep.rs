//! Deterministic complex documented-domain coverage.
//!
//! Every plain complex-capable scalar root function receives 1,024 samples in
//! each precision. Name-derived SplitMix64 seeds, modulus/argument strata, exact
//! signed-zero cut lips and separate landmarks complement the accuracy oracles.
//! Failures report the function, precision, index, seed and exact argument bits.
//! No environment switches, ignored cases or timing assertions control coverage.
#![allow(clippy::unwrap_used, clippy::panic)]

#[path = "support/domain_inventory.rs"]
mod inventory;

use scioxide_core::{
    Complex, Complex64 as C, NumericalIssue, RealScalar, SciOxideError, SciOxideErrorView,
};
use scioxide_special as sf;
use std::collections::BTreeSet;
use std::f64::consts::PI;

const COUNT: usize = 1024;
const SEED: u64 = 1166;
type Args = [f64; 12];

fn args(values: &[f64]) -> Args {
    let mut result = [0.0; 12];
    result[..values.len()].copy_from_slice(values);
    result
}
fn coordinates(values: &[[f64; 2]]) -> Args {
    let mut result = [0.0; 12];
    for (pair, value) in result.chunks_exact_mut(2).zip(values) {
        pair.copy_from_slice(value);
    }
    result
}
fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn c<S: RealScalar>(p: Args, n: usize) -> Complex<S> {
    Complex::new(r(p[2 * n]), r(p[2 * n + 1]))
}
fn widen<S: RealScalar>(z: Complex<S>) -> C {
    C::new(z.re.into_f64(), z.im.into_f64())
}
fn spheroidal_kind(selector: f64) -> sf::SpheroidalKind {
    if selector == 0.0 {
        sf::SpheroidalKind::Prolate
    } else {
        sf::SpheroidalKind::Oblate
    }
}
fn radial_kind(selector: f64) -> sf::RadialKind {
    match selector as u32 {
        0 => sf::RadialKind::First,
        1 => sf::RadialKind::Second,
        2 => sf::RadialKind::Third,
        _ => sf::RadialKind::Fourth,
    }
}
fn seed(name: &str) -> u64 {
    name.bytes().fold(SEED, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
struct Samples {
    state: u64,
    index: usize,
}
impl Samples {
    fn unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((x ^ (x >> 31)) >> 11) as f64 * 2.0_f64.powi(-53)
    }
    fn linear(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn log(&mut self, lo: f64, hi: f64) -> f64 {
        self.linear(lo.ln(), hi.ln()).exp()
    }
    fn integer(&mut self, max: u32) -> f64 {
        (self.unit() * f64::from(max + 1)).floor()
    }
    fn cartesian(&mut self, lo: f64, hi: f64, imag: f64) -> [f64; 2] {
        [self.linear(lo, hi), self.linear(-imag, imag)]
    }
    fn polar(&mut self, lo: f64, hi: f64) -> [f64; 2] {
        // Equal occupancy of logarithmic modulus strata, independent of the lip.
        let fraction = ((self.index / 16) % 4) as f64 / 4.0;
        let width = (hi / lo).ln();
        let radius = (lo.ln() + width * (fraction + self.unit() / 4.0)).exp();
        let phase = self.linear(-PI, PI);
        match self.index % 16 {
            0 => [-radius, 0.0],
            1 => [-radius, -0.0],
            2 => [radius, 0.0],
            3 => [radius, -0.0],
            4 => [-radius, radius * 1e-7],
            5 => [-radius, -radius * 1e-7],
            6 => [0.0, radius],
            7 => [-0.0, radius],
            8 => [0.0, -radius],
            9 => [-0.0, -radius],
            _ => [radius * phase.cos(), radius * phase.sin()],
        }
    }
}
struct Entry {
    name: &'static str,
    domain: &'static str,
    sample: fn(&mut Samples) -> Args,
    f32: fn(Args) -> Result<C, SciOxideError>,
    f64: fn(Args) -> Result<C, SciOxideError>,
}
fn gamma_sample(s: &mut Samples) -> Args {
    coordinates(&[s.polar(1e-4, 40.0)])
}
fn bessel_sample(s: &mut Samples) -> Args {
    let mut nu = s.cartesian(-5.0, 5.0, 3.0);
    if s.index.is_multiple_of(2) {
        nu[1] = 0.0;
    }
    if s.index % 32 < 4 {
        nu[0] = nu[0].round() + if s.index % 32 < 2 { 0.0 } else { 1e-8 };
        nu[1] *= 1e-8;
    }
    let (lo, hi) = match (s.index / 2) % 5 {
        0 => (1e-6, 0.1),
        1 => (0.1, 2.0),
        2 => (2.0, 10.0),
        3 => (10.0, 30.0),
        _ => (30.0, 100.0),
    };
    let mut p = coordinates(&[nu, s.polar(lo, hi)]);
    p[4] = s.integer(3);
    p
}
fn hyper_sample(s: &mut Samples) -> Args {
    coordinates(&[
        s.cartesian(0.1, 4.0, 1.0),
        s.cartesian(0.1, 5.0, 1.0),
        s.cartesian(0.1, 5.0, 1.0),
        s.polar(1e-3, 10.0),
    ])
}
fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    macro_rules! add {
        ($p:ident,$S:ident; $domain:literal; $sample:expr; $($name:ident($($arg:expr),*));+ $(;)?) => {
            $(entries.push(Entry {
                name: stringify!($name), domain: $domain, sample: $sample,
                f32: |$p| { type $S=f32; sf::$name($($arg),*).map(widen::<$S>) },
                f64: |$p| { type $S=f64; sf::$name($($arg),*).map(widen::<$S>) },
            });)+
        };
    }
    add!(p,S; "finite complex z, excluding nonpositive integer poles"; gamma_sample;
        gamma(c::<S>(p,0)); ln_gamma(c::<S>(p,0)); digamma(c::<S>(p,0)));
    add!(p,S; "integer order >= 0; complex z away from nonpositive integers"; |s| {let mut p=gamma_sample(s);p[2]=s.integer(5);p};
        polygamma(p[2] as usize,c::<S>(p,0)));
    add!(p,S; "complex gamma parameters away from poles"; |s| coordinates(&[s.cartesian(0.1,10.0,3.0),s.cartesian(0.1,10.0,3.0)]);
        beta(c::<S>(p,0),c::<S>(p,1)); ln_beta(c::<S>(p,0),c::<S>(p,1));
        pochhammer(c::<S>(p,0),c::<S>(p,1)); pochhammer_relative(c::<S>(p,0),c::<S>(p,1)));
    add!(p,S; "complex a,b away from gamma poles; principal z and 1-z powers"; |s| coordinates(&[s.cartesian(0.1,5.0,1.0),s.cartesian(0.1,5.0,1.0),s.polar(1e-3,3.0)]);
        beta_inc(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2)));
    add!(p,S; "Re(a)>0; nonzero complex z on either cut lip"; |s| {
        let mut a=s.cartesian(0.1,20.0,5.0);
        if s.index.is_multiple_of(2) {a[1]*=0.02;}
        let z=match (s.index/2)%3 {0=>s.polar(1e-3,0.1),1=>{let z=s.polar(0.8,1.2);[z[0]*a[0],z[1]*a[0]]},_=>s.polar(1.0,50.0)};
        coordinates(&[a,z])
    }; gamma_lower(c::<S>(p,0),c::<S>(p,1)); gamma_upper(c::<S>(p,0),c::<S>(p,1)); gamma_p(c::<S>(p,0),c::<S>(p,1)); gamma_q(c::<S>(p,0),c::<S>(p,1)));
    add!(p,S; "entire complex z; typed scalar range failures admitted"; |s| coordinates(&[s.polar(1e-6,30.0)]);
        erf(c::<S>(p,0)); erfc(c::<S>(p,0)); erfcx(c::<S>(p,0)); erfi(c::<S>(p,0)); dawson(c::<S>(p,0)); faddeeva(c::<S>(p,0));
        fresnel_c(c::<S>(p,0)); fresnel_s(c::<S>(p,0)); airy_ai(c::<S>(p,0)); airy_bi(c::<S>(p,0)); airy_ai_scaled(c::<S>(p,0)); airy_bi_scaled(c::<S>(p,0)));
    add!(p,S; "complex z away from erfc zeros and unresolved image-cut cases"; |s| coordinates(&[s.polar(1e-6,6.0)]); log_erfc(c::<S>(p,0)));
    add!(p,S; "real x, sigma>0, gamma>=0"; |s| args(&[s.linear(-20.0,20.0),s.log(0.01,5.0),s.linear(0.0,5.0)]);
        voigt_complex(r::<S>(p[0]),r::<S>(p[1]),r::<S>(p[2])));
    add!(p,S; "real/complex order; z!=0 on both signed cut lips"; bessel_sample;
        bessel_j(c::<S>(p,0),c::<S>(p,1)); bessel_y(c::<S>(p,0),c::<S>(p,1)); bessel_i(c::<S>(p,0),c::<S>(p,1)); bessel_k(c::<S>(p,0),c::<S>(p,1));
        bessel_j_scaled(c::<S>(p,0),c::<S>(p,1)); bessel_y_scaled(c::<S>(p,0),c::<S>(p,1)); bessel_i_scaled(c::<S>(p,0),c::<S>(p,1)); bessel_k_scaled(c::<S>(p,0),c::<S>(p,1));
        hankel_1(c::<S>(p,0),c::<S>(p,1)); hankel_2(c::<S>(p,0),c::<S>(p,1)); hankel_1_scaled(c::<S>(p,0),c::<S>(p,1)); hankel_2_scaled(c::<S>(p,0),c::<S>(p,1));
        bessel_j_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32); bessel_y_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32);
        bessel_i_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32); bessel_k_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32);
        bessel_i_scaled_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32); bessel_k_scaled_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32);
        hankel_1_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32); hankel_2_derivative(c::<S>(p,0),c::<S>(p,1),p[4] as u32));
    add!(p,S; "integer order >=0; nonzero complex z"; |s| {let mut p=coordinates(&[s.polar(1e-4,50.0)]);p[2]=s.integer(12);p};
        spherical_jn(p[2] as u32,c::<S>(p,0)); spherical_yn(p[2] as u32,c::<S>(p,0)); spherical_in(p[2] as u32,c::<S>(p,0)); spherical_kn(p[2] as u32,c::<S>(p,0));
        spherical_in_scaled(p[2] as u32,c::<S>(p,0)); spherical_kn_scaled(p[2] as u32,c::<S>(p,0));
        spherical_hankel_1(p[2] as u32,c::<S>(p,0)); spherical_hankel_2(p[2] as u32,c::<S>(p,0));
        spherical_hankel_1_scaled(p[2] as u32,c::<S>(p,0)); spherical_hankel_2_scaled(p[2] as u32,c::<S>(p,0));
        parabolic_d_complex(p[2] as u32,c::<S>(p,0)));
    add!(p,S; "finite complex z, integer degree >=0"; |s| {let mut p=coordinates(&[s.polar(1e-3,20.0)]);p[2]=s.integer(32);p}; generalized_binomial(c::<S>(p,0),p[2] as usize));
    add!(p,S; "finite complex s, excluding zeta pole s=1"; |s| coordinates(&[s.cartesian(-10.0,20.0,30.0)]);
        riemann_zeta(c::<S>(p,0)); dirichlet_eta(c::<S>(p,0)));
    add!(p,S; "s!=1; complex q away from nonpositive integer poles"; |s| coordinates(&[s.cartesian(-3.0,10.0,5.0),s.cartesian(0.1,5.0,3.0)]);
        hurwitz_zeta(c::<S>(p,0),c::<S>(p,1)));
    add!(p,S; "principal complex continuation, signed lips at z>=1"; |s| coordinates(&[s.polar(1e-6,10.0)]);
        dilog(c::<S>(p,0)); spence(c::<S>(p,0)));
    add!(p,S; "complex order; principal z continuation excluding (s,z)=(1,1)"; |s| coordinates(&[s.cartesian(-3.0,5.0,2.0),s.polar(1e-3,5.0)]);
        polylog(c::<S>(p,0),c::<S>(p,1)));
    add!(p,S; "complex s,a away from poles; principal z continuation"; |s| coordinates(&[s.polar(1e-3,3.0),s.cartesian(-2.0,4.0,1.0),s.cartesian(0.1,3.0,1.0)]);
        lerch_phi(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2)));
    add!(p,S; "nonzero complex z; integer sheets -2..2"; |s| {
        let z=match (s.index/5)%4 {0=>s.polar(1e-30,0.3),1=>s.polar(0.3,3.0),2=>s.polar(3.0,100.0),_=>s.polar(1e3,1e6)};
        let mut p=coordinates(&[z]);p[2]=(s.index%5) as f64-2.0;p
    }; lambert_w_complex(c::<S>(p,0),p[2] as i32));
    add!(p,S; "finite complex z; conventional omega unwinding"; |s| coordinates(&[s.cartesian(-30.0,30.0,30.0)]);
        wright_omega(c::<S>(p,0)));
    add!(p,S; "finite complex m, principal cuts; m!=1 for K"; |s| coordinates(&[s.polar(1e-5,3.0)]);
        elliptic_k(c::<S>(p,0)); elliptic_e(c::<S>(p,0)));
    add!(p,S; "complex amplitude and parameter, no interior singularity"; |s| coordinates(&[s.cartesian(-0.6,0.6,0.5),s.polar(1e-4,2.0),s.cartesian(0.0,0.8,0.3)]);
        elliptic_f(c::<S>(p,0),c::<S>(p,1)); elliptic_e_inc(c::<S>(p,0),c::<S>(p,1)); elliptic_pi(c::<S>(p,2),c::<S>(p,1)));
    add!(p,S; "nonzero Carlson arguments with compatible signed cut lips"; |s| coordinates(&[s.polar(1e-3,10.0),s.polar(1e-3,10.0),s.polar(1e-3,10.0),s.polar(1e-3,10.0)]);
        carlson_rf(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2)); carlson_rd(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2)); carlson_rg(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2));
        carlson_rc(c::<S>(p,0),c::<S>(p,1)); carlson_rj(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2),c::<S>(p,3)));
    add!(p,S; "finite complex u,m away from Jacobi poles"; |s| coordinates(&[s.cartesian(-3.0,3.0,1.0),s.cartesian(0.1,0.9,0.2)]);
        jacobi_sn(c::<S>(p,0),c::<S>(p,1)); jacobi_cn(c::<S>(p,0),c::<S>(p,1)); jacobi_dn(c::<S>(p,0),c::<S>(p,1)); jacobi_amplitude(c::<S>(p,0),c::<S>(p,1)));
    add!(p,S; "denominator parameter not a nonpositive integer; complex z"; hyper_sample;
        hyp0f1(c::<S>(p,1),c::<S>(p,3));
        hyp2f0(c::<S>(p,0),c::<S>(p,1),c::<S>(p,3)); hyperu(c::<S>(p,0),c::<S>(p,1),c::<S>(p,3)));
    add!(p,S; "finite complex z, including all phases up to |z|=1e4"; |s| {
        let mut p=hyper_sample(s);if s.index.is_multiple_of(2){let z=s.polar(10.0,1e4);p[6]=z[0];p[7]=z[1];}p
    };hyp1f1(c::<S>(p,0),c::<S>(p,1),c::<S>(p,3)));
    add!(p,S; "complex Gauss continuation, including |1-z| down to 1e-15"; |s| {
        let mut p=hyper_sample(s);if s.index.is_multiple_of(2){let z=s.polar(1e-15,1e-2);p[6]=1.0+z[0];p[7]=if z[1]==0.0{z[0].abs()}else{z[1]};}p
    };hyp2f1(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2),c::<S>(p,3));
        hyp2f1_regularized(c::<S>(p,0),c::<S>(p,1),c::<S>(p,2),c::<S>(p,3)));
    add!(p,S; "pFq series: p<=q or p=q+1 with |z|<1; nonpole denominators"; |s| {
        let mut p=hyper_sample(s);let z=s.polar(1e-3,0.9);p[6]=z[0];p[7]=z[1];p[8]=s.integer(2);p
    }; hyp_pfq(&[c::<S>(p,0),c::<S>(p,1)][..p[8] as usize],&[c::<S>(p,2)],c::<S>(p,3)));
    add!(p,S; "real l>-1/2, eta finite; nonzero complex radius"; |s| {let mut p=coordinates(&[s.polar(0.1,20.0)]);p[2]=s.linear(-0.4,5.0);p[3]=s.linear(-3.0,3.0);p};
        coulomb_f(r::<S>(p[2]),r::<S>(p[3]),c::<S>(p,0)); coulomb_g(r::<S>(p[2]),r::<S>(p[3]),c::<S>(p,0));
        coulomb_f_prime(r::<S>(p[2]),r::<S>(p[3]),c::<S>(p,0)); coulomb_g_prime(r::<S>(p[2]),r::<S>(p[3]),c::<S>(p,0)); coulomb_spherical_f(r::<S>(p[2]),r::<S>(p[3]),c::<S>(p,0)));
    add!(p,S; "n>=1, 0<=l<n, real charge>0; finite complex radius"; |s| {let mut p=coordinates(&[s.polar(1e-4,30.0)]);p[2]=1.0+s.integer(8);p[3]=s.integer(p[2] as u32-1);p[4]=s.log(0.1,5.0);p};
        hydrogenic_radial(p[2] as u32,p[3] as u32,r::<S>(p[4]),c::<S>(p,0)));
    add!(p,S; "0<h2<k2, n>=0, 1<=p<=2n+1; complex exterior |s|>k"; |s| {let mut p=coordinates(&[s.polar(1.01,3.0)]);p[2]=s.linear(0.2,0.8);p[3]=s.integer(4);p[4]=1.0+s.integer(2*p[3] as u32);p};
        ellipsoidal_harmonic(r::<S>(p[2]), r::<S>(1.0), p[3] as usize, p[4] as usize, c::<S>(p,0), scioxide_special::EllipsoidalSigns::default());
        ellipsoidal_harmonic_second_kind(r::<S>(p[2]),r::<S>(1.0),p[3] as usize,p[4] as usize,c::<S>(p,0)));
    add!(p,S; "integer order n >= 1; real q; complex z with |Im z| <= 1"; |s| {let mut p=coordinates(&[s.cartesian(-10.0,10.0,1.0)]);p[2]=1.0+s.integer(7);p[3]=s.linear(-25.0,25.0);p};
        mathieu_ce(p[2] as u32,r::<S>(p[3]),c::<S>(p,0)); mathieu_se(p[2] as u32,r::<S>(p[3]),c::<S>(p,0)));
    add!(p,S; "integer order n >= 1; real q != 0; complex z with -1 <= Re z <= 2, |Im z| <= 1; four radial kinds"; |s| {
        let mut p=coordinates(&[s.cartesian(-1.0,2.0,1.0)]);p[2]=1.0+s.integer(5);p[3]=s.linear(0.1,10.0);
        if s.index.is_multiple_of(3) {p[3] = -p[3];}
        p[4]=s.integer(3);p
    }; mathieu_mc(p[2] as u32,r::<S>(p[3]),c::<S>(p,0),radial_kind(p[4])); mathieu_ms(p[2] as u32,r::<S>(p[3]),c::<S>(p,0),radial_kind(p[4])));
    add!(p,S; "prolate or oblate; integer 0 <= m <= n; real c; complex x with |Re x| <= 1, |Im x| <= 0.5"; |s| {
        let mut p=coordinates(&[s.cartesian(-1.0,1.0,0.5)]);p[2]=s.integer(1);p[3]=s.integer(4);p[4]=p[3]+s.integer(6);p[5]=s.linear(-8.0,8.0);p
    }; spheroidal_angular(spheroidal_kind(p[2]),p[3] as u32,p[4] as u32,r::<S>(p[5]),c::<S>(p,0)));
    add!(p,S; "prolate or oblate; integer 0 <= m <= n; real c != 0; complex x with 1.1 <= Re x <= 4, |Im x| <= 0.5, continued near the focus; four radial kinds"; |s| {
        // One sample in four lies where the higher kinds are continued from the expansion.
        let x=if s.index.is_multiple_of(4) {s.cartesian(1.1,1.5,0.5)} else {s.cartesian(1.5,4.0,0.5)};
        let mut p=coordinates(&[x]);p[2]=s.integer(1);p[3]=s.integer(3);p[4]=p[3]+s.integer(4);p[5]=s.linear(0.5,5.0);p[6]=s.integer(3);p
    }; spheroidal_radial(spheroidal_kind(p[2]),p[3] as u32,p[4] as u32,r::<S>(p[5]),c::<S>(p,0),radial_kind(p[6])));
    add!(p,S; "integer l>=0, |m|<=l, theta in [0,pi], finite phi"; |s| {let l=s.integer(16);args(&[l,s.integer(2*l as u32)-l,s.linear(0.0,f64::from(std::f32::consts::PI.next_down())),s.linear(-10.0,10.0)])};
        spherical_harmonic(p[0] as usize,p[1] as i32,r::<S>(p[2]),r::<S>(p[3])));
    entries
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Finite,
    Limit,
    Overflow,
}
fn classify(result: &Result<C, SciOxideError>, limit: Option<C>) -> Result<Class, String> {
    match result {
        Ok(z) if z.is_finite() => Ok(Class::Finite),
        Ok(z) if Some(*z) == limit && !z.re.is_nan() && !z.im.is_nan() => Ok(Class::Limit),
        Ok(z) => Err(format!("undocumented nonfinite value {z:?}")),
        Err(error) => match error.view() {
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            } => Ok(Class::Overflow),
            _ => Err(format!("unexpected {:?}: {error}", error.kind())),
        },
    }
}
fn range_supported(class: Class, wider: &Result<C, SciOxideError>) -> bool {
    match wider {
        Ok(z) if z.is_finite() => match class {
            Class::Overflow => [z.re, z.im]
                .into_iter()
                .any(|v| v.abs() > f64::from(f32::MAX)),
            _ => false,
        },
        _ => classify(wider, None) == Ok(class),
    }
}
fn check(
    entry: &Entry,
    mut p: Args,
    single: bool,
    index: usize,
    limit: Option<C>,
    source_seed: u64,
) -> Result<Class, String> {
    if single {
        p = p.map(|v| f64::from(v as f32));
    }
    let result = if single {
        (entry.f32)(p)
    } else {
        (entry.f64)(p)
    };
    let class = classify(&result, limit).and_then(|class| {
        if single && class == Class::Overflow && !range_supported(class, &(entry.f64)(p)) {
            Err(format!("unjustified binary32 range failure: {result:?}"))
        } else {
            Ok(class)
        }
    });
    class.map_err(|message| {
        format!(
            "{} f{} index={index} seed={:#x} args={p:?} bits={:x?}; {}: {message}",
            entry.name,
            if single { 32 } else { 64 },
            source_seed,
            p.map(|v| if single {
                u64::from((v as f32).to_bits())
            } else {
                v.to_bits()
            }),
            entry.domain
        )
    })
}
#[test]
fn every_public_complex_function_has_a_domain_entry() {
    let entries = entries();
    let table: BTreeSet<_> = entries.iter().map(|e| e.name.to_owned()).collect();
    let public = inventory::functions(true);
    assert_eq!(table.len(), entries.len(), "duplicate table entry");
    assert!(entries.iter().all(|e| !e.domain.is_empty()));
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
            let p = (entry.sample)(&mut samples);
            match check(&entry, p, single, index, None, seed(entry.name)) {
                Ok(Class::Finite) => finite += 1,
                Ok(_) => {}
                Err(message) => {
                    failures.push(message);
                    break;
                }
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
fn documented_domains_f32() {
    sweep(true);
}

#[test]
fn classification_rejects_unexpected_errors_and_unjustified_ranges() {
    for z in [
        C::new(f64::NAN, 0.0),
        C::new(0.0, f64::NAN),
        C::new(f64::INFINITY, 0.0),
        C::new(0.0, f64::NEG_INFINITY),
    ] {
        assert!(classify(&Ok(z), None).is_err());
    }
    let limit = C::new(f64::NEG_INFINITY, 0.0);
    assert_eq!(classify(&Ok(limit), Some(limit)), Ok(Class::Limit));
    for issue in [
        NumericalIssue::Overflow,
        NumericalIssue::Underflow,
        NumericalIssue::LossOfSignificance,
    ] {
        let result = Err(SciOxideError::numerical_instability(issue, "synthetic"));
        // A rounded zero is a value, so an underflow error is never accepted.
        assert_eq!(
            classify(&result, None).is_ok(),
            issue == NumericalIssue::Overflow
        );
    }
    for error in [
        SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new(),
            "synthetic",
        ),
        SciOxideError::domain_error(
            scioxide_core::argument!("z"),
            scioxide_core::DomainReason::Pole,
            "synthetic",
        ),
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Intermediate, "synthetic"),
        <f32 as RealScalar>::try_from_f64_lossy(f64::MAX).unwrap_err(),
    ] {
        assert!(classify(&Err(error), None).is_err());
    }
    assert!(!range_supported(Class::Overflow, &Ok(C::new(1.0, 0.0))));
    assert!(range_supported(Class::Overflow, &Ok(C::new(0.0, f64::MAX))));
}

fn landmarks(single: bool) {
    let entries = entries();
    let evaluate = |name: &str, values: &[f64]| {
        let entry = entries.iter().find(|e| e.name == name).unwrap();
        check(entry, args(values), single, COUNT, None, seed(entry.name)).unwrap();
    };
    let delta = if single {
        2.0_f64.powi(-18)
    } else {
        2.0_f64.powi(-40)
    };
    for n in 0..=4 {
        for side in [-1.0, 1.0] {
            let x = -f64::from(n);
            for z in [
                [x + side * delta, 0.0],
                [x + side * delta, -0.0],
                [x, side * delta],
            ] {
                for name in ["gamma", "ln_gamma", "digamma", "polygamma"] {
                    evaluate(name, &[z[0], z[1], 2.0]);
                }
            }
        }
    }
    for lip in [0.0, -0.0] {
        for branch in -2..=2 {
            for x in [
                -1.0 / std::f64::consts::E - delta,
                -1.0 / std::f64::consts::E,
                -1.0 / std::f64::consts::E + delta,
                -3.0,
                -0.3,
            ] {
                evaluate("lambert_w_complex", &[x, lip, f64::from(branch)]);
            }
        }
        for name in ["gamma_lower", "gamma_upper", "gamma_p", "gamma_q"] {
            evaluate(
                name,
                &[
                    2.7881610989887076,
                    0.03470671342846998,
                    0.02842616704752339,
                    -0.005068492501103518,
                ],
            );
            evaluate(name, &[2.5, 0.1, 0.0, lip]);
        }
        for name in ["dilog", "spence"] {
            for x in [0.0, 1.0 - delta, 1.0, 1.0 + delta, 2.0] {
                evaluate(name, &[x, lip]);
            }
        }
        for name in ["hyp2f1", "hyp2f1_regularized"] {
            for x in [1.0 - delta, 1.0, 1.0 + delta] {
                evaluate(name, &[1.0, 0.0, 1.0, 0.0, 3.0, 0.0, x, lip]);
            }
        }
        for name in ["hyp0f1", "hyp1f1", "hyp2f1", "hyperu"] {
            evaluate(name, &[0.5, 0.1, -2.0, delta, 2.0, 0.2, -0.5, lip]);
        }
        evaluate("carlson_rf", &[0.0, lip, 1.0, 0.0, 2.0, 0.0]);
        evaluate("carlson_rd", &[0.0, lip, 1.0, 0.0, 2.0, 0.0]);
        evaluate("carlson_rg", &[0.0, lip, 0.0, lip, 0.0, lip]);
        evaluate("lambert_w_complex", &[0.0, lip, 0.0]);
        evaluate("pochhammer_relative", &[0.5, 0.5, 0.0, lip]);
        for name in ["spherical_jn", "spherical_in", "parabolic_d_complex"] {
            evaluate(name, &[0.0, lip, 0.0]);
        }
    }
    for sign in [-1.0, 1.0] {
        for dx in [-delta, 0.0, delta] {
            for dy in [-delta, 0.0, delta] {
                evaluate("wright_omega", &[-1.0 + dx, sign * (PI + dy)]);
            }
        }
    }
    let maximum = if single {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let tiny = if single {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    for z in [
        [maximum, 0.0],
        [maximum, maximum],
        [-maximum, maximum],
        [-maximum, 0.0],
        [0.0, maximum],
        [tiny, 0.0],
        [tiny, tiny],
        [-tiny, -0.0],
    ] {
        for branch in -2..=2 {
            evaluate("lambert_w_complex", &[z[0], z[1], f64::from(branch)]);
        }
    }
    for x in [-1000.0, -700.0, 1.0, 600.0, 1000.0] {
        evaluate("wright_omega", &[x, 0.0]);
    }
    for name in ["gamma", "erf", "erfc", "erfcx", "erfi", "faddeeva"] {
        evaluate(name, &[200.0, 1.0]);
    }
}
#[test]
fn landmarks_f64() {
    landmarks(false);
}
#[test]
fn landmarks_f32() {
    landmarks(true);
}

// Preserve every input of the former Bessel-only Issue #1151 target, using the
// common classifier and function table. Its independent seed stays fixed.
fn bessel_regressions(single: bool) {
    let entries = entries();
    let names = [
        "bessel_j", "bessel_y", "bessel_i", "bessel_k", "hankel_1", "hankel_2",
    ];
    let mut rng = Samples {
        state: 1151,
        index: 0,
    };
    for index in 0..2048 {
        let mut nr = rng.linear(-5.0, 5.0);
        let mut ni = if index % 2 == 0 {
            0.0
        } else {
            rng.linear(-3.0, 3.0)
        };
        if index % 32 < 4 {
            nr = nr.round() + if index % 32 < 2 { 0.0 } else { 1e-8 };
            ni *= 1e-8;
        }
        let (lo, hi): (f64, f64) = match (index / 2) % 5 {
            0 => (1e-6, 0.1),
            1 => (0.1, 2.0),
            2 => (2.0, 10.0),
            3 => (10.0, 30.0),
            _ => (30.0, 100.0),
        };
        let radius = rng.linear(lo.ln(), hi.ln()).exp();
        let phase = rng.linear(-PI, PI);
        let (zr, zi) = match index % 16 {
            0 | 1 => (-radius, 0.0),
            2 | 3 => (-radius, -0.0),
            _ => (radius * phase.cos(), radius * phase.sin()),
        };
        for name in names {
            let entry = entries.iter().find(|e| e.name == name).unwrap();
            let class = check(entry, args(&[nr, ni, zr, zi]), single, index, None, 1151)
                .unwrap_or_else(|message| panic!("Issue #1151 original seed=1151: {message}"));
            assert!(
                single || class == Class::Finite,
                "Issue #1151 seed=1151 {name} index={index}: {class:?}"
            );
        }
    }
}
#[test]
fn bessel_regressions_f64() {
    bessel_regressions(false);
}
#[test]
fn bessel_regressions_f32() {
    bessel_regressions(true);
}
