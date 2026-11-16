//! Full-range scalar functions and a finite complex logarithm at extreme scale.
use scioxide_core::{Complex, RealScalar};
fn main() {
    let sine = RealScalar::sin(1e20_f64);
    let decimal = RealScalar::exp10(3.0_f64);
    let logarithm = Complex::new(f64::MAX, f64::MAX).ln();
    println!("sin(1e20)={sine}; 10^3={decimal}; log(MAX+i MAX)={logarithm:?}");
}
