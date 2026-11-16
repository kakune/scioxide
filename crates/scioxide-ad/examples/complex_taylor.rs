//! Taylor coefficients of complex functions with jets as parts.
use scioxide_ad::{ComplexElementary, Jet};
use scioxide_core::{Complex, SciOxideError};

fn main() -> Result<(), SciOxideError> {
    // z(t) = 0.3 + (1 + 2i)·t, so exp(z(t)) has cₖ = e^0.3·(1 + 2i)ᵏ/k!.
    let z = Complex::new(
        Jet::<f64, 5>::new([0.3, 1.0, 0.0, 0.0, 0.0]),
        Jet::new([0.0, 2.0, 0.0, 0.0, 0.0]),
    );
    let w = z.exp();
    let mut expected = Complex::new(0.3_f64.exp(), 0.0);
    for k in 0..5 {
        assert!((w.re.coefficients()[k] - expected.re).abs() < 1e-13);
        assert!((w.im.coefficients()[k] - expected.im).abs() < 1e-13);
        println!(
            "c{k} = {} + {}i",
            w.re.coefficients()[k],
            w.im.coefficients()[k]
        );
        expected = expected * Complex::new(1.0, 2.0) / Complex::new((k + 1) as f64, 0.0);
    }
    // ln(exp z) = z near the real axis.
    let back = w.ln();
    assert!((back.im.coefficients()[1] - 2.0).abs() < 1e-13);
    Ok(())
}
