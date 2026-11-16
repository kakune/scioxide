//! Exact counting, analytic continuation and overflow-safe sequence output.
use scioxide_core::Complex;
use scioxide_special::{
    CombinatorialOptions, bernoulli_with_options, pochhammer, stirling_second_kind_with_options,
};
pub(crate) fn main() -> Result<(), scioxide_core::SciOxideError> {
    let count = scioxide_core::checked_stirling_second_kind(10, 3)?;
    println!("exact S(10,3)={count}");
    let value = pochhammer(Complex::new(3.25_f64, 0.5), Complex::new(0.25, 0.125))?;
    println!("continued rising factorial={value}");
    let large = bernoulli_with_options::<f64>(1000, CombinatorialOptions::default())?;
    println!(
        "B1000: {:?}, log={:?}",
        large.scalar_representation(),
        large.logarithmic()
    );
    let s = stirling_second_kind_with_options::<f64>(100, 50, CombinatorialOptions::default())?;
    println!(
        "S(100,50)={:?}; work={}",
        s.scaled(),
        s.counts().iterations()
    );
    Ok(())
}
