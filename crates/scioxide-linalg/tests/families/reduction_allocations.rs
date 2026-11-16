//! Allocation contracts for public dense reductions.

use scioxide_core::Complex;
use scioxide_linalg::Vector;

#[test]
fn public_dot_and_norm_paths_do_not_allocate() {
    let real = Vector::from_vec(
        (0..10_003)
            .map(|index| ((index * 17 + 3) % 41) as f64 - 20.0)
            .collect(),
    );
    let complex = Vector::from_vec(
        (0..10_003)
            .map(|index| {
                Complex::new(
                    ((index * 11 + 5) % 37) as f32 - 18.0,
                    ((index * 7 + 1) % 29) as f32 - 14.0,
                )
            })
            .collect(),
    );

    real.view()
        .dotu_scoped(
            &real,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    real.view()
        .norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    scioxide_linalg::VectorView::from(&complex)
        .dotc_left_scoped(
            complex.view(),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    complex
        .view()
        .norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
}
