//! Trace shifts cover the whole augmentation and restore factors in extended range.
use super::*;
use scioxide_core::{AdjointOperator, Scaled, SciOxideErrorKind};
use scioxide_linalg::ExponentialActionReport;

fn action<S: NumericScalar>(
    op: &dyn AdjointOperator<S>,
    options: ExponentialAction<S>,
    time: S::Real,
    phi: usize,
    rhs: &[S],
) -> (Vec<S>, ExponentialActionReport) {
    let plan = options.with_max_phi(phi);
    let mut workspace = plan.workspace(op, 1).unwrap();
    let mut output = vec![S::ZERO; rhs.len()];
    let report = plan
        .apply_into(
            op,
            scioxide_linalg::PhiFunction::phi(phi, time),
            BlockVectorView::new(rhs, rhs.len(), 1).unwrap(),
            BlockVectorViewMut::new(&mut output, rhs.len(), 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    (output, report)
}

fn bound(a: &Matrix<f64>, shift: f64) -> f64 {
    (0..a.cols())
        .map(|j| {
            (0..a.rows())
                .map(|i| (a.view().get(i, j).unwrap() - if i == j { shift } else { 0. }).abs())
                .sum::<f64>()
        })
        .fold(0., f64::max)
}

#[test]
fn negative_spectrum_trace_shift_reaches_working_precision_with_less_work() {
    let n = 30;
    let diagonal: Vec<_> = (0..n)
        .map(|i| -9. + 8.7 * i as f64 / (n - 1) as f64)
        .collect();
    let mut a = Matrix::zeros(n, n).unwrap();
    for (i, &d) in diagonal.iter().enumerate() {
        a.view_mut().set(i, i, d).unwrap();
    }
    let mu = diagonal.iter().sum::<f64>() / n as f64;
    let op = a.view().try_finite().unwrap();
    let (unshifted, before) = action(&op, ExponentialAction::new(9.), 1., 0, &vec![1.; n]);
    let (shifted, after) = action(
        &op,
        ExponentialAction::new(bound(&a, mu)).with_shift(mu),
        1.,
        0,
        &vec![1.; n],
    );
    let reference: Vec<_> = diagonal.iter().map(|d| d.exp()).collect();
    let norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
    let error = difference_norm(&shifted, &reference) / norm;
    assert!(error <= 1e-15, "relative error {error:e}");
    assert!(
        after.applications() <= before.applications(),
        "{} > {}",
        after.applications(),
        before.applications()
    );
    assert!(difference_norm(&shifted, &reference) <= difference_norm(&unshifted, &reference));
}

#[test]
fn shifted_phi_zero_one_two_agree_on_nonnormal_forty_by_forty() {
    let n = 40;
    let mut a = Matrix::zeros(n, n).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, -0.2 - 0.01 * i as f64).unwrap();
        if i + 1 < n {
            a.view_mut().set(i, i + 1, 0.5).unwrap();
        }
        if i + 2 < n {
            a.view_mut().set(i, i + 2, -0.1).unwrap();
        }
    }
    let mu = (0..n).map(|i| a.view().get(i, i).unwrap()).sum::<f64>() / n as f64;
    let rhs: Vec<_> = (0..n).map(|i| 1. + i as f64 / n as f64).collect();
    let op = a.view().try_finite().unwrap();
    for time in [1., -0.5] {
        for phi in 0..=2 {
            let (reference, _) =
                action(&op, ExponentialAction::new(bound(&a, 0.)), time, phi, &rhs);
            let (shifted, _) = action(
                &op,
                ExponentialAction::new(bound(&a, mu)).with_shift(mu),
                time,
                phi,
                &rhs,
            );
            let norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
            let error = difference_norm(&shifted, &reference) / norm;
            assert!(error <= 1e-14, "phi {phi}, time {time}: {error:e}");
        }
    }
}

#[test]
fn restoration_survives_underflowing_and_overflowing_exponential_factors() {
    for (mu, rhs_scale) in [(-800., 1e300), (800., 1e-300)] {
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                mu, 1., 0., 0., 0., mu, 1., 0., 0., 0., mu, 1., 0., 0., 0., mu,
            ],
        )
        .unwrap();
        let op = a.view().try_finite().unwrap();
        let (output, _) = action(
            &op,
            ExponentialAction::new(1.).with_shift(mu),
            1.,
            0,
            &[rhs_scale; 4],
        );
        let base = Scaled::from_log(1., mu)
            .unwrap()
            .mul(Scaled::from_value(rhs_scale).unwrap())
            .unwrap()
            .round()
            .unwrap();
        for (&actual, factor) in output.iter().zip([8. / 3., 2.5, 2., 1.]) {
            assert!(actual.is_finite());
            assert!(
                (actual / (base * factor) - 1.).abs() <= 5e-13,
                "mu {mu}: {actual:e}"
            );
        }
    }
}

#[test]
fn complex_shift_restores_phase_and_uses_conjugate_adjoint() {
    let mu = Complex::new(-0.5, 20.);
    let one = Complex::new(1., 0.);
    // A large shifted bound selects power estimates. The shifted square is
    // nonzero, so estimation reaches its adjoint probes instead of exiting early.
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            mu + Complex::from_real(0.25),
            Complex::new(0., 1000.),
            Complex::ZERO,
            mu - Complex::from_real(0.25),
        ],
    )
    .unwrap();
    let op = a.view().try_finite().unwrap();
    for time in [0.5, -0.5] {
        let (output, _) = action(
            &op,
            ExponentialAction::new(1000.25).with_shift(mu),
            time,
            0,
            &[one; 2],
        );
        let factor = (mu * Complex::from_real(time)).exp();
        let reference = [
            factor * Complex::new((0.25 * time).exp(), 1000. * (0.25 * time).sinh() / 0.25),
            factor * Complex::from_real((-0.25 * time).exp()),
        ];
        for (&actual, expected) in output.iter().zip(reference) {
            assert!((actual - expected).norm() / expected.norm() <= 1e-13);
        }
    }
}

#[test]
fn shift_validation_and_zero_shift_and_time() {
    let a = Matrix::from_vec(2, 2, vec![1., 0.5, 0., -1.]).unwrap();
    let op = a.view().try_finite().unwrap();
    for mu in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = ExponentialAction::new(1.)
            .with_shift(mu)
            .workspace_requirements(&op, 1)
            .unwrap_err();
        assert_eq!(err.kind(), SciOxideErrorKind::NonFinite);
    }
    let complex = Matrix::<Complex<f64>>::identity(2).unwrap();
    let err = ExponentialAction::new(1.)
        .with_shift(Complex::new(0., f64::NAN))
        .workspace_requirements(&complex.view().try_finite().unwrap(), 1)
        .unwrap_err();
    assert_eq!(err.kind(), SciOxideErrorKind::NonFinite);
    for (bound, kind) in [
        (f64::NAN, SciOxideErrorKind::NonFinite),
        (-1.0, SciOxideErrorKind::InvalidInput),
    ] {
        let err = ExponentialAction::<f64>::new(bound)
            .workspace_requirements(&op, 1)
            .unwrap_err();
        assert_eq!(err.kind(), kind);
    }
    let options = ExponentialAction::new(1.5);
    assert_eq!(options.shift(), 0.);
    for phi in 0..=2 {
        let (default, _) = action(&op, options, 1., phi, &[1., 2.]);
        let (zero, _) = action(&op, options.with_shift(0.), 1., phi, &[1., 2.]);
        assert_eq!(default, zero);
        let (at_zero, _) = action(&op, options.with_shift(800.), 0., phi, &[1., 2.]);
        let divisor = if phi == 2 { 2. } else { 1. };
        assert_eq!(at_zero, [1. / divisor, 2. / divisor]);
    }
}

fn scalar_shift<S: NumericScalar>() {
    let real = |x| S::Real::try_from_f64_lossy(x).unwrap();
    let mu = S::from_components(real(-2.), if S::IS_REAL { real(0.) } else { real(0.25) });
    let a = Matrix::from_vec(2, 2, vec![mu, S::ONE, S::ZERO, mu]).unwrap();
    let op = a.view().try_finite().unwrap();
    let rhs = [S::ONE, S::from_real(real(2.))];
    let factor = S::from_components(
        real(-2.).exp() * mu.imaginary_component().cos(),
        real(-2.).exp() * mu.imaginary_component().sin(),
    );
    let (out, _) = action(
        &op,
        ExponentialAction::new(real(1.)).with_shift(mu),
        real(1.),
        0,
        &rhs,
    );
    for (&actual, expected) in out
        .iter()
        .zip([S::from_real(real(3.)) * factor, rhs[1] * factor])
    {
        assert!((actual - expected).abs() <= real(128.) * S::Real::EPSILON * expected.abs());
    }
}

#[test]
fn shift_supports_all_four_scalar_types() {
    scalar_shift::<f32>();
    scalar_shift::<f64>();
    scalar_shift::<Complex<f32>>();
    scalar_shift::<Complex<f64>>();
}

/// `φₚ(d) b = eᵈ b / dᵖ − Σ_{j<p} dʲ⁻ᵖ b / j!`, with the first term formed in
/// log space for `d > 0`.
fn phi_scalar(p: usize, d: f64, b: f64) -> f64 {
    let p_int = i32::try_from(p).unwrap();
    let leading = if d > 0. {
        b.signum() * (d - f64::from(p_int) * d.ln() + b.abs().ln()).exp()
    } else {
        d.exp() / d.powi(p_int) * b
    };
    let mut factorial = 1.;
    let mut sum = 0.;
    for j in 0..p_int {
        if j > 0 {
            factorial *= f64::from(j);
        }
        sum += d.powi(j - p_int) / factorial;
    }
    leading - sum * b
}

#[test]
fn phi_actions_converge_for_stiff_and_strongly_shifted_operators() {
    // Strongly decaying unshifted operators, and operators near `μ I` whose
    // shift `μ` drives the trailing augmented block: in both cases the
    // series of each step cancels. The input scale keeps `φₚ(A) b` finite.
    for (centre, shift, p, scale) in [
        (-800., 0., 1, 1.),
        (-5000., 0., 2, 1.),
        (800., 800., 1, 1e-300),
        (800., 800., 3, 1e-300),
    ] {
        let diagonal = [centre - 1., centre + 0.5, centre + 1.];
        let mut a = Matrix::zeros(3, 3).unwrap();
        for (i, &d) in diagonal.iter().enumerate() {
            a.view_mut().set(i, i, d).unwrap();
        }
        let op = a.view().try_finite().unwrap();
        let rhs = [scale, -2. * scale, 3. * scale];
        let options = ExponentialAction::new(bound(&a, shift)).with_shift(shift);
        let (out, _) = action(&op, options, 1., p, &rhs);
        for ((&actual, &d), &b) in out.iter().zip(&diagonal).zip(&rhs) {
            let expected = phi_scalar(p, d, b);
            assert!(
                (actual - expected).abs() <= 1e-10 * expected.abs(),
                "{centre} {p}: {actual:e} vs {expected:e}"
            );
        }
    }
}
