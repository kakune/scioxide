//! Large-radius regime selection and incremental recurrence work.
#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex, ExecutionContext, NumericScalar, RealScalar};
use scioxide_special::*;

fn bands<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let r = |x| S::try_from_f64_lossy(x).unwrap();
    for (nu, x) in [(560., 1e5), (999., 1e5), (2000., 1e6), (6000., 1e7)] {
        for result in [
            bessel_j_with_options(r(nu), r(x), BesselOptions::default()),
            bessel_y_with_options(r(nu), r(x), BesselOptions::default()),
        ] {
            let result = result.unwrap();
            assert!(result.termination().is_success());
            assert!(result.counts().iterations() < 4096);
        }
    }
    // Outside the adjacent Hankel admission band, beyond the turning circle,
    // Debye's adjacent K values replace Miller recurrences whose depth grows
    // with the radius. Off-axis and real orders share the path (#1676).
    for (nu, x, y) in [
        (Complex::new(500., 0.5), 4e4, 0.),
        (Complex::new(600., 0.5), 5e4, 0.),
        (Complex::new(1100., 0.5), 66000., 0.),
        (Complex::new(1100., 0.), 66000., 1e-3),
        (Complex::new(2000., 0.5), 1e6, 0.),
        (Complex::new(6000., -2.), 1e7, -1.),
    ] {
        let order = Complex::new(r(nu.re), r(nu.im));
        let z = Complex::new(r(x), r(y));
        for result in [
            bessel_j_with_options(order, z, BesselOptions::default()),
            bessel_i_with_options(order, Complex::new(-z.im, z.re), BesselOptions::default()),
            bessel_y_with_options(order, z, BesselOptions::default()),
        ] {
            let result = result.unwrap();
            assert!(result.termination().is_success());
            assert!(result.counts().iterations() < 512);
        }
        let limited = bessel_j_with_options(
            order,
            z,
            BesselOptions::default().with_iteration_budget(Budget::try_from(16).unwrap()),
        );
        assert_eq!(publication::budget_counts(limited).iterations(), 16);
    }
    for (l, eta) in [(0., 200.), (200., 0.), (300., 300.)] {
        let mut previous_count = usize::MAX;
        for x in [7e4, 1e5, 1e6] {
            let result =
                coulomb_wave_with_options(r(l), r(eta), r(x), CoulombOptions::default()).unwrap();
            assert!(result.is_success());
            let count = result.f().counts().iterations();
            // Phase reduction can need a few extra terms at a larger radius.
            assert!(count < 4096 && count <= previous_count.saturating_add(16));
            previous_count = count;
        }
    }
}

#[test]
fn both_precisions_complete_with_reused_storage_or_raising_the_budget() {
    bands::<f32>();
    bands::<f64>();
}

#[test]
fn large_coulomb_sequences_reuse_seeds_and_publish_atomically() {
    let mut output = [CoulombWaveResult::<f64>::zero(); 4];
    let mut staging = output;
    for (l, eta) in [(0., 200.), (200., 0.), (300., 300.)] {
        coulomb_wave_sequence_into(
            l,
            eta,
            7e4,
            CoulombOptions::default(),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (offset, wave) in output.iter().enumerate() {
            let reference =
                coulomb_wave_with_options(l + offset as f64, eta, 7e4, CoulombOptions::default())
                    .unwrap();
            for (a, b) in [wave.f(), wave.g(), wave.f_prime(), wave.g_prime()]
                .into_iter()
                .zip([
                    reference.f(),
                    reference.g(),
                    reference.f_prime(),
                    reference.g_prime(),
                ])
            {
                assert!(
                    (a.try_value().unwrap() - b.try_value().unwrap()).abs()
                        <= a.value().unwrap().absolute_error()
                            + b.value().unwrap().absolute_error()
                );
            }
        }
        let before = output;
        let error = coulomb_wave_sequence_into(
            l,
            eta,
            7e4,
            CoulombOptions::default().with_iteration_budget(Budget::ONE),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(publication::exhausted_counts(&error).iterations(), 1);
        for (actual, expected) in output.iter().zip(before) {
            assert_eq!(actual.f(), expected.f());
            assert_eq!(actual.g(), expected.g());
            assert_eq!(actual.f_prime(), expected.f_prime());
            assert_eq!(actual.g_prime(), expected.g_prime());
        }
    }
}
