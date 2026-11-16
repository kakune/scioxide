//! Independent Arb regressions for dense systems beyond interval LU wrapping.
#![cfg(test)]
use scioxide_core::{
    AlgebraicScalar, Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorKind,
};
use scioxide_linalg::{LuDecomposition, Matrix};
use scioxide_validated::{
    DenseIntervalOperator, LinearCertificate, LinearVerifier, LuIntervalInverse, ValidatedOptions,
    point_enclosure,
};

fn uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f64 - 8388608.0) / 8388608.0
}
fn cases<S: NumericScalar>(width: &str, complex: bool) {
    let mut count = 0;
    for line in include_str!("data/verified-dense.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] != width || fields[1] != if complex { "1" } else { "0" } {
            continue;
        }
        let n: usize = fields[2].parse().unwrap();
        let mut state: u64 = fields[3].parse().unwrap();
        let mut scalar = || {
            S::from_components(
                S::Real::try_from_f64_exact(uniform(&mut state)).unwrap(),
                S::Real::try_from_f64_exact(if complex { uniform(&mut state) } else { 0.0 })
                    .unwrap(),
            )
        };
        let matrix = Matrix::from_vec(n, n, (0..n * n).map(|_| scalar()).collect()).unwrap();
        let rhs: Vec<_> = (0..n).map(|_| point_enclosure(scalar()).unwrap()).collect();
        let factor = LuDecomposition::factor(&matrix).unwrap();
        let bytes = LuIntervalInverse::<S>::dense_storage_bytes(n).unwrap();
        let limited = ValidatedOptions::new().with_byte_limit(bytes - 1);
        assert_eq!(
            LuIntervalInverse::from_dense(&factor, limited)
                .err()
                .unwrap()
                .kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let options = ValidatedOptions::new().with_byte_limit(bytes);
        let inverse = LuIntervalInverse::from_dense(&factor, options).unwrap();
        let system = DenseIntervalOperator::point(matrix.view()).unwrap();
        let plan = LinearVerifier::new(&system, &inverse, ValidatedOptions::new()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut certificate = LinearCertificate::new(n).unwrap();
        let report = plan
            .solve_into(
                &rhs,
                certificate.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            report.is_verified(),
            "f{width} complex={complex} n={n}: {report:?}"
        );
        assert!(plan.verify(&rhs, certificate.view()).unwrap().is_verified());
        let boxes = certificate.enclosure().unwrap();
        let decode = |word: &str| {
            let bits = u64::from_str_radix(word, 16).unwrap();
            S::Real::try_from_f64_exact(if width == "32" {
                f64::from(f32::from_bits(bits as u32))
            } else {
                f64::from_bits(bits)
            })
            .unwrap()
        };
        let mut max_relative = 0.0_f64;
        for (i, (bounds, &x)) in boxes.iter().zip(certificate.view().center).enumerate() {
            let k = 5 + 4 * i;
            assert!(
                bounds.re.contains(decode(fields[k]))
                    && bounds.re.contains(decode(fields[k + 1]))
                    && bounds.im.contains(decode(fields[k + 2]))
                    && bounds.im.contains(decode(fields[k + 3])),
                "n={n}, component={i}"
            );
            let r = certificate.view().radii[i];
            for (radius, center) in [(r.re, x.real_component()), (r.im, x.imaginary_component())] {
                if center != S::Real::ZERO {
                    max_relative = max_relative.max(radius.into_f64() / center.into_f64().abs());
                }
            }
        }
        if width == "64" {
            let arb: f64 = fields[4].parse().unwrap();
            assert!(
                max_relative <= 10.0 * arb,
                "n={n} complex={complex}: SciOxide={max_relative:e}, Arb53={arb:e}"
            );
        }
        count += 1;
    }
    assert_eq!(count, if width == "64" { 3 } else { 1 });
}
#[test]
fn random_real64() {
    cases::<f64>("64", false);
}
#[test]
fn random_complex64() {
    cases::<Complex<f64>>("64", true);
}
#[test]
fn random_real32() {
    cases::<f32>("32", false);
}
#[test]
fn random_complex32() {
    cases::<Complex<f32>>("32", true);
}

#[test]
fn hilbert_and_singular_systems_are_inconclusive() {
    for n in 12..=14 {
        let entries = (0..n)
            .flat_map(|i| (0..n).map(move |j| 1.0 / (i + j + 1) as f64))
            .collect();
        let matrix = Matrix::from_vec(n, n, entries).unwrap();
        let factor = LuDecomposition::factor(&matrix).unwrap();
        let inverse = LuIntervalInverse::from_dense(&factor, ValidatedOptions::new()).unwrap();
        let system = DenseIntervalOperator::point(matrix.view()).unwrap();
        let plan = LinearVerifier::new(&system, &inverse, ValidatedOptions::new()).unwrap();
        let rhs = vec![point_enclosure(0.0).unwrap(); n];
        let (report, certificate) = plan.solve(&rhs).unwrap();
        assert!(certificate.is_none());
        assert!(
            matches!(
                report.status(),
                scioxide_validated::LinearVerificationStatus::Inconclusive(_)
            ),
            "Hilbert n={n}: {report:?}"
        );
    }
    let singular = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let identity = Matrix::identity(2).unwrap();
    let inverse = LuDecomposition::factor(&identity).unwrap();
    let inverse = LuIntervalInverse::from_dense(&inverse, ValidatedOptions::new()).unwrap();
    let system = DenseIntervalOperator::point(singular.view()).unwrap();
    let plan = LinearVerifier::new(&system, &inverse, ValidatedOptions::new()).unwrap();
    let rhs = vec![point_enclosure(0.0).unwrap(); 2];
    assert!(matches!(
        plan.solve(&rhs).unwrap().0.status(),
        scioxide_validated::LinearVerificationStatus::Inconclusive(_)
    ));
}
