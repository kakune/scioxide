//! One-call discrete cosine and sine transforms; see [`crate::RealTransformPlan`].
use crate::{
    DctType, DstType, FftDirection, Normalization, RealTransformKind, RealTransformPlan,
    RealTransformPlanSpec,
};
use alloc::vec::Vec;
use scioxide_core::SciOxideError;
use scioxide_core::backend::try_filled;

/// Computes the discrete cosine transform of type I, II, III or IV.
///
/// For type II, the most common one,
///
/// ```text
/// y[k] = Σⱼ x[j] cos(π (j + ½) k / n),   0 ≤ j, k < n.
/// ```
///
/// [`RealTransformPlan`] defines all four types, their endpoint weights and
/// their scale factors under each [`Normalization`]. The result has the length
/// of `input` and costs `O(n log n)`. [`idct`] is the inverse. Use a
/// [`RealTransformPlan`] to transform many sequences of the same length.
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty, or has fewer than two values for
///   type I.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a scale factor or a result is not
///   representable.
///
/// # Examples
///
/// A constant sequence has only a zero-frequency coefficient, equal to the sum
/// for type II. Inverse normalization leaves the forward transform unscaled.
///
/// ```
/// use scioxide_fft::{DctType, Normalization, dct};
///
/// let coefficients = dct(&[1.0_f64; 4], DctType::II, Normalization::Inverse)?;
/// assert!((coefficients[0] - 4.0).abs() < 1e-12);
/// assert!(coefficients[1..].iter().all(|x| x.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - V. Britanak, P. C. Yip and K. R. Rao, *Discrete Cosine and Sine
///   Transforms: General Properties, Fast Algorithms and Integer
///   Approximations*, Academic Press, 2007.
/// - J. Makhoul, "A fast cosine transform in one and two dimensions",
///   *IEEE Trans. Acoust. Speech Signal Process.* 28(1), 1980, 27–34.
pub fn dct<S: scioxide_core::RealScalar>(
    input: &[S],
    ty: DctType,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(input, ty.into(), FftDirection::Forward, normalization)
}

/// Computes the inverse of the discrete cosine transform of the given type.
///
/// `ty` names the forward type to invert. The inverse of types I and IV is a
/// scaled transform of the same type, and the inverse of type II is a scaled
/// type III and vice versa. With the same [`Normalization`] as [`dct`], other
/// than `Unscaled`, `idct(dct(x))` returns `x` up to rounding; `Unscaled`
/// returns `x` times the factor listed in [`RealTransformPlan`].
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty, or has fewer than two values for
///   type I.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a scale factor or a result is not
///   representable.
///
/// # Examples
///
/// The inverse takes the forward type, so `DctType::II` runs a type-III
/// transform.
///
/// ```
/// use scioxide_fft::{DctType, Normalization, dct, idct};
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0];
/// let coefficients = dct(&input, DctType::II, Normalization::Orthonormal)?;
/// let recovered = idct(&coefficients, DctType::II, Normalization::Orthonormal)?;
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn idct<S: scioxide_core::RealScalar>(
    input: &[S],
    ty: DctType,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(input, ty.into(), FftDirection::Inverse, normalization)
}

/// Computes the discrete sine transform of type I, II, III or IV.
///
/// For type II,
///
/// ```text
/// y[k] = Σⱼ x[j] sin(π (j + ½)(k + 1) / n),   0 ≤ j, k < n.
/// ```
///
/// [`RealTransformPlan`] defines all four types, their endpoint weights and
/// their scale factors under each [`Normalization`]. Every type accepts a
/// single value. The result has the length of `input` and costs `O(n log n)`.
/// [`idst`] is the inverse.
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a scale factor or a result is not
///   representable.
///
/// # Examples
///
/// For `x = [1, 1]`, type II gives `y[0] = sin(π/4) + sin(3π/4) = √2` and
/// `y[1] = sin(π/2) + sin(3π/2) = 0`. Inverse normalization leaves the forward
/// transform unscaled.
///
/// ```
/// use scioxide_fft::{DstType, Normalization, dst};
///
/// let coefficients = dst(&[1.0_f64; 2], DstType::II, Normalization::Inverse)?;
/// assert!((coefficients[0] - 2.0_f64.sqrt()).abs() < 1e-12);
/// assert!(coefficients[1].abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - V. Britanak, P. C. Yip and K. R. Rao, *Discrete Cosine and Sine
///   Transforms: General Properties, Fast Algorithms and Integer
///   Approximations*, Academic Press, 2007.
/// - J. Makhoul, "A fast cosine transform in one and two dimensions",
///   *IEEE Trans. Acoust. Speech Signal Process.* 28(1), 1980, 27–34.
pub fn dst<S: scioxide_core::RealScalar>(
    input: &[S],
    ty: DstType,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(input, ty.into(), FftDirection::Forward, normalization)
}

/// Computes the inverse of the discrete sine transform of the given type.
///
/// `ty` names the forward type to invert, with the pairing and scaling rules
/// of [`idct`]: types I and IV invert to themselves, and types II and III
/// invert to each other.
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a scale factor or a result is not
///   representable.
///
/// # Examples
///
/// The inverse takes the forward type, so `DstType::II` runs a type-III
/// transform.
///
/// ```
/// use scioxide_fft::{DstType, Normalization, dst, idst};
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0];
/// let coefficients = dst(&input, DstType::II, Normalization::Orthonormal)?;
/// let recovered = idst(&coefficients, DstType::II, Normalization::Orthonormal)?;
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn idst<S: scioxide_core::RealScalar>(
    input: &[S],
    ty: DstType,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(input, ty.into(), FftDirection::Inverse, normalization)
}

fn real_transform<S: scioxide_core::RealScalar>(
    input: &[S],
    kind: RealTransformKind,
    direction: FftDirection,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    let plan = RealTransformPlan::new(RealTransformPlanSpec::new(
        input.len(),
        kind,
        direction,
        normalization,
    ))?;
    let quantity = match (kind, direction) {
        (RealTransformKind::Dct(_), FftDirection::Forward) => {
            scioxide_core::quantity!("dct_output")
        }
        (RealTransformKind::Dct(_), FftDirection::Inverse) => {
            scioxide_core::quantity!("inverse_dct_output")
        }
        (RealTransformKind::Dst(_), FftDirection::Forward) => {
            scioxide_core::quantity!("dst_output")
        }
        (RealTransformKind::Dst(_), FftDirection::Inverse) => {
            scioxide_core::quantity!("inverse_dst_output")
        }
    };
    let mut output = try_filled(input.len(), S::ZERO, quantity)?;
    let mut workspace = plan.workspace(())?;
    plan.execute_into(
        input,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_empty() {
        let error = dct::<f64>(&[], DctType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_idct_empty() {
        let error = idct::<f64>(&[], DctType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_dct_type1_too_short() {
        let error = dct(&[1.0_f64], DctType::I, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_dct_non_finite_input() {
        let error = dct(&[1.0_f64, f64::NAN], DctType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_idct_non_finite_input() {
        let error = idct(
            &[1.0_f64, f64::INFINITY],
            DctType::II,
            Normalization::Inverse,
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_dct_type1_constant() {
        // DCT-I of [1, 1, 1]: X[0] = 4, rest = 0
        let input = [1.0_f64, 1.0, 1.0];
        let result = dct(&input, DctType::I, Normalization::Inverse).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
        assert!(result[2].abs() < 1e-12);
    }

    #[test]
    fn test_dct_type2_dc() {
        // DC term (k=0): cos(0) = 1 for all n, so X[0] = sum(x)
        let input = [1.0_f64, 2.0, 3.0, 4.0];
        let result = dct(&input, DctType::II, Normalization::Inverse).unwrap();
        assert!((result[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn normalized_dcts_scale_before_growth() {
        let input = [f64::MAX * 0.125; 16];
        for ty in [DctType::I, DctType::II, DctType::III, DctType::IV] {
            for normalization in [Normalization::Forward, Normalization::Orthonormal] {
                let result = dct(&input, ty, normalization).unwrap();
                assert!(result.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn test_dst_empty() {
        let error = dst::<f64>(&[], DstType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_idst_empty() {
        let error = idst::<f64>(&[], DstType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_dst_non_finite_input() {
        let error = dst(&[1.0_f64, f64::NAN], DstType::II, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_idst_non_finite_input() {
        let error = idst(
            &[1.0_f64, f64::INFINITY],
            DstType::II,
            Normalization::Inverse,
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_dst_type1_known() {
        // DST-I of [1, 1, 1] with N=3, N+1=4:
        // X[k] = Σ x[n] sin(π(n+1)(k+1)/4)
        // X[0] = sin(π/4) + sin(π/2) + sin(3π/4) = √2/2 + 1 + √2/2 = 1 + √2
        let input = [1.0_f64, 1.0, 1.0];
        let result = dst(&input, DstType::I, Normalization::Inverse).unwrap();
        assert_eq!(result.len(), 3);
        let expected_0 = 1.0 + std::f64::consts::SQRT_2;
        assert!((result[0] - expected_0).abs() < 1e-12);
    }

    #[test]
    fn test_dst_type2_single() {
        // DST-II of [a]: X[0] = a * sin(π*1*1/2) = a * sin(π/2) = a
        let input = [3.0_f64];
        let result = dst(&input, DstType::II, Normalization::Inverse).unwrap();
        assert!((result[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_dsts_scale_before_growth() {
        let input = [f64::MAX * 0.125; 16];
        for ty in [DstType::I, DstType::II, DstType::III, DstType::IV] {
            for normalization in [Normalization::Forward, Normalization::Orthonormal] {
                let result = dst(&input, ty, normalization).unwrap();
                assert!(result.iter().all(|value| value.is_finite()));
            }
        }
    }
}
