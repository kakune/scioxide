//! Shared, pure validation for PDE inputs and numerical values.
use scioxide_core::{
    AlgebraicScalar, Argument, Dimensions, InvalidInputReason, NonFiniteOrigin, NumericScalar,
    RealScalar, SciOxideError,
};

/// Fixed partition-slot bound of parallel PDE kernels; it caps every policy.
pub(crate) const MAX_WORKERS: usize = 32;

/// The partition-slot bound as a core worker limit.
pub(crate) const WORKER_LIMIT: core::num::NonZeroUsize =
    match core::num::NonZeroUsize::new(MAX_WORKERS) {
        Some(limit) => limit,
        None => core::num::NonZeroUsize::MIN,
    };

/// Most workers `policy` can resolve to under [`WORKER_LIMIT`], for storage bounds.
pub(crate) fn worker_bound(policy: scioxide_core::ExecutionPolicy) -> usize {
    match policy {
        scioxide_core::ExecutionPolicy::Sequential => 1,
        scioxide_core::ExecutionPolicy::Parallel(workers) => workers.get().min(MAX_WORKERS),
        scioxide_core::ExecutionPolicy::Auto => MAX_WORKERS,
    }
}

pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}
pub(crate) fn shape(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(actual),
        ))
    }
}
pub(crate) fn finite<T: NumericScalar>(
    value: T,
    origin: NonFiniteOrigin,
) -> Result<T, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(origin, "value is non-finite"))
    }
}
pub(crate) fn finite_slice<T: NumericScalar>(
    values: &[T],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    values
        .iter()
        .try_for_each(|&value| finite(value, origin).map(|_| ()))
}
pub(crate) fn poison<T: NumericScalar>(values: &mut [T]) {
    values.fill(T::from_real(T::Real::NAN));
}

/// Replace a finite sampled tensor tangent by its Hermitian part, the derivative
/// of the projection [`positive_tensor`] applies to admitted tensors.
pub(crate) fn hermitian_part<V: NumericScalar>(
    tensor: &mut [V],
    n: usize,
) -> Result<(), SciOxideError> {
    shape(
        tensor.len(),
        n.checked_mul(n)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("diffusion")))?,
    )?;
    finite_slice(tensor, NonFiniteOrigin::Callback)?;
    for row in 0..n {
        tensor[row * n + row] = V::from_real(tensor[row * n + row].real_component());
        for col in 0..row {
            let (a, b) = (tensor[row * n + col], tensor[col * n + row]);
            let projected = V::from_components(
                scioxide_core::checked_midpoint(a.real_component(), b.real_component())?,
                scioxide_core::checked_midpoint(a.imaginary_component(), -b.imaginary_component())?,
            );
            tensor[row * n + col] = projected;
            tensor[col * n + row] = projected.conj();
        }
    }
    Ok(())
}

/// Admit a sampled material tensor, project rounding-level Hermitian defects,
/// and check positive definiteness with normalized, unpivoted Cholesky.
/// Only caller-owned factor scratch is overwritten; no factorization workspace
/// or numerical execution scope is needed for this coefficient check.
pub(crate) fn positive_tensor<V: NumericScalar>(
    tensor: &mut [V],
    factor: &mut [V],
    n: usize,
) -> Result<(), SciOxideError> {
    shape(
        tensor.len(),
        n.checked_mul(n)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("diffusion")))?,
    )?;
    shape(factor.len(), tensor.len())?;
    finite_slice(tensor, NonFiniteOrigin::Callback)?;
    let scale = tensor.iter().fold(V::Real::ZERO, |s, &x| {
        s.max(x.real_component().abs())
            .max(x.imaginary_component().abs())
    });
    let rejected = || {
        SciOxideError::domain_error(
            scioxide_core::argument!("diffusion"),
            scioxide_core::DomainReason::OutsideDomain,
            "diffusion tensor must be Hermitian positive definite",
        )
    };
    if scale == V::Real::ZERO {
        return Err(rejected());
    }
    let tolerance = V::Real::EPSILON * V::Real::try_from_usize_exact(n)? * V::Real::ONE.scalbn(5);
    for row in 0..n {
        if (tensor[row * n + row].imaginary_component() / scale).abs() > tolerance {
            return Err(rejected());
        }
        for col in 0..row {
            let a = tensor[row * n + col] / V::from_real(scale);
            let b = tensor[col * n + row].conj() / V::from_real(scale);
            if (a - b).abs() > tolerance {
                return Err(rejected());
            }
        }
    }
    for row in 0..n {
        tensor[row * n + row] = V::from_real(tensor[row * n + row].real_component());
        for col in 0..row {
            let a = tensor[row * n + col];
            let b = tensor[col * n + row];
            let projected = V::from_components(
                scioxide_core::checked_midpoint(a.real_component(), b.real_component())?,
                scioxide_core::checked_midpoint(a.imaginary_component(), -b.imaginary_component())?,
            );
            tensor[row * n + col] = projected;
            tensor[col * n + row] = projected.conj();
        }
    }
    if n <= 3 {
        let diagonal = |i: usize| tensor[i * n + i].real_component() / scale;
        let positive = |x: V::Real| x.is_finite() && x > V::Real::ZERO;
        let p0 = diagonal(0);
        if !positive(p0) {
            return Err(rejected());
        }
        if n == 1 {
            return Ok(());
        }
        let l10 = tensor[n] / V::from_real(scale) / V::from_real(p0.sqrt());
        let p1 = diagonal(1) - (l10 * l10.conj()).real_component();
        if !positive(p1) {
            return Err(rejected());
        }
        if n == 2 {
            return Ok(());
        }
        let l20 = tensor[6] / V::from_real(scale) / V::from_real(p0.sqrt());
        let l21 = (tensor[7] / V::from_real(scale) - l20 * l10.conj()) / V::from_real(p1.sqrt());
        let p2 =
            diagonal(2) - (l20 * l20.conj()).real_component() - (l21 * l21.conj()).real_component();
        return if positive(p2) {
            Ok(())
        } else {
            Err(rejected())
        };
    }
    for row in 0..n {
        for col in 0..=row {
            let mut value = tensor[row * n + col] / V::from_real(scale);
            for k in 0..col {
                value = value - factor[row * n + k] * factor[col * n + k].conj();
            }
            factor[row * n + col] = if row == col {
                let pivot = value.real_component();
                if !pivot.is_finite() || pivot <= V::Real::ZERO {
                    return Err(rejected());
                }
                V::from_real(pivot.sqrt())
            } else {
                finite(value / factor[col * n + col], NonFiniteOrigin::Intermediate)?
            };
        }
    }
    Ok(())
}
