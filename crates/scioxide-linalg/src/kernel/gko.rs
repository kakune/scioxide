//! Contiguous Cauchy generator reconstruction and complex updates.
use scioxide_core::backend::KernelDispatch;
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar};

/// Native real-plane dots for a complex upper-triangular substitution row.
/// The caller checks the resulting candidate before publication.
pub(crate) fn upper_product<T: NumericScalar>(
    upper: &[T],
    offset: usize,
    rhs: &[T],
    start: usize,
    dispatch: KernelDispatch,
) -> Result<Option<Complex<T::Real>>, scioxide_core::SciOxideError> {
    if !T::IS_REAL
        || rhs.len() / 2 - start < 32
        || dispatch.backend() == scioxide_core::backend::KernelBackend::Scalar
    {
        return Ok(None);
    }
    let (ur, ui) = upper.split_at(upper.len() / 2);
    let (yr, yi) = rhs.split_at(rhs.len() / 2);
    let yr = &yr[start..];
    let yi = &yi[start..];
    let ur = &ur[offset..offset + yr.len()];
    let ui = &ui[offset..offset + yr.len()];
    let Some(rr) = super::arch::dot(ur, yr, dispatch)? else {
        return Ok(None);
    };
    let Some(ii) = super::arch::dot(ui, yi, dispatch)? else {
        return Ok(None);
    };
    let Some(ri) = super::arch::dot(ur, yi, dispatch)? else {
        return Ok(None);
    };
    let Some(ir) = super::arch::dot(ui, yr, dispatch)? else {
        return Ok(None);
    };
    Ok(Some(Complex::new(
        (rr - ii).real_component(),
        (ri + ir).real_component(),
    )))
}
// Complex values occupy two native slots for real T, one for complex T.
// This keeps borrowed scalar workspaces usable without casts or extra arenas.
#[inline]
pub(crate) fn width<T: NumericScalar>() -> usize {
    if T::IS_REAL { 2 } else { 1 }
}

#[inline]
pub(crate) fn complex<T: NumericScalar>(z: T) -> Complex<T::Real> {
    Complex::new(z.real_component(), z.imaginary_component())
}

#[inline]
pub(crate) fn get<T: NumericScalar>(values: &[T], i: usize) -> Complex<T::Real> {
    if T::IS_REAL {
        Complex::new(
            values[i].real_component(),
            values[values.len() / 2 + i].real_component(),
        )
    } else {
        complex(values[i])
    }
}

#[inline]
pub(crate) fn put<T: NumericScalar>(values: &mut [T], i: usize, z: Complex<T::Real>) {
    if T::IS_REAL {
        values[i] = T::from_real(z.re);
        let imaginary = values.len() / 2 + i;
        values[imaginary] = T::from_real(z.im);
    } else {
        values[i] = T::from_components(z.re, z.im);
    }
}

/// First largest Euclidean column entry, with independent reduction lanes.
pub(crate) fn pivot<T: NumericScalar>(column: &[T], start: usize) -> usize {
    let n = column.len() / width::<T>();
    let count = n - start;
    let mut maxima = [T::Real::ZERO; 4];
    let mut indices = [start; 4];
    for block in 0..count / 4 {
        for lane in 0..4 {
            let i = start + 4 * block + lane;
            let norm = get(column, i).norm_sqr();
            if norm > maxima[lane] {
                maxima[lane] = norm;
                indices[lane] = i;
            }
        }
    }
    for lane in 0..count % 4 {
        let i = n - count % 4 + lane;
        let norm = get(column, i).norm_sqr();
        if norm > maxima[lane] {
            maxima[lane] = norm;
            indices[lane] = i;
        }
    }
    let mut largest = T::Real::ZERO;
    let mut pivot = start;
    for (norm, i) in maxima.into_iter().zip(indices) {
        if norm > largest || (norm == largest && i < pivot) {
            largest = norm;
            pivot = i;
        }
    }
    if !largest.is_normal() {
        largest = T::Real::ZERO;
        for i in start..n {
            let norm = get(column, i).norm();
            if norm > largest {
                largest = norm;
                pivot = i;
            }
        }
    }
    pivot
}

// Distinct roots lie on the unit circle, so the squared separation is bounded
// by four. Check the resulting value at the reconstructed row/column boundary.
#[inline]
fn reconstruct<R: RealScalar>(numerator: Complex<R>, separation: Complex<R>) -> Complex<R> {
    (numerator * separation.conj()) * (R::ONE / separation.norm_sqr())
}

// Independent complex AXPY entries, with one finite reduction at the boundary.
// Separate real/imaginary planes permit ordinary loop vectorization for real T.
#[inline(always)]
pub(crate) fn subtract_product_scalar<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: Complex<T::Real>,
) -> bool {
    let mut finite = true;
    if T::IS_REAL {
        let (tr, ti) = target.split_at_mut(target.len() / 2);
        let (sr, si) = source.split_at(source.len() / 2);
        let re = T::from_real(factor.re);
        let im = T::from_real(factor.im);
        for (((tr, ti), sr), si) in tr[start..start + count]
            .iter_mut()
            .zip(&mut ti[start..start + count])
            .zip(&sr[source_start..source_start + count])
            .zip(&si[source_start..source_start + count])
        {
            let next_re = *tr - (re * *sr - im * *si);
            let next_im = *ti - (re * *si + im * *sr);
            finite &= next_re.has_finite_components() & next_im.has_finite_components();
            *tr = next_re;
            *ti = next_im;
        }
    } else {
        let factor = T::from_components(factor.re, factor.im);
        for (target, source) in target[start..start + count]
            .iter_mut()
            .zip(&source[source_start..source_start + count])
        {
            *target = *target - factor * *source;
            finite &= target.has_finite_components();
        }
    }
    finite
}

// Reconstruct a contiguous generator row/column. Slicing once outside the
// arithmetic loop permits vectorization of division and finite validation.
#[inline(always)]
pub(crate) fn reconstruct_line_scalar<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (Complex<T::Real>, Complex<T::Real>),
    node: Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
) -> bool {
    let (a, b, nodes) = sources;
    let (wa, wb) = weights;
    let mut finite = true;
    if T::IS_REAL {
        let (ar, ai) = a.split_at(a.len() / 2);
        let (br, bi) = b.split_at(b.len() / 2);
        let (nr, ni) = nodes.split_at(nodes.len() / 2);
        let (out_re, out_im) = output.split_at_mut(output.len() / 2);
        let output_range = offset..offset + range.len();
        let wr = T::from_real(wa.re);
        let wi = T::from_real(wa.im);
        let vr = T::from_real(wb.re);
        let vi = T::from_real(wb.im);
        let node_re = T::from_real(node.re);
        let node_im = T::from_real(node.im);
        for (((((((ar, ai), br), bi), nr), ni), out_re), out_im) in ar[range.clone()]
            .iter()
            .zip(&ai[range.clone()])
            .zip(&br[range.clone()])
            .zip(&bi[range.clone()])
            .zip(&nr[range.clone()])
            .zip(&ni[range])
            .zip(&mut out_re[output_range.clone()])
            .zip(&mut out_im[output_range])
        {
            let pr = (*ar * wr - *ai * wi) + (*br * vr - *bi * vi);
            let pi = (*ar * wi + *ai * wr) + (*br * vi + *bi * vr);
            let dr = *nr - node_re;
            let di = *ni - node_im;
            let inverse = T::ONE / (dr * dr + di * di);
            *out_re = (pr * dr + pi * di) * inverse;
            *out_im = (pi * dr - pr * di) * inverse;
            finite &= out_re.has_finite_components() & out_im.has_finite_components();
        }
    } else {
        for (i, j) in range.enumerate() {
            let value = reconstruct(get(a, j) * wa + get(b, j) * wb, get(nodes, j) - node);
            finite &= value.is_finite();
            put(output, offset + i, value);
        }
    }
    finite
}

pub(crate) fn subtract_product<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: Complex<T::Real>,
    dispatch: KernelDispatch,
) -> bool {
    if let Some(finite) = super::arch::gko_subtract_product(
        target,
        start,
        source,
        source_start,
        count,
        factor,
        dispatch,
    ) {
        return finite;
    }
    subtract_product_scalar(target, start, source, source_start, count, factor)
}

pub(crate) fn reconstruct_line<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (Complex<T::Real>, Complex<T::Real>),
    node: Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
    dispatch: KernelDispatch,
) -> bool {
    if let Some(finite) = super::arch::gko_reconstruct_line(
        sources,
        weights,
        node,
        range.clone(),
        output,
        offset,
        dispatch,
    ) {
        return finite;
    }
    reconstruct_line_scalar(sources, weights, node, range, output, offset)
}
