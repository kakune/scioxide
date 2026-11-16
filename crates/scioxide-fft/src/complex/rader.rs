//! Prime-length cyclic convolution and its modular permutations.
use super::*;
#[derive(Debug, Clone)]
pub(super) struct RaderPlan<S: scioxide_core::RealScalar> {
    pub(super) n: usize,
    pub(super) convolution: SmoothPlan<S>,
    pub(super) input_permutation: Vec<usize>,
    pub(super) output_permutation: Vec<usize>,
    pub(super) kernel_spectrum: Vec<Complex<S>>,
    pub(super) convolution_scale: S,
}

impl<S: scioxide_core::RealScalar> RaderPlan<S> {
    pub(super) fn new(
        n: usize,
        dispatch: KernelDispatch,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        // Rader: for prime n with generator g, the non-zero indices are
        // {g^q}, and for k = g^p
        //   X[g^p] = x[0] + sum_q x[g^-q] * w^(g^(p-q)),  w = exp(-2*pi*i/n),
        // a cyclic convolution of length n - 1 of a_q = x[g^-q] with
        // b_m = w^(g^m). `rader_is_suitable` guarantees n - 1 is smooth, so the
        // convolution is a plain SmoothPlan and planning never recurses.
        let convolution_len = n - 1;
        let convolution = SmoothPlan::new(convolution_len, dispatch, length)?;
        let generator = primitive_root(n, length)?;
        let mut powers = try_with_capacity(
            convolution_len,
            scioxide_core::quantity!("rader_permutation_powers"),
        )?;
        let mut power = 1_usize;
        for _ in 0..convolution_len {
            powers.push(power);
            power = mul_mod(power, generator, n);
        }

        // g^-q = g^(n-1-q), so the input order [g^0, g^(n-2), ..., g^1] is the
        // power table reversed after its leading 1.
        let mut input_permutation = try_with_capacity(
            convolution_len,
            scioxide_core::quantity!("rader_input_permutation"),
        )?;
        input_permutation.push(1);
        input_permutation.extend(powers[1..].iter().rev().copied());
        let output_permutation = powers;

        let zero = Complex::new(S::ZERO, S::ZERO);
        let mut kernel = try_filled(
            convolution_len,
            zero,
            scioxide_core::quantity!("rader_convolution_kernel"),
        )?;
        for (value, exponent) in kernel.iter_mut().zip(output_permutation.iter().copied()) {
            *value = reduced_root::<S>(exponent as u128, n as u128, false, length)?;
        }
        // Retain the unscaled kernel spectrum. Apply the prepared convolution
        // normalization in the final assembly, preserving the product order.
        let mut kernel_spectrum = try_filled(
            convolution_len,
            zero,
            scioxide_core::quantity!("rader_convolution_kernel_spectrum"),
        )?;
        convolution.transform_into(&kernel, &mut kernel_spectrum, false);

        let convolution_scale = scalar_ratio_u128::<S>(1, convolution.n as u128, length)?;
        Ok(Self {
            n,
            convolution,
            input_permutation,
            output_permutation,
            kernel_spectrum,
            convolution_scale,
        })
    }

    // The convolution length is plan state, so its invariant names `self`.
    pub(super) fn execute<const BOUNDED: bool>(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        let convolution_len = self.n - 1;
        let (values, work) = scratch[..2 * convolution_len].split_at_mut(convolution_len);
        let zero = Complex::new(S::ZERO, S::ZERO);
        // The kernel spectrum is built for the forward sign only; the inverse
        // uses IDFT(x) = conj(DFT(conj(x))), conjugating on the way in and out.
        // X[0] (the plain sum) is outside the multiplicative group and x[0]
        // contributes to every other bin additively, so both are handled
        // outside the convolution.
        let input_zero = if inverse { data[0].conj() } else { data[0] };
        let dc = data.iter().copied().fold(zero, |sum, value| {
            let value = if BOUNDED {
                crate::validation::bounded(value)
            } else {
                value
            };
            sum + if inverse { value.conj() } else { value }
        });
        values
            .iter_mut()
            .zip(self.input_permutation.iter().copied())
            .for_each(|(destination, source)| {
                let value = data[source];
                *destination = if inverse { value.conj() } else { value };
            });
        self.convolution.transform_into(values, work, false);
        work.iter_mut()
            .zip(self.kernel_spectrum.iter().copied())
            .for_each(|(value, kernel)| *value *= kernel);
        self.convolution.transform_into(work, values, true);

        let directed = |value: Complex<S>| if inverse { value.conj() } else { value };
        data[0] = directed(dc) * scale;
        self.output_permutation
            .iter()
            .copied()
            .zip(values.iter().copied())
            .for_each(|(destination, value)| {
                data[destination] = directed(input_zero + value * self.convolution_scale) * scale;
            });
    }

    pub(super) fn heap_bytes(&self) -> Result<usize, SciOxideError> {
        [
            self.convolution.heap_bytes()?,
            crate::memory::vector_bytes(&self.input_permutation),
            crate::memory::vector_bytes(&self.output_permutation),
            crate::memory::vector_bytes(&self.kernel_spectrum),
        ]
        .into_iter()
        .try_fold(0, |bytes, value| {
            checked_add(bytes, value, scioxide_core::quantity!("fft_plan_bytes"))
        })
    }
}

pub(super) fn primitive_root(prime: usize, length: Argument) -> Result<usize, SciOxideError> {
    let phi = prime - 1;
    let mut factors =
        try_with_capacity(16, scioxide_core::quantity!("rader_primitive_root_factors"))?;
    let mut remaining = phi;
    let mut divisor = 2_usize;
    while divisor <= remaining / divisor {
        if remaining.is_multiple_of(divisor) {
            factors.push(divisor);
            while remaining.is_multiple_of(divisor) {
                remaining /= divisor;
            }
        }
        divisor += usize::from(divisor == 2) + 1;
    }
    if remaining > 1 {
        factors.push(remaining);
    }
    // g is a generator of (Z/p)^* iff g^((p-1)/q) != 1 for every prime q
    // dividing p - 1; the smallest such g is chosen so plans are deterministic.
    for candidate in 2..prime {
        if factors
            .iter()
            .copied()
            .all(|factor| pow_mod(candidate, phi / factor, prime) != 1)
        {
            return Ok(candidate);
        }
    }
    Err(SciOxideError::invalid_input(
        length,
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        "Rader primitive-root search failed",
    ))
}

pub(super) fn pow_mod(mut base: usize, mut exponent: usize, modulus: usize) -> usize {
    let mut result = 1_usize;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

// Widen to u128 so the product of two residues below a usize modulus cannot
// overflow.
pub(super) fn mul_mod(left: usize, right: usize, modulus: usize) -> usize {
    ((left as u128 * right as u128) % modulus as u128) as usize
}
