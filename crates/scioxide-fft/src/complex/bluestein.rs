//! Chirp convolution for awkward exact lengths.
use super::*;
use crate::chirp::square_mod_twice_len;
#[derive(Debug, Clone)]
pub(super) struct BluesteinPlan<S: scioxide_core::RealScalar> {
    pub(super) n: usize,
    pub(super) convolution: SmoothPlan<S>,
    pub(super) chirp: Vec<Complex<S>>,
    pub(super) kernel_spectrum: Vec<Complex<S>>,
    pub(super) convolution_scale: S,
}

impl<S: scioxide_core::RealScalar> BluesteinPlan<S> {
    pub(super) fn new(
        n: usize,
        workspace_len: usize,
        dispatch: KernelDispatch,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        // Bluestein: jk = (j^2 + k^2 - (k - j)^2) / 2 turns the DFT into
        //   X[k] = c[k] * sum_j (x[j] * c[j]) * conj(c[k - j]),
        //   c[j] = exp(-i*pi*j^2/n),
        // a linear convolution with lags in (-n, n). Any cyclic length of at
        // least 2n - 1 avoids wrap-around aliasing, so the smallest smooth
        // length above it is used to keep the private FFTs mixed-radix.
        let convolution = SmoothPlan::new(workspace_len, dispatch, length)?;
        let zero = Complex::new(S::ZERO, S::ZERO);
        let mut chirp = try_with_capacity(n, scioxide_core::quantity!("bluestein_chirp"))?;
        let mut kernel = try_filled(
            workspace_len,
            zero,
            scioxide_core::quantity!("bluestein_convolution_kernel"),
        )?;
        for index in 0..n {
            let exponent = square_mod_twice_len(index, n, length)?;
            let value = reduced_root::<S>(exponent, 2_u128 * n as u128, false, length)?;
            let kernel_value = value.conj();
            chirp.push(value);
            // conj(c) is even in the lag, so negative lag -index is stored
            // cyclically at workspace_len - index; the gap between the two
            // halves stays zero.
            kernel[index] = kernel_value;
            if index != 0 {
                kernel[workspace_len - index] = kernel_value;
            }
        }
        let mut kernel_spectrum = try_filled(
            workspace_len,
            zero,
            scioxide_core::quantity!("bluestein_convolution_kernel_spectrum"),
        )?;
        convolution.transform_into(&kernel, &mut kernel_spectrum, false);
        let convolution_scale = scalar_ratio_u128::<S>(1, convolution.n as u128, length)?;
        Ok(Self {
            n,
            convolution,
            chirp,
            kernel_spectrum,
            convolution_scale,
        })
    }

    // The workspace length is plan state, so its invariant names `self`.
    pub(super) fn execute<const BOUNDED: bool>(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
    ) {
        let workspace_len = self.convolution.n;
        let (values, work) = scratch[..2 * workspace_len].split_at_mut(workspace_len);
        // Scratch may hold stale data; the zero padding beyond n is part of
        // the linear-convolution embedding. The inverse direction reuses the
        // forward chirps through IDFT(x) = conj(DFT(conj(x))).
        values[self.n..].fill(Complex::new(S::ZERO, S::ZERO));
        values[..self.n]
            .iter_mut()
            .zip(data.iter().copied())
            .zip(self.chirp.iter().copied())
            .for_each(|((destination, value), chirp)| {
                let value = if BOUNDED {
                    crate::validation::bounded(value)
                } else {
                    value
                };
                let directed = if inverse { value.conj() } else { value };
                *destination = directed * chirp;
            });
        self.convolution.transform_into(values, work, false);
        work.iter_mut()
            .zip(self.kernel_spectrum.iter().copied())
            .for_each(|(value, kernel)| *value *= kernel);
        self.convolution.transform_into(work, values, true);
        data.iter_mut()
            .zip(values[..self.n].iter().copied())
            .zip(self.chirp.iter().copied())
            .for_each(|((destination, value), chirp)| {
                let transformed = (value * self.convolution_scale) * chirp;
                *destination = (if inverse {
                    transformed.conj()
                } else {
                    transformed
                }) * scale;
            });
    }

    pub(super) fn heap_bytes(&self) -> Result<usize, SciOxideError> {
        [
            self.convolution.heap_bytes()?,
            crate::memory::vector_bytes(&self.chirp),
            crate::memory::vector_bytes(&self.kernel_spectrum),
        ]
        .into_iter()
        .try_fold(0, |bytes, value| {
            checked_add(bytes, value, scioxide_core::quantity!("fft_plan_bytes"))
        })
    }
}
