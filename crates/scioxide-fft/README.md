# scioxide-fft

Fast Fourier transforms and related spectral transforms for Rust.
The crate computes discrete Fourier transforms of complex, real and Hermitian data in one or more dimensions, DCT and DST of types I–IV, the chirp z-transform and zoom FFT, nonuniform FFTs of types 1–3, and Hankel transforms on logarithmic (FFTLog) and Bessel-zero grids, for `f32` and `f64`.
Every transform takes an explicit normalization.

## Main entry points

- Fourier transforms: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`.
- N-D transforms over selected axes: `fftn`, `ifftn`, `rfftn`, `irfftn`, `hfftn`, `ihfftn`.
- Cosine and sine transforms: `dct`, `idct`, `dst`, `idst`, `dctn`, `idctn`, `dstn`, `idstn`.
- Contours and nonuniform data: `czt`, `zoom_fft`, `nufft_type1`, `nufft_type2`, `nufft_type3`.
- Hankel transforms: `fht`, `ifht`, `fftlog_low_ringing_offset`, `dht`, `idht`.
- Frequency grids and lengths: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `fftshift_nd`, `ifftshift_nd`, `next_fast_len`, `prev_fast_len`.
- Reusable plans: `FftPlan`, `FftNdPlan`, `RealTransformPlan`, `RealTransformNdPlan`, `CztPlan`, `NufftPlan`, `FftLogPlan`, `DiscreteHankelPlan`, and the plan cache `FftPlanCache`.
- External FFT kernels: `FftProvider`, `FftPlanningContext`.

## Feature flags

- `std` (default): runtime detection of CPU features (AVX2 with FMA, NEON).
  Without it the crate is `no_std`, needs `alloc`, and `new` prepares plans for the portable scalar kernels.

See the [API documentation](https://docs.rs/scioxide-fft) for a quick start, a guide to choosing a method, the transform conventions, and the error and accuracy rules.
Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
