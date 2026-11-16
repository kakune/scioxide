# scioxide-signal

Digital signal processing for Rust.
The crate convolves and correlates signals, designs FIR and IIR filters and applies them to records or streams, converts sample rates, and estimates spectra with periodograms, Welch averaging and the Lomb–Scargle periodogram.
It computes short-time Fourier and discrete wavelet transforms, analytic signals, smoothing and peak detection, and simulates continuous- and discrete-time state-space systems, for real and complex `f32` and `f64` samples.

## Main entry points

- Convolution: `convolve`, `correlate`, `convolve_circular`, `convolve2d`, `ConvolutionNdPlan`, `SeparableFir2dPlan`, `OverlapPlan`, `OverlapStream`.
- IIR filters: `design_iir`, `select_iir_order`, `iir_prototype`, `design_notch_peak`, `design_comb`, `design_gammatone`, `SecondOrderSections::filter`, `SecondOrderSections::forward_backward`, `IirFilter`, `TransferFunction`, `PoleZeroGain`, `SecondOrderSections`.
- FIR filters: `design_fir_window`, `design_fir_least_squares`, `design_fir_frequency_sampling`, `design_fir_remez`, `minimum_phase`.
- Resampling: `resample`, `resample_periodic`, `PolyphasePlan`, `PolyphaseStream`, `FourierResamplePlan`.
- Windows: `window`, `Window`, `window_metrics`, `DpssTapers`.
- Spectra: `power_spectrum`, `cross_spectrum`, `welch`, `lomb_scargle`, `SpectralEstimatePlan`, `LombScargle`.
- Time–frequency: `StftPlan`, `StftAnalysisStream`, `StftSynthesisStream`, `WaveletPlan`, `Wavelet2dPlan`, `hilbert`, `envelope`, `RieszPlan`.
- Smoothing and features: `savitzky_golay`, `median_filter`, `find_peaks`, `detrend`, `vector_strength`, `waveform`, `LocalFilterPlan`, `SavitzkyGolayPlan`, `WhittakerHendersonPlan`, `SplineFilter`, `RidgePlan`.
- State-space systems: `StateSpace`, `FrequencyResponsePlan`, `TimeResponsePlan`.

## Feature flags

- `std` (default): threads and clocks from `scioxide-core`.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-signal) for a quick start, a guide to choosing a method, and the conventions, error and accuracy rules.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
