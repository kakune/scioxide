//! Independent defining sums for small real-transform test oracles.

use scioxide_fft::{DctType, DstType, Normalization};

fn inverse_dct_type(ty: DctType) -> DctType {
    match ty {
        DctType::I => DctType::I,
        DctType::II => DctType::III,
        DctType::III => DctType::II,
        DctType::IV => DctType::IV,
    }
}

fn direct_dct_raw(input: &[f64], ty: DctType) -> Vec<f64> {
    let n = input.len();
    match ty {
        DctType::I => (0..n)
            .map(|frequency| {
                input[0]
                    + if frequency.is_multiple_of(2) {
                        input[n - 1]
                    } else {
                        -input[n - 1]
                    }
                    + 2.0
                        * input[1..n - 1]
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, value)| {
                                value
                                    * (std::f64::consts::PI * frequency as f64 * (index + 1) as f64
                                        / (n - 1) as f64)
                                        .cos()
                            })
                            .sum::<f64>()
            })
            .collect(),
        DctType::II => (0..n)
            .map(|frequency| {
                input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            * (std::f64::consts::PI * (2 * index + 1) as f64 * frequency as f64
                                / (2 * n) as f64)
                                .cos()
                    })
                    .sum()
            })
            .collect(),
        DctType::III => (0..n)
            .map(|output_index| {
                input[0]
                    + 2.0
                        * input[1..]
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(index, value)| {
                                value
                                    * (std::f64::consts::PI
                                        * (index + 1) as f64
                                        * (2 * output_index + 1) as f64
                                        / (2 * n) as f64)
                                        .cos()
                            })
                            .sum::<f64>()
            })
            .collect(),
        DctType::IV => (0..n)
            .map(|frequency| {
                input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            * (std::f64::consts::PI
                                * (2 * index + 1) as f64
                                * (2 * frequency + 1) as f64
                                / (4 * n) as f64)
                                .cos()
                    })
                    .sum()
            })
            .collect(),
    }
}

pub(crate) fn direct_dct(
    input: &[f64],
    configured_type: DctType,
    normalization: Normalization,
    inverse: bool,
) -> Vec<f64> {
    let raw_type = if inverse {
        inverse_dct_type(configured_type)
    } else {
        configured_type
    };
    let mut adjusted = input.to_vec();
    if normalization == Normalization::Orthonormal {
        match raw_type {
            DctType::I => {
                adjusted[0] *= std::f64::consts::SQRT_2;
                let last = adjusted.len() - 1;
                adjusted[last] *= std::f64::consts::SQRT_2;
            }
            DctType::III => adjusted[0] *= std::f64::consts::SQRT_2,
            DctType::II | DctType::IV => {}
        }
    }
    let mut output = direct_dct_raw(&adjusted, raw_type);
    if normalization == Normalization::Orthonormal {
        match raw_type {
            DctType::I => {
                let scale = (2.0 * (input.len() - 1) as f64).sqrt().recip();
                output.iter_mut().for_each(|value| *value *= scale);
                output[0] /= std::f64::consts::SQRT_2;
                let last = output.len() - 1;
                output[last] /= std::f64::consts::SQRT_2;
            }
            DctType::II => {
                let scale = (2.0 / input.len() as f64).sqrt();
                output.iter_mut().for_each(|value| *value *= scale);
                output[0] /= std::f64::consts::SQRT_2;
            }
            DctType::III => {
                let scale = (2.0 * input.len() as f64).sqrt().recip();
                output.iter_mut().for_each(|value| *value *= scale);
            }
            DctType::IV => {
                let scale = (2.0 / input.len() as f64).sqrt();
                output.iter_mut().for_each(|value| *value *= scale);
            }
        }
    } else if (!inverse && normalization == Normalization::Forward)
        || (inverse && normalization == Normalization::Inverse)
    {
        let scale = dct_roundtrip_factor(configured_type, input.len()).recip();
        output.iter_mut().for_each(|value| *value *= scale);
    }
    output
}

fn inverse_dst_type(ty: DstType) -> DstType {
    match ty {
        DstType::I => DstType::I,
        DstType::II => DstType::III,
        DstType::III => DstType::II,
        DstType::IV => DstType::IV,
    }
}

fn direct_dst_raw(input: &[f64], ty: DstType) -> Vec<f64> {
    let n = input.len();
    match ty {
        DstType::I => (0..n)
            .map(|frequency| {
                input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            * (std::f64::consts::PI * (index + 1) as f64 * (frequency + 1) as f64
                                / (n + 1) as f64)
                                .sin()
                    })
                    .sum()
            })
            .collect(),
        DstType::II => (0..n)
            .map(|frequency| {
                input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            * (std::f64::consts::PI
                                * (2 * index + 1) as f64
                                * (frequency + 1) as f64
                                / (2 * n) as f64)
                                .sin()
                    })
                    .sum()
            })
            .collect(),
        DstType::III => (0..n)
            .map(|output_index| {
                (if output_index.is_multiple_of(2) {
                    input[n - 1]
                } else {
                    -input[n - 1]
                }) + 2.0
                    * input[..n - 1]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(index, value)| {
                            value
                                * (std::f64::consts::PI
                                    * (index + 1) as f64
                                    * (2 * output_index + 1) as f64
                                    / (2 * n) as f64)
                                    .sin()
                        })
                        .sum::<f64>()
            })
            .collect(),
        DstType::IV => (0..n)
            .map(|frequency| {
                input
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, value)| {
                        value
                            * (std::f64::consts::PI
                                * (2 * index + 1) as f64
                                * (2 * frequency + 1) as f64
                                / (4 * n) as f64)
                                .sin()
                    })
                    .sum()
            })
            .collect(),
    }
}

pub(crate) fn direct_dst(
    input: &[f64],
    configured_type: DstType,
    normalization: Normalization,
    inverse: bool,
) -> Vec<f64> {
    let raw_type = if inverse {
        inverse_dst_type(configured_type)
    } else {
        configured_type
    };
    let mut adjusted = input.to_vec();
    if normalization == Normalization::Orthonormal && raw_type == DstType::III {
        let last = adjusted.len() - 1;
        adjusted[last] *= std::f64::consts::SQRT_2;
    }
    let mut output = direct_dst_raw(&adjusted, raw_type);
    if normalization == Normalization::Orthonormal {
        match raw_type {
            DstType::I => {
                let scale = (2.0 / (input.len() + 1) as f64).sqrt();
                output.iter_mut().for_each(|value| *value *= scale);
            }
            DstType::II => {
                let scale = (2.0 / input.len() as f64).sqrt();
                output.iter_mut().for_each(|value| *value *= scale);
                let last = output.len() - 1;
                output[last] /= std::f64::consts::SQRT_2;
            }
            DstType::III => {
                let scale = (2.0 * input.len() as f64).sqrt().recip();
                output.iter_mut().for_each(|value| *value *= scale);
            }
            DstType::IV => {
                let scale = (2.0 / input.len() as f64).sqrt();
                output.iter_mut().for_each(|value| *value *= scale);
            }
        }
    } else if (!inverse && normalization == Normalization::Forward)
        || (inverse && normalization == Normalization::Inverse)
    {
        let scale = dst_roundtrip_factor(configured_type, input.len()).recip();
        output.iter_mut().for_each(|value| *value *= scale);
    }
    output
}

pub(crate) fn dct_roundtrip_factor(ty: DctType, n: usize) -> f64 {
    match ty {
        DctType::I => 2.0 * (n - 1) as f64,
        DctType::II | DctType::III => n as f64,
        DctType::IV => n as f64 / 2.0,
    }
}

pub(crate) fn dst_roundtrip_factor(ty: DstType, n: usize) -> f64 {
    match ty {
        DstType::I => (n + 1) as f64 / 2.0,
        DstType::II | DstType::III => n as f64,
        DstType::IV => n as f64 / 2.0,
    }
}
