//! Independent supplied-zero fixtures; generate_hankel_zeros.py, mpmath 1.3.0, 160/240 digits.
#![expect(clippy::unwrap_used)]
use scioxide_core::RealScalar;
pub fn zeros<S: RealScalar>(order: S) -> [S; 33] {
    const J0: [f64; 33] = [
        f64::from_bits(0x40033d152e971b40),
        f64::from_bits(0x4016148f5b2c2e45),
        f64::from_bits(0x40214eb56cccdeca),
        f64::from_bits(0x40279544008272b6),
        f64::from_bits(0x402ddca13ef271d2),
        f64::from_bits(0x403212313f8a19f6),
        f64::from_bits(0x4035362dd173f792),
        f64::from_bits(0x40385a3b930156dd),
        f64::from_bits(0x403b7e54a5fd5f11),
        f64::from_bits(0x403ea27591cbbed2),
        f64::from_bits(0x4040e34e13a66fe6),
        f64::from_bits(0x404275637a9619ec),
        f64::from_bits(0x4044077a7ed6293a),
        f64::from_bits(0x40459992c65d0d8d),
        f64::from_bits(0x40472bac0f810810),
        f64::from_bits(0x4048bdc6293f0657),
        f64::from_bits(0x404a4fe0ee444c7b),
        f64::from_bits(0x404be1fc41a4c607),
        f64::from_bits(0x404d74180c9e41f6),
        f64::from_bits(0x404f06343d0971d4),
        f64::from_bits(0x40504c28621f11e6),
        f64::from_bits(0x40511536cb22d72b),
        f64::from_bits(0x4051de4554a1c2dd),
        f64::from_bits(0x4052a753fa820480),
        f64::from_bits(0x40537062b9535d16),
        f64::from_bits(0x405439718e2e379c),
        f64::from_bits(0x40550280769a2196),
        f64::from_bits(0x4055cb8f7079c7b4),
        f64::from_bits(0x4056949e79fb1f0c),
        f64::from_bits(0x40575dad918abf9a),
        f64::from_bits(0x405826bcb5c9b623),
        f64::from_bits(0x4058efcbe5854256),
        f64::from_bits(0x4059b8db1fb01801),
    ];
    const J225: [f64; 33] = [
        f64::from_bits(0x4015cdee31b1f094),
        f64::from_bits(0x402183f5a88acb56),
        f64::from_bits(0x4027f218ff54f66c),
        f64::from_bits(0x402e502ad9efa971),
        f64::from_bits(0x40325359f8137de8),
        f64::from_bits(0x40357c8a4ac57c8c),
        f64::from_bits(0x4038a47517f5fa8c),
        f64::from_bits(0x403bcb8a08bbbd60),
        f64::from_bits(0x403ef20ad67b987e),
        f64::from_bits(0x40410c1060570b27),
        f64::from_bits(0x40429ef37eca7b22),
        f64::from_bits(0x404431b81e5da765),
        f64::from_bits(0x4045c464e0573174),
        f64::from_bits(0x404756fe9ac11854),
        f64::from_bits(0x4048e988ea05d0e0),
        f64::from_bits(0x404a7c068e68cc4f),
        f64::from_bits(0x404c0e79a9ecbf65),
        f64::from_bits(0x404da0e3ea70c10f),
        f64::from_bits(0x404f3346a706a31f),
        f64::from_bits(0x405062d17a68864a),
        f64::from_bits(0x40512bfcdb1138f7),
        f64::from_bits(0x4051f525d2d08350),
        f64::from_bits(0x4052be4caf4dcb60),
        f64::from_bits(0x40538771b1ad2e48),
        f64::from_bits(0x4054509510fbefe2),
        f64::from_bits(0x405519b6fc120b33),
        f64::from_bits(0x4055e2d79b0bdc32),
        f64::from_bits(0x4056abf710752389),
        f64::from_bits(0x405775157a37dd43),
        f64::from_bits(0x40583e32f25c96c6),
        f64::from_bits(0x4059074f8fa67feb),
        f64::from_bits(0x4059d06b6612f416),
        f64::from_bits(0x405a998687426353),
    ];
    assert!(order == S::ZERO || order == S::HALF || order == S::try_from_f64_lossy(2.25).unwrap());
    core::array::from_fn(|i| {
        S::try_from_f64_lossy(if order == S::ZERO {
            J0[i]
        } else if order == S::HALF {
            (i + 1) as f64 * core::f64::consts::PI
        } else {
            J225[i]
        })
        .unwrap()
    })
}
