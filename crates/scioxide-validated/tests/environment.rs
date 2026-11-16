//! Control changes are confined to this test thread and restored before assertions.
#![cfg(test)]

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod aarch64 {
    use scioxide_core::{InvalidInputReason, SciOxideError};
    use scioxide_validated::{
        DecoratedInterval as D, RootClaim, ScalarRootCertificate, ScalarRootVerifier,
    };

    fn control() -> u64 {
        let value: u64;
        // SAFETY: FPCR is readable at EL0; this changes no state.
        unsafe {
            core::arch::asm!("mrs {p}, fpcr", p=out(reg) value, options(nostack,preserves_flags));
        }
        value
    }
    fn status() -> u64 {
        let value: u64;
        // SAFETY: FPSR is readable at EL0; this changes no state.
        unsafe {
            core::arch::asm!("mrs {p}, fpsr", p=out(reg) value, options(nostack,preserves_flags));
        }
        value
    }
    fn set_control(value: u64) {
        // SAFETY: callers change only documented rounding/underflow bits on
        // this test thread and restore the original value before assertions.
        unsafe {
            core::arch::asm!("msr fpcr, {p}", p=in(reg) value, options(nostack,preserves_flags));
        }
    }
    fn rejected<T>(result: &Result<T, SciOxideError>) -> bool {
        matches!(result,Err(error) if matches!(error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::UnsupportedConfiguration, .. }))
    }
    #[test]
    fn rejects_changed_controls_before_primitive_and_composite_arithmetic() {
        let x = D::new(std::hint::black_box(f64::MAX), f64::MAX).unwrap();
        let two = D::point(2.0).unwrap();
        let original = control();
        for mask in [1 << 24, 1 << 19, 1 << 22, 2 << 22, 3 << 22] {
            let changed = original | mask;
            let before = status();
            set_control(changed);
            let results = [
                x.add(&x),
                x.sub(&two),
                x.mul(&two),
                x.div(&two),
                x.sqrt(),
                x.square(),
                x.reciprocal(),
                x.neg(),
                x.abs(),
                x.hypot(&two).map(|r| r.hull()),
                x.cosh().map(|r| r.hull()),
                x.sin().map(|r| r.hull()),
                x.cos().map(|r| r.hull()),
                x.powi(2).map(|r| r.hull()),
                x.powf(&two).map(|r| r.hull()),
                x.atan2(&two).map(|r| r.hull()),
                x.log(&two).map(|r| r.hull()),
            ];
            let observed = control();
            let after = status();
            set_control(original);
            assert!(results.iter().all(rejected));
            assert_eq!(observed, changed);
            assert_eq!(after, before, "rejected calls must not evaluate endpoints");
            assert!(x.add(&two).is_ok());
        }
    }
    #[test]
    fn readmits_environment_after_user_callback() {
        let original = control();
        let x = D::new(-1.0, 1.0).unwrap();
        let one = D::point(1.0).unwrap();
        let value = |_: D<f64>| {
            set_control(original | (1 << 24));
            Ok(x)
        };
        let derivative = |_: D<f64>| Ok(one);
        let result = ScalarRootVerifier::new(
            &value,
            &derivative,
            scioxide_validated::ValidatedOptions::new(),
        )
        .verify(ScalarRootCertificate {
            region: x.set(),
            center: 0.0,
            claim: RootClaim::Exists,
        });
        let observed = control();
        set_control(original);
        assert!(rejected(&result));
        assert_eq!(observed, original | (1 << 24));
    }
}
#[cfg(all(
    any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ),
    not(miri)
))]
#[test]
fn rejects_changed_controls_without_mutating_them() {
    use scioxide_core::{InvalidInputReason, SciOxideError};
    use scioxide_validated::{
        AffineContext, AffineForm, AffineOptions, AffineProjection, AffineState, DecoratedInterval,
    };
    fn rejected<T>(result: Result<T, SciOxideError>) -> bool {
        match result {
            Err(error) => matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ),
            Ok(_) => false,
        }
    }
    let interval = DecoratedInterval::new(1.0_f64, 2.0).unwrap();
    let context = AffineContext::new(0, AffineOptions::new()).unwrap();
    let affine = AffineForm::constant(&context, 1.0_f64).unwrap();
    let mut projected = AffineForm::<f64>::zero(&context).unwrap();
    let before = projected.report();
    let singular = scioxide_linalg::Matrix::from_vec(1, 1, vec![0.0_f64]).unwrap();
    let mut original = 0_u32;
    let mut original_x87 = 0_u16;
    // SAFETY: correctly sized live stack destinations; read-only control capture.
    unsafe {
        core::arch::asm!("stmxcsr [{p}]",p=in(reg) &mut original,options(nostack,preserves_flags));
        core::arch::asm!("fnstcw [{p}]",p=in(reg) &mut original_x87,options(nostack,preserves_flags));
    }
    for mask in [1 << 15, 1 << 6, 1 << 13, 2 << 13, 3 << 13] {
        let changed = original | mask;
        let mut observed = 0_u32;
        // SAFETY: only documented MXCSR bits are changed on this test thread.
        unsafe {
            core::arch::asm!("ldmxcsr [{p}]",p=in(reg) &changed,options(nostack,preserves_flags));
        }
        let linear = scioxide_validated::solve(&singular, &[1.0]);
        let result = interval.add(&interval);
        let elementary = interval.sin();
        let projection = affine
            .view()
            .project_into(AffineProjection::Real, projected.view_mut());
        let initialized = AffineState::<f64>::zero(&context);
        // SAFETY: restore the original control before any assertion can unwind.
        unsafe {
            core::arch::asm!("stmxcsr [{p}]",p=in(reg) &mut observed,options(nostack,preserves_flags));
            core::arch::asm!("ldmxcsr [{p}]",p=in(reg) &original,options(nostack,preserves_flags));
        }
        assert!(rejected(result));
        assert!(rejected(elementary));
        assert!(rejected(projection));
        assert!(rejected(linear));
        assert_eq!(projected.report(), before);
        assert!(initialized.is_ok());
        assert_eq!(observed, changed);
    }
    for mode in [1_u16, 2, 3] {
        let changed = original_x87 | (mode << 10);
        // SAFETY: change only x87 rounding bits, restoring before assertions.
        unsafe {
            core::arch::asm!("fldcw [{p}]",p=in(reg) &changed,options(nostack,preserves_flags));
        }
        let result = interval.neg();
        let elementary = interval.exp();
        // SAFETY: restore the saved x87 control word.
        unsafe {
            core::arch::asm!("fldcw [{p}]",p=in(reg) &original_x87,options(nostack,preserves_flags));
        }
        assert!(rejected(result));
        assert!(rejected(elementary));
    }
    assert!(interval.add(&interval).is_ok());
}

#[cfg(all(
    any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ),
    not(miri)
))]
#[test]
fn rejects_enabled_traps_before_overflow_arithmetic() {
    use scioxide_core::InvalidInputReason;
    use scioxide_validated::DecoratedInterval;
    let maximum = std::hint::black_box(f64::MAX);
    let a = DecoratedInterval::point(maximum).unwrap();
    let b = DecoratedInterval::point(2.0).unwrap();
    let mut original = 0_u32;
    // SAFETY: capture the current thread's MXCSR into a live stack object.
    unsafe {
        core::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut original, options(nostack, preserves_flags));
    }
    for exception in 0..6 {
        // Clear pending flags before enabling one trap. An erroneously evaluated
        // endpoint overflows; even masked premature arithmetic would set flags.
        let changed = (original & !0x3f) & !(1 << (exception + 7));
        let mut observed = 0_u32;
        // SAFETY: documented MXCSR bits only, confined to this test thread.
        unsafe {
            core::arch::asm!("ldmxcsr [{p}]", p = in(reg) &changed, options(nostack, preserves_flags));
        }
        let result = a.mul(&b);
        let elementary = a.exp();
        // SAFETY: restore controls before assertions, formatting, or unwinding.
        unsafe {
            core::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut observed, options(nostack, preserves_flags));
            core::arch::asm!("ldmxcsr [{p}]", p = in(reg) &original, options(nostack, preserves_flags));
        }
        assert!(match result {
            Err(error) => matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ),
            Ok(_) => false,
        });
        assert!(match elementary {
            Err(error) => matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ),
            Ok(_) => false,
        });
        assert_eq!(observed, changed);
    }
}
