//! Read-only, per-call check of the native IEEE rounding environment.

use scioxide_core::{InvalidInputReason, SciOxideError};

/// Check the current thread's rounding environment and report a rejection.
pub(crate) fn check() -> Result<(), SciOxideError> {
    if supported() {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("rounding_environment"),
            InvalidInputReason::UnsupportedConfiguration,
            "validated arithmetic requires a qualified x86/AArch64 nearest-even environment with gradual underflow",
        ))
    }
}

#[cfg(all(
    any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse2")
    ),
    not(miri)
))]
#[allow(
    unsafe_code,
    reason = "read-only admission of the current thread IEEE control registers"
)]
fn supported() -> bool {
    let mut mxcsr = core::mem::MaybeUninit::<u32>::uninit();
    let mut control = core::mem::MaybeUninit::<u16>::uninit();
    // SAFETY: Each instruction initializes its entire correctly sized, live
    // stack destination before assume_init reads it. Both only capture the
    // current thread's control state and do not change it.
    // No pure/nomem annotation permits hoisting or removal across arithmetic.
    let (mxcsr, control) = unsafe {
        core::arch::asm!("stmxcsr [{p}]", p = in(reg) mxcsr.as_mut_ptr(), options(nostack, preserves_flags));
        core::arch::asm!("fnstcw [{p}]", p = in(reg) control.as_mut_ptr(), options(nostack, preserves_flags));
        (mxcsr.assume_init(), control.assume_init())
    };
    // Require nearest-even rounding, gradual underflow and masked SSE traps;
    // require nearest-even rounding and masked traps in x87 as well.
    mxcsr & 0xffc0 == 0x1f80 && control & 0x0c3f == 0x003f
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[expect(
    unsafe_code,
    reason = "read-only admission of the current thread IEEE control registers"
)]
fn supported() -> bool {
    let control: u64;
    // SAFETY: FPCR is readable at EL0 and this instruction changes no state.
    unsafe {
        core::arch::asm!("mrs {p}, fpcr", p = out(reg) control, options(nostack, preserves_flags));
    }
    control & ((1 << 24) | (1 << 19) | (3 << 22) | (0x1f << 8) | (1 << 15) | 3) == 0
}

#[cfg(any(
    miri,
    not(any(
        all(target_arch = "x86", target_feature = "sse2"),
        target_arch = "x86_64",
        target_arch = "aarch64"
    ))
))]
fn supported() -> bool {
    false
}
