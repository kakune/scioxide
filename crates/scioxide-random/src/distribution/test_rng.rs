//! Scripted uniform words for count-sampler unit tests.
use crate::BitGenerator;
use alloc::vec::Vec;
use scioxide_core::SciOxideError;

/// Uniform words decoding to the scripted values, then to `fallback`.
pub(super) struct ScriptedRng {
    values: alloc::vec::IntoIter<f64>,
    fallback: f64,
    /// Words drawn so far.
    pub(super) draws: usize,
}

impl ScriptedRng {
    pub(super) fn new(values: Vec<f64>, fallback: f64) -> Self {
        Self {
            values: values.into_iter(),
            fallback,
            draws: 0,
        }
    }

    pub(super) fn constant(value: f64) -> Self {
        Self::new(Vec::new(), value)
    }
}

impl BitGenerator for ScriptedRng {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(0)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.draws += 1;
        Ok((self.values.next().unwrap_or(self.fallback) * 18_446_744_073_709_551_616.0) as u64)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
        dest.fill(0);
        Ok(())
    }
}
