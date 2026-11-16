//! The one per-draw provider wrapper: every request of a draw, including nested
//! draws, kernel retries and failed requests, is charged to one uniform-word
//! allowance. A request may consume several native transitions; uniform-word
//! normalization keeps its own finite rejection bound.
use crate::{BitGenerator, Rng};
use core::marker::PhantomData;
use scioxide_core::{RealScalar, SciOxideError};

/// Floats decoded from uniform words on the common 53-bit grid.
pub(crate) struct Decoded;
/// Floats forwarded from the caller's checked floating provider.
pub(crate) struct Forwarded;
/// The caller's validated `S` floats widened to `f64`.
pub(crate) struct Widened<S>(PhantomData<S>);

/// One draw's uniform-word allowance, shared by every nested draw and kernel
/// retry. `F` selects how floats are formed: [`Decoded`] from uniform words,
/// [`Forwarded`] from the caller's provider, or [`Widened`] from it in `f64`.
pub(crate) struct Words<'a, R: ?Sized, F = Decoded> {
    rng: &'a mut R,
    left: usize,
    floats: PhantomData<F>,
}
impl<'a, R: ?Sized> Words<'a, R, Decoded> {
    /// Decode floats from charged uniform words.
    pub(crate) fn new(rng: &'a mut R, left: usize) -> Self {
        Self {
            rng,
            left,
            floats: PhantomData,
        }
    }
}
impl<'a, R: ?Sized> Words<'a, R, Forwarded> {
    /// Charge, then forward the caller's own floating provider.
    pub(crate) fn forwarding(rng: &'a mut R, left: usize) -> Self {
        Self {
            rng,
            left,
            floats: PhantomData,
        }
    }
}
impl<'a, S, R: ?Sized> Words<'a, R, Widened<S>> {
    /// Charge, validate the caller's `S` float once, and widen it to `f64`.
    pub(crate) fn widening(rng: &'a mut R, left: usize) -> Self {
        Self {
            rng,
            left,
            floats: PhantomData,
        }
    }
}
impl<R: ?Sized, F> Words<'_, R, F> {
    #[inline]
    fn charge(&mut self) -> Result<(), SciOxideError> {
        crate::law::charge(&mut self.left)
    }
}
impl<R: BitGenerator + ?Sized, F> BitGenerator for Words<'_, R, F> {
    #[inline]
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.charge()?;
        self.rng.next_u64()
    }
    #[inline]
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.charge()?;
        self.rng.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.charge()?;
        self.rng.next_uniform_u64()
    }
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.charge()?;
        self.rng.next_uniform_u32()
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(out, 8, || self.next_u64())
    }
}
impl<S: RealScalar, R: BitGenerator + ?Sized> Rng<S> for Words<'_, R, Decoded> {
    #[inline]
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        Ok(crate::decode_u64(self.next_uniform_u64()?))
    }
    // Decoded uniform words lie in [0, 1) by construction.
    #[inline]
    fn checked_float(
        &mut self,
        _: scioxide_core::Argument,
        _: crate::traits::private::Token,
    ) -> Result<S, SciOxideError> {
        self.next_float()
    }
}
impl<S: RealScalar, R: Rng<S> + ?Sized> Rng<S> for Words<'_, R, Forwarded> {
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        self.charge()?;
        self.rng.next_float()
    }
    fn checked_float(
        &mut self,
        argument: scioxide_core::Argument,
        token: crate::traits::private::Token,
    ) -> Result<S, SciOxideError> {
        self.charge()?;
        self.rng.checked_float(argument, token)
    }
}
impl<S: RealScalar, R: Rng<S> + ?Sized> Rng<f64> for Words<'_, R, Widened<S>> {
    #[inline]
    fn next_float(&mut self) -> Result<f64, SciOxideError> {
        self.charge()?;
        Ok(
            crate::util::validated_unit_float(self.rng, scioxide_core::argument!("rng"))?
                .into_f64(),
        )
    }
    // `next_float` already validated the caller's provider value once.
    #[inline]
    fn checked_float(
        &mut self,
        _: scioxide_core::Argument,
        _: crate::traits::private::Token,
    ) -> Result<f64, SciOxideError> {
        self.next_float()
    }
}

/// Decode one uniform word as `f64` on the common 53-bit grid.
#[inline]
pub(crate) fn uniform<R: BitGenerator + ?Sized>(rng: &mut R) -> Result<f64, SciOxideError> {
    Ok(crate::decode_u64(rng.next_uniform_u64()?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Ranlux, SeedableRng};

    // Native uniform normalization is preserved, every request is charged
    // once, and exhaustion consumes nothing further.
    #[test]
    fn requests_preserve_normalization_and_budget() {
        let mut rng = Ranlux::from_seed(42);
        let mut expected = rng.clone();
        let mut words = Words::new(&mut rng, 2);
        assert_eq!(
            words.next_uniform_u32().unwrap(),
            expected.next_uniform_u32().unwrap()
        );
        let decoded: f64 = words.next_float().unwrap();
        assert_eq!(decoded, uniform(&mut expected).unwrap());
        assert_eq!(words.left, 0);
        assert!(words.next_uniform_u64().is_err());
        assert_eq!(rng.position(), expected.position());

        let mut rng = Ranlux::from_seed(42);
        let mut expected = rng.clone();
        let mut widened = Words::<_, Widened<f32>>::widening(&mut rng, 2);
        let wide = widened.next_float().unwrap();
        let narrow: f32 = expected.next_float().unwrap();
        assert_eq!(wide, f64::from(narrow));
        let mut forwarded = Words::forwarding(&mut rng, 1);
        let forward: f64 = forwarded.next_float().unwrap();
        assert_eq!(forward, expected.next_float().unwrap());
        assert!(Rng::<f64>::next_float(&mut forwarded).is_err());
    }
}
