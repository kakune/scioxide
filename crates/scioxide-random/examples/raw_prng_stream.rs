//! Emit a caller-selected seeded stream; no ambient generator selection.
//!
//! `cargo run --release -p scioxide-random --example raw_prng_stream -- ENGINE SEED [BYTES]`
//!
//! New families emit packed native bits: 24/48 for RANLUX variants, 31 for
//! MRG/CMRG and 32 for feedback/MT. This removes representation padding, not
//! intrinsic generator bias. Existing engine names retain their old u64 stream.
//! Optional BYTES bounds output; omission streams until a broken pipe.

use scioxide_random::*;
use std::env;
use std::io::{self, BufWriter, Write};

struct Stream {
    generator: Box<dyn BitGenerator>,
    width: u32,
    pending: u128,
    available: u32,
}
impl Stream {
    fn from_name(name: &str, seed: u64) -> io::Result<Self> {
        let (generator, width): (Box<dyn BitGenerator>, u32) = match name {
            "splitmix64" => (Box::new(SplitMix64::from_seed(seed)), 64),
            "xoshiro256ss" => (Box::new(Xoshiro256StarStar::from_seed(seed)), 64),
            "xoshiro128ss" => (Box::new(Xoshiro128StarStar::from_seed(seed)), 64),
            "pcg64" => (Box::new(Pcg64::from_seed(seed)), 64),
            "lcg64" => (Box::new(Lcg64::from_seed(seed)), 64),
            "mrg32k3a" => (Box::new(Mrg32k3a::from_seed(seed)), 64),
            "mt19937" => (Box::new(Mt19937::from_seed(seed)), 32),
            "taus2" => (Box::new(Taus2::from_seed(seed)), 32),
            "taus113" => (Box::new(Taus113::from_seed(seed)), 32),
            "gfsr4" => (Box::new(Gfsr4::from_seed(seed)), 32),
            "mrg" => (Box::new(Mrg::from_seed(seed)), 31),
            "cmrg" => (Box::new(Cmrg::from_seed(seed)), 31),
            "ranlux0" => (
                Box::new(Ranlux::with_luxury(seed, RanluxLuxury::Level0)),
                24,
            ),
            "ranlux1" => (
                Box::new(Ranlux::with_luxury(seed, RanluxLuxury::Level1)),
                24,
            ),
            "ranlux2" => (
                Box::new(Ranlux::with_luxury(seed, RanluxLuxury::Level2)),
                24,
            ),
            "ranlux3" => (Box::new(Ranlux::from_seed(seed)), 24),
            "ranlux4" => (
                Box::new(Ranlux::with_luxury(seed, RanluxLuxury::Level4)),
                24,
            ),
            "ranlxs0" => (
                Box::new(Ranlxs::with_luxury(seed, RanlxsLuxury::Level0)),
                24,
            ),
            "ranlxs1" => (Box::new(Ranlxs::from_seed(seed)), 24),
            "ranlxs2" => (
                Box::new(Ranlxs::with_luxury(seed, RanlxsLuxury::Level2)),
                24,
            ),
            "ranlxd1" => (Box::new(Ranlxd::from_seed(seed)), 48),
            "ranlxd2" => (
                Box::new(Ranlxd::with_luxury(seed, RanlxdLuxury::Level2)),
                48,
            ),
            _ => {
                return Err(invalid(format!(
                    "unknown engine {name}; see the raw_prng_stream source for names"
                )));
            }
        };
        Ok(Self {
            generator,
            width,
            pending: 0,
            available: 0,
        })
    }
    fn fill(&mut self, output: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        for byte in output {
            if self.available < 8 {
                let word = if self.width > 32 {
                    self.generator.next_u64()?
                } else {
                    u64::from(self.generator.next_u32()?)
                };
                self.pending |= u128::from(word) << self.available;
                self.available += self.width;
            }
            *byte = self.pending.to_le_bytes()[0];
            self.pending >>= 8;
            self.available -= 8;
        }
        Ok(())
    }
}
fn invalid(detail: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, detail.into())
}
fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    let name = args
        .next()
        .ok_or_else(|| invalid("usage: raw_prng_stream ENGINE SEED [BYTES]"))?;
    let seed = args
        .next()
        .ok_or_else(|| invalid("missing decimal seed"))?
        .parse()
        .map_err(|_| invalid("invalid decimal u64 seed"))?;
    let mut remaining = args
        .next()
        .map(|s| s.parse::<u64>().map_err(|_| invalid("invalid byte count")))
        .transpose()?;
    if args.next().is_some() {
        return Err(invalid("too many arguments"));
    }
    let mut stream = Stream::from_name(&name, seed)?;
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    let mut buffer = [0; 65536];
    loop {
        let size = remaining.map_or(buffer.len(), |left| {
            usize::try_from(left)
                .unwrap_or(usize::MAX)
                .min(buffer.len())
        });
        if size == 0 {
            return output.flush();
        }
        stream.fill(&mut buffer[..size]).map_err(io::Error::other)?;
        if let Err(error) = output.write_all(&buffer[..size]) {
            return if error.kind() == io::ErrorKind::BrokenPipe {
                Ok(())
            } else {
                Err(error)
            };
        }
        remaining = remaining.map(|left| left - size as u64);
    }
}
