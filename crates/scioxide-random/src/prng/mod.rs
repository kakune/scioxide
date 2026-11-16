//! Pseudo-random number generator engines.
//!
//! Each engine implements [`BitGenerator`](crate::traits::BitGenerator) and
//! [`SeedableRng`](crate::traits::SeedableRng) at minimum, with additional
//! traits for skip-ahead, jumping, splitting, and state serialization.

mod lcg;
mod mrg32k3a;
mod pcg;
mod splitmix;
mod words;
mod xoshiro;

pub use lcg::Lcg64;
pub use mrg32k3a::Mrg32k3a;
pub use pcg::Pcg64;
pub use splitmix::SplitMix64;
pub use xoshiro::{Xoshiro128StarStar, Xoshiro256StarStar};

mod classic;
mod counter;
mod feedback;
mod mrg;
mod mt19937;
mod ranlux;
pub use counter::{Philox4x64, Threefry4x64, philox4x64_block, threefry4x64_block};
pub use feedback::{Gfsr4, Taus2, Taus113};
pub use mrg::{Cmrg, Mrg};
pub use mt19937::Mt19937;
pub use ranlux::{Ranlux, RanluxLuxury, Ranlxd, RanlxdLuxury, Ranlxs, RanlxsLuxury};
