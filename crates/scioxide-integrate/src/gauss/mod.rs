//! Gaussian rules for classical weight functions and Gauss–Kronrod pairs.

pub(crate) mod kronrod;
mod plan;
pub use kronrod::{GaussKronrod, GaussKronrodRule};
pub use plan::{
    GaussianDomain, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
};
