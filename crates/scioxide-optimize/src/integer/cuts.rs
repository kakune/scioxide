//! Restricted cut families with independently checked derivation conditions.
use super::*;
use crate::LinearConstraintSense as Sense;

use model::{Model, Row};

#[derive(Debug)]
pub(super) enum Proof {
    IntegerRow(usize),
    Cover { row: usize, variables: Vec<usize> },
}

// A downward bound on a sum of positive represented coefficients. Rounding
// uncertainty can reject a cover, never authorize an invalid cut.
fn cover_sum<S: RealScalar + NumericScalar<Real = S>>(
    source: &Row<S>,
    variables: &[usize],
    sign: S,
) -> Option<S> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut sum = z;
    for &j in variables {
        let a = source.entries.iter().find(|(k, _)| *k == j)?.1 * sign;
        if a <= z {
            return None;
        }
        sum += a;
        if !sum.is_finite() {
            return None;
        }
        sum = sum.next_after(z);
    }
    Some(sum)
}

pub(super) fn verify<S: RealScalar + NumericScalar<Real = S>>(
    m: &Model<S>,
    cut: &Row<S>,
    proof: &Proof,
) -> Result<bool, SciOxideError> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let index = match proof {
        Proof::IntegerRow(i) => *i,
        Proof::Cover { row, .. } => *row,
    };
    let Some(source) = m.rows.get(index) else {
        return Ok(false);
    };
    let sign = if source.sense == Sense::GreaterEqual {
        -one
    } else {
        one
    };
    if cut.sense != Sense::LessEqual {
        return Ok(false);
    }
    match proof {
        Proof::IntegerRow(_) => Ok(source
            .entries
            .iter()
            .all(|&(j, a)| m.integer.contains(&j) && a.round() == a)
            && cut.entries.len() == source.entries.len()
            && cut
                .entries
                .iter()
                .zip(&source.entries)
                .all(|(&(j, a), &(k, b))| j == k && a == sign * b)
            && cut.rhs == (source.rhs * sign).floor()),
        Proof::Cover { variables, .. } => {
            if source.sense == Sense::Equal
                || source.entries.iter().any(|&(j, a)| {
                    a * sign < z
                        || !m.integer.contains(&j)
                        || !matches!(m.bounds[j],(Some(l),Some(u)) if l>=z&&u<=one)
                })
                || variables.is_empty()
                || variables.windows(2).any(|v| v[0] >= v[1])
                || cut.entries.len() != variables.len()
                || !cut
                    .entries
                    .iter()
                    .zip(variables)
                    .all(|(&(j, a), &k)| j == k && a == one)
            {
                return Ok(false);
            }
            Ok(
                cover_sum(source, variables, sign).is_some_and(|v| v > source.rhs * sign)
                    && cut.rhs == S::try_from_usize_exact(variables.len() - 1)?,
            )
        }
    }
}

pub(super) struct CutWorkspace<S> {
    order: Vec<(usize, S)>,
    variables: Vec<usize>,
    entries: Vec<(usize, S)>,
    pub output: Vec<Row<S>>,
}
impl<S> Default for CutWorkspace<S> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            variables: Vec::new(),
            entries: Vec::new(),
            output: Vec::new(),
        }
    }
}
impl<S> CutWorkspace<S> {
    pub(super) fn bytes(&self) -> usize {
        (self.order.capacity() + self.entries.capacity()) * core::mem::size_of::<(usize, S)>()
            + self.variables.capacity() * core::mem::size_of::<usize>()
            + self.output.capacity() * core::mem::size_of::<Row<S>>()
    }
}
#[cfg(test)]
pub(super) fn separate<'a, S: 'a + RealScalar + NumericScalar<Real = S>>(
    m: &Model<S>,
    x: &[S],
    existing: impl Iterator<Item = &'a Row<S>> + Clone,
    remaining: usize,
    nonzeros: usize,
    t: scioxide_core::FeasibilityTolerance<S>,
) -> Result<Vec<Row<S>>, SciOxideError> {
    separate_with_workspace(
        m,
        x,
        existing,
        remaining,
        nonzeros,
        t,
        &mut CutWorkspace::default(),
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn separate_with_workspace<'a, S: 'a + RealScalar + NumericScalar<Real = S>>(
    m: &Model<S>,
    x: &[S],
    existing: impl Iterator<Item = &'a Row<S>> + Clone,
    remaining: usize,
    nonzeros: usize,
    t: scioxide_core::FeasibilityTolerance<S>,
    work: &mut CutWorkspace<S>,
) -> Result<Vec<Row<S>>, SciOxideError> {
    let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let maximum = m.rows.iter().map(|r| r.entries.len()).max().unwrap_or(0);
    for values in [&mut work.order, &mut work.entries] {
        scioxide_core::backend::try_reserve(
            values,
            maximum.saturating_sub(values.len()),
            scioxide_core::quantity!("mip_cut_scratch"),
        )?;
    }
    let additional = maximum.saturating_sub(work.variables.len());
    scioxide_core::backend::try_reserve(
        &mut work.variables,
        additional,
        scioxide_core::quantity!("mip_cover_scratch"),
    )?;
    let mut out = core::mem::take(&mut work.output);
    out.clear();
    scioxide_core::backend::try_reserve(
        &mut out,
        remaining.min(m.rows.len()),
        scioxide_core::quantity!("mip_cuts"),
    )?;
    let mut nnz = 0;
    for (i, source) in m.rows.iter().enumerate() {
        if out.len() >= remaining {
            break;
        }
        let sign = if source.sense == Sense::GreaterEqual {
            -one
        } else {
            one
        };
        let mut proposal = None;
        if source
            .entries
            .iter()
            .all(|&(j, a)| m.integer.contains(&j) && a.round() == a)
            && (source.rhs * sign).floor() < source.rhs * sign
        {
            let mut entries = core::mem::take(&mut work.entries);
            entries.clear();
            entries.extend(source.entries.iter().map(|&(j, a)| (j, a * sign)));
            proposal = Some((
                Row {
                    entries,
                    rhs: (source.rhs * sign).floor(),
                    sense: Sense::LessEqual,
                    provenance: Some((MipCutKind::IntegerRow, i)),
                },
                Proof::IntegerRow(i),
            ));
        } else if source.sense != Sense::Equal
            && source.entries.iter().all(|&(j, a)| {
                a * sign > z
                    && m.integer.contains(&j)
                    && matches!(m.bounds[j],(Some(l),Some(u)) if l>=z&&u<=one)
            })
        {
            let order = &mut work.order;
            order.clear();
            order.extend_from_slice(&source.entries);
            order.sort_unstable_by(|(a, _), (b, _)| {
                x[*b]
                    .partial_cmp(&x[*a])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            let variables = &mut work.variables;
            variables.clear();
            for &(j, _) in order.iter() {
                variables.push(j);
                if cover_sum(source, variables, sign).is_some_and(|v| v > source.rhs * sign) {
                    break;
                }
            }
            variables.sort_unstable();
            if cover_sum(source, variables, sign).is_some_and(|v| v > source.rhs * sign) {
                // Deterministic deletion makes the greedy cover minimal.
                let mut k = variables.len();
                while k > 0 {
                    k -= 1;
                    let j = variables.remove(k);
                    if !cover_sum(source, variables, sign).is_some_and(|v| v > source.rhs * sign) {
                        variables.insert(k, j);
                    }
                }
                if variables.is_empty() {
                    continue;
                }
                let mut entries = core::mem::take(&mut work.entries);
                entries.clear();
                entries.extend(variables.iter().map(|&j| (j, one)));
                let rhs = S::try_from_usize_exact(variables.len() - 1)?;
                proposal = Some((
                    Row {
                        entries,
                        rhs,
                        sense: Sense::LessEqual,
                        provenance: Some((MipCutKind::BinaryCover, i)),
                    },
                    Proof::Cover {
                        row: i,
                        variables: core::mem::take(&mut work.variables),
                    },
                ));
            }
        }
        let Some((cut, proof)) = proposal else {
            continue;
        };
        let lhs = cut.value(x)?;
        let admissible = cut.entries.len() <= nonzeros.saturating_sub(nnz)
            && !existing.clone().any(|r| r == &cut)
            && !out.contains(&cut)
            && lhs - cut.rhs
                > t.threshold(NumericScalar::abs(lhs).max(NumericScalar::abs(cut.rhs)))?;
        if admissible {
            if !verify(m, &cut, &proof)? {
                return Err(invalid(
                    scioxide_core::argument!("state"),
                    "cut derivation failed independent checks",
                ));
            }
            nnz += cut.entries.len();
            out.push(Row {
                entries: copy(&cut.entries, scioxide_core::quantity!("accepted_cut"))?,
                rhs: cut.rhs,
                sense: cut.sense,
                provenance: cut.provenance,
            });
        }
        work.entries = cut.entries;
        if let Proof::Cover { variables, .. } = proof {
            work.variables = variables;
        }
    }
    Ok(out)
}
