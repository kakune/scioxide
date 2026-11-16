//! Concordance counts for Kendall's tau and the weighted Kendall coefficient.
use super::KendallWeightCombination;
use crate::arithmetic::{Arithmetic, Compensated};
use crate::ranking::{TieSums, finite_cmp, tie_runs};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{RealScalar, SciOxideError};

#[derive(Clone, Copy, Default)]
struct Observation {
    rank: [usize; 2],
    untied: [usize; 2],
    // Each observation's number of concordant minus discordant partners.
    balance: isize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Ties {
    pub sums: TieSums,
    pub distinct: usize,
}

pub(super) struct Counts {
    orders: [Vec<usize>; 2],
    observations: Vec<Observation>,
    pub ties: [Ties; 2],
    pub score: f64,
}

fn prefix(tree: &[usize], mut end: usize) -> usize {
    let mut sum = 0;
    while end > 0 {
        sum += tree[end - 1];
        end &= end - 1;
    }
    sum
}

impl Counts {
    pub(super) fn new<S: RealScalar>(x: &[S], y: &[S]) -> Result<Self, SciOxideError> {
        let mut orders = [Vec::new(), Vec::new()];
        let mut observations = try_filled(
            x.len(),
            Observation::default(),
            scioxide_core::quantity!("kendall_observations"),
        )?;
        let mut ties = [Ties::default(); 2];
        for (axis, (a, b)) in [(x, y), (y, x)].into_iter().enumerate() {
            let order = &mut orders[axis];
            *order = try_with_capacity(x.len(), scioxide_core::quantity!("kendall_order"))?;
            order.extend(0..x.len());
            order.sort_unstable_by(|&i, &j| {
                finite_cmp(&a[i], &a[j])
                    .then_with(|| finite_cmp(&b[i], &b[j]))
                    .then(i.cmp(&j))
            });
            for (run, group) in tie_runs(order, |&i, &j| a[i] == a[j]) {
                let tie = &mut ties[axis];
                tie.sums.add(run.len());
                for &i in group {
                    observations[i].rank[axis] = tie.distinct;
                    observations[i].untied[axis] = x.len() - run.len();
                }
                tie.distinct += 1;
            }
        }
        let mut tree = try_filled(
            ties[1].distinct,
            0_usize,
            scioxide_core::quantity!("kendall_counts"),
        )?;
        // Two sweeps compute per-observation influences. Integer subtraction is
        // exact; additive weighting can therefore cancel counts before weighting.
        for reverse in [false, true] {
            tree.fill(0);
            let mut seen = 0;
            let mut start = 0;
            while start < x.len() {
                let position = |k| orders[0][if reverse { x.len() - 1 - k } else { k }];
                let mut end = start + 1;
                while end < x.len()
                    && observations[position(end)].rank[0] == observations[position(start)].rank[0]
                {
                    end += 1;
                }
                for k in start..end {
                    let observation = &mut observations[position(k)];
                    let below = prefix(&tree, observation.rank[1]);
                    let above = seen - prefix(&tree, observation.rank[1] + 1);
                    // Vec allocation bounds the observation count below isize::MAX.
                    observation.balance += if reverse {
                        above as isize - below as isize
                    } else {
                        below as isize - above as isize
                    };
                }
                for k in start..end {
                    let mut index = observations[position(k)].rank[1];
                    while index < tree.len() {
                        tree[index] += 1;
                        index |= index + 1;
                    }
                }
                seen += end - start;
                start = end;
            }
        }
        // Every pair appears at its two endpoints; i128 also covers the largest
        // possible slice on a 64-bit target without losing integer cancellation.
        let score = (observations.iter().map(|o| o.balance as i128).sum::<i128>() / 2) as f64;
        Ok(Self {
            orders,
            observations,
            ties,
            score,
        })
    }

    pub(super) fn importance(&self, axis: usize, weights: &mut [f64]) {
        let mut rank = 0;
        // Reverse lexicographic groups, retaining original-index order within
        // joint ties, exactly as the public default importance convention.
        for group in self.orders[axis]
            .chunk_by(|&i, &j| self.observations[i].rank == self.observations[j].rank)
            .rev()
        {
            for &i in group {
                weights[i] = 1.0 / (rank + 1) as f64;
                rank += 1;
            }
        }
    }

    pub(super) fn weighted<N: Arithmetic>(
        &self,
        weights: &[f64],
        combination: KendallWeightCombination,
    ) -> Result<[N; 3], SciOxideError> {
        let mut sums = [Compensated::<N>::zero(); 3];
        if combination == KendallWeightCombination::Additive {
            for (observation, &weight) in self.observations.iter().zip(weights) {
                let w = N::number(weight)?;
                for (sum, count) in sums.iter_mut().zip([
                    observation.balance as f64,
                    observation.untied[0] as f64,
                    observation.untied[1] as f64,
                ]) {
                    *sum = sum.added(w.mul(N::number(count)?)?)?;
                }
            }
        } else {
            // Each marginal counts only pairs crossing distinct value groups.
            // Never subtract tie mass from a larger rounded total.
            for axis in 0..2 {
                let mut previous = Compensated::<N>::zero();
                for group in self.orders[axis].chunk_by(|&i, &j| {
                    self.observations[i].rank[axis] == self.observations[j].rank[axis]
                }) {
                    for &i in group {
                        let w = N::number(weights[i])?;
                        for part in previous.parts() {
                            sums[axis + 1] = sums[axis + 1].added(w.mul(part)?)?;
                        }
                    }
                    for &i in group {
                        previous = previous.added(N::number(weights[i])?)?;
                    }
                }
            }
            let size = self.ties[1].distinct;
            let mut tree = try_filled(
                size,
                N::ZERO_SUM,
                scioxide_core::quantity!("kendall_pair_masses"),
            )?;
            // Direct lower and upper queries avoid total-minus-prefix loss of
            // tiny positive masses. Carry every node's correction into products.
            for reverse in [false, true] {
                tree.fill(N::ZERO_SUM);
                for group in self.orders[0]
                    .chunk_by(|&i, &j| self.observations[i].rank[0] == self.observations[j].rank[0])
                {
                    for &i in group {
                        let rank = self.observations[i].rank[1];
                        let mut end = if reverse { size - 1 - rank } else { rank };
                        let w = N::number(if reverse { -weights[i] } else { weights[i] })?;
                        while end > 0 {
                            for part in N::sum_parts(tree[end - 1]) {
                                sums[0] = sums[0].added(w.mul(part)?)?;
                            }
                            end &= end - 1;
                        }
                    }
                    for &i in group {
                        let rank = self.observations[i].rank[1];
                        let mut index = if reverse { size - 1 - rank } else { rank };
                        let w = N::number(weights[i])?;
                        while index < size {
                            N::sum_add(&mut tree[index], w)?;
                            index |= index + 1;
                        }
                    }
                }
            }
        }
        Ok([sums[0].total()?, sums[1].total()?, sums[2].total()?])
    }
}
