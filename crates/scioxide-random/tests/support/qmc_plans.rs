//! Deterministic QMC plan configurations shared by replay and resource checks.
#![allow(clippy::unwrap_used)]
use scioxide_random::{DirectionNumberSet, QmcPlan, QmcScramble, SobolDirection};
pub fn plans() -> Vec<QmcPlan> {
    vec![
        QmcPlan::sobol(3).unwrap(),
        QmcPlan::sobol(3)
            .unwrap()
            .with_randomization(
                scioxide_random::QmcRandomization::LinearMatrixShift,
                &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(19),
            )
            .unwrap(),
        QmcPlan::halton(3).unwrap(),
        QmcPlan::halton(3)
            .unwrap()
            .with_scramble(QmcScramble::DigitPermutation(928))
            .unwrap(),
        QmcPlan::halton(3)
            .unwrap()
            .with_permutations(vec![vec![0, 1], vec![0, 2, 1], vec![0, 4, 3, 2, 1]])
            .unwrap(),
        QmcPlan::reverse_halton(3).unwrap(),
        QmcPlan::niederreiter_base2(3).unwrap(),
        QmcPlan::sobol_with_direction_numbers(
            3,
            DirectionNumberSet::Custom(vec![
                SobolDirection::new(1, 0, vec![1]).unwrap(),
                SobolDirection::new(2, 1, vec![1, 3]).unwrap(),
            ]),
        )
        .unwrap(),
    ]
}
