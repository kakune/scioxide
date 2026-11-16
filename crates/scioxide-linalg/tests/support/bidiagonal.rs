//! Destination-rounded independent bidiagonal SVD references.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use std::vec::Vec;

pub(crate) struct Bidiagonal<R> {
    pub name: &'static str,
    pub diagonal: Vec<R>,
    pub off_diagonal: Vec<R>,
    pub expected: Vec<R>,
}

pub(crate) fn cases<R: RealScalar>() -> impl Iterator<Item = Bidiagonal<R>> {
    include_str!("../data/svd_bidiagonal.tsv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            if fields[0].parse::<usize>().unwrap() != core::mem::size_of::<R>() * 8 {
                return None;
            }
            let parse = |field: &str| {
                field
                    .split(',')
                    .map(|value| R::try_from_f64_lossy(value.parse().unwrap()).unwrap())
                    .collect()
            };
            let case = Bidiagonal {
                name: fields[1],
                diagonal: parse(fields[2]),
                off_diagonal: parse(fields[3]),
                expected: parse(fields[4]),
            };
            assert_eq!(case.diagonal.len(), case.expected.len());
            assert_eq!(case.off_diagonal.len() + 1, case.diagonal.len());
            Some(case)
        })
}
