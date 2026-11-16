//! Immutable sorted CSR rows for tessellation incidence and adjacency.
use crate::tessellation::Result;
use crate::tessellation::storage::Storage;
use crate::tessellation::work::POLL_UNITS;
use scioxide_core::SolveContext;
use scioxide_core::backend::checked_add;

use crate::incidence::Incidence;

impl Incidence {
    pub(super) fn from_pairs(
        count: usize,
        pairs: impl Iterator<Item = (usize, usize)>,
        context: Option<&SolveContext<'_, '_>>,
        storage: &Storage,
    ) -> Result<Self> {
        let frame = storage.frame();

        // Poll after at most `POLL_UNITS` collected pairs, and after sorting.
        let mut unpolled = 0;
        let mut check = || -> Result<()> {
            unpolled += 1;
            if unpolled >= POLL_UNITS {
                unpolled = 0;
                if let Some(context) = context {
                    context.execution().check()?;
                }
            }
            Ok(())
        };
        let quantity = scioxide_core::quantity!("tessellation_incidence");
        let mut sorted = storage.capacity(pairs.size_hint().0, quantity)?;
        for pair in pairs {
            check()?;
            storage.reserve(&mut sorted, 1)?;
            sorted.push(pair);
        }
        let result = Incidence::from_pairs_with(
            sorted,
            |length| {
                let offsets = storage.filled(checked_add(count, 1, quantity)?, 0, quantity)?;
                let entries = storage.capacity(length, quantity)?;
                Ok((offsets, entries))
            },
            || match context {
                Some(context) => context.execution().check(),
                None => Ok(()),
            },
        )?;
        frame.keep(0, result.retained_bytes());
        Ok(result)
    }
}
