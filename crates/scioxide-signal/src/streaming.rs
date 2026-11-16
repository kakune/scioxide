//! Progress counts of stream calls.
/// Counts reported by one stream call, in samples per channel.
///
/// See [Streams](crate#streams).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamProgress {
    pub(crate) consumed: usize,
    pub(crate) produced: usize,
    pub(crate) first_output_index: usize,
    pub(crate) replacements: usize,
}

impl StreamProgress {
    /// Returns the number of input samples the call consumed.
    pub const fn consumed(self) -> usize {
        self.consumed
    }

    /// Returns the number of output samples the call wrote.
    pub const fn produced(self) -> usize {
        self.produced
    }

    /// Returns the index, counted from the start of the output record, of the
    /// first sample the call wrote.
    pub const fn first_output_index(self) -> usize {
        self.first_output_index
    }

    /// Returns the number of replacement indices a Hampel filter stream wrote,
    /// or zero for other streams.
    pub const fn replacements(self) -> usize {
        self.replacements
    }
}
