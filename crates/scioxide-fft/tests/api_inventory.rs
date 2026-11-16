//! The authoritative scioxide-fft root census.
//!
//! Every crate-root export is imported here as an external consumer, and a
//! test fails when this list and the `pub use` items of `src/lib.rs` differ.
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_fft::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    AxisWorkspaceRequest, CztAlgorithm, CztContour, CztLimits, CztPlan, CztPlanSpec, DctType,
    DiscreteHankelDiagnostics, DiscreteHankelPlan, DiscreteHankelPlanSpec, DstType, FftAlgorithmKind,
    FftBackend, FftBackendStrategy, FftCachePolicy, FftCacheStats, FftDirection, FftInput,
    FftLayout, FftLogLimits, FftLogDiagnostics, FftLogPlan, FftLogPlanSpec, FftNdInput, FftNdOutput, FftNdPlan,
    FftNdPlanSpec, FftOutput, FftPlan, FftPlanCache, FftPlanDiagnostics, FftPlanSpec,
    FftPlanningContext, FftProvider, FftProviderLimits, FftProviderPlan, FftProviderPolicy,
    FftWorkspace, FftWorkspaceRequirements, HankelLimits, HermitianSpectrum, HermitianSpectrumNd,
    Normalization, NufftDiagnostics, NufftLimits, NufftPlan,
    NufftPlanSpec, NufftTransform, RealTransformKind, RealTransformNdPlan, RealTransformNdPlanSpec,
    RealTransformPlan, RealTransformPlanSpec, RealTransformNdWorkspace,
    RealTransformNdWorkspaceRequirements, czt, dct, dctn, dht, dst, dstn, fft, fftfreq,
    fftlog_low_ringing_offset, fftn, fftshift, fftshift_nd, fht, hfft, hfftn, idct, idctn, idst,
    idht, idstn, ifft, ifftn, ifftshift, ifftshift_nd, ifht, ihfft, ihfftn, irfft, irfftn, next_fast_len,
    nufft_type1, nufft_type2, nufft_type3, prev_fast_len, rfft, rfftfreq, rfftn, zoom_fft,
}

/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("pub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn inventory_lists_every_root_export_once() {
    let listed: BTreeSet<_> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    let root = root_exports(include_str!("../src/lib.rs"));
    let unlisted: Vec<_> = root.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&root).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "update tests/api_inventory.rs\nunlisted: {unlisted:?}\nstale: {stale:?}"
    );
}

#[test]
fn root_exports_read_single_and_braced_statements() {
    let source = "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\nuse c::Hidden;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}
