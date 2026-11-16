/** @typedef {import('@astrojs/starlight/route-data').StarlightRouteData['sidebar'][number]} Entry */

/** @param {Entry} entry @returns {boolean} */
function containsCurrent(entry) {
  return entry.type === 'link' ? entry.isCurrent : entry.entries.some(containsCurrent);
}

/** @param {Entry} entry @returns {Extract<Entry, { type: 'link' }> | undefined} */
function firstLink(entry) {
  if (entry.type === 'link') return entry;
  for (const child of entry.entries) {
    const link = firstLink(child);
    if (link) return link;
  }
}

/**
 * Keep the current chapter's detail and one overview link for every other chapter.
 * Starlight has already resolved locale/base URLs and previous/next links.
 * @param {Entry[]} sidebar
 * @returns {Entry[]}
 */
export function chapterSidebar(sidebar) {
  return sidebar.flatMap((entry) => {
    if (entry.type === 'link' || containsCurrent(entry)) return [entry];
    const overview = firstLink(entry);
    return overview ? [{ ...overview, label: entry.label }] : [];
  });
}
