const diagrams = new Map<HTMLElement, () => void>();

function initializeDiagrams() {
  for (const figure of document.querySelectorAll<HTMLElement>('.mermaid-diagram')) {
    if (diagrams.has(figure)) continue;
    let scrollLeft: number | undefined;
    let restoring = true;
    // Give keyboard users the same native scrollport as touch and pointer users.
    figure.tabIndex = 0;
    const rememberPosition = () => {
      if (!restoring && figure.dataset.state === 'ready') scrollLeft = figure.scrollLeft;
    };
    const positionDiagram = () => {
      restoring = true;
      if (figure.dataset.state !== 'ready') return;
      requestAnimationFrame(() => {
        if (!figure.isConnected || figure.dataset.state !== 'ready') return;
        const firstNode = figure.querySelector<SVGElement>('svg .node');
        if (scrollLeft !== undefined) {
          // Theme changes replace the SVG and temporarily collapse the scrollport.
          figure.scrollLeft = scrollLeft;
        } else if (firstNode) {
          const node = firstNode.getBoundingClientRect();
          const frame = figure.getBoundingClientRect();
          // Wide decision trees otherwise open on empty space to the left of their root.
          figure.scrollLeft += node.x + node.width / 2 - frame.x - frame.width / 2;
        }
        scrollLeft = figure.scrollLeft;
        // Ignore queued scroll events from the temporary collapse until restoration paints.
        requestAnimationFrame(() => { restoring = false; });
      });
    };
    const observer = new MutationObserver(positionDiagram);
    figure.addEventListener('scroll', rememberPosition, { passive: true });
    diagrams.set(figure, () => {
      observer.disconnect();
      figure.removeEventListener('scroll', rememberPosition);
    });
    observer.observe(figure, { attributes: true, attributeFilter: ['data-state'] });
    positionDiagram();
  }
}

initializeDiagrams();
document.addEventListener('astro:page-load', initializeDiagrams);
document.addEventListener('astro:before-swap', () => {
  for (const cleanup of diagrams.values()) cleanup();
  diagrams.clear();
});
