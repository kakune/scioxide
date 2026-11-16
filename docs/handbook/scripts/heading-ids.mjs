import GithubSlugger from 'github-slugger';

export function headingId(text, slugger) {
  return slugger.slug(text);
}

export function remarkHeadingIds() {
  return (tree, file) => {
    const slugger = new GithubSlugger();
    const source = String(file);
    const plainText = (node) => {
      // Starlight restores unhandled inline directives after this transform.
      if (node.type === 'textDirective' && node.data === undefined) {
        return source.slice(node.position.start.offset, node.position.end.offset);
      }
      return node.value ?? (node.children ?? []).map(plainText).join('');
    };
    const visit = (node) => {
      if (node.type === 'heading') {
        node.data ??= {};
        node.data.hProperties ??= {};
        node.data.hProperties.id = headingId(plainText(node), slugger);
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}
