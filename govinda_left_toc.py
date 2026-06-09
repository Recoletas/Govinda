"""
Inject each markdown page's H2 (and H3) headings as nested children in the
left navigation, so single-file pages (spec.md, plan.md) expose their sections
without needing the right-side TOC.

Heavily inspired by mkdocs-awesome-nav but trimmed to one job.
"""
from mkdocs.plugins import BasePlugin
from mkdocs.structure.nav import Section, Link


def _toc_to_nav(toc_items, base_url, level=2):
    """Recursively convert TOC items (level 2+3) into a list of Section/Link.

    base_url is the page URL (no anchor); H2 url = base_url + '#' + id.
    """
    result = []
    for t in toc_items:
        if t.level < level:
            continue
        if t.level > level + 1:
            # Skip H4+ for now; flatten into parent level
            continue
        anchor_url = base_url + t.url  # t.url is "#anchor-id"
        if t.children:
            sub = _toc_to_nav(t.children, base_url, level=level + 1)
            # Wrap as Section so the H2 itself is clickable + shows sub-items
            result.append(Section(title=t.title, children=[Link(title=t.title, url=anchor_url)] + sub))
        else:
            result.append(Link(title=t.title, url=anchor_url))
    return result


def _wrap_page(page_item, page):
    """If page has H2 headings, replace the leaf Link/Page with a Section."""
    if not page or not page.toc or not page.toc.items:
        return page_item
    h2 = [t for t in page.toc.items if t.level == 2]
    if not h2:
        return page_item
    sub = _toc_to_nav(page.toc.items, page.url, level=2)
    # Section title = page title; first child = link to page top, then H2 sections
    return Section(
        title=page_item.title,
        children=[Link(title=page_item.title, url=page.url)] + sub,
    )


class GovindaLeftTocPlugin(BasePlugin):
    def on_nav(self, nav, config, files):
        # files: list[File]. Map URL → File → Page for quick lookup.
        page_by_url = {}
        for f in files:
            if f.page is not None and not f.is_documentation_page():
                continue
            if f.page is not None:
                page_by_url[f.page.url] = f.page

        def walk(item):
            if isinstance(item, Section):
                item.children = [walk(c) for c in item.children]
                return item
            # Leaf: it's a Page or Link to a page.
            page = page_by_url.get(item.url)
            if page is None:
                return item
            return _wrap_page(item, page)

        nav.items = [walk(i) for i in nav.items]
        return nav
