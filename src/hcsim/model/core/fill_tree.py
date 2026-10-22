"""The fill tree: how the fill categories nest.

```
fills
├── hires
│   ├── external_hires
│   └── conversion_hires
└── internal_starts
```

A parent is exactly the sum of its children, so a forecast can carry any cut through the tree:
the leaves, a parent alone, or both. The structure is checked when this module is imported, so
a broken tree can't load at all:

- one root, every node with one parent, and no cycles;
- every node a headcount-grain guidance category;
- leaves are terms of the coherency identity, with the same sign, and inner nodes aren't;
- an inner node's symbol is its leaves' symbols, joined in order;
- an inner node's measure is exactly the sum of its children's.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Final

from hcsim.model.core.categories import CATEGORIES, Grain, GuidanceCategory
from hcsim.model.core.measures import CATEGORY_MEASURES, TransTypeMeasure
from hcsim.model.core.transactions import TransType

C = GuidanceCategory
Tree = Mapping[GuidanceCategory, tuple[GuidanceCategory, ...]]


class FillTreeError(ValueError):
    """A fill tree whose structure doesn't hold."""


FILL_TREE: Final[Tree] = MappingProxyType(
    {
        C.FILLS: (C.HIRES, C.INTERNAL_STARTS),
        C.HIRES: (C.EXTERNAL_HIRES, C.CONVERSION_HIRES),
    }
)


def nodes(tree: Tree = FILL_TREE) -> frozenset[GuidanceCategory]:
    return frozenset(tree) | {child for children in tree.values() for child in children}


def roots(tree: Tree = FILL_TREE) -> list[GuidanceCategory]:
    children = {child for kids in tree.values() for child in kids}
    return [node for node in tree if node not in children]


def parent(category: GuidanceCategory, tree: Tree = FILL_TREE) -> GuidanceCategory | None:
    for node, children in tree.items():
        if category in children:
            return node
    return None


def ancestors(category: GuidanceCategory, tree: Tree = FILL_TREE) -> list[GuidanceCategory]:
    """From the nearest parent up to the root."""
    found = []
    node = parent(category, tree)
    while node is not None:
        found.append(node)
        node = parent(node, tree)
    return found


def leaves(category: GuidanceCategory, tree: Tree = FILL_TREE) -> list[GuidanceCategory]:
    """The leaves under `category`, in order; a leaf is its own."""
    children = tree.get(category)
    if not children:
        return [category]
    return [leaf for child in children for leaf in leaves(child, tree)]


def cover(
    node: GuidanceCategory, carried: Collection[GuidanceCategory], tree: Tree = FILL_TREE
) -> tuple[list[GuidanceCategory], list[GuidanceCategory]]:
    """The deepest carried nodes that cover `node`'s leaves, each leaf once, and the leaves
    they leave uncovered. Children are preferred when they cover every leaf below them; the
    node itself when they don't and it's carried; and otherwise as much as the children can."""
    children = tree.get(node, ())
    if not children:
        return ([node], []) if node in carried else ([], [node])
    parts = [cover(child, carried, tree) for child in children]
    terms = [t for found, _ in parts for t in found]
    missing = [m for _, gaps in parts for m in gaps]
    if missing and node in carried:
        return [node], []
    return terms, missing


def identity_terms(
    carried: Collection[GuidanceCategory], tree: Tree = FILL_TREE
) -> tuple[dict[GuidanceCategory, int], list[GuidanceCategory]]:
    """The forecast-coherency identity's terms for a forecast carrying `carried`, with their
    signs, and the identity's flows it can't cover. Under each fill-tree root the terms are the
    deepest carried cover, so a fill is never counted twice; every other flow term is a term
    when carried, and missing when not."""
    terms: dict[GuidanceCategory, int] = {}
    missing: list[GuidanceCategory] = []
    filled = set()
    for root in roots(tree):
        found, gaps = cover(root, carried, tree)
        filled |= set(leaves(root, tree))
        for node in found:
            signs = {CATEGORIES[leaf].coherency_sign for leaf in leaves(node, tree)}
            (sign,) = signs  # the fill tree's structure check keeps a node's leaves one sign
            assert sign is not None
            terms[node] = sign
        missing += gaps
    for category, spec in CATEGORIES.items():
        if spec.coherency_sign in (1, -1) and category not in filled:
            if category in carried:
                terms[category] = spec.coherency_sign
            else:
                missing.append(category)
    return terms, missing


def is_leaf(category: GuidanceCategory, tree: Tree = FILL_TREE) -> bool:
    return category in nodes(tree) and category not in tree


def check(tree: Tree) -> list[str]:
    """Every structural problem in `tree`."""
    problems = []
    counts = Counter(child for children in tree.values() for child in children)
    shared = sorted(c.value for c, n in counts.items() if n > 1)
    if shared:
        problems.append(f"nodes with more than one parent: {shared}")
    found = roots(tree)
    if len(found) != 1:
        problems.append(f"the tree needs one root, not {sorted(c.value for c in found)}")
    for node in tree:
        seen: set[GuidanceCategory] = set()
        current: GuidanceCategory | None = node
        while current is not None:
            if current in seen:
                problems.append(f"a cycle through {node.value!r}")
                break
            seen.add(current)
            current = parent(current, tree)
    if problems:  # the rest walks the tree, which needs it well formed
        return problems
    for node in sorted(nodes(tree)):
        spec = CATEGORIES[node]
        if spec.grain is not Grain.HEADCOUNT:
            problems.append(f"{node.value!r} has {spec.grain.value} grain, not headcount")
        if node in tree:
            if spec.coherency_sign is not None:
                problems.append(f"inner node {node.value!r} is a term of the identity")
            symbol = " + ".join(CATEGORIES[leaf].symbol for leaf in leaves(node, tree))
            if spec.symbol != symbol:
                problems.append(f"{node.value!r} has symbol {spec.symbol!r}, not {symbol!r}")
            problems += _measure_problems(node, tree)
        elif spec.coherency_sign not in (1, -1):
            problems.append(f"leaf {node.value!r} isn't a term of the identity")
    signs = {CATEGORIES[leaf].coherency_sign for root in roots(tree) for leaf in leaves(root, tree)}
    if len(signs) > 1:
        problems.append("the leaves mix inflows and outflows")
    return problems


def _measure_problems(node: GuidanceCategory, tree: Tree) -> list[str]:
    measures = [CATEGORY_MEASURES[c] for c in (node, *tree[node])]
    if not all(isinstance(m, TransTypeMeasure) for m in measures):
        return [f"{node.value!r} and its children must all be transaction measures"]
    total: Counter[TransType] = Counter()
    for child in tree[node]:
        measure = CATEGORY_MEASURES[child]
        assert isinstance(measure, TransTypeMeasure)
        total.update(dict(measure.terms))
    own = CATEGORY_MEASURES[node]
    assert isinstance(own, TransTypeMeasure)
    summed = {code: sign for code, sign in total.items() if sign}
    if dict(own.terms) != summed:
        return [f"{node.value!r}'s measure isn't the sum of its children's"]
    return []


_problems = check(FILL_TREE)
if _problems:  # pragma: no cover - raised at import, tested through `check`
    raise FillTreeError("the fill tree doesn't hold: " + "; ".join(_problems))
