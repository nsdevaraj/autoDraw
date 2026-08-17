"""Read the SVGDepot token index; mirrors iconPathsFromIndex in src/svgdepot-corpus.mjs."""

INDEX_SCHEMA_VERSION = 1


def icon_paths_from_index(index):
    """Expand the index's packed (packId, filename) pairs into repository-relative paths."""
    if index.get('schemaVersion') != INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported SVGDepot index schema version: {index.get('schemaVersion')}")
    icons = index.get('icons')
    packs = index.get('packs')
    categories = index.get('categories')
    if not isinstance(icons, list) or not isinstance(packs, list) or not isinstance(categories, list):
        raise ValueError('SVGDepot index must contain icons, packs, and categories')

    paths = []
    for icon_id, entry in enumerate(icons):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f'SVGDepot index entry {icon_id} is malformed')
        pack_id, filename = entry
        if not isinstance(pack_id, int) or pack_id < 0 or pack_id >= len(packs):
            raise ValueError(f'SVGDepot index entry {icon_id} is malformed')
        pack = packs[pack_id]
        if not isinstance(pack, list) or not isinstance(filename, str):
            raise ValueError(f'SVGDepot index entry {icon_id} is malformed')
        if not isinstance(pack[0], int) or pack[0] < 0 or pack[0] >= len(categories):
            raise ValueError(f'SVGDepot index entry {icon_id} has an unknown category')
        segments = [categories[pack[0]], pack[1], filename]
        paths.append({'packId': pack_id, 'path': '/'.join(part for part in segments if part)})
    return paths
