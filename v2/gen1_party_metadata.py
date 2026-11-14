"""Generation-I Pokémon identity helpers for Red/Blue monitoring."""
from __future__ import annotations
from typing import Any

# Pokémon Red/Blue internal species ID -> National Pokédex number.
# This is the ROM's PokedexOrder lookup represented as a direct mapping.
INTERNAL_TO_NATIONAL_DEX: dict[int, int] = {
    0x01: 112,
    0x02: 115,
    0x03:  32,
    0x04:  35,
    0x05:  21,
    0x06: 100,
    0x07:  34,
    0x08:  80,
    0x09:   2,
    0x0A: 103,
    0x0B: 108,
    0x0C: 102,
    0x0D:  88,
    0x0E:  94,
    0x0F:  29,
    0x10:  31,
    0x11: 104,
    0x12: 111,
    0x13: 131,
    0x14:  59,
    0x15: 151,
    0x16: 130,
    0x17:  90,
    0x18:  72,
    0x19:  92,
    0x1A: 123,
    0x1B: 120,
    0x1C:   9,
    0x1D: 127,
    0x1E: 114,
    0x21:  58,
    0x22:  95,
    0x23:  22,
    0x24:  16,
    0x25:  79,
    0x26:  64,
    0x27:  75,
    0x28: 113,
    0x29:  67,
    0x2A: 122,
    0x2B: 106,
    0x2C: 107,
    0x2D:  24,
    0x2E:  47,
    0x2F:  54,
    0x30:  96,
    0x31:  76,
    0x33: 126,
    0x35: 125,
    0x36:  82,
    0x37: 109,
    0x39:  56,
    0x3A:  86,
    0x3B:  50,
    0x3C: 128,
    0x40:  83,
    0x41:  48,
    0x42: 149,
    0x46:  84,
    0x47:  60,
    0x48: 124,
    0x49: 146,
    0x4A: 144,
    0x4B: 145,
    0x4C: 132,
    0x4D:  52,
    0x4E:  98,
    0x52:  37,
    0x53:  38,
    0x54:  25,
    0x55:  26,
    0x58: 147,
    0x59: 148,
    0x5A: 140,
    0x5B: 141,
    0x5C: 116,
    0x5D: 117,
    0x60:  27,
    0x61:  28,
    0x62: 138,
    0x63: 139,
    0x64:  39,
    0x65:  40,
    0x66: 133,
    0x67: 136,
    0x68: 135,
    0x69: 134,
    0x6A:  66,
    0x6B:  41,
    0x6C:  23,
    0x6D:  46,
    0x6E:  61,
    0x6F:  62,
    0x70:  13,
    0x71:  14,
    0x72:  15,
    0x74:  85,
    0x75:  57,
    0x76:  51,
    0x77:  49,
    0x78:  87,
    0x7B:  10,
    0x7C:  11,
    0x7D:  12,
    0x7E:  68,
    0x80:  55,
    0x81:  97,
    0x82:  42,
    0x83: 150,
    0x84: 143,
    0x85: 129,
    0x88:  89,
    0x8A:  99,
    0x8B:  91,
    0x8D: 101,
    0x8E:  36,
    0x8F: 110,
    0x90:  53,
    0x91: 105,
    0x93:  93,
    0x94:  63,
    0x95:  65,
    0x96:  17,
    0x97:  18,
    0x98: 121,
    0x99:   1,
    0x9A:   3,
    0x9B:  73,
    0x9D: 118,
    0x9E: 119,
    0xA3:  77,
    0xA4:  78,
    0xA5:  19,
    0xA6:  20,
    0xA7:  33,
    0xA8:  30,
    0xA9:  74,
    0xAA: 137,
    0xAB: 142,
    0xAD:  81,
    0xB0:   4,
    0xB1:   7,
    0xB2:   5,
    0xB3:   8,
    0xB4:   6,
    0xB9:  43,
    0xBA:  44,
    0xBB:  45,
    0xBC:  69,
    0xBD:  70,
    0xBE:  71,
}

# Fallback metadata for the already-persisted best-party snapshot. New live
# snapshots get all this data directly from the party's RAM structure.
LEGACY_TYPE_NAMES_BY_INTERNAL_ID: dict[int, tuple[str, ...]] = {
    0xB0: ("Fire",),
    0xA9: ("Rock", "Ground"),
}
LEGACY_TYPE_IDS_BY_INTERNAL_ID: dict[int, tuple[int, ...]] = {
    0xB0: (0x14, 0x14),
    0xA9: (0x05, 0x04),
}
LEGACY_STARTER_INTERNAL_IDS = frozenset({0xB0})
LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID: dict[int, tuple[str, ...]] = {
    0xA9: ("STRENGTH",),
}


def merge_missing_party_metadata(
    existing_party: list[dict[str, Any]],
    current_party: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Backfill absent diagnostics from a live same-worker party snapshot.

    Historical party order, level, HP, status and moves are deliberately kept.
    Only identity/type/starter/HM presentation metadata is copied or derived.
    """
    result = [dict(mon) if isinstance(mon, dict) else {} for mon in existing_party]
    current = [dict(mon) for mon in current_party if isinstance(mon, dict)]
    by_slot_and_species: dict[tuple[int, int], dict[str, Any]] = {}
    by_species: dict[int, dict[str, Any]] = {}
    for mon in current:
        try:
            species = int(mon.get("species_index", -1))
            slot = int(mon.get("slot", -1))
        except (TypeError, ValueError):
            continue
        by_slot_and_species[(slot, species)] = mon
        by_species.setdefault(species, mon)

    changed = False
    fields = ("national_dex", "type_ids", "types", "is_starter", "hm_preferred_carrier_for")
    for index, old in enumerate(result):
        if not isinstance(old, dict):
            continue
        try:
            species = int(old.get("species_index", -1))
            slot = int(old.get("slot", index + 1))
        except (TypeError, ValueError):
            continue
        fresh = by_slot_and_species.get((slot, species)) or by_species.get(species)
        if fresh is not None:
            for field in fields:
                old_value = old.get(field)
                fresh_value = fresh.get(field)
                old_missing = old_value is None or old_value == [] or old_value == ""
                fresh_present = fresh_value is not None and fresh_value != [] and fresh_value != ""
                if old_missing and fresh_present:
                    old[field] = fresh_value
                    changed = True

        # Safe fallback for this project's legacy Charmander/Geodude snapshot.
        fallbacks = {
            "national_dex": INTERNAL_TO_NATIONAL_DEX.get(species),
            "type_ids": LEGACY_TYPE_IDS_BY_INTERNAL_ID.get(species),
            "types": LEGACY_TYPE_NAMES_BY_INTERNAL_ID.get(species),
            "is_starter": species in LEGACY_STARTER_INTERNAL_IDS,
            "hm_preferred_carrier_for": LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID.get(species),
        }
        for field, value in fallbacks.items():
            old_value = old.get(field)
            old_missing = old_value is None or old_value == [] or old_value == ""
            if old_missing and value is not None:
                old[field] = list(value) if isinstance(value, tuple) else value
                changed = True
    return result, changed
