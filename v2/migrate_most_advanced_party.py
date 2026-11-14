#!/usr/bin/env python3
"""Backfill sprites, types and labels in the persisted best-party snapshot."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from gen1_party_metadata import (
    INTERNAL_TO_NATIONAL_DEX,
    LEGACY_TYPE_NAMES_BY_INTERNAL_ID,
    LEGACY_TYPE_IDS_BY_INTERNAL_ID,
    LEGACY_STARTER_INTERNAL_IDS,
    LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID,
)


def migrate(path: Path, dry_run: bool = False) -> tuple[int, int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    party = data.get("party_snapshot")
    if not isinstance(party, list):
        raise ValueError(f"No party_snapshot list in {path}")
    dex_changes = type_changes = starter_changes = role_changes = 0
    for mon in party:
        if not isinstance(mon, dict):
            continue
        try:
            internal = int(mon.get("species_index", -1))
        except (TypeError, ValueError):
            continue
        dex = INTERNAL_TO_NATIONAL_DEX.get(internal)
        if dex is not None and mon.get("national_dex") != dex:
            mon["national_dex"] = dex
            dex_changes += 1
        if not mon.get("types") and internal in LEGACY_TYPE_NAMES_BY_INTERNAL_ID:
            mon["types"] = list(LEGACY_TYPE_NAMES_BY_INTERNAL_ID[internal])
            type_changes += 1
        if not mon.get("type_ids") and internal in LEGACY_TYPE_IDS_BY_INTERNAL_ID:
            mon["type_ids"] = list(LEGACY_TYPE_IDS_BY_INTERNAL_ID[internal])
        if "is_starter" not in mon:
            mon["is_starter"] = internal in LEGACY_STARTER_INTERNAL_IDS
            starter_changes += 1
        if not mon.get("hm_preferred_carrier_for") and internal in LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID:
            mon["hm_preferred_carrier_for"] = list(LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID[internal])
            role_changes += 1
    if not dry_run and (dex_changes or type_changes or starter_changes or role_changes):
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    return dex_changes, type_changes, starter_changes, role_changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("runs/_trajectory_history/most_advanced_party.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dex, types, starters, roles = migrate(args.path, args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action}: National Dex={dex}, types={types}, starter labels={starters}, HM roles={roles}, file={args.path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
