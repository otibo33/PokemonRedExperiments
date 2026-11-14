from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from gen1_party_metadata import (
    INTERNAL_TO_NATIONAL_DEX,
    LEGACY_TYPE_NAMES_BY_INTERNAL_ID,
    LEGACY_TYPE_IDS_BY_INTERNAL_ID,
    LEGACY_STARTER_INTERNAL_IDS,
    LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID,
    merge_missing_party_metadata,
)
from migrate_most_advanced_party import migrate

ROOT = Path(__file__).resolve().parent


class PartyMetadataVisualFixTests(unittest.TestCase):
    def test_complete_gen1_internal_to_national_dex_mapping(self):
        self.assertEqual(len(INTERNAL_TO_NATIONAL_DEX), 151)
        self.assertEqual(set(INTERNAL_TO_NATIONAL_DEX.values()), set(range(1, 152)))
        self.assertEqual(INTERNAL_TO_NATIONAL_DEX[0x99], 1)
        self.assertEqual(INTERNAL_TO_NATIONAL_DEX[0xB0], 4)
        self.assertEqual(INTERNAL_TO_NATIONAL_DEX[0xA9], 74)
        self.assertEqual(INTERNAL_TO_NATIONAL_DEX[0x15], 151)
        self.assertNotIn(0x1F, INTERNAL_TO_NATIONAL_DEX)

    def test_legacy_current_party_metadata(self):
        self.assertEqual(LEGACY_TYPE_NAMES_BY_INTERNAL_ID[0xB0], ("Fire",))
        self.assertEqual(LEGACY_TYPE_NAMES_BY_INTERNAL_ID[0xA9], ("Rock", "Ground"))
        self.assertEqual(LEGACY_TYPE_IDS_BY_INTERNAL_ID[0xA9], (0x05, 0x04))
        self.assertIn(0xB0, LEGACY_STARTER_INTERNAL_IDS)
        self.assertEqual(LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID[0xA9], ("STRENGTH",))

    def test_merge_preserves_historical_state_and_backfills(self):
        old = [{"slot": 1, "species_index": 0xB0, "species": "Charmander", "level": 38,
                "hp": 4, "max_hp": 99, "moves": ["Scratch", "Ember"]}]
        current = [{"slot": 1, "species_index": 0xB0, "species": "Charmander", "level": 39,
                    "national_dex": 4, "type_ids": [0x14, 0x14], "types": ["Fire"],
                    "is_starter": True, "hm_preferred_carrier_for": []}]
        merged, changed = merge_missing_party_metadata(old, current)
        self.assertTrue(changed)
        self.assertEqual(merged[0]["national_dex"], 4)
        self.assertEqual(merged[0]["types"], ["Fire"])
        self.assertTrue(merged[0]["is_starter"])
        self.assertEqual(merged[0]["level"], 38)
        self.assertEqual(merged[0]["hp"], 4)
        self.assertEqual(merged[0]["moves"], ["Scratch", "Ember"])

    def test_legacy_fallbacks_work_without_current_party_match(self):
        old = [{"slot": 1, "species_index": 0xB0}, {"slot": 2, "species_index": 0xA9}]
        merged, changed = merge_missing_party_metadata(old, [{"slot": 1, "species_index": 0x77}])
        self.assertTrue(changed)
        self.assertEqual(merged[0]["national_dex"], 4)
        self.assertEqual(merged[0]["types"], ["Fire"])
        self.assertTrue(merged[0]["is_starter"])
        self.assertEqual(merged[1]["national_dex"], 74)
        self.assertEqual(merged[1]["types"], ["Rock", "Ground"])
        self.assertEqual(merged[1]["hm_preferred_carrier_for"], ["STRENGTH"])

    def test_migration_backfills_snapshot_without_changing_progress(self):
        saved = {"worker_index": 13, "timestep": 509635778, "party_size": 3, "party_snapshot": [
            {"slot": 1, "species_index": 0xB0, "species": "Charmander", "level": 38, "hp": 4},
            {"slot": 2, "species_index": 0xA9, "species": "Geodude", "level": 14},
            {"slot": 3, "species_index": 0xA9, "species": "Geodude", "level": 14},
        ]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "most_advanced_party.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            counts = migrate(path)
            result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(counts, (3, 3, 3, 2))
        party = result["party_snapshot"]
        self.assertEqual([mon["national_dex"] for mon in party], [4, 74, 74])
        self.assertEqual(party[0]["types"], ["Fire"])
        self.assertEqual(party[1]["types"], ["Rock", "Ground"])
        self.assertEqual(party[2]["type_ids"], [0x05, 0x04])
        self.assertTrue(party[0]["is_starter"])
        self.assertFalse(party[1]["is_starter"])
        self.assertEqual(party[1]["hm_preferred_carrier_for"], ["STRENGTH"])
        self.assertEqual((result["worker_index"], result["timestep"], result["party_size"]),
                         (13, 509635778, 3))
        self.assertEqual(party[0]["level"], 38)
        self.assertEqual(party[0]["hp"], 4)

    def test_migration_dry_run_is_non_destructive(self):
        saved = {"party_snapshot": [{"slot": 1, "species_index": 0xB0}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "party.json"
            original = json.dumps(saved)
            path.write_text(original, encoding="utf-8")
            result = migrate(path, dry_run=True)
            self.assertEqual(result, (1, 1, 1, 0))
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_environment_snapshot_includes_sprite_and_type_fields(self):
        source = (ROOT / "red_gym_env_v2.py").read_text(encoding="utf-8")
        self.assertIn('"national_dex": national_dex', source)
        self.assertIn('"type_ids": list(type_ids)', source)

    def test_callback_refreshes_old_metadata_and_resolves_sprites(self):
        source = (ROOT / "tensorboard_callback.py").read_text(encoding="utf-8")
        self.assertIn("merge_missing_party_metadata", source)
        self.assertIn("metadata_upgraded", source)
        self.assertIn("INTERNAL_TO_NATIONAL_DEX.get(species_index)", source)
        self.assertIn("LEGACY_TYPE_NAMES_BY_INTERNAL_ID.get(species_index", source)
        self.assertIn("LEGACY_HM_PREFERRED_CARRIERS_BY_INTERNAL_ID.get(species_index", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
