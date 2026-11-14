from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / "red_gym_env_v2.py"
CALLBACK_PATH = ROOT / "tensorboard_callback.py"

# SHA-256 of the original RedGymEnv.__init__ assignments from the saved
# checkpoint-compatible source. The patched file must retain them exactly.
EXPECTED_MODEL_IO = {
    "valid_actions": "6afddb6fb889c6b1873fcd8b61b4a335cbcefcae53d6fe1acc791934f1ec736d",
    "action_space": "ffd4a26a78c5901137629b9a133be22e85dab9967536126cc3c20104616f7071",
    "observation_space": "d198012e64ab2852935468ed2bebf4c742b2a6b2d092804617b768362e1b5620",
}


def assignment_hash(init_node: ast.FunctionDef, attr_name: str) -> str:
    for node in ast.walk(init_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == attr_name
            ):
                value = ast.dump(node, include_attributes=False)
                return hashlib.sha256(value.encode("utf-8")).hexdigest()
    raise AssertionError(f"Could not find self.{attr_name} assignment in __init__")


def eval_literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Set):
        return {eval_literal(item) for item in node.elts}
    if isinstance(node, ast.Tuple):
        return tuple(eval_literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [eval_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {eval_literal(k): eval_literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "frozenset":
        values = eval_literal(node.args[0]) if node.args else []
        return frozenset(values)
    raise AssertionError(f"Unsupported constant expression: {ast.dump(node)}")


def module_assignments(tree: ast.Module) -> dict[str, object]:
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "HM_PREFERRED_TYPES", "CUT_GATE_MAP_IDS", "SURF_ROUTE_MAP_IDS",
                    "STRENGTH_BOULDER_EVENT_INDICES",
                }:
                    result[target.id] = eval_literal(node.value)
    return result


class HMAwarenessPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_source = ENV_PATH.read_text(encoding="utf-8")
        cls.callback_source = CALLBACK_PATH.read_text(encoding="utf-8")
        cls.env_ast = ast.parse(cls.env_source)
        cls.callback_ast = ast.parse(cls.callback_source)
        cls.env_class = next(
            node for node in cls.env_ast.body
            if isinstance(node, ast.ClassDef) and node.name == "RedGymEnv"
        )
        cls.env_methods = {
            node.name: node for node in cls.env_class.body
            if isinstance(node, ast.FunctionDef)
        }
        cls.callback_methods = {
            node.name: node for node in cls.callback_ast.body
            if isinstance(node, ast.ClassDef)
            for node in node.body if isinstance(node, ast.FunctionDef)
        }

    def test_model_io_unchanged(self):
        init = self.env_methods["__init__"]
        for name, expected_hash in EXPECTED_MODEL_IO.items():
            self.assertEqual(assignment_hash(init, name), expected_hash, name)

    def test_hm_preferred_type_roles(self):
        constants = module_assignments(self.env_ast)
        self.assertEqual(constants["HM_PREFERRED_TYPES"], {
            "cut": frozenset({0x16}),
            "fly": frozenset({0x02}),
            "surf": frozenset({0x15}),
            "strength": frozenset({0x01, 0x05}),
        })

    def test_cut_and_surf_route_proxies(self):
        constants = module_assignments(self.env_ast)
        self.assertEqual(constants["CUT_GATE_MAP_IDS"], frozenset({0x14, 0x1B}))
        self.assertEqual(constants["SURF_ROUTE_MAP_IDS"], frozenset({0x1E, 0x1F, 0x20}))

    def test_strength_boulder_event_list(self):
        constants = module_assignments(self.env_ast)
        events = constants["STRENGTH_BOULDER_EVENT_INDICES"]
        self.assertEqual(len(events), 13)
        self.assertTrue({0x50E, 0x50F, 0x538, 0x53F, 0x660, 0x666, 0x917,
                         0x9C0, 0x9C1, 0x9C8, 0x9C9, 0x9D0, 0x9D1}.issubset(events))

    def test_step_updates_field_use(self):
        step = self.env_methods["step"]
        calls = {
            node.func.attr for node in ast.walk(step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        }
        self.assertIn("update_hm_progress", calls)
        self.assertIn("update_hm_field_use_progress", calls)

    def test_field_use_and_holder_rewards_exist(self):
        for key in (
            '"hm_nonstarter_candidates"', '"hm_role_candidates"',
            '"hm_nonstarter_taught"', '"hm_role_taught"',
            '"hm_preferred_carrier_ready"', '"hm_cut_field_use"',
            '"hm_surf_field_use"', '"hm_strength_field_use"',
        ):
            self.assertIn(key, self.env_source)

    def test_callback_logs_hm_metrics_and_holder_labels(self):
        self.assertIn("hm_scalar_keys", self.callback_source)
        self.assertIn("hm_strength_boulder_moves", self.callback_source)
        self.assertIn("hm_preferred_carrier_for", self.callback_source)
        self.assertIn('identity = "STARTER" if mon.get("is_starter", species_index in LEGACY_STARTER_INTERNAL_IDS) else "TEAMMATE"', self.callback_source)
        self.assertIn("cls._party_hm_text", self.callback_source)

    def test_capture_diagnostics_remain_in_place(self):
        # The patch must not replace or modify the existing capture logic.
        self.assertIn("def update_pokemon_progress(self):", self.env_source)
        self.assertIn("CAPTURED_MON_SPECIES_ADDRESS", self.env_source)
        self.assertIn("capture_pending_steps", self.env_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
