from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Stub runtime-only libraries so the RAM/reward helpers can be tested without
# launching PyBoy or installing the full training stack in this build sandbox.
gym = types.ModuleType('gymnasium')
class Env: pass
class Spaces: pass
gym.Env = Env
gym.spaces = Spaces()
sys.modules['gymnasium'] = gym

pyboy = types.ModuleType('pyboy')
class PyBoy: pass
pyboy.PyBoy = PyBoy
pyboy_utils = types.ModuleType('pyboy.utils')
class WindowEvent: pass
pyboy_utils.WindowEvent = WindowEvent
sys.modules['pyboy'] = pyboy
sys.modules['pyboy.utils'] = pyboy_utils

mediapy = types.ModuleType('mediapy')
sys.modules['mediapy'] = mediapy

global_map = types.ModuleType('global_map')
global_map.local_to_global = lambda *args: (0, 0)
global_map.GLOBAL_MAP_SHAPE = (48, 48)
sys.modules['global_map'] = global_map

module_path = Path(__file__).with_name('red_gym_env_v2.py')
spec = importlib.util.spec_from_file_location('red_gym_env_capture_test', module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class FakeEnv(mod.RedGymEnv):
    def __init__(self):
        self.memory = {}
        self.starter_species = ""
        self.base_owned_count = 0
        self.max_owned_count = 0
        self.last_owned_count = 0
        self.last_captured_species = 0
        self.last_owned_dex_numbers = set()
        self.pending_new_owned_dex_steps = {}
        self.caught_dex_numbers = set()
        self.novel_capture_count = 0
        self.duplicate_capture_count = 0
        self.unclassified_capture_count = 0
        self.duplicate_capture_penalty_weight = 10.0
        self.party_duplicate_penalty_weight = 4.0
        self.lead_hp_critical_threshold = 0.30
        self.lead_hp_danger_weight = 12.0
        self.critical_lead_hp_step_penalty_weight = 0.25
        self.lead_faint_penalty_weight = 10.0
        self.critical_lead_hp_steps = 0
        self.lead_faint_count = 0
        self.center_low_hp_arrival_reward_weight = 3.0
        self.center_heal_reward_weight = 4.0
        self.center_visit_count = 0
        self.center_low_hp_visits = 0
        self.center_heal_count = 0
        self.center_hp_restored = 0.0
        self.center_maps_rewarded_at_low_hp = set()
        self.last_lead_health = 1.0
        self.last_map_id_for_healing = 0
        self.critical_lead_hp_steps = 0
        self.lead_faint_count = 0
        self.capture_pending_steps = 0
        self.pending_dex_entries = 0
        self.pending_dex_steps = 0
        self.capture_count = 0
        self.new_dex_entries = 0
        self.lifetime_best_owned_count = 0
        self.base_party_size = 1
        self.max_party_size = 1
        self.base_unique_party_count = 1
        self.max_unique_party_count = 1
        self.base_pokeballs = 5
        self.max_pokeballs = 5
        self.ball_goal = 10
        self.reward_scale = 0.5
        self.capture_reward_weight = 16.0
        self.dex_reward_weight = 2.0
        self.party_reward_weight = 3.0
        self.party_size_reward_weight = 12.0
        self.party_diversity_weight = 1.5
        self.party_balance_weight = 2.5
        self.party_level_tolerance = 8
        self.ball_reward_weight = 0.5
        self.hm_candidate_reward_weight = 2.0
        self.hm_item_reward_weight = 0.75
        self.hm_taught_reward_weight = 1.5
        self.hm_ready_reward_weight = 2.5
        # Defaults added by the HM-awareness extension.
        self.hm_nonstarter_candidate_reward_weight = 0.75
        self.hm_role_candidate_reward_weight = 1.0
        self.hm_nonstarter_taught_reward_weight = 1.5
        self.hm_role_taught_reward_weight = 2.0
        self.hm_preferred_ready_reward_weight = 1.5
        self.hm_cut_field_use_reward_weight = 1.5
        self.hm_surf_field_use_reward_weight = 1.5
        self.hm_strength_field_use_reward_weight = 0.5
        self.hm_candidate_loss_penalty_weight = 1.0
        self.hm_ready_loss_penalty_weight = 0.5
        self.base_hm_candidate_score = 0.0
        self.base_hm_item_score = 0.0
        self.base_hm_taught_score = 0.0
        self.base_hm_ready_score = 0.0
        self.max_hm_candidate_score = 0.0
        self.max_hm_item_score = 0.0
        self.max_hm_taught_score = 0.0
        self.max_hm_ready_score = 0.0
        self.base_hm_nonstarter_candidate_score = 0.0
        self.base_hm_role_candidate_score = 0.0
        self.base_hm_nonstarter_taught_score = 0.0
        self.base_hm_role_taught_score = 0.0
        self.base_hm_preferred_ready_score = 0.0
        self.max_hm_nonstarter_candidate_score = 0.0
        self.max_hm_role_candidate_score = 0.0
        self.max_hm_nonstarter_taught_score = 0.0
        self.max_hm_role_taught_score = 0.0
        self.max_hm_preferred_ready_score = 0.0
        self.hm_cut_gate_entries = 0
        self.hm_surf_route_entries = 0
        self.hm_strength_boulder_moves = 0
        self.current_hm_status = {}
        self.current_hm_party_compatible_count = 0
        self.current_hm_items_owned_count = 0
        self.current_hm_moves_taught_count = 0
        self.current_hm_field_ready_count = 0
        self.current_hm_critical_party_compatible_count = 0
        self.current_hm_critical_items_owned_count = 0
        self.current_hm_critical_moves_taught_count = 0
        self.current_hm_critical_ready_count = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.total_healing_rew = 0
        self.explore_weight = 0.25
        self.map_progress_weight = 5.0
        self.stuck_penalty_weight = 1.0
        self.max_map_progress = 0
        self.seen_coords = {}
        self._event_cache = [0] * (mod.event_flags_end - mod.event_flags_start)

    def read_m(self, address):
        return self.memory.get(address, 0)

    def update_max_event_rew(self): return 0
    def get_levels_reward(self): return 0
    def get_badges(self): return 0
    def get_current_coord_count_reward(self): return 0


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_dex_count_masks_unused_bit():
    env = FakeEnv()
    env.memory[0xD2F7] = 0b00000011  # two valid species
    env.memory[0xD309] = 0b11000000  # #151 bit valid; #152 bit ignored
    check(env.read_pokedex_owned_count() == 3, 'Dex bit counting/mask failed')


def test_party_and_bag_parsing():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 3
    env.memory[0xD164] = 25
    env.memory[0xD165] = 4
    env.memory[0xD166] = 25
    check(env.read_unique_party_count() == 2, 'Unique party count failed')
    env.memory[mod.NUM_BAG_ITEMS_ADDRESS] = 3
    for idx, (item, qty) in enumerate([(4, 6), (0x14, 2), (3, 3)]):
        env.memory[mod.BAG_ITEMS_START + idx * 2] = item
        env.memory[mod.BAG_ITEMS_START + idx * 2 + 1] = qty
    check(env.read_pokeball_count() == 9, 'catch-ball inventory count failed')


def test_capture_marker_then_dex():
    env = FakeEnv()
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 25
    env.update_pokemon_progress()
    check(env.capture_count == 1,
          'Successful capture marker must count immediately, without waiting for a new Dex bit')
    check(env.capture_pending_steps > 0,
          'Capture marker should retain a short diagnostic window for a following Dex update')
    env.memory[0xD2F7] = 1
    env.update_pokemon_progress()
    check(env.capture_count == 1,
          'Dex update following an already-counted capture must not double-count it')


def test_duplicate_species_captures_are_counted():
    env = FakeEnv()
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 25
    env.update_pokemon_progress()
    check(env.capture_count == 1, 'First successful catch was not counted')
    # The game clears wCapturedMonSpecies when a ball is used. Observe that
    # reset before the same species is caught again; a new Dex bit is not needed.
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0
    env.update_pokemon_progress()
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 25
    env.update_pokemon_progress()
    check(env.capture_count == 2,
          'A repeat catch of an already-owned species should count as another capture')


def test_dex_then_capture_marker():
    env = FakeEnv()
    env.memory[0xD2F7] = 1
    env.update_pokemon_progress()
    check(env.pending_dex_entries == 1 and env.capture_count == 0,
          'Unmatched Dex entry should be held briefly')
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 4
    env.update_pokemon_progress()
    check(env.capture_count == 1, 'Capture was not counted when marker arrived after Dex bit')


def test_evolution_or_gift_is_not_mislabelled_without_capture_marker():
    env = FakeEnv()
    env.memory[0xD2F7] = 1
    env.update_pokemon_progress()
    check(env.capture_count == 0, 'Dex increase alone must not count as a capture')


def test_rewards_are_monotonic_and_capture_is_weighted():
    env = FakeEnv()
    env.capture_count = 1
    env.novel_capture_count = 1
    env.max_owned_count = 2
    env.max_unique_party_count = 2
    env.max_pokeballs = 10
    rewards = env.get_game_state_reward()
    check(rewards['capture'] == 8.0, 'Capture reward weight/scale incorrect')
    check(rewards['dex'] == 2.0, 'Dex reward incorrect')
    check(rewards['party'] == 1.5, 'Party reward incorrect')
    check(rewards['balls'] == 1.25, 'Ball supply reward incorrect')
    env.memory[mod.PARTY_COUNT_ADDRESS] = 3
    rewards = env.get_game_state_reward()
    check(rewards['party_size'] == 3.0, 'Party-size reward should be one point per active slot at reward_scale=0.5')



def test_geodude_first_catch_is_novel_then_repeat_is_duplicate():
    env = FakeEnv()
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0xA9  # Geodude, National Dex #74
    env.update_pokemon_progress()
    check(env.capture_count == 1, 'First Geodude capture was not counted')
    check(env.novel_capture_count == 1, 'First Geodude should be a novel capture')
    check(env.duplicate_capture_count == 0, 'First Geodude was misclassified as a duplicate')

    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0
    env.update_pokemon_progress()
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0xA9
    env.update_pokemon_progress()
    check(env.capture_count == 2, 'Repeat Geodude catch was not counted')
    check(env.novel_capture_count == 1, 'Repeat Geodude incorrectly earned novelty')
    check(env.duplicate_capture_count == 1, 'Repeat Geodude was not classified as duplicate')


def test_dex_bit_before_capture_marker_is_still_novel():
    env = FakeEnv()
    # National Dex #74 is zero-based bit 73: byte 9, bit 1.
    env.memory[mod.POKEDEX_OWNED_START + 9] = 1 << 1
    env.update_pokemon_progress()
    check(74 in env.pending_new_owned_dex_steps, 'New Geodude Dex bit was not queued')
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0xA9
    env.update_pokemon_progress()
    check(env.novel_capture_count == 1,
          'Dex bit arriving immediately before the capture marker should remain novel')
    check(env.duplicate_capture_count == 0, 'Delayed capture marker was misclassified as duplicate')


def test_already_owned_geodude_is_duplicate_capture():
    env = FakeEnv()
    env.memory[mod.POKEDEX_OWNED_START + 9] = 1 << 1
    env.last_owned_dex_numbers = {74}
    env.memory[mod.CAPTURED_MON_SPECIES_ADDRESS] = 0xA9
    env.update_pokemon_progress()
    check(env.novel_capture_count == 0, 'Already-owned Geodude got a novelty reward')
    check(env.duplicate_capture_count == 1, 'Already-owned Geodude did not count as duplicate')


def test_duplicate_party_slots_are_counted_and_penalized():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 3
    env.memory[mod.PARTY_SPECIES_START] = 0xA9
    env.memory[mod.PARTY_SPECIES_START + 1] = 0xA9
    env.memory[mod.PARTY_SPECIES_START + 2] = 0x02
    check(env.get_party_duplicate_count() == 1,
          'Two Geodude in a three-member party should count as one redundant slot')
    summary = env.get_party_duplicate_summary()
    check(summary == [{'species_index': 0xA9, 'species': 'Geodude', 'count': 2}],
          'Duplicate party summary did not identify Geodude')
    rewards = env.get_game_state_reward()
    check(rewards['party_duplicates'] == -2.0,
          'Duplicate party slot penalty weight/scale is incorrect')


def test_repeat_capture_reward_is_negative_and_not_reissued_as_novelty():
    env = FakeEnv()
    env.novel_capture_count = 1
    before = env.get_game_state_reward()
    env.duplicate_capture_count = 1
    after = env.get_game_state_reward()
    check(after['capture'] == before['capture'],
          'A duplicate catch must not increase the novelty reward')
    check(after['duplicate_capture'] == -5.0,
          'A repeat catch should receive the configured duplicate penalty')



def test_lead_hp_fraction_uses_first_party_member_not_party_average():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 2
    # Lead Charmander: 4/99 HP; second member is at full health.
    env.memory[mod.PARTY_MON_DATA_START + 1] = 0
    env.memory[mod.PARTY_MON_DATA_START + 2] = 4
    env.memory[mod.PARTY_MON_DATA_START + 0x22] = 0
    env.memory[mod.PARTY_MON_DATA_START + 0x23] = 99
    second = mod.PARTY_MON_DATA_START + mod.PARTY_MON_DATA_STRIDE
    env.memory[second + 1] = 0
    env.memory[second + 2] = 99
    env.memory[second + 0x22] = 0
    env.memory[second + 0x23] = 99
    check(abs(env.read_lead_hp_fraction() - 4 / 99) < 1e-9,
          'Lead HP awareness must not be masked by a healthy reserve Pokémon')
    check(env.get_lead_hp_danger_score() > 0.8,
          'Lead at 4/99 HP should register as critically weak')




def test_critical_hp_exposure_and_fainting_are_penalized():
    env = FakeEnv()
    safe = env.get_game_state_reward()
    env.critical_lead_hp_steps = 1
    exposed = env.get_game_state_reward()
    check(exposed["critical_hp_exposure"] == -0.125,
          'Each critical-HP step should add a bounded penalty at reward_scale=0.5')
    check(exposed["critical_hp_exposure"] < safe["critical_hp_exposure"],
          'Remaining at critical HP should reduce cumulative reward')
    env.lead_faint_count = 1
    fainted = env.get_game_state_reward()
    check(fainted["lead_faint"] == -5.0,
          'A lead faint should receive the configured one-time penalty')


def test_center_low_hp_arrival_is_rewarded_once_per_center():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    base = mod.PARTY_MON_DATA_START
    env.memory[base + 1] = 0
    env.memory[base + 2] = 20
    env.memory[base + 0x22] = 0
    env.memory[base + 0x23] = 100
    env.last_lead_health = 0.20
    env.last_map_id_for_healing = 0x0C  # Route 1
    env.memory[0xD35E] = 0x29  # Viridian Pokémon Center
    env.update_center_healing_metrics()
    check(env.center_visit_count == 1, 'Center entry was not detected')
    check(env.center_low_hp_visits == 1, 'Low-HP Center arrival was not rewarded')
    env.last_map_id_for_healing = 0x0C
    env.update_center_healing_metrics()
    env.last_map_id_for_healing = 0x29
    env.update_center_healing_metrics()
    check(env.center_low_hp_visits == 1,
          'Repeated door entry to the same Center must not farm the arrival reward')


def test_center_healing_records_actual_lead_hp_recovery():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    base = mod.PARTY_MON_DATA_START
    env.memory[base + 1] = 0
    env.memory[base + 2] = 20
    env.memory[base + 0x22] = 0
    env.memory[base + 0x23] = 100
    env.last_lead_health = 0.20
    env.last_map_id_for_healing = 0x29
    env.memory[0xD35E] = 0x29
    env.update_center_healing_metrics()
    env.memory[base + 2] = 100
    env.update_center_healing_metrics()
    check(env.center_heal_count == 1, 'Actual Center recovery was not detected')
    check(abs(env.center_hp_restored - 0.80) < 1e-9,
          'Restored HP fraction was not recorded accurately')


def test_center_heal_is_not_inferred_outside_a_center():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    base = mod.PARTY_MON_DATA_START
    env.memory[base + 1] = 0
    env.memory[base + 2] = 20
    env.memory[base + 0x22] = 0
    env.memory[base + 0x23] = 100
    env.last_lead_health = 0.20
    env.last_map_id_for_healing = 0x0C
    env.memory[0xD35E] = 0x0C
    env.update_center_healing_metrics()
    env.memory[base + 2] = 100
    env.update_center_healing_metrics()
    check(env.center_heal_count == 0,
          'A heal outside a verified Center must not count as a Center visit')


def test_hm_internal_species_compatibility_is_red_blue_specific():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 2
    # Charizard (B4) can learn Cut/Strength in Red/Blue, but Fly is Yellow-only.
    env.memory[mod.PARTY_SPECIES_START] = 0xB4
    env.memory[mod.PARTY_SPECIES_START + 1] = 0xFF
    env.memory[mod.PARTY_MON_DATA_START] = 0xB4
    # Diglett (3B) has no HM compatibility in the Red/Blue chart; it must not count for Cut.
    env.memory[mod.PARTY_SPECIES_START + 1] = 0x3B
    env.memory[mod.PARTY_MON_DATA_START + mod.PARTY_MON_DATA_STRIDE] = 0x3B
    status = env.read_hm_status()
    check(status['cut']['party_can_learn'], 'Charizard should be a Cut candidate')
    check(status['strength']['party_can_learn'], 'Charizard should be a Strength candidate')
    check(not status['fly']['party_can_learn'], 'Charizard must not be counted as a Fly candidate in Red/Blue')
    check(mod.HM_COMPATIBLE_INTERNAL_SPECIES['cut'].__contains__(0xB4), 'Charizard Cut compatibility missing')
    # Verify Diglett is not incorrectly treated as a Cut learner in Red/Blue.
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.memory[mod.PARTY_SPECIES_START] = 0x3B
    env.memory[mod.PARTY_MON_DATA_START] = 0x3B
    status = env.read_hm_status()
    check(not status['cut']['party_can_learn'], 'Diglett must not be counted as a Cut candidate')


def test_hm_party_moves_item_and_badge_are_separate_gates():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.memory[mod.PARTY_SPECIES_START] = 0x40  # Farfetch'd: Cut + Fly candidate
    env.memory[mod.PARTY_MON_DATA_START] = 0x40
    env.memory[mod.PARTY_MON_DATA_START + mod.PARTY_MON_MOVE_OFFSET] = 0x0F  # Cut
    env.memory[mod.NUM_BAG_ITEMS_ADDRESS] = 1
    env.memory[mod.BAG_ITEMS_START] = 0xC4  # HM01 Cut
    env.memory[mod.BAG_ITEMS_START + 1] = 1
    env.memory[mod.BADGES_ADDRESS] = 0  # no Cascade Badge yet
    status = env.read_hm_status()['cut']
    check(status['party_can_learn'], 'Farfetchd should be compatible with Cut')
    check(status['item_owned'], 'HM01 item should be detected in the bag')
    check(status['party_knows_move'], 'Cut move should be read from the party record')
    check(not status['badge_owned'], 'Cascade Badge should be required')
    check(not status['field_ready'], 'Cut must not be field-ready without Cascade Badge')
    env.memory[mod.BADGES_ADDRESS] = 1 << 1
    status = env.read_hm_status()['cut']
    check(status['field_ready'], 'Cut should be field-ready once all three gates are met')


def test_hm_party_data_move_slots_and_items():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.memory[mod.PARTY_SPECIES_START] = 0x4E  # Krabby can learn Cut/Surf/Strength
    base = mod.PARTY_MON_DATA_START
    env.memory[base] = 0x4E
    env.memory[base + mod.PARTY_MON_MOVE_OFFSET] = 0x39  # Surf
    env.memory[base + mod.PARTY_MON_MOVE_OFFSET + 1] = 0x46  # Strength
    env.memory[mod.NUM_BAG_ITEMS_ADDRESS] = 2
    env.memory[mod.BAG_ITEMS_START] = 0xC6  # HM03 Surf
    env.memory[mod.BAG_ITEMS_START + 1] = 1
    env.memory[mod.BAG_ITEMS_START + 2] = 0xC7  # HM04 Strength
    env.memory[mod.BAG_ITEMS_START + 3] = 1
    env.memory[mod.BADGES_ADDRESS] = (1 << 4) | (1 << 3)
    status = env.read_hm_status()
    check(status['surf']['party_can_learn'] and status['strength']['party_can_learn'], 'Krabby HM compatibility failed')
    check(status['surf']['party_knows_move'] and status['strength']['party_knows_move'], 'Move-slot parsing failed')
    check(status['surf']['field_ready'] and status['strength']['field_ready'], 'Surf/Strength readiness failed')


def test_non_obvious_red_blue_hm_compatibility_entries():
    # Compatibility table uses Red/Blue internal species IDs, not National Dex IDs.
    expected = {
        "surf": 0x0B,       # Lickitung
        "strength": 0x1D,   # Pinsir
        "strength_magmar": 0x33,
        "strength_electabuzz": 0x35,
        "flash": 0x68,      # Jolteon
    }
    check(expected["surf"] in mod.HM_COMPATIBLE_INTERNAL_SPECIES["surf"],
          'Lickitung should be able to learn Surf')
    check(expected["strength"] in mod.HM_COMPATIBLE_INTERNAL_SPECIES["strength"],
          'Pinsir should be able to learn Strength')
    check(expected["strength_magmar"] in mod.HM_COMPATIBLE_INTERNAL_SPECIES["strength"],
          'Magmar should be able to learn Strength')
    check(expected["strength_electabuzz"] in mod.HM_COMPATIBLE_INTERNAL_SPECIES["strength"],
          'Electabuzz should be able to learn Strength')
    check(expected["flash"] in mod.HM_COMPATIBLE_INTERNAL_SPECIES["flash"],
          'Jolteon should be able to learn Flash')



def test_hm_coverage_retention_penalizes_losing_only_party_access():
    env = FakeEnv()
    env.current_hm_status = {
        name: {
            "party_can_learn": False,
            "item_owned": False,
            "party_knows_move": False,
            "badge_owned": False,
            "field_ready": False,
            "nonstarter_can_learn": False,
            "role_can_learn": False,
            "nonstarter_knows_move": False,
            "role_knows_move": False,
            "preferred_carrier_knows_move": False,
            "preferred_field_ready": False,
            "compatible_party_species": (),
            "nonstarter_party_species": (),
            "role_party_species": (),
        }
        for name in mod.HM_DEFINITIONS
    }
    env.current_hm_status["cut"].update({"party_can_learn": False})
    env.max_hm_candidate_score = 2.0
    env.max_hm_ready_score = 2.0
    rewards = env.get_game_state_reward()
    check(rewards["hm_candidate_retention"] == -1.0,
          'Losing all previously achieved HM-compatible party coverage should be penalized')
    check(rewards["hm_ready_retention"] == -0.5,
          'Losing previously achieved HM field readiness should be penalized')



def test_hm_critical_completion_metrics():
    env = FakeEnv()
    env.current_hm_status = {
        name: {
            "party_can_learn": name in {"cut", "surf", "strength"},
            "item_owned": False,
            "party_knows_move": False,
            "badge_owned": False,
            "field_ready": False,
            "nonstarter_can_learn": False,
            "role_can_learn": False,
            "nonstarter_knows_move": False,
            "role_knows_move": False,
            "preferred_carrier_knows_move": False,
            "preferred_field_ready": False,
            "compatible_party_species": (),
            "nonstarter_party_species": (),
            "role_party_species": (),
        }
        for name in mod.HM_DEFINITIONS
    }
    env.current_hm_critical_party_compatible_count = 3
    env.current_hm_critical_ready_count = 2
    env._refresh_hm_counts = lambda: None
    metrics = env.get_hm_info_metrics()
    check(metrics["hm_critical_candidates_complete"] == 1,
          "All three story-critical HM candidate flags should be reported")
    check(metrics["hm_critical_ready_complete"] == 0,
          "Critical HM readiness must remain incomplete until all three are usable")
    env.current_hm_status["strength"]["field_ready"] = True
    env.current_hm_critical_ready_count = 3
    metrics = env.get_hm_info_metrics()
    check(metrics["hm_critical_ready_complete"] == 1,
          "Critical HM readiness should report complete at 3/3")


def test_hm_progress_uses_high_water_scores():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.memory[mod.PARTY_SPECIES_START] = 0x4E  # Krabby gives Cut, Surf, Strength coverage
    env.memory[mod.PARTY_MON_DATA_START] = 0x4E
    env.reset_hm_progress_baseline()
    baseline = env.max_hm_candidate_score
    env.memory[mod.PARTY_COUNT_ADDRESS] = 2
    env.memory[mod.PARTY_SPECIES_START + 1] = 0x24  # Pidgey adds Fly
    env.memory[mod.PARTY_MON_DATA_START + mod.PARTY_MON_DATA_STRIDE] = 0x24
    env.update_hm_progress()
    check(env.max_hm_candidate_score > baseline, 'New HM-compatible species should increase candidate coverage')
    high = env.max_hm_candidate_score
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.update_hm_progress()
    check(env.max_hm_candidate_score == high, 'HM candidate reward score must be a high-water mark')




def test_party_snapshot_includes_names_level_hp_moves_and_hm_roles():
    env = FakeEnv()
    env.memory[mod.PARTY_COUNT_ADDRESS] = 1
    env.memory[mod.PARTY_SPECIES_START] = 0xB1  # Squirtle
    base = mod.PARTY_MON_DATA_START
    env.memory[base] = 0xB1
    env.memory[base + 0x01] = 0
    env.memory[base + 0x02] = 25
    env.memory[base + 0x04] = 0
    env.memory[base + 0x08] = 0x39  # Surf
    env.memory[base + 0x21] = 12
    env.memory[base + 0x22] = 0
    env.memory[base + 0x23] = 30
    party = env.read_party_snapshot()
    check(len(party) == 1, 'Party snapshot should contain the active party member')
    mon = party[0]
    check(mon['species'] == 'Squirtle', 'Internal species index should resolve to Squirtle')
    check(mon['level'] == 12 and mon['hp'] == 25 and mon['max_hp'] == 30,
          'Party level/HP fields were parsed incorrectly')
    check('Surf' in mon['moves'], 'HM moves should have readable labels')
    check('SURF' in mon['hm_can_learn'], 'Squirtle should be marked as a Surf learner')
    check('SURF' in mon['hm_knows'], 'Squirtle should be marked as knowing Surf')


def test_generation_one_move_name_table_is_complete():
    check(len(mod.MOVE_NAMES) == 0xA5, 'Move-name lookup must cover Gen-I move IDs 0x01..0xA5')
    check(mod.MOVE_NAMES[0x0F] == 'Cut', 'Cut move label incorrect')
    check(mod.MOVE_NAMES[0x13] == 'Fly', 'Fly move label incorrect')
    check(mod.MOVE_NAMES[0x39] == 'Surf', 'Surf move label incorrect')
    check(mod.MOVE_NAMES[0x46] == 'Strength', 'Strength move label incorrect')
    check(mod.MOVE_NAMES[0x94] == 'Flash', 'Flash move label incorrect')


def test_internal_species_name_map_known_anchors():
    check(mod.INTERNAL_SPECIES_NAMES[0x01] == 'Rhydon', 'Internal index 0x01 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0x15] == 'Mew', 'Internal index 0x15 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0x70] == 'Weedle', 'Internal index 0x70 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0x85] == 'Magikarp', 'Internal index 0x85 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0x99] == 'Bulbasaur', 'Internal index 0x99 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0xB0] == 'Charmander', 'Internal index 0xB0 name anchor failed')
    check(mod.INTERNAL_SPECIES_NAMES[0xB1] == 'Squirtle', 'Internal index 0xB1 name anchor failed')


for test in (
    test_dex_count_masks_unused_bit,
    test_party_and_bag_parsing,
    test_capture_marker_then_dex,
    test_dex_then_capture_marker,
    test_evolution_or_gift_is_not_mislabelled_without_capture_marker,
    test_rewards_are_monotonic_and_capture_is_weighted,
    test_geodude_first_catch_is_novel_then_repeat_is_duplicate,
    test_dex_bit_before_capture_marker_is_still_novel,
    test_already_owned_geodude_is_duplicate_capture,
    test_duplicate_party_slots_are_counted_and_penalized,
    test_repeat_capture_reward_is_negative_and_not_reissued_as_novelty,
    test_lead_hp_fraction_uses_first_party_member_not_party_average,
    test_critical_hp_exposure_and_fainting_are_penalized,
    test_center_low_hp_arrival_is_rewarded_once_per_center,
    test_center_healing_records_actual_lead_hp_recovery,
    test_center_heal_is_not_inferred_outside_a_center,
    test_hm_internal_species_compatibility_is_red_blue_specific,
    test_hm_party_moves_item_and_badge_are_separate_gates,
    test_hm_party_data_move_slots_and_items,
    test_non_obvious_red_blue_hm_compatibility_entries,
    test_hm_coverage_retention_penalizes_losing_only_party_access,
    test_hm_critical_completion_metrics,
    test_hm_progress_uses_high_water_scores,
    test_party_snapshot_includes_names_level_hp_moves_and_hm_roles,
    test_internal_species_name_map_known_anchors,
    test_generation_one_move_name_table_is_complete,
):
    test()
    print('PASS', test.__name__)
print('All capture-aware reward tests passed.')

