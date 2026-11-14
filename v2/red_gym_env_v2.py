import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE
from starter_diversity import StarterDiversityTracker
from gen1_party_metadata import INTERNAL_TO_NATIONAL_DEX

event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)

# Pokémon Red/Blue WRAM fields (English international layout).
PARTY_COUNT_ADDRESS = 0xD163
PARTY_SPECIES_START = 0xD164
POKEDEX_OWNED_START = 0xD2F7
POKEDEX_OWNED_END = 0xD30A  # exclusive; 19 bytes, 151 valid Pokédex bits
NUM_BAG_ITEMS_ADDRESS = 0xD31D
BAG_ITEMS_START = 0xD31E  # item ID, quantity pairs
BAG_ITEM_CAPACITY = 20
# Master, Ultra, Great, Poké, and Safari Balls in the English Red/Blue item table.
CATCH_BALL_ITEM_IDS = frozenset({0x01, 0x02, 0x03, 0x04})
CAPTURED_MON_SPECIES_ADDRESS = 0xD11C

# English Red/Blue map IDs for interiors whose nurse script heals the party.
# The Indigo Plateau lobby is included because its nurse runs the same Center script.
POKEMON_CENTER_MAP_NAMES = {
    0x29: "Viridian Pokémon Center",
    0x3A: "Pewter Pokémon Center",
    0x40: "Cerulean Pokémon Center",
    0x44: "Mt. Moon Pokémon Center",
    0x51: "Rock Tunnel Pokémon Center",
    0x59: "Vermilion Pokémon Center",
    0x85: "Celadon Pokémon Center",
    0x8D: "Lavender Pokémon Center",
    0x9A: "Fuchsia Pokémon Center",
    0xAB: "Cinnabar Pokémon Center",
    0xAE: "Indigo Plateau Pokémon Center",
    0xB6: "Saffron Pokémon Center",
}
POKEMON_CENTER_MAP_IDS = frozenset(POKEMON_CENTER_MAP_NAMES)

# Generation I Red/Blue party records: six 44-byte structures follow the
# party species list. Each structure stores four move IDs at offsets 0x08-0x0B.
PARTY_MON_DATA_START = 0xD16B
PARTY_MON_DATA_STRIDE = 0x2C
PARTY_MON_MOVE_OFFSET = 0x08
PARTY_MON_MOVE_SLOTS = 4
PARTY_MON_TYPE1_OFFSET = 0x05
PARTY_MON_TYPE2_OFFSET = 0x06
BADGES_ADDRESS = 0xD356

# Gen-I type IDs used by the party data structures (type1/type2).
TYPE_NAMES_BY_ID = {
    0x01: "Fighting",
    0x02: "Flying",
    0x05: "Rock",
    0x15: "Water",
    0x16: "Grass",
    0x03: "Poison",
    0x04: "Ground",
    0x06: "Bird",
    0x07: "Bug",
    0x08: "Ghost",
    0x14: "Fire",
    0x17: "Electric",
    0x18: "Psychic",
    0x19: "Ice",
    0x1A: "Dragon",
    0x00: "Normal",
}

# Preferred HM carrier roles. Compatibility remains governed by the game's
# species/HM compatibility table; type is a preference among legal carriers.
HM_PREFERRED_TYPES = {
    "cut": frozenset({0x16}),             # Grass
    "fly": frozenset({0x02}),             # Flying
    "surf": frozenset({0x15}),            # Water
    "strength": frozenset({0x01, 0x05}),  # Fighting or Rock
}

STARTER_LINEAGES_BY_NAME = {
    "Bulbasaur": frozenset({0x99, 0x09, 0x9A}),
    "Charmander": frozenset({0xB0, 0xB2, 0xB4}),
    "Squirtle": frozenset({0xB1, 0xB3, 0x1C}),
}

# A first entry into these outdoor route maps is a practical progression proxy
# for passing the corresponding field obstacle. It is deliberately logged as
# a route-entry proxy, not as a claim that every Cut/Surf animation was seen.
CUT_GATE_MAP_IDS = frozenset({0x14, 0x1B})       # Route 9 and Route 16
SURF_ROUTE_MAP_IDS = frozenset({0x1E, 0x1F, 0x20})  # Routes 19, 20 and 21

# Gen-I event-flag indices set by pushing specific Seafoam Islands / Victory
# Road boulders onto switches. These are direct evidence of Strength puzzle use.
STRENGTH_BOULDER_EVENT_INDICES = frozenset({
    0x50E, 0x50F,
    0x538, 0x53F,
    0x660, 0x666,
    0x917,
    0x9C0, 0x9C1,
    0x9C8, 0x9C9,
    0x9D0, 0x9D1,
})

# Red/Blue HM item IDs and move IDs from pret/pokered constants. HMs are
# reusable items, so possession remains separate from the move being taught.
HM_DEFINITIONS = {
    "cut":      {"item_id": 0xC4, "move_id": 0x0F, "badge_mask": 1 << 1, "badge": "Cascade", "required": True,  "weight": 2.0},
    "fly":      {"item_id": 0xC5, "move_id": 0x13, "badge_mask": 1 << 2, "badge": "Thunder", "required": False, "weight": 0.75},
    "surf":     {"item_id": 0xC6, "move_id": 0x39, "badge_mask": 1 << 4, "badge": "Soul", "required": True,  "weight": 3.0},
    "strength": {"item_id": 0xC7, "move_id": 0x46, "badge_mask": 1 << 3, "badge": "Rainbow", "required": True,  "weight": 2.5},
    "flash":    {"item_id": 0xC8, "move_id": 0x94, "badge_mask": 1 << 0, "badge": "Boulder", "required": False, "weight": 0.5},
}

# Pokémon species compatibility for the English Red/Blue release. These are
# the game's internal species indexes (the values stored at D164-D169 and in
# each party record), NOT National Pokédex numbers. Yellow-only compatibility
# entries are intentionally excluded (e.g. Charizard-Fly, # Butterfree-Flash, and Kabutops-Cut).
HM_COMPATIBLE_INTERNAL_SPECIES = {
    "cut": frozenset({
        0x99, 0x09, 0x9A, 0xB0, 0xB2, 0xB4, 0x72, 0x60, 0x61, 0xB9, 0xBA, 0xBB,
        0x6D, 0x2E, 0xBC, 0xBD, 0xBE, 0x18, 0x9B, 0x40,
        0x4E, 0x8A, 0x0B, 0x1E, 0x1A, 0x1D, 0x15,
    }),
    "fly": frozenset({
        0x24, 0x96, 0x97, 0x05, 0x23, 0x40, 0x46, 0x74,
        0xAB, 0x4A, 0x4B, 0x49, 0x15,
    }),
    "surf": frozenset({
        0xB1, 0xB3, 0x1C, 0x10, 0x07, 0x2F, 0x80, 0x47,
        0x6E, 0x6F, 0x18, 0x9B, 0x25, 0x08, 0x3A, 0x78,
        0x17, 0x8B, 0x4E, 0x8A, 0x0B, 0x01, 0x02, 0x5C, 0x5D,
        0x9D, 0x9E, 0x1B, 0x98, 0x16, 0x13, 0x69, 0x62,
        0x63, 0x5A, 0x5B, 0x84, 0x58, 0x59, 0x42, 0x15,
    }),
    "strength": frozenset({
        0xB0, 0xB2, 0xB4, 0xB1, 0xB3, 0x1C, 0x6C, 0x2D,
        0x60, 0x61, 0x10, 0x07, 0x04, 0x8E, 0x64, 0x65,
        0x2F, 0x80, 0x39, 0x75, 0x6E, 0x6F, 0x6A, 0x29,
        0x7E, 0xA9, 0x27, 0x31, 0x25, 0x08, 0x3A, 0x78,
        0x0E, 0x22, 0x4E, 0x8A, 0x0A, 0x11, 0x91, 0x2B,
        0x2C, 0x0B, 0x12, 0x01, 0x28, 0x02, 0x3C, 0x16,
        0x13, 0x42, 0x83, 0x1D, 0x33, 0x35, 0x15,
    }),
    "flash": frozenset({
        0x54, 0x55, 0x04, 0x8E, 0x64, 0x65, 0x94, 0x26,
        0x95, 0x25, 0x08, 0xAD, 0x36, 0x30, 0x81, 0x06,
        0x8D, 0x28, 0x1B, 0x98, 0x2A, 0x35, 0x68, 0xAA, 0x4B,
        0x83, 0x15,
    }),
}

# Weight story-critical HM capability more heavily than optional travel aids.
HM_CRITICAL_NAMES = frozenset({"cut", "surf", "strength"})

# Gen-I Red/Blue internal species-index names (1..190); not National Dex IDs.
STARTER_SPECIES_BY_INTERNAL_ID = {
    0x99: "Bulbasaur",
    0xB0: "Charmander",
    0xB1: "Squirtle",
}

INTERNAL_SPECIES_NAMES = {
    0x01: 'Rhydon',
    0x02: 'Kangaskhan',
    0x03: 'Nidoran♂',
    0x04: 'Clefairy',
    0x05: 'Spearow',
    0x06: 'Voltorb',
    0x07: 'Nidoking',
    0x08: 'Slowbro',
    0x09: 'Ivysaur',
    0x0A: 'Exeggutor',
    0x0B: 'Lickitung',
    0x0C: 'Exeggcute',
    0x0D: 'Grimer',
    0x0E: 'Gengar',
    0x0F: 'Nidoran♀',
    0x10: 'Nidoqueen',
    0x11: 'Cubone',
    0x12: 'Rhyhorn',
    0x13: 'Lapras',
    0x14: 'Arcanine',
    0x15: 'Mew',
    0x16: 'Gyarados',
    0x17: 'Shellder',
    0x18: 'Tentacool',
    0x19: 'Gastly',
    0x1A: 'Scyther',
    0x1B: 'Staryu',
    0x1C: 'Blastoise',
    0x1D: 'Pinsir',
    0x1E: 'Tangela',
    0x1F: 'MissingNo.',
    0x20: 'MissingNo.',
    0x21: 'Growlithe',
    0x22: 'Onix',
    0x23: 'Fearow',
    0x24: 'Pidgey',
    0x25: 'Slowpoke',
    0x26: 'Kadabra',
    0x27: 'Graveler',
    0x28: 'Chansey',
    0x29: 'Machoke',
    0x2A: 'Mr. Mime',
    0x2B: 'Hitmonlee',
    0x2C: 'Hitmonchan',
    0x2D: 'Arbok',
    0x2E: 'Parasect',
    0x2F: 'Psyduck',
    0x30: 'Drowzee',
    0x31: 'Golem',
    0x32: 'MissingNo.',
    0x33: 'Magmar',
    0x34: 'MissingNo.',
    0x35: 'Electabuzz',
    0x36: 'Magneton',
    0x37: 'Koffing',
    0x38: 'MissingNo.',
    0x39: 'Mankey',
    0x3A: 'Seel',
    0x3B: 'Diglett',
    0x3C: 'Tauros',
    0x3D: 'MissingNo.',
    0x3E: 'MissingNo.',
    0x3F: 'MissingNo.',
    0x40: "Farfetch'd",
    0x41: 'Venonat',
    0x42: 'Dragonite',
    0x43: 'MissingNo.',
    0x44: 'MissingNo.',
    0x45: 'MissingNo.',
    0x46: 'Doduo',
    0x47: 'Poliwag',
    0x48: 'Jynx',
    0x49: 'Moltres',
    0x4A: 'Articuno',
    0x4B: 'Zapdos',
    0x4C: 'Ditto',
    0x4D: 'Meowth',
    0x4E: 'Krabby',
    0x4F: 'MissingNo.',
    0x50: 'MissingNo.',
    0x51: 'MissingNo.',
    0x52: 'Vulpix',
    0x53: 'Ninetales',
    0x54: 'Pikachu',
    0x55: 'Raichu',
    0x56: 'MissingNo.',
    0x57: 'MissingNo.',
    0x58: 'Dratini',
    0x59: 'Dragonair',
    0x5A: 'Kabuto',
    0x5B: 'Kabutops',
    0x5C: 'Horsea',
    0x5D: 'Seadra',
    0x5E: 'MissingNo.',
    0x5F: 'MissingNo.',
    0x60: 'Sandshrew',
    0x61: 'Sandslash',
    0x62: 'Omanyte',
    0x63: 'Omastar',
    0x64: 'Jigglypuff',
    0x65: 'Wigglytuff',
    0x66: 'Eevee',
    0x67: 'Flareon',
    0x68: 'Jolteon',
    0x69: 'Vaporeon',
    0x6A: 'Machop',
    0x6B: 'Zubat',
    0x6C: 'Ekans',
    0x6D: 'Paras',
    0x6E: 'Poliwhirl',
    0x6F: 'Poliwrath',
    0x70: 'Weedle',
    0x71: 'Kakuna',
    0x72: 'Beedrill',
    0x73: 'MissingNo.',
    0x74: 'Dodrio',
    0x75: 'Primeape',
    0x76: 'Dugtrio',
    0x77: 'Venomoth',
    0x78: 'Dewgong',
    0x79: 'MissingNo.',
    0x7A: 'MissingNo.',
    0x7B: 'Caterpie',
    0x7C: 'Metapod',
    0x7D: 'Butterfree',
    0x7E: 'Machamp',
    0x7F: 'MissingNo.',
    0x80: 'Golduck',
    0x81: 'Hypno',
    0x82: 'Golbat',
    0x83: 'Mewtwo',
    0x84: 'Snorlax',
    0x85: 'Magikarp',
    0x86: 'MissingNo.',
    0x87: 'MissingNo.',
    0x88: 'Muk',
    0x89: 'MissingNo.',
    0x8A: 'Kingler',
    0x8B: 'Cloyster',
    0x8C: 'MissingNo.',
    0x8D: 'Electrode',
    0x8E: 'Clefable',
    0x8F: 'Weezing',
    0x90: 'Persian',
    0x91: 'Marowak',
    0x92: 'MissingNo.',
    0x93: 'Haunter',
    0x94: 'Abra',
    0x95: 'Alakazam',
    0x96: 'Pidgeotto',
    0x97: 'Pidgeot',
    0x98: 'Starmie',
    0x99: 'Bulbasaur',
    0x9A: 'Venusaur',
    0x9B: 'Tentacruel',
    0x9C: 'MissingNo.',
    0x9D: 'Goldeen',
    0x9E: 'Seaking',
    0x9F: 'MissingNo.',
    0xA0: 'MissingNo.',
    0xA1: 'MissingNo.',
    0xA2: 'MissingNo.',
    0xA3: 'Ponyta',
    0xA4: 'Rapidash',
    0xA5: 'Rattata',
    0xA6: 'Raticate',
    0xA7: 'Nidorino',
    0xA8: 'Nidorina',
    0xA9: 'Geodude',
    0xAA: 'Porygon',
    0xAB: 'Aerodactyl',
    0xAC: 'MissingNo.',
    0xAD: 'Magnemite',
    0xAE: 'MissingNo.',
    0xAF: 'MissingNo.',
    0xB0: 'Charmander',
    0xB1: 'Squirtle',
    0xB2: 'Charmeleon',
    0xB3: 'Wartortle',
    0xB4: 'Charizard',
    0xB5: 'MissingNo.',
    0xB6: 'MissingNo.',
    0xB7: 'MissingNo.',
    0xB8: 'MissingNo.',
    0xB9: 'Oddish',
    0xBA: 'Gloom',
    0xBB: 'Vileplume',
    0xBC: 'Bellsprout',
    0xBD: 'Weepinbell',
    0xBE: 'Victreebel',
}



# Generation-I Red/Blue move IDs (Data Crystal's Red/Blue notes table).
_MOVE_NAME_SEQUENCE = (
    "Pound", "Karate Chop", "Double Slap", "Comet Punch", "Mega Punch", "Pay Day", "Fire Punch", "Ice Punch", "Thunderpunch", "Scratch", "ViceGrip", "Guillotine", "Razor Wind", "Swords Dance", "Cut", "Gust", "Wing Attack", "Whirlwind", "Fly", "Bind", "Slam", "Vine Whip", "Stomp", "Double Kick", "Mega Kick", "Jump Kick", "Rolling Kick", "Sand-Attack", "Headbutt", "Horn Attack", "Fury Attack", "Horn Drill", "Tackle", "Body Slam", "Wrap", "Take Down", "Thrash", "Double-Edge", "Tail Whip", "Poison Sting", "Twineedle", "Pin Missile", "Leer", "Bite", "Growl", "Roar", "Sing", "Supersonic", "Sonicboom", "Disable", "Acid", "Ember", "Flamethrower", "Mist", "Water Gun", "Hydro Pump", "Surf", "Ice Beam", "Blizzard", "Psybeam", "Bubblebeam", "Aurora Beam", "Hyper Beam", "Peck", "Drill Peck", "Submission", "Low Kick", "Counter", "Seismic Toss", "Strength", "Absorb", "Mega Drain", "Leech Seed", "Growth", "Razor Leaf", "Solarbeam", "Poison Powder", "Stun Spore", "Sleep Powder", "Petal Dance", "String Shot", "Dragon Rage", "Fire Spin", "Thundershock", "Thunderbolt", "Thunder Wave", "Thunder", "Rock Throw", "Earthquake", "Fissure", "Dig", "Toxic", "Confusion", "Psychic", "Hypnosis", "Meditate", "Agility", "Quick Attack", "Rage", "Teleport", "Night Shade", "Mimic", "Screech", "Double Team", "Recover", "Harden", "Minimize", "Smokescreen", "Confuse Ray", "Withdraw", "Defense Curl", "Barrier", "Light Screen", "Haze", "Reflect", "Focus Energy", "Bide", "Metronome", "Mirror Move", "Selfdestruct", "Egg Bomb", "Lick", "Smog", "Sludge", "Bone Club", "Fire Blast", "Waterfall", "Clamp", "Swift", "Skull Bash", "Spike Cannon", "Constrict", "Amnesia", "Kinesis", "Softboiled", "Hi Jump Kick", "Glare", "Dream Eater", "Poison Gas", "Barrage", "Leech Life", "Lovely Kiss", "Sky Attack", "Transform", "Bubble", "Dizzy Punch", "Spore", "Flash", "Psywave", "Splash", "Acid Armor", "Crabhammer", "Explosion", "Fury Swipes", "Bonemerang", "Rest", "Rock Slide", "Hyper Fang", "Sharpen", "Conversion", "Tri Attack", "Super Fang", "Slash", "Substitute", "Struggle",
)
MOVE_NAMES = {move_id: name for move_id, name in enumerate(_MOVE_NAME_SEQUENCE, start=1)}

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        # Reward for reaching the next important story map. This setting was
        # already present in the original environment but was never used.
        self.map_progress_weight = float(config.get("map_progress_weight", 5.0))
        # A small shaping penalty encourages the agent to leave tiles where it
        # has been looping for a long time.
        self.stuck_penalty_weight = float(config.get("stuck_penalty_weight", 1.0))

        # Pokémon-specific objectives. These are reward-side additions only:
        # the observation keys and shapes stay unchanged for checkpoint compatibility.
        self.capture_reward_weight = float(config.get("capture_reward_weight", 32.0))
        self.dex_reward_weight = float(config.get("dex_reward_weight", 2.0))
        self.party_reward_weight = float(config.get("party_reward_weight", 8.0))
        # Novel captures and repeat captures are rewarded separately. These
        # additions do not change the checkpoint-facing observation/action spaces.
        self.duplicate_capture_penalty_weight = float(
            config.get("duplicate_capture_penalty_weight", 10.0)
        )
        self.party_duplicate_penalty_weight = float(
            config.get("party_duplicate_penalty_weight", 4.0)
        )
        if min(self.duplicate_capture_penalty_weight,
               self.party_duplicate_penalty_weight) < 0:
            raise ValueError("Duplicate-capture/party penalty weights must be >= 0")
        # Focus health awareness on the lead/starter instead of a party average
        # that can hide one critically weak member behind healthy reserves.
        self.lead_hp_critical_threshold = float(config.get("lead_hp_critical_threshold", 0.30))
        self.lead_hp_danger_weight = float(config.get("lead_hp_danger_weight", 12.0))
        self.critical_lead_hp_step_penalty_weight = float(
            config.get("critical_lead_hp_step_penalty_weight", 0.25)
        )
        self.lead_faint_penalty_weight = float(config.get("lead_faint_penalty_weight", 10.0))
        self.center_low_hp_arrival_reward_weight = float(
            config.get("center_low_hp_arrival_reward_weight", 3.0)
        )
        self.center_heal_reward_weight = float(config.get("center_heal_reward_weight", 4.0))
        if not 0.0 < self.lead_hp_critical_threshold < 1.0:
            raise ValueError("lead_hp_critical_threshold must be between 0 and 1")
        if min(self.lead_hp_danger_weight,
               self.critical_lead_hp_step_penalty_weight,
               self.lead_faint_penalty_weight,
               self.center_low_hp_arrival_reward_weight,
               self.center_heal_reward_weight) < 0:
            raise ValueError("Health/Center reward weights must be >= 0")
        # Reward keeping additional Pokémon in the active party. This is a
        # current-state potential (party_size / 6), so adding a teammate earns
        # reward and depositing one later loses that potential. It complements
        # the distinct-species high-water reward below.
        self.party_size_reward_weight = float(config.get("party_size_reward_weight", 12.0))
        # Current-state shaping rewards a diverse active party and keeps its
        # members within a reasonable level band. These are reward-only changes;
        # policy observations and action spaces remain unchanged.
        self.party_diversity_weight = float(config.get("party_diversity_weight", 2.5))
        self.party_balance_weight = float(config.get("party_balance_weight", 10.0))
        self.party_level_tolerance = int(config.get("party_level_tolerance", 10))
        if min(self.party_size_reward_weight, self.party_diversity_weight, self.party_balance_weight) < 0:
            raise ValueError("Party size/diversity/balance reward weights must be >= 0")
        # Shared population-level reward: whichever starter is currently least
        # represented across workers/runs receives the largest one-time bonus.
        # This changes rewards only; the policy still chooses every action.
        self.starter_diversity_weight = float(
            config.get("starter_diversity_weight", 40.0)
        )
        if self.starter_diversity_weight < 0:
            raise ValueError("starter_diversity_weight must be >= 0")
        if self.party_level_tolerance < 0:
            raise ValueError("party_level_tolerance must be >= 0")
        self.ball_reward_weight = float(config.get("ball_reward_weight", 4.0))
        self.ball_goal = int(config.get("ball_goal", 10))
        # HM planning is shaped as a progression ladder: have a compatible
        # party member, acquire the HM, teach the move, then meet the badge gate.
        self.hm_candidate_reward_weight = float(config.get("hm_candidate_reward_weight", 2.0))
        self.hm_item_reward_weight = float(config.get("hm_item_reward_weight", 0.75))
        self.hm_taught_reward_weight = float(config.get("hm_taught_reward_weight", 1.5))
        self.hm_ready_reward_weight = float(config.get("hm_ready_reward_weight", 2.5))
        # Prefer teaching HMs to a non-starter, and prefer a carrier whose type
        # matches the role (Grass/Cut, Flying/Fly, Water/Surf, Fighting-or-Rock/Strength).
        self.hm_nonstarter_candidate_reward_weight = float(
            config.get("hm_nonstarter_candidate_reward_weight", 0.75)
        )
        self.hm_role_candidate_reward_weight = float(
            config.get("hm_role_candidate_reward_weight", 1.0)
        )
        self.hm_nonstarter_taught_reward_weight = float(
            config.get("hm_nonstarter_taught_reward_weight", 1.5)
        )
        self.hm_role_taught_reward_weight = float(
            config.get("hm_role_taught_reward_weight", 2.0)
        )
        self.hm_preferred_ready_reward_weight = float(
            config.get("hm_preferred_ready_reward_weight", 1.5)
        )
        # Reward verified HM field progression. Cut/Surf use route-entry proxies;
        # Strength uses the game's actual boulder-switch event flags.
        self.hm_cut_field_use_reward_weight = float(
            config.get("hm_cut_field_use_reward_weight", 1.5)
        )
        self.hm_surf_field_use_reward_weight = float(
            config.get("hm_surf_field_use_reward_weight", 1.5)
        )
        self.hm_strength_field_use_reward_weight = float(
            config.get("hm_strength_field_use_reward_weight", 0.5)
        )
        # Once the policy has built HM coverage, discourage it from depositing
        # the only compatible/ready party member and losing field access.
        self.hm_candidate_loss_penalty_weight = float(
            config.get("hm_candidate_loss_penalty_weight", 1.0)
        )
        self.hm_ready_loss_penalty_weight = float(
            config.get("hm_ready_loss_penalty_weight", 0.5)
        )
        if self.hm_candidate_loss_penalty_weight < 0 or self.hm_ready_loss_penalty_weight < 0:
            raise ValueError("HM coverage-loss penalty weights must be >= 0")
        if self.ball_goal < 0:
            raise ValueError("ball_goal must be >= 0")

        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.starter_diversity_tracker = StarterDiversityTracker(
            self.s_path / "starter_choice_counts.sqlite3",
            reward_weight=self.starter_diversity_weight,
        )
        self.starter_choice_counts = self.starter_diversity_tracker.get_counts()
        self._starter_choice_recorded = False
        self._warned_preselected_starter_state = False
        self.starter_species = None
        self.starter_diversity_reward = 0.0
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Episode maps are reset on every Gymnasium reset, but these lifetime
        # trackers deliberately survive those resets within this worker.
        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.lifetime_explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)
        self.lifetime_seen_coords = set()
        self.lifetime_event_flags = {}
        self.best_map_progress = -1
        self.best_map_id = -1
        self.episodes_completed = 0
        self.lifetime_best_owned_count = 0

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "level": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self.enc_freqs,), dtype=np.float32
                ),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        head = "null" if config["headless"] else "SDL2"

        #log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            #debugging=False,
            #disable_input=False,
            window=head, sound_emulated=False
        )

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options=None):
        # Gymnasium uses this to initialize its RNG and keep reset API compliant.
        super().reset(seed=seed)
        self.seed = seed
        options = options or {}
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)
        # Do not clear lifetime_explore_map / lifetime_seen_coords here. SB3
        # automatically resets completed workers, and these trackers preserve
        # exploration history across those episode boundaries.

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self._starter_choice_recorded = False
        self.starter_species = None
        self.starter_diversity_reward = 0.0

        # Read event memory once. The same cached bytes are used by the reward
        # calculation and observation, avoiding two full RAM scans per step.
        self.refresh_event_cache()
        self.base_event_flags = sum(
            self.bit_count(int(value)) for value in self._event_cache
        )

        self.current_event_flags_set = {}
        self.out_of_bounds_coords = 0

        # Baseline Pokémon state comes from the loaded save-state. Progress is
        # measured relative to this baseline, so pre-existing catches are not
        # rewarded again every time an episode starts.
        self.base_owned_count = self.read_pokedex_owned_count()
        self.max_owned_count = self.base_owned_count
        self.last_owned_count = self.base_owned_count
        self.base_party_size = min(max(self.read_m(PARTY_COUNT_ADDRESS), 0), 6)
        self.max_party_size = self.base_party_size
        if self.base_party_size > 0:
            initial_species = int(self.read_m(PARTY_SPECIES_START))
            initial_starter = STARTER_SPECIES_BY_INTERNAL_ID.get(initial_species)
            if initial_starter is not None:
                # If the configured state already contains a starter, it was
                # not selected by this episode's policy and earns no novelty bonus.
                self._starter_choice_recorded = True
                self.starter_species = initial_starter
            if not self._warned_preselected_starter_state:
                label = initial_starter or f"internal species 0x{initial_species:02X}"
                print(
                    "[StarterDiversity] WARNING: the configured init_state "
                    f"already contains a party ({self.base_party_size} member(s); "
                    f"first slot: {label}). The agent cannot choose its starter "
                    "from this state. Use a verified pre-selection state with "
                    "party count 0."
                )
                self._warned_preselected_starter_state = True
        self.base_unique_party_count = self.read_unique_party_count()
        self.max_unique_party_count = self.base_unique_party_count
        self.base_pokeballs = self.read_pokeball_count()
        self.max_pokeballs = self.base_pokeballs
        self.last_captured_species = self.read_m(CAPTURED_MON_SPECIES_ADDRESS)
        self.capture_pending_steps = 0
        self.pending_dex_entries = 0
        self.pending_dex_steps = 0
        # Keep species-level ownership so the capture marker can be classified
        # even when its Dex bit appears one environment step before/after it.
        self.last_owned_dex_numbers = self.read_pokedex_owned_dex_numbers()
        self.pending_new_owned_dex_steps = {}
        self.caught_dex_numbers = set()
        self.capture_count = 0
        self.novel_capture_count = 0
        self.duplicate_capture_count = 0
        self.unclassified_capture_count = 0
        self.new_dex_entries = 0
        self.last_lead_health = self.read_lead_hp_fraction()
        self.last_map_id_for_healing = int(self.read_m(0xD35E))
        self.critical_lead_hp_steps = 0
        self.lead_faint_count = 0
        self.center_visit_count = 0
        self.center_low_hp_visits = 0
        self.center_heal_count = 0
        self.center_hp_restored = 0.0
        self.center_maps_rewarded_at_low_hp = set()
        self.lifetime_best_owned_count = max(
            self.lifetime_best_owned_count, self.base_owned_count
        )
        self.reset_hm_progress_baseline()
        self.reset_hm_field_use_baseline()

        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        # Include the initial tile/map in the episode's baseline. This prevents
        # the first action from receiving a misleading exploration bonus.
        self.update_map_progress()
        self.update_seen_coords()
        self.update_explore_map()
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        # Preserve the existing grayscale/downscaled representation used by
        # the policy. Do not change this path when improving TensorBoard images.
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]  # (144, 160, 1)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def get_tensorboard_frame(self, scale=4):
        """Return a full-colour Game Boy frame enlarged for TensorBoard.

        The emulator's native 160x144 frame is enlarged with nearest-neighbour
        replication. This preserves pixel edges and does not alter the model's
        72x80 grayscale observation.

        Args:
            scale: Integer enlargement factor. 4 produces a 640x576 RGB frame.

        Returns:
            uint8 RGB image in HWC layout, suitable for Stable-Baselines3's
            logger.Image(..., "HWC").
        """
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
            raise ValueError(f"scale must be a positive integer; got {scale!r}")

        frame = np.asarray(self.pyboy.screen.ndarray)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        else:
            raise ValueError(
                "Unexpected PyBoy screen format: "
                f"expected HxW or HxWx3+ array, got shape {frame.shape}"
            )

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if scale > 1:
            frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

        return np.ascontiguousarray(frame).copy()
    
    def _get_obs(self):
        
        screen = self.render()

        self.update_recent_screens(screen)
        
        # normalize to approx 0-1
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_lead_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions
        }

        return observation

    def step(self, action):

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.refresh_event_cache()

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()
        self.update_center_healing_metrics()

        self.party_size = self.read_m(PARTY_COUNT_ADDRESS)
        self.update_pokemon_progress()
        self.update_hm_progress()
        self.update_hm_field_use_progress()
        self._record_starter_choice_if_available()

        self.last_health = self.read_hp_fraction()

        # Update progression BEFORE calculating the reward.
        self.update_map_progress()

        new_reward = self.update_reward()
        # Record the post-action state and the progression values just rewarded.
        self.append_agent_stats(action)

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # Keep event names current periodically, and always refresh them on
        # the terminal step before SubprocVecEnv automatically resets the env.
        if self.step_count % 100 == 0 or step_limit_reached:
            self.update_event_flag_names()

        current_map = self.get_game_coords()[2]
        info = {
            "map_progress": int(self.max_map_progress),
            "best_map_progress": int(self.best_map_progress),
            "best_map_id": int(self.best_map_id),
            "map_id": int(current_map),
            "seen_tiles": len(self.seen_coords),
            "lifetime_seen_tiles": len(self.lifetime_seen_coords),
            "badges": int(self.get_badges()),
            "pokedex_owned": int(self.read_pokedex_owned_count()),
            "lifetime_best_owned_count": int(self.lifetime_best_owned_count),
            "new_dex_entries": int(self.new_dex_entries),
            "captures": int(self.capture_count),
            "novel_captures": int(self.novel_capture_count),
            "duplicate_captures": int(self.duplicate_capture_count),
            "unclassified_captures": int(self.unclassified_capture_count),
            "party_duplicate_count": int(self.get_party_duplicate_count()),
            "party_duplicate_species": self.get_party_duplicate_summary(),
            "lead_hp_fraction": float(self.read_lead_hp_fraction()),
            "lead_hp_danger": float(self.get_lead_hp_danger_score()),
            "lead_hp_critical": int(
                self.read_lead_hp_fraction() <= self.lead_hp_critical_threshold
            ),
            "critical_lead_hp_steps": int(self.critical_lead_hp_steps),
            "lead_faints": int(self.lead_faint_count),
            "current_map_is_pokecenter": int(current_map in POKEMON_CENTER_MAP_IDS),
            "center_visits": int(self.center_visit_count),
            "center_low_hp_visits": int(self.center_low_hp_visits),
            "center_heals": int(self.center_heal_count),
            "center_hp_restored": float(self.center_hp_restored),
            "novel_capture_reward_score": float(self.progress_reward.get("capture", 0.0)),
            "duplicate_capture_penalty_score": float(self.progress_reward.get("duplicate_capture", 0.0)),
            "party_duplicate_penalty_score": float(self.progress_reward.get("party_duplicates", 0.0)),
            "critical_hp_exposure_penalty_score": float(self.progress_reward.get("critical_hp_exposure", 0.0)),
            "lead_faint_penalty_score": float(self.progress_reward.get("lead_faint", 0.0)),
            "center_return_reward_score": float(self.progress_reward.get("center_low_hp_arrival", 0.0)),
            "center_heal_reward_score": float(self.progress_reward.get("center_heal", 0.0)),
            "capture_marker_species": int(self.read_m(CAPTURED_MON_SPECIES_ADDRESS)),
            "capture_marker_active": int(
                self.read_m(CAPTURED_MON_SPECIES_ADDRESS) not in (0, 0xFF)
            ),
            "capture_pending_steps": int(self.capture_pending_steps),
            "pending_dex_entries": int(self.pending_dex_entries),
            "party_size": int(self.get_party_size()),
            "party_fill_score": float(self.get_party_fill_score()),
            "party_species_count": int(self.read_unique_party_count()),
            "party_diversity_score": float(self.get_party_diversity_score()),
            "party_level_balance_eligible": int(self.get_party_level_balance_eligible()),
            "party_level_balance_score": float(self.get_party_level_balance_score()),
            "pokedex_owned_delta_current": int(max(
                self.read_pokedex_owned_count() - self.base_owned_count, 0
            )),
            "party_level_spread": int(self.get_party_level_spread()),
            "party_level_spread_target": int(self.party_level_tolerance),
            "party_level_spread_excess": int(self.get_party_level_spread_excess()),
            "party_level_spread_within_target": int(self.is_party_level_spread_within_target()),
            "pokeballs": int(self.read_pokeball_count()),
            "party_snapshot": self.read_party_snapshot(),
            "starter_species": self.starter_species or "",
            "starter_choice_recorded": int(self._starter_choice_recorded),
            "starter_diversity_reward": float(self.starter_diversity_reward),
            **self.get_hm_info_metrics(),
            "episodes_completed": int(self.episodes_completed),
        }

        if step_limit_reached:
            # Stable-Baselines3 VecEnv workers auto-reset after done=True.
            # Put copies of all terminal data in info now; callbacks must use
            # this payload instead of querying env attributes after the reset.
            self.episodes_completed += 1
            terminal_stats = dict(self.agent_stats[-1]) if self.agent_stats else {}
            info["episode_snapshot"] = {
                "episode": int(self.reset_count),
                "episode_steps": int(self.step_count + 1),
                "agent_stats": terminal_stats,
                "explore_map": self.explore_map.copy(),
                "lifetime_explore_map": self.lifetime_explore_map.copy(),
                "map_progress": int(self.max_map_progress),
                "best_map_progress": int(self.best_map_progress),
                "best_map_id": int(self.best_map_id),
                "map_id": int(current_map),
                "seen_tiles": int(len(self.seen_coords)),
                "lifetime_seen_tiles": int(len(self.lifetime_seen_coords)),
                "badges": int(self.get_badges()),
                "pokedex_owned": int(self.read_pokedex_owned_count()),
                "new_dex_entries": int(self.new_dex_entries),
                "captures": int(self.capture_count),
                "novel_captures": int(self.novel_capture_count),
                "duplicate_captures": int(self.duplicate_capture_count),
                "unclassified_captures": int(self.unclassified_capture_count),
                "party_duplicate_count": int(self.get_party_duplicate_count()),
                "party_duplicate_species": self.get_party_duplicate_summary(),
                "lead_hp_fraction": float(self.read_lead_hp_fraction()),
                "lead_hp_danger": float(self.get_lead_hp_danger_score()),
                "lead_hp_critical": int(
                    self.read_lead_hp_fraction() <= self.lead_hp_critical_threshold
                ),
                "critical_lead_hp_steps": int(self.critical_lead_hp_steps),
                "lead_faints": int(self.lead_faint_count),
                "current_map_is_pokecenter": int(current_map in POKEMON_CENTER_MAP_IDS),
                "center_visits": int(self.center_visit_count),
                "center_low_hp_visits": int(self.center_low_hp_visits),
                "center_heals": int(self.center_heal_count),
                "center_hp_restored": float(self.center_hp_restored),
                "novel_capture_reward_score": float(self.progress_reward.get("capture", 0.0)),
                "duplicate_capture_penalty_score": float(self.progress_reward.get("duplicate_capture", 0.0)),
                "party_duplicate_penalty_score": float(self.progress_reward.get("party_duplicates", 0.0)),
                "critical_hp_exposure_penalty_score": float(self.progress_reward.get("critical_hp_exposure", 0.0)),
                "lead_faint_penalty_score": float(self.progress_reward.get("lead_faint", 0.0)),
                "center_return_reward_score": float(self.progress_reward.get("center_low_hp_arrival", 0.0)),
                "center_heal_reward_score": float(self.progress_reward.get("center_heal", 0.0)),
                "capture_marker_species": int(self.read_m(CAPTURED_MON_SPECIES_ADDRESS)),
                "capture_marker_active": int(
                    self.read_m(CAPTURED_MON_SPECIES_ADDRESS) not in (0, 0xFF)
                ),
                "capture_pending_steps": int(self.capture_pending_steps),
                "pending_dex_entries": int(self.pending_dex_entries),
                "party_size": int(self.get_party_size()),
                "party_fill_score": float(self.get_party_fill_score()),
                "party_species_count": int(self.read_unique_party_count()),
                "party_diversity_score": float(self.get_party_diversity_score()),
                "party_level_balance_eligible": int(self.get_party_level_balance_eligible()),
                "party_level_balance_score": float(self.get_party_level_balance_score()),
                "pokedex_owned_delta_current": int(max(
                    self.read_pokedex_owned_count() - self.base_owned_count, 0
                )),
                "party_level_spread": int(self.get_party_level_spread()),
                "party_level_spread_target": int(self.party_level_tolerance),
                "party_level_spread_excess": int(self.get_party_level_spread_excess()),
                "party_level_spread_within_target": int(self.is_party_level_spread_within_target()),
                "party_snapshot": self.read_party_snapshot(),
                "starter_species": self.starter_species or "",
                "starter_choice_recorded": int(self._starter_choice_recorded),
                "starter_diversity_reward": float(self.starter_diversity_reward),
                "pokeballs": int(self.read_pokeball_count()),
                **self.get_hm_info_metrics(),
                "lifetime_best_owned_count": int(self.lifetime_best_owned_count),
                "episodes_completed": int(self.episodes_completed),
                "event_flags": dict(self.current_event_flags_set),
                "lifetime_event_flags": dict(self.lifetime_event_flags),
            }
            info["episodes_completed"] = int(self.episodes_completed)

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, info
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        press_step = 8
        self.pyboy.tick(press_step, False, False)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step, True, False)
        if self.save_video and self.fast_video:
            self.add_video_frame()
        
    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "best_map_progress": self.best_map_progress,
                "lifetime_coord_count": len(self.lifetime_seen_coords),
                "map_progress_reward": self.progress_reward.get("map_progress", 0.0),
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "lead_hp": self.read_lead_hp_fraction(),
                "lead_hp_danger": self.get_lead_hp_danger_score(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "pokedex_owned": self.read_pokedex_owned_count(),
                "new_dex_entries": self.new_dex_entries,
                "captures": self.capture_count,
                "novel_captures": self.novel_capture_count,
                "duplicate_captures": self.duplicate_capture_count,
                "unclassified_captures": self.unclassified_capture_count,
                "party_size": self.read_m(PARTY_COUNT_ADDRESS),
                "party_species_count": self.read_unique_party_count(),
                "party_duplicate_count": self.get_party_duplicate_count(),
                "pokeballs": self.read_pokeball_count(),
                "capture_reward": self.progress_reward.get("capture", 0.0),
                "duplicate_capture_penalty": self.progress_reward.get("duplicate_capture", 0.0),
                "party_duplicate_penalty": self.progress_reward.get("party_duplicates", 0.0),
                "dex_reward": self.progress_reward.get("dex", 0.0),
                "party_reward": self.progress_reward.get("party", 0.0),
                "ball_reward": self.progress_reward.get("balls", 0.0),
                "hm_party_compatible": self.current_hm_party_compatible_count,
                "hm_items_owned": self.current_hm_items_owned_count,
                "hm_moves_taught": self.current_hm_moves_taught_count,
                "hm_field_ready": self.current_hm_field_ready_count,
                "hm_critical_ready": self.current_hm_critical_ready_count,
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad*4, self.coords_pad*4), 
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        # if not in battle
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
            self.lifetime_seen_coords.add(coord_string)
            #self.seen_coords[coord_string] = self.step_count

    def get_current_coord_count_reward(self):
        """Return a 0..1 penalty intensity for repeatedly staying on one tile.

        The old threshold of 600 actions was so high that it rarely helped the
        agent escape an early loop. The penalty now ramps up after 10 visits and
        reaches its cap at 50 visits. It is a state-potential term, so moving to
        a less-visited tile gives some reward back.
        """
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        count = self.seen_coords.get(coord_string, 0)
        return float(np.clip((count - 10) / 40.0, 0.0, 1.0))

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        r, c = self.get_global_coords()
        height, width = self.explore_map.shape
        if 0 <= r < height and 0 <= c < width:
            self.explore_map[r, c] = 255
            self.lifetime_explore_map[r, c] = 255
        else:
            # Avoid printing on every step if a map coordinate is unmapped.
            self.out_of_bounds_coords += 1

    def get_lifetime_explore_map(self):
        """Return a safe copy of exploration accumulated across episode resets."""
        return self.lifetime_explore_map.copy()

    def get_explore_map(self):
        """Return a fixed-size, zero-padded local map around the player.

        The old NumPy slice could become smaller than 24x24 near a map edge
        (or behave unexpectedly for negative indices), violating the declared
        observation shape. This implementation always returns 48x48.
        """
        r, c = self.get_global_coords()
        height, width = self.explore_map.shape
        size = self.coords_pad * 2
        out = np.zeros((size, size), dtype=np.uint8)

        if not (0 <= r < height and 0 <= c < width):
            return out.repeat(2, axis=0).repeat(2, axis=1)

        src_r0 = max(r - self.coords_pad, 0)
        src_r1 = min(r + self.coords_pad, height)
        src_c0 = max(c - self.coords_pad, 0)
        src_c1 = min(c + self.coords_pad, width)

        dst_r0 = src_r0 - (r - self.coords_pad)
        dst_c0 = src_c0 - (c - self.coords_pad)
        out[
            dst_r0:dst_r0 + (src_r1 - src_r0),
            dst_c0:dst_c0 + (src_c1 - src_c0),
        ] = self.explore_map[src_r0:src_r1, src_c0:src_c1]

        return out.repeat(2, axis=0).repeat(2, axis=1)
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # These values are only used by the recurrent policy's reward memory.
        scale = max(abs(self.reward_scale), 1e-8)
        explore_scale = max(abs(self.explore_weight * self.reward_scale), 1e-8)
        return (
            prog["level"] * 100 / scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / explore_scale,
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:,:, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:,:, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    def read_m(self, addr):
        #return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def _read_event_bytes(self):
        return np.fromiter(
            (self.read_m(address) for address in range(event_flags_start, event_flags_end)),
            dtype=np.uint8,
            count=event_flags_end - event_flags_start,
        )

    def refresh_event_cache(self):
        self._event_cache = self._read_event_bytes()
        return self._event_cache

    def read_event_bits(self):
        # np.unpackbits defaults to MSB-first, matching the original f"{value:08b}".
        if not hasattr(self, "_event_cache"):
            self.refresh_event_cache()
        return np.unpackbits(self._event_cache).astype(np.int8).tolist()

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def _record_starter_choice_if_available(self):
        """Award one novelty bonus after this episode's starter enters the party."""
        if self._starter_choice_recorded:
            return False
        if int(self.read_m(PARTY_COUNT_ADDRESS)) < 1:
            return False

        internal_species = int(self.read_m(PARTY_SPECIES_START))
        starter_name = STARTER_SPECIES_BY_INTERNAL_ID.get(internal_species)
        if starter_name is None:
            return False

        reward, counts = self.starter_diversity_tracker.record_choice(starter_name)
        self._starter_choice_recorded = True
        self.starter_species = starter_name
        self.starter_diversity_reward = float(reward)
        self.starter_choice_counts = counts
        return True

    def get_starter_choice_counts(self):
        """Expose persistent population counts for TensorBoard diagnostics."""
        self.starter_choice_counts = self.starter_diversity_tracker.get_counts()
        return dict(self.starter_choice_counts)

    def read_party_species(self):
        """Return valid species IDs in the current party, respecting party count."""
        count = min(max(int(self.read_m(PARTY_COUNT_ADDRESS)), 0), 6)
        species = [
            int(self.read_m(PARTY_SPECIES_START + index))
            for index in range(count)
        ]
        return [value for value in species if value not in (0, 0xFF)]

    def read_unique_party_count(self):
        return len(set(self.read_party_species()))

    def get_party_diversity_score(self):
        """Current active-party species diversity, normalized to 0..1.

        Duplicate species do not increase the score. The score is based on
        the active party, not the full Pokédex, so PC deposits are visible.
        """
        return min(self.read_unique_party_count(), 6) / 6.0

    def get_party_duplicate_count(self):
        """Count redundant active-party slots (two Geodude means one)."""
        species = self.read_party_species()
        return max(0, len(species) - len(set(species)))

    def get_party_duplicate_summary(self):
        """Describe duplicate species in the active party for diagnostics."""
        counts = {}
        for species_index in self.read_party_species():
            counts[species_index] = counts.get(species_index, 0) + 1
        return [
            {
                "species_index": int(species_index),
                "species": INTERNAL_SPECIES_NAMES.get(
                    species_index, f"Unknown species 0x{species_index:02X}"
                ),
                "count": int(count),
            }
            for species_index, count in sorted(counts.items()) if count > 1
        ]

    def get_party_size(self):
        """Return the valid active party size (0..6)."""
        return min(max(int(self.read_m(PARTY_COUNT_ADDRESS)), 0), 6)

    def get_party_fill_score(self):
        """Active party completeness, normalized to 0..1."""
        return self.get_party_size() / 6.0

    def _get_active_party_levels(self):
        """Return valid levels for current party members only."""
        levels = []
        for mon in self.read_party_snapshot():
            if not isinstance(mon, dict):
                continue
            try:
                level = int(mon.get("level", 0))
            except (TypeError, ValueError):
                continue
            if level > 0:
                levels.append(level)
        return levels

    def get_party_level_spread(self):
        """Return the max-minus-min level difference for the active party."""
        levels = self._get_active_party_levels()
        return max(levels) - min(levels) if len(levels) >= 2 else 0

    def get_party_level_spread_excess(self):
        """How many levels the active party exceeds the target spread by."""
        return max(0, self.get_party_level_spread() - self.party_level_tolerance)

    def is_party_level_spread_within_target(self):
        """True only when at least two active Pokémon can be compared."""
        levels = self._get_active_party_levels()
        return len(levels) >= 2 and (
            max(levels) - min(levels) <= self.party_level_tolerance
        )

    def get_party_level_balance_eligible(self):
        """Return 1 when at least two valid active-party levels are available."""
        return int(len(self._get_active_party_levels()) >= 2)

    def get_party_level_balance_score(self):
        """Return 0..1 balance, but do not call a one-Pokémon party balanced.

        At least two valid party members are required. With two or more,
        spreads up to the configured target score 100%; larger spreads score
        target/spread. Party completeness and species variety are separate
        objectives and are logged independently.
        """
        levels = self._get_active_party_levels()
        if len(levels) < 2:
            return 0.0
        spread = max(levels) - min(levels)
        target = max(0, int(self.party_level_tolerance))
        if spread <= target:
            return 1.0
        if target == 0:
            return 0.0
        return float(max(0.0, min(1.0, target / float(spread))))

    def read_pokedex_owned_dex_numbers(self):
        """Return owned National-Dex numbers from the Gen-I bitfield."""
        raw = [
            int(self.read_m(address))
            for address in range(POKEDEX_OWNED_START, POKEDEX_OWNED_END)
        ]
        owned = set()
        for zero_based in range(151):
            if raw[zero_based // 8] & (1 << (zero_based % 8)):
                owned.add(zero_based + 1)
        return owned

    def read_pokedex_owned_count(self):
        """Count Pokédex-owned bits (ignore the unused #152 bit)."""
        return len(self.read_pokedex_owned_dex_numbers())

    def read_pokeball_count(self):
        """Read the total quantity of catch-capable balls in the bag."""
        items = self.read_bag_item_counts()
        return sum(items.get(item_id, 0) for item_id in CATCH_BALL_ITEM_IDS)

    def read_bag_item_counts(self):
        """Return bag item quantities keyed by the Gen-I Red/Blue item ID."""
        count = min(max(int(self.read_m(NUM_BAG_ITEMS_ADDRESS)), 0), BAG_ITEM_CAPACITY)
        items = {}
        for index in range(count):
            item_address = BAG_ITEMS_START + index * 2
            item_id = int(self.read_m(item_address))
            quantity = max(int(self.read_m(item_address + 1)), 0)
            if item_id not in (0, 0xFF) and quantity:
                items[item_id] = items.get(item_id, 0) + quantity
        return items

    def read_party_hm_records(self):
        """Read (internal species index, four move IDs) for each party member."""
        count = min(max(int(self.read_m(PARTY_COUNT_ADDRESS)), 0), 6)
        records = []
        for slot in range(count):
            base = PARTY_MON_DATA_START + slot * PARTY_MON_DATA_STRIDE
            species_index = int(self.read_m(base))
            moves = tuple(
                int(self.read_m(base + PARTY_MON_MOVE_OFFSET + move_slot))
                for move_slot in range(PARTY_MON_MOVE_SLOTS)
            )
            type_ids = (
                int(self.read_m(base + PARTY_MON_TYPE1_OFFSET)),
                int(self.read_m(base + PARTY_MON_TYPE2_OFFSET)),
            )
            # Party's ordered species list is authoritative; use it when the
            # structure byte is transient/empty during menu transitions.
            if species_index in (0, 0xFF):
                species_list = self.read_party_species()
                species_index = species_list[slot] if slot < len(species_list) else 0
            records.append({"slot": slot + 1, "species_index": species_index, "moves": moves, "type_ids": type_ids})
        return records

    def read_party_snapshot(self):
        """Return a JSON-safe, human-readable party snapshot for TensorBoard.

        Red/Blue party records are 44 bytes each. The party table starts at
        0xD16B; current HP is +0x01, status +0x04, moves +0x08..0x0B,
        level +0x21, and max HP +0x22.
        """
        snapshot = []
        records = self.read_party_hm_records()
        for slot, record in enumerate(records):
            base = PARTY_MON_DATA_START + slot * PARTY_MON_DATA_STRIDE
            species_index = int(record["species_index"])
            species_name = INTERNAL_SPECIES_NAMES.get(
                species_index, f"Unknown species 0x{species_index:02X}"
            )
            national_dex = INTERNAL_TO_NATIONAL_DEX.get(species_index)
            level = int(self.read_m(base + 0x21))
            hp = int(self.read_hp(base + 0x01))
            max_hp = int(self.read_hp(base + 0x22))
            status_byte = int(self.read_m(base + 0x04))
            status_names = []
            for bit, label in ((0x04, "Sleep"), (0x08, "Poison"),
                               (0x10, "Burn"), (0x20, "Freeze"),
                               (0x40, "Paralysis")):
                if status_byte & bit:
                    status_names.append(label)
            moves = []
            for move_id in record["moves"]:
                move_id = int(move_id)
                if move_id in (0, 0xFF):
                    continue
                moves.append(MOVE_NAMES.get(move_id, f"Unknown move 0x{move_id:02X}"))
            type_ids = tuple(int(value) for value in record.get("type_ids", (0, 0)))
            type_names = [TYPE_NAMES_BY_ID.get(value, f"Type 0x{value:02X}")
                          for value in dict.fromkeys(type_ids)]
            type_names = [name for name in type_names if name]
            is_starter = self.is_starter_party_record(record)
            compatible_hms = [name.upper() for name in HM_DEFINITIONS
                              if species_index in HM_COMPATIBLE_INTERNAL_SPECIES[name]]
            taught_hms = [name.upper() for name, definition in HM_DEFINITIONS.items()
                          if definition["move_id"] in record["moves"]]
            preferred_carrier_hms = [
                name.upper() for name, preferred_types in HM_PREFERRED_TYPES.items()
                if species_index in HM_COMPATIBLE_INTERNAL_SPECIES[name]
                and any(type_id in preferred_types for type_id in type_ids)
            ]
            snapshot.append({
                "slot": slot + 1,
                "species_index": species_index,
                "national_dex": national_dex,
                "species": species_name,
                "type_ids": list(type_ids),
                "types": type_names,
                "is_starter": bool(is_starter),
                "hm_preferred_carrier_for": preferred_carrier_hms,
                "level": level,
                "hp": hp,
                "max_hp": max_hp,
                "status": ", ".join(status_names) if status_names else "OK",
                "moves": moves,
                "hm_can_learn": compatible_hms,
                "hm_knows": taught_hms,
            })
        return snapshot

    def is_starter_party_record(self, record):
        """Identify the original starter, including its evolved forms.

        The game's RAM layout does not assign a convenient persistent object ID
        to party Pokémon. Use the recorded starter evolution line when known;
        before starter detection, slot 1 is treated as the starter candidate.
        """
        species_index = int(record.get("species_index", 0))
        slot = int(record.get("slot", 0))
        if self.starter_species in STARTER_LINEAGES_BY_NAME:
            return species_index in STARTER_LINEAGES_BY_NAME[self.starter_species]
        return slot == 1

    @staticmethod
    def hm_record_matches_role(name, record):
        preferred_types = HM_PREFERRED_TYPES.get(name)
        if not preferred_types:
            return False
        return any(
            int(type_id) in preferred_types
            for type_id in record.get("type_ids", ())
        )

    def read_hm_status(self):
        """Inspect HM items, compatible carriers, teaching and badge readiness.

        Species/HM compatibility is mandatory. Preferred type is a secondary
        carrier role, and a non-starter is preferred whenever one is available.
        """
        bag_items = self.read_bag_item_counts()
        party = self.read_party_hm_records()
        badge_byte = int(self.read_m(BADGES_ADDRESS))
        result = {}
        for name, definition in HM_DEFINITIONS.items():
            compatible = [
                mon for mon in party
                if mon["species_index"] in HM_COMPATIBLE_INTERNAL_SPECIES[name]
            ]
            nonstarter_compatible = [
                mon for mon in compatible if not self.is_starter_party_record(mon)
            ]
            role_compatible = [
                mon for mon in nonstarter_compatible
                if self.hm_record_matches_role(name, mon)
            ]
            knows = [
                mon for mon in party
                if definition["move_id"] in mon["moves"]
            ]
            nonstarter_knows = [
                mon for mon in knows if not self.is_starter_party_record(mon)
            ]
            role_knows = [
                mon for mon in knows
                if not self.is_starter_party_record(mon)
                and self.hm_record_matches_role(name, mon)
            ]
            item_owned = bag_items.get(definition["item_id"], 0) > 0
            badge_owned = bool(badge_byte & definition["badge_mask"])
            knows_preferred_carrier = bool(
                role_knows if role_compatible
                else nonstarter_knows if nonstarter_compatible
                else knows
            )
            result[name] = {
                "item_owned": item_owned,
                "party_can_learn": bool(compatible),
                "party_knows_move": bool(knows),
                "nonstarter_can_learn": bool(nonstarter_compatible),
                "role_can_learn": bool(role_compatible),
                "nonstarter_knows_move": bool(nonstarter_knows),
                "role_knows_move": bool(role_knows),
                "preferred_carrier_knows_move": knows_preferred_carrier,
                "badge_owned": badge_owned,
                "field_ready": bool(item_owned and knows and badge_owned),
                "preferred_field_ready": bool(
                    item_owned and knows_preferred_carrier and badge_owned
                ),
                "compatible_party_species": tuple(
                    mon["species_index"] for mon in compatible
                ),
                "nonstarter_party_species": tuple(
                    mon["species_index"] for mon in nonstarter_compatible
                ),
                "role_party_species": tuple(
                    mon["species_index"] for mon in role_compatible
                ),
            }
        return result

    @staticmethod
    def hm_weighted_score(status, field):
        return sum(
            float(HM_DEFINITIONS[name]["weight"])
            for name, values in status.items() if values[field]
        )

    def reset_hm_progress_baseline(self):
        status = self.read_hm_status()
        self.current_hm_status = status
        self.base_hm_candidate_score = self.hm_weighted_score(status, "party_can_learn")
        self.base_hm_item_score = self.hm_weighted_score(status, "item_owned")
        self.base_hm_taught_score = self.hm_weighted_score(status, "party_knows_move")
        self.base_hm_ready_score = self.hm_weighted_score(status, "field_ready")
        self.base_hm_nonstarter_candidate_score = self.hm_weighted_score(status, "nonstarter_can_learn")
        self.base_hm_role_candidate_score = self.hm_weighted_score(status, "role_can_learn")
        self.base_hm_nonstarter_taught_score = self.hm_weighted_score(status, "nonstarter_knows_move")
        self.base_hm_role_taught_score = self.hm_weighted_score(status, "role_knows_move")
        self.base_hm_preferred_ready_score = self.hm_weighted_score(status, "preferred_field_ready")
        self.max_hm_candidate_score = self.base_hm_candidate_score
        self.max_hm_item_score = self.base_hm_item_score
        self.max_hm_taught_score = self.base_hm_taught_score
        self.max_hm_ready_score = self.base_hm_ready_score
        self.max_hm_nonstarter_candidate_score = self.base_hm_nonstarter_candidate_score
        self.max_hm_role_candidate_score = self.base_hm_role_candidate_score
        self.max_hm_nonstarter_taught_score = self.base_hm_nonstarter_taught_score
        self.max_hm_role_taught_score = self.base_hm_role_taught_score
        self.max_hm_preferred_ready_score = self.base_hm_preferred_ready_score
        self._refresh_hm_counts()

    def _refresh_hm_counts(self):
        status = self.current_hm_status
        self.current_hm_party_compatible_count = sum(
            int(values["party_can_learn"]) for values in status.values()
        )
        self.current_hm_items_owned_count = sum(
            int(values["item_owned"]) for values in status.values()
        )
        self.current_hm_moves_taught_count = sum(
            int(values["party_knows_move"]) for values in status.values()
        )
        self.current_hm_field_ready_count = sum(
            int(values["field_ready"]) for values in status.values()
        )
        self.current_hm_critical_party_compatible_count = sum(
            int(values["party_can_learn"]) for name, values in status.items()
            if HM_DEFINITIONS[name]["required"]
        )
        self.current_hm_critical_items_owned_count = sum(
            int(values["item_owned"]) for name, values in status.items()
            if HM_DEFINITIONS[name]["required"]
        )
        self.current_hm_critical_moves_taught_count = sum(
            int(values["party_knows_move"]) for name, values in status.items()
            if HM_DEFINITIONS[name]["required"]
        )
        self.current_hm_critical_ready_count = sum(
            int(values["field_ready"]) for name, values in status.items()
            if HM_DEFINITIONS[name]["required"]
        )

    def update_hm_progress(self):
        status = self.read_hm_status()
        self.current_hm_status = status
        self.max_hm_candidate_score = max(
            self.max_hm_candidate_score,
            self.hm_weighted_score(status, "party_can_learn"),
        )
        self.max_hm_item_score = max(
            self.max_hm_item_score, self.hm_weighted_score(status, "item_owned")
        )
        self.max_hm_taught_score = max(
            self.max_hm_taught_score,
            self.hm_weighted_score(status, "party_knows_move"),
        )
        self.max_hm_ready_score = max(
            self.max_hm_ready_score, self.hm_weighted_score(status, "field_ready")
        )
        self.max_hm_nonstarter_candidate_score = max(
            self.max_hm_nonstarter_candidate_score,
            self.hm_weighted_score(status, "nonstarter_can_learn"),
        )
        self.max_hm_role_candidate_score = max(
            self.max_hm_role_candidate_score,
            self.hm_weighted_score(status, "role_can_learn"),
        )
        self.max_hm_nonstarter_taught_score = max(
            self.max_hm_nonstarter_taught_score,
            self.hm_weighted_score(status, "nonstarter_knows_move"),
        )
        self.max_hm_role_taught_score = max(
            self.max_hm_role_taught_score,
            self.hm_weighted_score(status, "role_knows_move"),
        )
        self.max_hm_preferred_ready_score = max(
            self.max_hm_preferred_ready_score,
            self.hm_weighted_score(status, "preferred_field_ready"),
        )
        self._refresh_hm_counts()

    def read_event_index(self, event_index):
        """Read a pokered event bit by its zero-based event-constant index."""
        event_index = int(event_index)
        if event_index < 0:
            return False
        address = event_flags_start + (event_index // 8)
        bit_mask = 1 << (event_index % 8)
        return bool(int(self.read_m(address)) & bit_mask)

    def reset_hm_field_use_baseline(self):
        """Reset field-use counters and seed visited-map/boulder flag state."""
        current_map = int(self.get_game_coords()[2])
        self.last_hm_field_map_id = current_map
        self.hm_cut_gate_maps_seen = set()
        self.hm_surf_route_maps_seen = set()
        if current_map in CUT_GATE_MAP_IDS:
            self.hm_cut_gate_maps_seen.add(current_map)
        if current_map in SURF_ROUTE_MAP_IDS:
            self.hm_surf_route_maps_seen.add(current_map)
        self.hm_cut_gate_entries = 0
        self.hm_surf_route_entries = 0
        self.hm_strength_boulder_moves = 0
        self.hm_strength_boulder_flag_state = {
            index: self.read_event_index(index)
            for index in STRENGTH_BOULDER_EVENT_INDICES
        }

    def update_hm_field_use_progress(self):
        """Count first-time HM-gated route entries and confirmed Strength events."""
        current_map = int(self.get_game_coords()[2])
        if current_map != self.last_hm_field_map_id:
            if current_map in CUT_GATE_MAP_IDS:
                if current_map not in self.hm_cut_gate_maps_seen:
                    if self.current_hm_status["cut"]["field_ready"]:
                        self.hm_cut_gate_entries += 1
                    self.hm_cut_gate_maps_seen.add(current_map)
            if current_map in SURF_ROUTE_MAP_IDS:
                if current_map not in self.hm_surf_route_maps_seen:
                    if self.current_hm_status["surf"]["field_ready"]:
                        self.hm_surf_route_entries += 1
                    self.hm_surf_route_maps_seen.add(current_map)
            self.last_hm_field_map_id = current_map

        if self.current_hm_status["strength"]["field_ready"]:
            for index in STRENGTH_BOULDER_EVENT_INDICES:
                current = self.read_event_index(index)
                previous = self.hm_strength_boulder_flag_state.get(index, False)
                if current and not previous:
                    self.hm_strength_boulder_moves += 1
                self.hm_strength_boulder_flag_state[index] = current
        else:
            # Track state even before Strength becomes usable, so an already-set
            # story flag cannot be mistaken for a new use later.
            for index in STRENGTH_BOULDER_EVENT_INDICES:
                self.hm_strength_boulder_flag_state[index] = self.read_event_index(index)

    def get_hm_info_metrics(self):
        """Flatten compact HM status fields for SB3 info/TensorBoard."""
        status = self.current_hm_status
        metrics = {
            "hm_party_compatible": int(self.current_hm_party_compatible_count),
            "hm_items_owned": int(self.current_hm_items_owned_count),
            "hm_moves_taught": int(self.current_hm_moves_taught_count),
            "hm_field_ready": int(self.current_hm_field_ready_count),
            "hm_critical_party_compatible": int(self.current_hm_critical_party_compatible_count),
            "hm_critical_items_owned": int(self.current_hm_critical_items_owned_count),
            "hm_critical_moves_taught": int(self.current_hm_critical_moves_taught_count),
            "hm_critical_ready": int(self.current_hm_critical_ready_count),
            "hm_critical_candidates_complete": int(
                self.current_hm_critical_party_compatible_count == len(HM_CRITICAL_NAMES)
            ),
            "hm_critical_ready_complete": int(
                self.current_hm_critical_ready_count == len(HM_CRITICAL_NAMES)
            ),
            "hm_nonstarter_candidates": int(sum(v["nonstarter_can_learn"] for v in status.values())),
            "hm_role_candidates": int(sum(v["role_can_learn"] for v in status.values())),
            "hm_nonstarter_taught": int(sum(v["nonstarter_knows_move"] for v in status.values())),
            "hm_role_taught": int(sum(v["role_knows_move"] for v in status.values())),
            "hm_preferred_carrier_ready": int(sum(v["preferred_field_ready"] for v in status.values())),
            "hm_cut_gate_entries": int(self.hm_cut_gate_entries),
            "hm_surf_route_entries": int(self.hm_surf_route_entries),
            "hm_strength_boulder_moves": int(self.hm_strength_boulder_moves),
            "hm_candidate_coverage_gap": float(max(
                self.max_hm_candidate_score
                - self.hm_weighted_score(status, "party_can_learn"), 0.0
            )),
            "hm_ready_coverage_gap": float(max(
                self.max_hm_ready_score - self.hm_weighted_score(status, "field_ready"),
                0.0,
            )),
        }
        for name, values in status.items():
            metrics[f"hm_{name}_party_compatible"] = int(values["party_can_learn"])
            metrics[f"hm_{name}_item_owned"] = int(values["item_owned"])
            metrics[f"hm_{name}_taught"] = int(values["party_knows_move"])
            metrics[f"hm_{name}_badge"] = int(values["badge_owned"])
            metrics[f"hm_{name}_usable"] = int(values["field_ready"])
            metrics[f"hm_{name}_nonstarter_candidate"] = int(values["nonstarter_can_learn"])
            metrics[f"hm_{name}_role_candidate"] = int(values["role_can_learn"])
            metrics[f"hm_{name}_nonstarter_taught"] = int(values["nonstarter_knows_move"])
            metrics[f"hm_{name}_role_taught"] = int(values["role_knows_move"])
            metrics[f"hm_{name}_preferred_carrier_ready"] = int(values["preferred_field_ready"])
        return metrics

    def update_pokemon_progress(self):
        """Update Dex and party progress; distinguish novel and repeat catches.

        The successful-capture marker at 0xD11C remains authoritative for the
        total catch counter. Novelty is based on National-Dex ownership bits,
        with a short pending window for Dex bits and markers seen on adjacent
        environment steps.
        """
        owned_dex_now = self.read_pokedex_owned_dex_numbers()
        owned_count = len(owned_dex_now)
        previous_owned_dex = set(getattr(self, "last_owned_dex_numbers", set()))
        newly_owned_dex = owned_dex_now - previous_owned_dex
        owned_delta = max(owned_count - self.last_owned_count, 0)
        captured_species = int(self.read_m(CAPTURED_MON_SPECIES_ADDRESS))

        # Age pending Dex bits after giving this step's marker a chance to claim
        # them. A gift/evolution without a marker simply expires from this map.
        pending = getattr(self, "pending_new_owned_dex_steps", {})
        pending = {dex: ttl - 1 for dex, ttl in pending.items() if ttl > 1}

        capture_signal = captured_species not in (0, 0xFF) and (
            captured_species != self.last_captured_species
        )

        if capture_signal:
            self.capture_count += 1
            captured_dex = INTERNAL_TO_NATIONAL_DEX.get(captured_species)
            if captured_dex is None:
                self.unclassified_capture_count = getattr(
                    self, "unclassified_capture_count", 0
                ) + 1
            elif (
                (
                    captured_dex not in previous_owned_dex
                    and captured_dex not in self.caught_dex_numbers
                )
                or captured_dex in newly_owned_dex
                or captured_dex in pending
            ):
                self.novel_capture_count = getattr(self, "novel_capture_count", 0) + 1
                self.caught_dex_numbers.add(captured_dex)
                pending.pop(captured_dex, None)
            else:
                self.duplicate_capture_count = getattr(
                    self, "duplicate_capture_count", 0
                ) + 1

            if self.pending_dex_entries > 0 and self.pending_dex_steps > 0:
                # Retain the earlier aggregate pairing diagnostics.
                self.pending_dex_entries = max(self.pending_dex_entries - 1, 0)
                if self.pending_dex_entries == 0:
                    self.pending_dex_steps = 0
            else:
                self.capture_pending_steps = 4

        if owned_delta > 0:
            self.new_dex_entries = max(
                self.new_dex_entries,
                max(owned_count - self.base_owned_count, 0),
            )
            if capture_signal or self.capture_pending_steps > 0:
                self.capture_pending_steps = 0
            else:
                self.pending_dex_entries += owned_delta
                self.pending_dex_steps = 4

        for dex_number in newly_owned_dex:
            if dex_number not in getattr(self, "caught_dex_numbers", set()):
                pending[dex_number] = 4
        self.pending_new_owned_dex_steps = pending

        if self.capture_pending_steps > 0 and not owned_delta and not capture_signal:
            self.capture_pending_steps -= 1
        if self.pending_dex_steps > 0 and not capture_signal and not owned_delta:
            self.pending_dex_steps -= 1
            if self.pending_dex_steps == 0:
                self.pending_dex_entries = 0

        self.max_owned_count = max(self.max_owned_count, owned_count)
        self.lifetime_best_owned_count = max(
            self.lifetime_best_owned_count, self.max_owned_count
        )

        party_size = min(max(int(self.read_m(PARTY_COUNT_ADDRESS)), 0), 6)
        unique_party_count = self.read_unique_party_count()
        pokeballs = self.read_pokeball_count()
        self.max_party_size = max(self.max_party_size, party_size)
        self.max_unique_party_count = max(
            self.max_unique_party_count, unique_party_count
        )
        self.max_pokeballs = max(self.max_pokeballs, pokeballs)

        self.last_owned_count = owned_count
        self.last_owned_dex_numbers = owned_dex_now
        self.last_captured_species = captured_species

    def get_all_events_reward(self):
        # Count cached event flags and exclude the museum ticket as before.
        if not hasattr(self, "_event_cache"):
            self.refresh_event_cache()
        event_count = int(np.unpackbits(self._event_cache).sum())
        ticket_offset = museum_ticket[0] - event_flags_start
        ticket_set = (int(self._event_cache[ticket_offset]) >> museum_ticket[1]) & 1
        return max(event_count - self.base_event_flags - ticket_set, 0)

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            "level": self.reward_scale * self.get_levels_reward() * 0.5,
            "heal": self.reward_scale * self.total_healing_rew * 10,
            # Penalize the lead becoming critically weak; recovery removes this
            # state-potential penalty and makes starter survival observable.
            "lead_hp_danger": (
                -self.reward_scale * self.lead_hp_danger_weight
                * self.get_lead_hp_danger_score()
            ),
            # A small per-step cost makes remaining at dangerous HP progressively
            # worse, instead of relying only on the one-time HP-state potential.
            "critical_hp_exposure": (
                -self.reward_scale * self.critical_lead_hp_step_penalty_weight
                * self.critical_lead_hp_steps
            ),
            "lead_faint": (
                -self.reward_scale * self.lead_faint_penalty_weight
                * self.lead_faint_count
            ),
            # Reward verified low-HP Center returns and actual recovery there.
            "center_low_hp_arrival": (
                self.reward_scale * self.center_low_hp_arrival_reward_weight
                * self.center_low_hp_visits
            ),
            "center_heal": (
                self.reward_scale * self.center_heal_reward_weight
                * self.center_heal_count
            ),
            #"op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            #"dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 10,
            # Important story-map milestones. This uses the config value that
            # existed in the original file but was not connected to rewards.
            "map_progress": (
                self.reward_scale * self.map_progress_weight * self.max_map_progress
            ),
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,
            "stuck": (
                -self.reward_scale * self.stuck_penalty_weight
                * self.get_current_coord_count_reward()
            ),
            # Only a species not previously registered in the Dex earns the
            # collection reward. Total and repeat catches remain visible in info.
            "capture": (
                self.reward_scale * self.capture_reward_weight
                * self.novel_capture_count
            ),
            # Repeat catches are diagnostic events, not new collection progress.
            "duplicate_capture": (
                -self.reward_scale * self.duplicate_capture_penalty_weight
                * self.duplicate_capture_count
            ),
            # Penalize redundant active-party slots, not extra species in the PC.
            # As a state potential, depositing a duplicate reverses this penalty.
            "party_duplicates": (
                -self.reward_scale * self.party_duplicate_penalty_weight
                * self.get_party_duplicate_count()
            ),
            # Reward general Dex growth too (for example, evolutions or gifts),
            # but less than an actual capture.
            "dex": (
                self.reward_scale * self.dex_reward_weight
                * max(self.max_owned_count - self.base_owned_count, 0)
            ),
            # Encourage carrying a varied active team, without requiring six
            # members before the agent can progress.
            "party": (
                self.reward_scale * self.party_reward_weight
                * max(self.max_unique_party_count - self.base_unique_party_count, 0)
            ),
            # A direct potential penalty applies only when the active-party
            # max/min level spread exceeds the 10-level target. Bringing the
            # weaker members closer reduces this penalty; once the spread is
            # within target, no further balancing penalty is applied.
            # Current party size is a potential: each extra occupied slot
            # earns reward, while depositing a teammate removes that potential.
            "party_size": (
                self.reward_scale * self.party_size_reward_weight
                * self.get_party_fill_score()
            ),
            "party_diversity": (
                self.reward_scale * self.party_diversity_weight
                * self.get_party_diversity_score()
            ),
            "party_balance": (
                -self.reward_scale * self.party_balance_weight
                * (self.get_party_level_spread_excess() / max(self.party_level_tolerance, 1))
            ),
            # Reward stocking a modest reserve of Poké Balls. The high-water
            # score cannot be farmed by buying, throwing, and buying the same
            # balls repeatedly; using a ball is never directly penalized.
            "balls": (
                self.reward_scale * self.ball_reward_weight
                * max(
                    min(self.max_pokeballs, self.ball_goal)
                    - min(self.base_pokeballs, self.ball_goal),
                    0,
                )
            ),
            # A shared count in session storage rewards underrepresented
            # starters across all parallel workers. The policy remains in
            # control; no action or species is forced by the environment.
            "starter_diversity": (
                self.reward_scale * float(getattr(self, "starter_diversity_reward", 0.0))
            ),
            # HM progression ladder. Candidate coverage encourages catching a
            # compatible team member; the next stages reward obtaining the HM,
            # teaching its move, and having the correct badge to use it outside.
            "hm_candidates": (
                self.reward_scale * self.hm_candidate_reward_weight
                * max(self.max_hm_candidate_score - self.base_hm_candidate_score, 0.0)
            ),
            # High-water rewards stop repeated menu/farming exploits; these
            # current-state penalties additionally teach the agent not to lose
            # the only compatible or already field-ready party member.
            "hm_candidate_retention": (
                -self.reward_scale * self.hm_candidate_loss_penalty_weight
                * max(
                    self.max_hm_candidate_score
                    - self.hm_weighted_score(self.current_hm_status, "party_can_learn"),
                    0.0,
                )
            ),
            "hm_items": (
                self.reward_scale * self.hm_item_reward_weight
                * max(self.max_hm_item_score - self.base_hm_item_score, 0.0)
            ),
            "hm_taught": (
                self.reward_scale * self.hm_taught_reward_weight
                * max(self.max_hm_taught_score - self.base_hm_taught_score, 0.0)
            ),
            "hm_ready": (
                self.reward_scale * self.hm_ready_reward_weight
                * max(self.max_hm_ready_score - self.base_hm_ready_score, 0.0)
            ),
            "hm_ready_retention": (
                -self.reward_scale * self.hm_ready_loss_penalty_weight
                * max(
                    self.max_hm_ready_score
                    - self.hm_weighted_score(self.current_hm_status, "field_ready"),
                    0.0,
                )
            ),
            "hm_nonstarter_candidates": (
                self.reward_scale * self.hm_nonstarter_candidate_reward_weight
                * max(self.max_hm_nonstarter_candidate_score - self.base_hm_nonstarter_candidate_score, 0.0)
            ),
            "hm_role_candidates": (
                self.reward_scale * self.hm_role_candidate_reward_weight
                * max(self.max_hm_role_candidate_score - self.base_hm_role_candidate_score, 0.0)
            ),
            "hm_nonstarter_taught": (
                self.reward_scale * self.hm_nonstarter_taught_reward_weight
                * max(self.max_hm_nonstarter_taught_score - self.base_hm_nonstarter_taught_score, 0.0)
            ),
            "hm_role_taught": (
                self.reward_scale * self.hm_role_taught_reward_weight
                * max(self.max_hm_role_taught_score - self.base_hm_role_taught_score, 0.0)
            ),
            "hm_preferred_carrier_ready": (
                self.reward_scale * self.hm_preferred_ready_reward_weight
                * max(self.max_hm_preferred_ready_score - self.base_hm_preferred_ready_score, 0.0)
            ),
            "hm_cut_field_use": (
                self.reward_scale * self.hm_cut_field_use_reward_weight
                * self.hm_cut_gate_entries
            ),
            "hm_surf_field_use": (
                self.reward_scale * self.hm_surf_field_use_reward_weight
                * self.hm_surf_route_entries
            ),
            "hm_strength_field_use": (
                self.reward_scale * self.hm_strength_field_use_reward_weight
                * self.hm_strength_boulder_moves
            ),
        }

        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_lead_hp_fraction(self):
        """Return the first party member's HP fraction (normally the lead)."""
        if self.get_party_size() < 1:
            return 1.0
        base = PARTY_MON_DATA_START
        current_hp = self.read_hp(base + 0x01)
        maximum_hp = self.read_hp(base + 0x22)
        if maximum_hp <= 0:
            return 1.0
        return float(max(0.0, min(1.0, current_hp / maximum_hp)))

    def get_lead_hp_danger_score(self):
        """Return 0 when safe and 1 at 0 HP below the critical threshold."""
        fraction = self.read_lead_hp_fraction()
        threshold = self.lead_hp_critical_threshold
        return float(max(0.0, min(1.0, (threshold - fraction) / threshold)))

    def update_center_healing_metrics(self):
        """Track low-HP Center arrivals and actual lead-HP recovery in Centers."""
        current_map = int(self.read_m(0xD35E))
        current_health = self.read_lead_hp_fraction()
        previous_map = int(self.last_map_id_for_healing)
        previous_health = float(self.last_lead_health)
        current_is_center = current_map in POKEMON_CENTER_MAP_IDS
        previous_was_center = previous_map in POKEMON_CENTER_MAP_IDS

        if current_is_center and not previous_was_center:
            self.center_visit_count += 1
            if (
                previous_health <= self.lead_hp_critical_threshold
                and current_map not in self.center_maps_rewarded_at_low_hp
            ):
                # One reward per Center per episode prevents door-cycle farming.
                self.center_low_hp_visits += 1
                self.center_maps_rewarded_at_low_hp.add(current_map)

        if previous_health > 0.0 and current_health <= 0.0:
            self.lead_faint_count += 1
        if current_health <= self.lead_hp_critical_threshold:
            self.critical_lead_hp_steps += 1

        hp_gain = max(0.0, current_health - previous_health)
        if current_is_center and hp_gain >= 0.10:
            self.center_heal_count += 1
            self.center_hp_restored += min(hp_gain, 1.0)

        self.last_lead_health = current_health
        self.last_map_id_for_healing = current_map

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)
    
    def update_map_progress(self):
        map_idx = int(self.read_m(0xD35E))
        progress = int(self.get_map_progress(map_idx))
        self.max_map_progress = max(self.max_map_progress, progress)
        if progress > self.best_map_progress:
            self.best_map_progress = progress
            self.best_map_id = map_idx

    def update_event_flag_names(self):
        """Refresh named event flags for this episode and lifetime history."""
        for offset, value in enumerate(self._event_cache):
            address = event_flags_start + offset
            for idx, bit in enumerate(f"{int(value):08b}"):
                if bit == "1":
                    key = f"0x{address:X}-{idx}"
                    name = self.event_names.get(key)
                    if name is not None:
                        self.current_event_flags_set[key] = name
                        self.lifetime_event_flags[key] = name
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def close(self):
        """Close any open video writers and stop the emulator cleanly."""
        for writer_name in ("full_frame_writer", "model_frame_writer", "map_frame_writer"):
            writer = getattr(self, writer_name, None)
            if writer is not None:
                try:
                    writer.close()
                finally:
                    setattr(self, writer_name, None)

        stop = getattr(self.pyboy, "stop", None)
        if callable(stop):
            stop()
