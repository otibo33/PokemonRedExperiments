from __future__ import annotations
import importlib.util, sys, types
from pathlib import Path

gym = types.ModuleType("gymnasium")
class Env: pass
class Spaces: pass
gym.Env, gym.spaces = Env, Spaces()
sys.modules["gymnasium"] = gym
pyboy = types.ModuleType("pyboy"); pyboy.PyBoy = type("PyBoy", (), {})
pyboy_utils = types.ModuleType("pyboy.utils"); pyboy_utils.WindowEvent = type("WindowEvent", (), {})
sys.modules["pyboy"] = pyboy; sys.modules["pyboy.utils"] = pyboy_utils
sys.modules["mediapy"] = types.ModuleType("mediapy")
global_map = types.ModuleType("global_map"); global_map.local_to_global=lambda *a:(0,0); global_map.GLOBAL_MAP_SHAPE=(48,48)
sys.modules["global_map"] = global_map

p=Path(__file__).with_name("red_gym_env_v2.py")
spec=importlib.util.spec_from_file_location("balance_env_test",p); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)

def check(ok,msg):
    if not ok: raise AssertionError(msg)

class Fake(mod.RedGymEnv):
    def __init__(self, levels, species=None):
        self.party_level_tolerance=10
        self.levels=list(levels)
        self.species=list(species if species is not None else range(1,len(levels)+1))
        self.memory={mod.PARTY_COUNT_ADDRESS: len(self.levels)}
    def read_m(self, address): return self.memory.get(address, 0)
    def read_party_snapshot(self): return [{"level":v} for v in self.levels]
    def read_unique_party_count(self): return len(set(self.species))

def test_single_member_is_not_balance_eligible():
    env=Fake([20])
    check(env.get_party_level_balance_score()==0.0,"one-member party must not receive balance credit")
    check(env.get_party_level_balance_eligible()==0,"one-member party must be ineligible")
    check(not env.is_party_level_spread_within_target(),"one-member party must not count as within target")

def test_spread_at_or_below_ten_is_full_score():
    env=Fake([20,10])
    check(env.get_party_level_balance_score()==1.0,"spread 10 must score 100%")
    check(env.get_party_level_balance_eligible()==1,"two-member party must be eligible")
    check(env.is_party_level_spread_within_target(),"spread 10 must be within target")

def test_spread_above_ten_uses_target_over_spread():
    check(abs(Fake([40,16]).get_party_level_balance_score()-(10/24))<1e-9,"spread 24 should score 41.67%")

def test_duplicate_species_do_not_increase_diversity():
    varied=Fake([10,10,10],[25,4,7]); duplicate=Fake([10,10,10],[25,25,25])
    check(varied.get_party_diversity_score()>duplicate.get_party_diversity_score(),"unique species must be preferred")

def test_party_fill_score_reflects_occupied_slots():
    check(abs(Fake([10]).get_party_fill_score()-(1/6))<1e-9,"one Pokémon should be 1/6 filled")
    check(abs(Fake([10,10,10]).get_party_fill_score()-0.5)<1e-9,"three Pokémon should be half filled")
    check(Fake([10]*6).get_party_fill_score()==1.0,"six Pokémon should be fully filled")

for fn in (test_single_member_is_not_balance_eligible,test_spread_at_or_below_ten_is_full_score,test_spread_above_ten_uses_target_over_spread,test_duplicate_species_do_not_increase_diversity,test_party_fill_score_reflects_occupied_slots):
    fn(); print("PASS",fn.__name__)
print("All party balance/completeness tests passed.")
