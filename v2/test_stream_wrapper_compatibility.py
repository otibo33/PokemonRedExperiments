from __future__ import annotations

import warnings
import unittest

import gymnasium as gym
from gymnasium import spaces

from stream_agent_wrapper import StreamWrapper


class _AttributeEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        self.current_hm_critical_moves_taught_count = 4
        self.lifetime_event_flags = {"got_cut": "SS Anne Captain"}

    def read_unique_party_count(self):
        return 3

    def get_starter_choice_counts(self):
        return {"Bulbasaur": 1, "Charmander": 2, "Squirtle": 0}


class StreamWrapperCompatibilityTests(unittest.TestCase):
    def make_wrapper_without_network(self):
        # Bypass StreamWrapper.__init__, which intentionally opens a websocket.
        wrapper = StreamWrapper.__new__(StreamWrapper)
        gym.Wrapper.__init__(wrapper, _AttributeEnv())
        return wrapper

    def test_legacy_sb3_getattr_delegation_does_not_emit_gymnasium_warning(self):
        wrapper = self.make_wrapper_without_network()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            # SB3 2.3.2 SubprocVecEnv uses getattr() for get_attr() and env_method().
            self.assertEqual(wrapper.current_hm_critical_moves_taught_count, 4)
            self.assertEqual(wrapper.read_unique_party_count(), 3)
            self.assertEqual(
                wrapper.get_starter_choice_counts(),
                {"Bulbasaur": 1, "Charmander": 2, "Squirtle": 0},
            )
            self.assertEqual(
                wrapper.lifetime_event_flags,
                {"got_cut": "SS Anne Captain"},
            )

        gymnasium_wrapper_warnings = [
            str(item.message)
            for item in captured
            if "to get variables from other wrappers is deprecated" in str(item.message)
        ]
        self.assertEqual(gymnasium_wrapper_warnings, [])

    def test_wrapper_own_attributes_take_precedence(self):
        wrapper = self.make_wrapper_without_network()
        wrapper.upload_interval = 300
        self.assertEqual(wrapper.upload_interval, 300)

    def test_unknown_attributes_still_raise_attribute_error(self):
        wrapper = self.make_wrapper_without_network()
        with self.assertRaises(AttributeError):
            _ = wrapper.not_a_real_environment_attribute


if __name__ == "__main__":
    unittest.main(verbosity=2)
