"""Regression tests for StreamWrapper's Gymnasium attribute delegation."""

import warnings
import unittest

from stream_agent_wrapper import _resolve_wrapper_attr


class _BaseEnvironment:
    def __init__(self):
        self.current_hm_critical_moves_taught_count = 3
        self.lifetime_event_flags = {"SSAnne": "done"}

    def read_unique_party_count(self):
        return 4


class _LegacyGymWrapper:
    """Model the deprecated fallback used by older Gymnasium wrappers."""

    def __init__(self, env):
        self.env = env
        self.wrapper_value = "intermediate"

    def __getattr__(self, name):
        warnings.warn(
            "env.<attribute> forwarding is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self.env, name)


class TestWrapperAttributeResolution(unittest.TestCase):
    def setUp(self):
        self.base = _BaseEnvironment()
        self.inner = _LegacyGymWrapper(self.base)
        self.outer = _LegacyGymWrapper(self.inner)

    def test_base_environment_values_do_not_trigger_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hm_count = _resolve_wrapper_attr(
                self.outer, "current_hm_critical_moves_taught_count"
            )
            event_flags = _resolve_wrapper_attr(self.outer, "lifetime_event_flags")

        self.assertEqual(hm_count, 3)
        self.assertEqual(event_flags, {"SSAnne": "done"})
        self.assertEqual(caught, [])

    def test_base_environment_methods_do_not_trigger_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            party_count = _resolve_wrapper_attr(
                self.outer, "read_unique_party_count"
            )()

        self.assertEqual(party_count, 4)
        self.assertEqual(caught, [])

    def test_intermediate_wrapper_attributes_are_preserved(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = _resolve_wrapper_attr(self.outer, "wrapper_value")

        self.assertEqual(value, "intermediate")
        self.assertEqual(caught, [])

    def test_missing_attribute_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _resolve_wrapper_attr(self.outer, "does_not_exist")


if __name__ == "__main__":
    unittest.main()
