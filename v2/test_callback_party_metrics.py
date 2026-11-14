from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

# Stub only SB3/TensorBoard symbols; exercise the real callback metric helper.
callbacks = types.ModuleType("stable_baselines3.common.callbacks")
class BaseCallback: pass
callbacks.BaseCallback = BaseCallback
logger = types.ModuleType("stable_baselines3.common.logger")
class Image:
    def __init__(self, *args, **kwargs): pass
logger.Image = Image
stable = types.ModuleType("stable_baselines3")
common = types.ModuleType("stable_baselines3.common")
common.callbacks, common.logger = callbacks, logger
stable.common = common
sys.modules["stable_baselines3"] = stable
sys.modules["stable_baselines3.common"] = common
sys.modules["stable_baselines3.common.callbacks"] = callbacks
sys.modules["stable_baselines3.common.logger"] = logger

tensorboard = types.ModuleType("torch.utils.tensorboard")
tensorboard.SummaryWriter = type("SummaryWriter", (), {})
sys.modules["torch.utils.tensorboard"] = tensorboard

path = Path(__file__).with_name("tensorboard_callback.py")
spec = importlib.util.spec_from_file_location("party_callback_test", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

metrics = module.TensorboardCallback._party_level_metrics
cases = [
    ([], 0.0, 0),
    ([{"level": 20}], 0.0, 0),
    ([{"level": 20}, {"level": 10}], 1.0, 1),
    ([{"level": 40}, {"level": 16}], 10.0 / 24.0, 1),
]
for party, expected_score, expected_eligible in cases:
    result = metrics(party, 10)
    assert abs(result["party_level_balance_score"] - expected_score) < 1e-9
    assert result["party_level_balance_eligible"] == expected_eligible
print("PASS TensorBoard callback matches environment balance eligibility/score")
