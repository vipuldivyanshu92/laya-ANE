"""Run with: python examples/quickstart.py"""

import json
from pathlib import Path

import laya_ane as laya

root = Path(__file__).parent
agent = laya.load("models/laya-coreml")
result = agent.predict(
    json.loads((root / "state.json").read_text()),
    json.loads((root / "questions.json").read_text()),
)
print(json.dumps(result, indent=2, ensure_ascii=False))
