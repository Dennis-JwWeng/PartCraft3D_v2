"""Live Gate C test: judge real FLUX before/after pairs via the VLM.

Calls the actual Gate C judge (_call_quality_judge_vlm + _passes_quality_thresholds)
against a running sglang VLM server, on the pairs produced by run_pipeline_minimal.py.
"""
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI
from partcraft.pipeline_v3.vlm_core import (
    _make_2d_pair_collage, _call_quality_judge_vlm,
    _passes_quality_thresholds, _QE_DEFS,
)

URL = "http://localhost:8002/v1"
D = Path("outputs/minimal/08/bdd36c94f3f74f22b02b8a069c8d97b7")
CASES = [
    dict(edit_id="scale_1", edit_type="scale", part="wooden bowl body",
         prompt="Make the wooden bowl taller and deeper", ep={}),
    dict(edit_id="modification_2", edit_type="modification", part="wooden ring base",
         prompt="Give the wooden bowl a tall cylindrical pedestal foot at its base",
         ep={"new_part_desc": "tall cylindrical pedestal foot"}),
]


def _model_id():
    with urllib.request.urlopen(f"{URL}/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


async def main():
    model = _model_id()
    print(f"VLM model id: {model}\n")
    client = AsyncOpenAI(base_url=URL, api_key="EMPTY")
    for c in CASES:
        coll = _make_2d_pair_collage(D / f"{c['edit_id']}_input.png",
                                     D / f"{c['edit_id']}_edited.png")
        if coll is None:
            print(f"{c['edit_id']}: missing images"); continue
        j = await _call_quality_judge_vlm(
            client, model, coll,
            edit_prompt=c["prompt"], edit_type=c["edit_type"],
            object_desc="a wooden bowl", part_label=c["part"],
            target_part_desc=c["part"], edit_params=c["ep"])
        ok = _passes_quality_thresholds(j, c["edit_type"], _QE_DEFS) if j else False
        print(f"=== {c['edit_id']} ({c['edit_type']}) ===")
        print(json.dumps(j, ensure_ascii=False, indent=2) if j else "VLM no response")
        print("GATE_C:", "PASS" if ok else "FAIL", "\n")


if __name__ == "__main__":
    asyncio.run(main())
