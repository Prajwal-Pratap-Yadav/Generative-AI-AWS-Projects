"""Audit preserved SageMaker notebooks offline; never execute notebook cells or AWS calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

NOTEBOOKS = ("Model_Evaluation_UdacityGenAIAWS.ipynb", "Model_FineTuning.ipynb")


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def notebook_summary(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) > 10_000_000:
        raise ValueError("Notebook exceeds the 10 MB audit limit")
    book = json.loads(raw)
    if not isinstance(book.get("cells"), list):
        raise ValueError("Notebook cells array is required")
    code, prompts, saved_responses = [], [], 0
    for cell in book["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        code.append(source)
        # AST parsing reads constants only. Install magics and dynamic payloads are not run.
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(t, ast.Name) and t.id == "payload" for t in node.targets
            ):
                continue
            try:
                payload = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if not isinstance(payload, dict) or not isinstance(
                payload.get("inputs"), str
            ):
                continue
            if not normalize_prompt(payload["inputs"]):
                raise ValueError("Empty evaluation prompt")
            prompts.append(
                {
                    "prompt": payload["inputs"],
                    "parameters": payload.get("parameters", {}),
                }
            )
            for output in cell.get("outputs", []):
                text = output.get("text", [])
                text = "".join(text) if isinstance(text, list) else text
                saved_responses += int("generation" in text or "> " in text)
    normalized = [normalize_prompt(p["prompt"]) for p in prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate normalized prompts would make pairing ambiguous")
    source = "\n".join(code)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "cells": len(book["cells"]),
        "code_cells": len(code),
        "literal_prompts": prompts,
        "saved_response_cells": saved_responses,
        "deployment_calls": len(re.findall(r"\.deploy\s*\(", source)),
        "training_calls": len(re.findall(r"\.fit\s*\(", source)),
        "contains_endpoint_cleanup": ".delete_endpoint(" in source,
        "contains_model_cleanup": ".delete_model(" in source,
        "contains_automatic_eula_acceptance": bool(
            re.search(r"accept_eula[\"']?\s*[:=]\s*[\"']?true", source)
        ),
        "instance_types": sorted(set(re.findall(r"ml\.[a-z0-9]+\.[a-z0-9]+", source))),
    }


def compare_prompts(left: dict, right: dict) -> dict:
    a = {normalize_prompt(p["prompt"]): p for p in left["literal_prompts"]}
    b = {normalize_prompt(p["prompt"]): p for p in right["literal_prompts"]}
    shared = sorted(a.keys() & b.keys())
    return {
        "matched_prompts": len(shared),
        "unpaired_base": sorted(a.keys() - b.keys()),
        "unpaired_finetuned": sorted(b.keys() - a.keys()),
        "all_matched_generation_parameters_equal": all(
            a[p]["parameters"] == b[p]["parameters"] for p in shared
        ),
        "interpretation": "Matching prompt/settings pairs establish comparability only; no quality improvement is measured.",
    }


def audit(root: Path) -> dict:
    summaries = [notebook_summary(root / name) for name in NOTEBOOKS]
    corpus = (root / "ITDataset.txt").read_bytes()
    return {
        "scope": "Static notebook/corpus inspection; no installs, cloud requests, training or generation",
        "notebooks": summaries,
        "comparison": compare_prompts(*summaries),
        "corpus": {
            "path": "ITDataset.txt",
            "bytes": len(corpus),
            "sha256": hashlib.sha256(corpus).hexdigest(),
            "words": len(corpus.decode("utf-8").split()),
            "redistribution_rights_verified": False,
        },
        "limits": [
            "Three saved prompt pairs are not a blinded evaluation or a statistical benchmark.",
            "Cloud instance names and SDK commands are historical notebook settings, not current recommendations.",
            "Review model/data terms and explicit resource cleanup before any separately authorized cloud run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/local/notebook-audit.json")
    )
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "comparison": result["comparison"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
