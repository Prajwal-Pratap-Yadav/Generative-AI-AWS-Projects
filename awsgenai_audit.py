"""Inspect preserved SageMaker source and saved text without executing notebook cells."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

NOTEBOOKS = ("Model_Evaluation_UdacityGenAIAWS.ipynb", "Model_FineTuning.ipynb")
MAX_BYTES = 10_000_000


def bounded_read(path: Path) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{path.name} exceeds the 10 MB audit limit")
    return raw


def normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("Prompts must be strings")
    return " ".join(prompt.casefold().split())


def _json_value(value) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_value(item) for key, item in value.items()
        )
    return False


def _source(cell: dict) -> str:
    value = cell.get("source", [])
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "".join(value)
    if isinstance(value, str):
        return value
    raise ValueError("Cell source must be a string or a list of strings")


def _saved_text(cell: dict) -> str:
    parts = []
    for output in cell.get("outputs", []):
        for value in (
            output.get("text", ""),
            output.get("data", {}).get("text/plain", ""),
        ):
            if isinstance(value, list) and all(isinstance(line, str) for line in value):
                parts.append("".join(value))
            elif isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _payload_assignment(node) -> bool:
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else [node.target]
        if isinstance(node, ast.AnnAssign)
        else []
    )
    return any(
        isinstance(target, ast.Name) and target.id == "payload" for target in targets
    )


def notebook_summary(path: Path) -> dict:
    raw = bounded_read(path)
    book = json.loads(raw)
    if not isinstance(book, dict) or not isinstance(book.get("cells"), list):
        raise ValueError("Notebook cells array is required")
    prompts, unparsed, skipped = [], [], []
    code_cells = saved_responses = 0
    methods = {"deploy": 0, "fit": 0, "delete_endpoint": 0, "delete_model": 0}
    instances, eula = set(), False
    for index, cell in enumerate(book["cells"]):
        if not isinstance(cell, dict):
            raise ValueError("Every notebook cell must be an object")
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = _source(cell)
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError):
            # Notebook shell/magic cells are visible omissions, never executed.
            unparsed.append(index)
            if re.search(r"(?m)^\s*payload\s*(?::[^=\n]+)?=", source):
                skipped.append(
                    {
                        "cell_index": index,
                        "line": None,
                        "reason": "payload cell is not parseable Python",
                    }
                )
            continue
        top_level = {id(node) for node in tree.body}
        cell_prompts = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in methods:
                    methods[node.func.attr] += 1
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    try:
                        value = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        continue
                    if (
                        keyword.arg == "instance_type"
                        and isinstance(value, str)
                        and re.fullmatch(r"ml\.[a-z0-9]+\.[a-z0-9]+", value)
                    ):
                        instances.add(value)
                    if keyword.arg == "accept_eula" and (
                        value is True or value == "true"
                    ):
                        eula = True
                    if keyword.arg == "environment" and isinstance(value, dict):
                        eula |= (
                            value.get("accept_eula") is True
                            or value.get("accept_eula") == "true"
                        )
                    if keyword.arg == "custom_attributes" and isinstance(value, str):
                        eula |= bool(
                            re.search(r"(?:^|;)\s*accept_eula=true(?:;|$)", value)
                        )
            if not _payload_assignment(node):
                continue
            location = {"cell_index": index, "line": node.lineno}
            if id(node) not in top_level:
                skipped.append(
                    {
                        **location,
                        "reason": "payload is inside control flow or a definition",
                    }
                )
                continue
            try:
                payload = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                skipped.append(
                    {**location, "reason": "payload is dynamic or unsupported"}
                )
                continue
            if not isinstance(payload, dict) or not isinstance(
                payload.get("inputs"), str
            ):
                skipped.append(
                    {**location, "reason": "payload has no literal string inputs"}
                )
                continue
            if not normalize_prompt(payload["inputs"]):
                raise ValueError("Empty evaluation prompt")
            parameters = payload.get("parameters", {})
            if not isinstance(parameters, dict) or not _json_value(parameters):
                raise ValueError("Generation parameters must be a finite JSON object")
            prompts.append(
                {"prompt": payload["inputs"], "parameters": parameters, **location}
            )
            cell_prompts += 1
        # Count output-bearing cells once, even if a cell assigns multiple payloads.
        if cell_prompts and re.search(
            r"(?m)^\s*>\s|['\"]generation['\"]\s*:", _saved_text(cell)
        ):
            saved_responses += 1
    normalized = [normalize_prompt(prompt["prompt"]) for prompt in prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate normalized prompts would make pairing ambiguous")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "cells": len(book["cells"]),
        "code_cells": code_cells,
        "literal_prompts": prompts,
        "saved_response_cells": saved_responses,
        "deployment_calls": methods["deploy"],
        "training_calls": methods["fit"],
        "contains_endpoint_cleanup": bool(methods["delete_endpoint"]),
        "contains_model_cleanup": bool(methods["delete_model"]),
        "contains_automatic_eula_acceptance": eula,
        "instance_types": sorted(instances),
        "unparsed_code_cells": unparsed,
        "skipped_payloads": skipped,
        "call_count_scope": "AST call sites in parseable cells; not execution counts",
        "prompt_scope": "Top-level literal payload assignments only; no code execution or control-flow inference",
    }


def compare_prompts(left: dict, right: dict) -> dict:
    a = {normalize_prompt(p["prompt"]): p for p in left["literal_prompts"]}
    b = {normalize_prompt(p["prompt"]): p for p in right["literal_prompts"]}
    shared = sorted(a.keys() & b.keys())
    equal = (
        all(a[p]["parameters"] == b[p]["parameters"] for p in shared)
        if shared
        else None
    )
    explicit = bool(shared) and all(
        a[p]["parameters"] and b[p]["parameters"] for p in shared
    )
    status = (
        "no_pairs"
        if not shared
        else "unpaired_prompts"
        if a.keys() != b.keys()
        else "parameter_mismatch"
        if not equal
        else "implicit_parameters"
        if not explicit
        else "unsupported_payloads"
        if left.get("skipped_payloads") or right.get("skipped_payloads")
        else "matched"
    )
    return {
        "pairing_status": status,
        "matched_prompts": len(shared),
        "unpaired_base": sorted(a.keys() - b.keys()),
        "unpaired_finetuned": sorted(b.keys() - a.keys()),
        "all_matched_generation_parameters_equal": equal,
        "all_matched_parameters_explicit": explicit,
        "all_matched_prompt_bytes_equal": all(
            a[p]["prompt"] == b[p]["prompt"] for p in shared
        )
        if shared
        else None,
        "interpretation": "Matching prompt/settings pairs establish comparability only; no quality improvement is measured.",
    }


def audit(root: Path) -> dict:
    summaries = [notebook_summary(root / name) for name in NOTEBOOKS]
    corpus = bounded_read(root / "ITDataset.txt")
    return {
        "schema_version": 2,
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
            "Saved responses are historical; this is not a blinded quality benchmark.",
            "Call-site counts exclude comments and strings, and do not prove execution or successful cleanup.",
            "Unparseable cells and unsupported payloads are reported explicitly.",
            "Cloud instance names and SDK commands are historical settings, not current recommendations.",
            "Review model/data terms and explicit cleanup before any separately authorized cloud run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/local/notebook-audit.json")
    )
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "comparison": result["comparison"]}, indent=2
        )
    )
    if result["comparison"]["pairing_status"] != "matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
