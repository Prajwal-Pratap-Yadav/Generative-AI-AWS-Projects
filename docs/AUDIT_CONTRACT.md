# Offline audit contract

Schema version 2 separates observations from missing evidence. The tool parses Python syntax and saved text; it does not execute code, install notebook packages, accept terms, or call AWS.

## What is counted

Deployment, training, and cleanup counts come from AST method-call sites in parseable cells. Comments and quoted examples do not count as calls. These are source call sites, including any defined code, not execution or billing counts. The original fine-tuning notebook contains one actual `.fit(...)` call and a second commented example; the old text search counted both.

Only top-level literal `payload` assignments become prompt records. Records include cell and line locations. Dynamic assignments, payloads inside definitions/control flow, and payload-containing cells that cannot be parsed are reported as unsupported. Notebook install/magic cells remain visible in `unparsed_code_cells`. Saved response-bearing cells are counted once, including text/plain outputs, even when a cell assigns multiple payloads.

## Pairing status

An empty intersection yields `pairing_status=no_pairs` and null parameter/byte equality, rather than an accidental true result from an empty comparison. Unpaired prompts, different parameters, implicit defaults, and unsupported payloads prevent a complete match. The CLI writes the diagnostic JSON and exits with status 1 for an incomplete comparison. Malformed input exits with status 2.

Case and whitespace normalization are explicit matching rules. The three original prompt pairs match under those rules and share generation settings, but their prompt bytes are not all identical because the first prompt differs in capitalization. Matching is not evidence of improved model quality. The audit remains a small qualitative-study inspection, with unresolved corpus rights and no new cloud run.

[Regression tests](../tests/test_evidence_contract.py) exercise missing evidence, commented calls, nested/dynamic payloads, malformed parameter types, output counting, bounded reads, and the failing CLI contract. The [recorded report](../reports/notebook-audit.json) retains hashes of the original notebook and corpus bytes.
