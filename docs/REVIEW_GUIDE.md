# AWS base versus fine-tuned model study: review guide

Inspect a historical cloud experiment safely through static parsing and a paired-prompt comparability audit.

## Follow the implementation

| Start here | File | What to inspect |
| --- | --- | --- |
| Offline audit | [awsgenai_audit.py](../awsgenai_audit.py) | Literal AST extraction, paired settings, lifecycle calls, and source hashes. |
| Audit tests | [tests/test_audit.py](../tests/test_audit.py) | No-code-execution contracts, malformed pairs, and setting mismatches. |
| Recorded report | [reports/notebook-audit.json](../reports/notebook-audit.json) | Three matched prompts and the limits of the comparison. |
| Study protocol | [docs/STUDY_PROTOCOL.md](../docs/STUDY_PROTOCOL.md) | Evaluation design, corpus rights, and cloud lifecycle boundaries. |

## Run the checks

Run from the repository root after the setup described in the [README](../README.md). Review fixtures and recorded evidence before starting any full experiment.

```bash
python -m unittest discover -s tests -v
python -m awsgenai_audit
```

## Interpret the evidence

The command needs only Python 3.11+ and does not run AWS notebook cells. Three saved response pairs do not establish a quality improvement. Cloud commands and instance names are historical. Corpus rights remain unresolved, and cleanup code alone does not prove resources were deleted.

The linked reports retain their original experiment dates and protocols. A new passing CI run verifies its named checks; it does not silently refresh historical measurements.

## Discuss the design

- Why compare prompt settings before interpreting response differences?
- What independent evaluation would distinguish memorization from useful adaptation?
