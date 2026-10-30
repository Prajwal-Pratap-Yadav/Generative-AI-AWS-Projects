# AWS Generative AI: Base vs Fine-Tuned Model Study

A historical SageMaker JumpStart exercise with an offline audit of paired prompts, generation settings, and resource lifecycle calls.

[![Checks](https://github.com/Prajwal-Pratap-Yadav/Generative-AI-AWS-Projects/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal-Pratap-Yadav/Generative-AI-AWS-Projects/actions/workflows/ci.yml)

Start with the [review guide](docs/REVIEW_GUIDE.md), [study protocol](docs/STUDY_PROTOCOL.md), and [recorded audit](reports/notebook-audit.json).

## Study question

How do captured responses to the same prompts differ before and after domain fine-tuning? Two preserved notebooks explore a Llama 2 7B JumpStart workflow using a short text corpus about ubiquitous computing and data management. The evidence is a small qualitative comparison, with no measured quality improvement or held-out benchmark.

| Artifact | Role |
| --- | --- |
| `Model_Evaluation_UdacityGenAIAWS.ipynb` | Historical base-model deployment and three saved prompt responses |
| `Model_FineTuning.ipynb` | Historical fine-tuning workflow and three saved prompt responses |
| `ITDataset.txt` | Existing corpus; provenance and redistribution rights unverified |
| [awsgenai_audit.py](awsgenai_audit.py) | Static source inspection and paired-prompt comparison |
| [tests/test_audit.py](tests/test_audit.py) | Safe parsing and comparison contracts |

## Run the offline review

Python 3.11+ is sufficient; no AWS SDK, credentials, account, or paid resources are required:

```bash
git clone https://github.com/Prajwal-Pratap-Yadav/Generative-AI-AWS-Projects.git
cd Generative-AI-AWS-Projects
python -m unittest discover -s tests -v
python -m awsgenai_audit
```

The audit parses literal payload assignments using Python's AST; it never evaluates arbitrary notebook code. It writes `reports/local/notebook-audit.json`. CI runs the same offline checks and uploads that report.

## Findings

The notebooks contain **three matched prompts** after case and whitespace normalization. All three pairs share generation settings: 64 maximum new tokens, temperature 0.6, top-p 0.9, and `return_full_text=False`. Matching inputs make the saved comparison easier to inspect; stochastic outputs, three examples, and the absence of blinded ratings prevent a quality conclusion.

Both notebooks contain deployment, endpoint cleanup, and model cleanup calls. The fine-tuning notebook also contains training calls. Presence of cleanup code does not establish that every resource was actually deleted. The recorded instance type `ml.g5.2xlarge`, SDK choices, and model identifiers describe the historical experiment, not current availability or recommendations.

## Cloud execution boundary

The original notebooks can install packages, accept model terms inline, start training, and deploy billable endpoints. They are preserved historical material rather than a current one-click deployment recipe. Any future cloud reproduction needs a separately reviewed environment, explicit model-term acceptance, least-privilege access, a cost budget, and cleanup verified against actual resources. The offline audit does none of these cloud actions.

The existing corpus's rights are unresolved. No derived dataset or model weights are added by this audit. Review [study limitations and next steps](docs/STUDY_PROTOCOL.md) before a new experiment.

## License

Repository code: [MIT](LICENSE). This does not replace the model's terms or establish rights to the third-party corpus and reports.
