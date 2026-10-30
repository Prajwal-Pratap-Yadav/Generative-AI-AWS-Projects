# Study protocol and evidence limits

## Existing comparison

The base and fine-tuned notebooks each contain three literal prompt payloads and three saved response cells. The audit matches prompts after case folding and whitespace normalization, compares generation parameters, and hashes the original notebook and corpus files. It extracts only static literals; dynamic expressions and unrelated cells are not executed.

All three paired payloads use the same generation settings. This supports an input-comparability claim only. Case normalization can hide capitalization differences; a future experiment should store identical prompt bytes. A sampled response at temperature 0.6 is not a stable estimate of model quality. No claim about improved correctness, generalization, latency, or cost follows from these saved examples.

## Concrete next evaluation

Build a rights-cleared training corpus and a separate evaluation set before tuning. Define a factuality and relevance rubric in advance, blind the model identities, and collect repeated generations using a recorded protocol. Compare per-prompt outcomes and uncertainty against the unchanged base model. Record tokens, latency, and actual billed resources separately. Preserve prompt and artifact hashes without publishing account identifiers or credentials.

## Lifecycle review

Historical source contains deployment/training calls, inline model-term acceptance, and endpoint/model deletion calls. Static presence of these calls does not prove successful execution or complete cleanup. A future authorized cloud run should make resource creation explicit and verify cleanup even after failures. Current CI uses only the standard library, performs no notebook execution, and requires no cloud secrets.

The included corpus is 22,866 bytes (3,455 whitespace-delimited words). Its origin and redistribution rights remain unverified. Existing PDFs and notebooks are preserved; the audit does not create new corpus derivatives or publish model weights.
