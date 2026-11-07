# Optional ONNX teacher service

This optional service labels training comments. It is independent of the ML
student: it imports no student service, PyTorch, GGUF or llama.cpp code.
The initial configuration uses the pinned large NLI checkpoint. Its scores are experimental silver labels, not human gold.

The original five services remain the default. `dataset` / `reviewed` training and ready
student predictions do not call this service. Explicitly choose the label
source `service` to use it during training. Teacher is excluded from the five
mandatory health checks returned by Web Master's `/api/status`.

The default student is a small TF-IDF classifier trained from the prepared dataset.
It neither loads a neural encoder nor calls Teacher. The explicit `semantic`
variant uses frozen Base NLI and 54 learned coefficients; this optional Teacher
uses its separately configured Large checkpoint. Teacher availability does not
establish the quality of its labels.

## Run with the monitor

```sh
python3 demo/start.py --config demo/1k.yaml
```

This starts the optional `training` profile. The ordinary demo still trains
from `dataset`; it does not request new neural labels. Teacher Swagger is at
[localhost:8082/docs](http://127.0.0.1:8082/docs). `ports.teacher` in the demo YAML
changes the published loopback port; ML uses `http://teacher:8000` internally.

The image build installs runtime dependencies without downloading the roughly
1.75 GB of configured model/tokenizer assets. `/health`, `/metadata` and `/docs`
start without the NN. The first in-scope `POST /label` downloads and verifies
the assets in the persistent `teacher-encoder-weights` volume. To prefetch them
explicitly after starting Teacher:

```sh
docker compose -p hn-monitor-demo-1k --profile training exec teacher python prepare.py
```

To use Teacher, create a model with `POST /services/ml/models` in the main Swagger,
then call `POST /services/ml/models/{id}/train` with
`{"teacher": "service", "mode": "from_scratch", "limit": 100}`.
Poll `GET /services/ml/jobs/{id}` for the result. The chosen entity must have imported
comments from at least two threads.

For a native run, select a writable cache and add the optional process:

```sh
export TEACHER_ENCODER_CACHE_DIR="$PWD/.local-data/teacher/encoder-weights"
python demo/run.py --config demo/1k.yaml
```

At the default base port 8100, Teacher Swagger uses 8105 and the main gateway
uses 8103. `native.base_port` in the YAML moves these together. With `with_teacher: false`,
the original five services start. Teacher failure does not stop those processes.

## REST and label cache

`/health` reports the service and whether the lazy model session is loaded.
A 200 response before loading is expected. `/metadata` supplies the revision,
asset checksums and exact input/output adapter protocol without loading the NN.
`POST /label` accepts `entity`, optional `aliases` and `matching`, and 1–100
comments in `items`. Each comment has `id`, `story_id`, `text`, and optional
`story_title`, `parent_text`, `parent_id`. `title`/`parent` are also accepted.
The response is `{revision, items: [{comment_id, probabilities, input_hash}]}`.
The six finite, nonnegative probabilities sum to one and follow A/B/C/D/E/F
from `lib.sentiment`; scores are calculated by the caller. Outside the shared
mention scope, the response is F with probability one and no model is loaded.

Clients should import `input_hash` (also named `teacher_input_hash`) from
`lib.teacher_contracts`. It hashes the complete raw text/context, entity,
aliases and effective mention rules. The revision hashes the pinned model and
tokenizer assets plus the adapter protocol. Changes to those inputs change the
revision; deployment cache paths do not.

`runtime/ml/teacher_client.py` first pins `/metadata` for the training job,
then batches requests. Storage caches the labels under
`entity + service-<revision> + comment_id`; a row is reused only if the full
input hash matches. Duplicate or missing IDs, foreign IDs, wrong hashes and a
response revision different from the pinned one are rejected. A new teacher
revision or a corrected reviewed target makes supervision fresh for
incremental training while preserving train/holdout thread roles. Reviewed
corrections are applied to training rows after the split.

Invalid request bodies return 422; unavailable or invalid model
assets/inference return 503. An unavailable teacher fails an explicitly
requested `service` training job while preserving the previous ready student.
Student prediction never calls Teacher. Inference calls and lazy initialization
are locked; ONNX Runtime uses two CPU threads by default.

## Replace the model

To change the HF model, edit YAML or point `TEACHER_CONFIG` to another YAML:

- Pin `model_weights` and `tokenizer` using `{path, is_local, sha256}`. Optional
  `size_bytes` verifies file size too. Relative local sources are resolved from
  the config's directory. Only self-contained ONNX exports are supported.
- For `backend: sequence_classifier`, configure source-logit order with
  `class_mapping`, such as `[B, C, D]` for negative/neutral/positive. Multiple
  source classes may map to the same A–F class and their probability mass sums.
- Pair inputs use `sequence: {input_format: pair, text_template: '{text}',
  target_template: '{entity}'}`. For models requiring a literal separator, use
  `input_format: text` and a template such as `'{text} </s> {entity}'`.
- NLI uses `nli.premise_template`, three entity-specific `nli.hypotheses`, and
  `entailment_index`. Their three raw entailment logits are softmax-normalized
  and mapped using `class_mapping`; metadata calls these uncalibrated
  `hypothesis_scores_softmax`, not calibrated model confidence.
- Match the tokenizer's `pad_token_id`, `pad_token`, and model `max_length`.

Templates can reference `{entity}`, `{text}`, `{story_title}`, `{parent_text}`
and their `{title}`/`{parent}` aliases. The default NLI tone uses only the current
comment; parent/title content is used by the mention gate.

The Dockerfile creates a writable cache owned by UID/GID 10001 and runs without
root. Compose preserves downloaded assets in `teacher-encoder-weights`.
Cached files are named by
SHA-256, verified before use, and atomically installed by
`lib/model_assets.py`. `TEACHER_ENCODER_CACHE_DIR` selects the deployment cache;
`prepare.py --cache-dir` can override it for prefetching. No runtime network is
needed once those assets are present. After replacing pinned weights in YAML,
prefetch the matching assets before an offline run. Native prefetching uses
`python runtime/teacher/prepare.py` with the same cache environment variable.
