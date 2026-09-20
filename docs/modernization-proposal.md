# heliaEDGE modernization: initial scope and contracts

Status: proposal, 2026-09-20; implementation and support claims remain pending.
Source baseline: `99a12249acc1ab7fb04364a1e223c44051783086` (0.6.2).
Consumer architecture: <https://github.com/AmbiqAI/sleepkit/pull/33>.

## 1. Dependency baseline and import isolation

Proposed installation contracts:

- `helia-edge`: framework-independent file, factory and future evidence helpers;
  importing these must not import Keras, TensorFlow or Torch.
- `helia-edge[tensorflow]`: Keras plus TensorFlow.
- `helia-edge[torch]`: Keras plus Torch, without TensorFlow.
- Preserve the existing `litert` extra and document its conversion/runtime
  dependencies separately; Torch training does not imply Torch-to-LiteRT support.

Move Keras and backend packages out of base dependencies. Retain existing public
import paths using lazy module/symbol resolution, including nested package imports.
Accessing a feature with a missing dependency should identify the required extra.
Backend selection remains Keras's `KERAS_BACKEND`, set before Keras imports;
heliaEDGE will not silently switch it. Serialization registration needs explicit
regression coverage because eager imports currently register custom objects.

Test TF 2.21 / Keras 3.15.1 as the consumer compatibility candidate, not an
unverified minimum or latest-version claim. SleepKit's verified environment contains
heliaEDGE **0.4.1**, not this checkout's 0.6.2; its passing tests do not certify EDGE
main. Resolve and test Torch separately. Keep the locked SleepKit environment
unchanged; use separate environments for EDGE validation.
Start with Linux CPU/Python 3.12 isolation lanes; record exact resolved versions.
Validate the currently advertised Python 3.12–3.14 range and platform wheel
availability before publishing revised constraints. Keep Metal an opt-in platform
capability, subject to compatibility verification.

Acceptance: wheel installation in base-only, TF-only and Torch-only environments;
fresh-process imports; checks for absent opposite-backend packages; existing public
imports and saved-model registration; TF training and export regression checks.
Backend CI must install separate environments, not merely change an environment
variable in a combined installation.

## 2. First trainer: MaskedAutoencoder

Keep the existing constructor and `calculate_loss(x, test=False)` return tuple.
Propose a separately callable `reconstruction_targets(x, training=False)` returning
`(target_patches, predicted_patches)`. It computes the forward path without
compilation, optimizer updates or metric updates. `calculate_loss` composes it with
the compiled Keras loss; native loops can apply their own objective to its outputs.
Confirm the name and tensor ordering through fixtures before publishing the API.

Port the trainer and its patch encoder together: replace TF-specific batched gathers
with public Keras operations, use tracked Keras random state, and propagate
`training` to encoder/decoder and compatible layer calls. Preserve ordinary
callable compatibility deliberately, with its train/eval limitations documented.
The current `test` argument is unused in the forward path; correct this with a
Dropout/BatchNormalization regression fixture. Track scalar loss separately from
prediction metrics.

Keep TF GradientTape and Torch autograd/optimizer execution in small explicit
backend methods. Expose portable computation without requiring callers to adopt a
new trainer superclass. Test native TF and Torch optimizers on the component as
well as Keras `fit()`/`evaluate()`. Existing native user models can independently
use generic helpers; this does not promise arbitrary native models work inside
every Keras trainer.

Acceptance: deterministic-mask forward and gradient comparisons; actual parameter
updates; repeated-step gradient clearing; evaluation without parameter updates;
train/eval state; sample-weight policy; tracked parameters/randomness; model/layer
save/reload and applicable optimizer restoration. Record mixed precision,
compilation, distribution and cross-backend checkpoint limits separately.
`training=False` controls layer behavior, not deterministic masking; control masks
or seeds explicitly for parity fixtures. SleepKit has no current MAE consumer and
must not count as independent consumer proof for this trainer API.

Keras reference patterns:
<https://keras.io/guides/custom_train_step_in_torch/> and
<https://keras.io/guides/writing_a_custom_training_loop_in_torch/>.

## 3. Loader correctness and timing

The current generator helper drops remainder IDs, divides by zero for empty IDs,
and calls `map(None, ...)` when optional preprocessing is omitted. It interleaves
generators within a process; its worker count is not a process-pool contract.

Preserve its public positional arguments. Specify invalid-worker handling, typed
empty datasets, identity behavior for absent preprocessing and complete finite
coverage. A balanced partition alone cannot establish repeated-stream sampling
correctness: arbitrary generator callbacks may emit unequal numbers of windows or
implement custom subject weights.

Before selecting repeated-stream behavior, obtain the consumer's finite/repeated,
replacement, subject/window/class weighting and epoch-budget contract. Propose a
global logical ID/window schedule owned by the caller as the correctness reference;
partition execution only where fixtures prove the declared distribution. Do not
silently reinterpret an opaque generator as uniform window sampling. Worker-count
independent coverage and worker-count independent ordering are separate promises.

Acceptance fixtures include empty IDs, fewer IDs than workers, remainder IDs,
invalid worker counts, missing preprocessing, unequal windows per subject and
finite versus repeated schedules. Verify exact finite counts and declared repeated
sampling probabilities before performance comparisons.

Measure cold preparation, warm loading, resident-batch model steps and end-to-end
training separately. Record hardware, exact environment, logical sample schedule,
batch size, memory limits, warmup, repetitions and device synchronization. Use
synthetic fixtures for mechanics; only consumer workloads establish useful speedups.

### SleepKit finite reference contract

The consumer response and schedule are maintained by SleepKit under
`/home/adamp/Ambiq/adks/sleepkit-evaluation-evidence/coordination/`:
`sleepkit-edge-consumer-contract.md` and `sleepkit-finite-schedule.json`.

First integration targets finite native contexts without replacement, subject/class
balancing or sample weighting. Preserve subject boundaries, gaps, eligibility,
labels, timestamps and masks. Keep final partial batches. Ordered validation/test
must match the caller's logical schedule, not worker completion order. Training's
existing bounded TF shuffle is a distinct declared ordering policy; controlled
backend comparisons use an identical explicit batch schedule.

The synthetic fixture has five subjects with 1/3/0/2/7 contexts: 13 examples,
batch size 4, four batches and a final batch of one. Exercise worker counts 1/2/4/8
with exact coverage and ordered reconstruction. This supplies no timing evidence.
Keep training-only normalizer fitting over all valid training feature frames
distinct from context eligibility. Legacy repeated staging/apnea paths require
separate characterization; do not migrate their sampling by inference.

SleepKit supplied observed legacy import paths and executable synthetic
train/save/export/release checks, reporting 46 passing tests in its locked 0.4.1
environment. Those checks do not exercise EDGE trainers or custom-object
serialization. Add representative EDGE custom-layer checks independently, and
triage stale legacy imports before promising compatibility.

## Coordination and delivery

Deliver focused PRs for dependency/import isolation, the first trainer and loader
correctness/benchmark evidence. Record tested commit pins before SleepKit integration;
do not invent a future version pin. SleepKit supplies representative imports,
saved-model/export fixtures and its sampling contract, and owns its integration PR.

Artifact extraction follows consumer proof. Candidate contracts are explicit file
inventories, tensor signatures, hashes and evidence binding; no new public schema
is fixed here. Require a second real consumer before promoting a shared API.
Dataset semantics, golden recipes, thresholds, licenses and release decisions stay
with each KIT. This proposal makes no changes to SleepKit or other KIT repositories.
