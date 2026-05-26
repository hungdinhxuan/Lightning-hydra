# Prompt for AI Agent: Integrate Distillation with MDT Loss via Inheritance-First Design

## Objective

Design and implement a **continual learning extension** for an existing audio deepfake detection training pipeline by adding **teacher-student distillation** while **preserving the current MDT loss and existing training code**. The implementation must prioritize **adding new code instead of modifying old code**, using inheritance, wrappers, composition, or plugin-style extension wherever possible.

The target method should be suitable for research experimentation and ablation under the framing:

**Replay + MDT + Distillation for Continual Audio Deepfake Detection**

***

## Core Requirements

### 1. Preserve existing code

- Do **not** rewrite the current training pipeline unless absolutely necessary.
- Do **not** change the semantics of the current MDT loss.
- Do **not** remove or alter the existing CE / MDT / replay logic in-place if the same effect can be achieved through extension.
- Prefer:
  - subclassing existing trainer / lightning module / model wrapper,
  - adding a new loss module,
  - adding a new training strategy class,
  - wrapper-based teacher forwarding,
  - config-driven composition.

### 2. Add distillation as an extension

Implement teacher-student distillation as a **new optional feature**:

- Teacher = checkpoint before continual adaptation.
- Teacher must be frozen, in `eval()` mode, with `no_grad()`.
- Student = current trainable model.
- Distillation should initially support:
  - **logit distillation** via KL divergence with temperature,
  - optional **feature distillation** if easy to access without invasive refactor.

### 3. Keep MDT loss intact

The current MDT loss must remain a first-class component of optimization.

The total objective should be:

$$
L_{total} = L_{ce} + L_{mdt} + \lambda_{kd} L_{kd} + \lambda_{feat} L_{feat}
$$

Where:
- `L_ce` = existing classification loss.
- `L_mdt` = existing MDT loss, unchanged in behavior.
- `L_kd` = new distillation loss.
- `L_feat` = optional feature distillation loss.

If feature distillation is too invasive, keep it disabled by default.

***

## Design Principles

### A. Inheritance-first / extension-first

The preferred implementation strategy is:

1. **Keep old classes untouched whenever possible.**
2. Add a **new derived module** or **new training strategy** for continual distillation.
3. Encapsulate new functionality in clearly separated files.
4. Make it easy to compare:
   - baseline existing pipeline,
   - baseline + replay,
   - baseline + replay + MDT,
   - baseline + replay + MDT + distillation.

### B. Config-driven behavior

All new functionality must be activated via configuration, not hardcoded logic.

Add a new config section such as:

```yaml
distill:
  enabled: true
  teacher_ckpt_path: /path/to/teacher.ckpt
  temperature: 4.0
  lambda_kd: 0.3
  lambda_feat: 0.0
  apply_on: replay_only   # choices: replay_only, all
  feature_layer: null
```

### C. Minimal disruption

The implementation should avoid breaking:
- current optimizer logic,
- current scheduler logic,
- current replay data pipeline,
- current logging system,
- current LoRA / adapter loading flow.

***

## What to Implement

## 1. New modules/classes

Create new code components instead of editing core logic directly.

Suggested structure:

- `losses/distillation.py`
  - `LogitKDLoss`
  - optional `FeatureKDLoss`

- `models/teacher_wrapper.py`
  - utility for loading and freezing the teacher model

- `strategies/continual_distill_strategy.py`
  - logic for selecting which samples get KD loss
  - support `replay_only` and `all`

- `modules/<new_lightning_module>.py`
  - subclass current training module
  - reuse parent logic as much as possible
  - add distillation-specific forward / loss aggregation / logging

If the codebase uses PyTorch Lightning, prefer subclassing the current `LightningModule` rather than editing it directly.

***

## 2. Training-step integration

Implement a new training path that:

1. Runs the student forward as usual.
2. Computes existing losses exactly as before.
3. Runs teacher forward under `torch.no_grad()`.
4. Applies KD only on the configured sample subset.
5. Adds KD loss to the total objective.
6. Logs all components separately.

### Pseudocode target

```python
def training_step(batch, batch_idx):
    student_outputs = self.forward(batch)

    ce_loss = self.compute_ce_loss(student_outputs, batch)
    mdt_loss = self.compute_mdt_loss(student_outputs, batch)

    kd_loss = 0.0
    feat_kd_loss = 0.0

    if self.distill_enabled:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch)

        kd_mask = self.build_kd_mask(batch, mode=self.hparams.distill.apply_on)

        kd_loss = self.logit_kd_loss(
            student_outputs["logits"][kd_mask],
            teacher_outputs["logits"][kd_mask],
            temperature=self.hparams.distill.temperature,
        )

        if self.feature_kd_enabled:
            feat_kd_loss = self.feature_kd_loss(
                student_outputs["features"][kd_mask],
                teacher_outputs["features"][kd_mask],
            )

    total_loss = ce_loss + mdt_loss + self.lambda_kd * kd_loss + self.lambda_feat * feat_kd_loss

    self.log("train/ce_loss", ce_loss)
    self.log("train/mdt_loss", mdt_loss)
    self.log("train/kd_loss", kd_loss)
    self.log("train/feat_kd_loss", feat_kd_loss)
    self.log("train/loss", total_loss)

    return total_loss
```

***

## 3. Distillation scope

Implement at least two modes:

### Mode 1: `replay_only`
Apply KD only on replay / old samples.

This should be the default, because the goal is to preserve old knowledge with minimal side effects.

### Mode 2: `all`
Apply KD on the entire batch.

This is optional for comparison and ablation.

If replay/new identity is available in batch metadata, use it directly. If not, propose the least invasive method to expose that information.

***

## 4. Logging and metrics

The new implementation must log all important components separately so the method is usable for ablation and paper writing.

Required logs:
- `train/mdt_loss`
- `train/kd_loss`
- `train/feat_kd_loss` (if used)
- `train/loss`
- `val/old_acc`
- `val/new_acc`
- `val/old_eer` if available
- `val/new_eer` if available
- `forgetting_score`

If possible, define:

```text
forgetting_score = old_reference_metric - current_old_metric
```

where `old_reference_metric` is the score before continual adaptation.

***

## 5. Evaluation support

The implementation must support evaluating performance separately on:

- old validation set,
- new validation set,
- optional mixed validation set.

The goal is to expose the stability-plasticity tradeoff clearly.

***

## 6. Deliverables expected from the agent

Return the following in a structured format:

### A. Architecture plan
Describe how the new distillation components integrate with the current training stack while preserving the original MDT pipeline.

### B. Files to add
List every new file that should be created.

### C. Files to minimally touch
If modifying existing files is unavoidable, list them explicitly and keep changes as small as possible.

### D. Code skeleton
Provide class/function skeletons for:
- teacher loading,
- KD loss,
- derived training module,
- KD mask selection.

### E. YAML config example
Provide a complete example config for the new continual distillation mode.

### F. Validation checklist
Explain how to verify:
- old behavior remains unchanged when `distill.enabled = false`,
- MDT still works,
- teacher is frozen,
- KD only affects configured samples,
- forgetting is reduced relative to baseline.

### G. Ablation plan
Provide a minimal ablation schedule:

1. baseline current pipeline
2. baseline + replay
3. baseline + replay + MDT
4. baseline + replay + MDT + KD(replay_only)
5. baseline + replay + MDT + KD(all)

***

## Constraints

- Prefer **new code over editing old code**.
- Prefer **inheritance over replacement**.
- Prefer **composition over invasive refactor**.
- Avoid hardcoding paths or hyperparameters.
- Preserve backwards compatibility.
- Make the method easy to disable and easy to compare against the baseline.
- Optimize for experimental correctness, clarity, and maintainability.

***

## Recommended implementation order

1. Add `LogitKDLoss` as a standalone module.
2. Add a frozen teacher loader utility.
3. Create a new derived training module that reuses the parent training behavior.
4. Add `replay_only` KD mask logic.
5. Add config switches and logging.
6. Verify equivalence with baseline when distillation is disabled.
7. Add optional feature KD only if cleanly supported.

***

## Final instruction

Implement the method so that the existing project can support a new research mode named conceptually as:

**Replay + MDT + Distillation for Continual Audio Deepfake Detection**

The implementation must be suitable for:
- clean experimentation,
- hyperparameter tuning,
- ablation studies,
- paper writing,
- and long-term maintenance.

When uncertain, choose the design that **adds the fewest changes to the existing codebase while keeping the new method scientifically valid**.