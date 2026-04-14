---
name: exec-plan
description: "Use when creating, updating, or executing a self-contained Execution Plan (ExecPlan) for the Aerial Object Classification & Detection project. Covers milestone planning, data validation checkpoints, PyTorch/YOLOv8 training phases, Streamlit/Docker deployment, and living document management with Progress, Decision Log, Surprises & Discoveries, and Retrospective tracking. Use for onboarding, resuming partially-complete work, or planning any multi-step pipeline phase."
argument-hint: "Describe the milestone or phase to plan, e.g. 'Phase 1 data validation' or 'full pipeline plan'"
---

# Execution Plan (ExecPlan) Skill

## Purpose

This skill produces and maintains **self-contained Execution Plans** — living documents that enable a complete novice to implement any feature or phase of the Aerial Object Classification & Detection pipeline end-to-end, without prior knowledge of this repository.

An ExecPlan is not a todo list. It is a **comprehensive specification** that combines purpose, context, step-by-step instructions, observable outcomes, and a running record of decisions and discoveries.

## When to Use

- Starting a new pipeline phase (data validation, preprocessing, model training, deployment)
- Planning a feature that spans multiple files or systems
- Onboarding someone who has never seen this codebase
- Resuming work on a partially completed milestone
- Any task that requires more than 3 coordinated steps

## Key Terms

- **ExecPlan**: A markdown document at `docs/exec-plans/<name>.md` that contains all knowledge needed to implement a feature. It is written so that someone with Python experience but zero knowledge of this repo can follow it start-to-finish.
- **Milestone**: A discrete unit of work within an ExecPlan that produces a verifiable outcome (e.g., "data validation passes with zero orphaned files").
- **Observable outcome**: A command you can run and output you can see that proves the milestone is complete (not "code was written" but "running `python -m src.data_validation` prints `✓ 0 orphan images, 0 invalid labels`").
- **Living document**: The ExecPlan is updated as work progresses. Every decision, surprise, and course change is recorded in-place so the document always reflects current reality.

## Non-Negotiable Requirements

1. **Self-contained**: The ExecPlan contains ALL context needed. No "see architecture doc" or "as defined previously." Repeat yourself if necessary.
2. **Living**: Updated at every stopping point. Progress, discoveries, and decisions are recorded inline.
3. **Novice-friendly**: Every term of art is defined. Every file path is repository-relative. Every command shows its working directory.
4. **Observable**: Every milestone ends with a command to run and output to expect. Behavior a human can verify, not internal attributes.
5. **Idempotent**: Steps can be run multiple times safely. Destructive operations have backups or fallbacks.

## Procedure

### Step 1: Gather Context (Read Before You Write)

Before writing a single line of the plan, read the source material thoroughly:

1. Read [project_summary.md](../../../project_summary.md) — understand the business problem, workflow, datasets, deliverables
2. Read [aerial-detection.agent.md](../../agents/aerial-detection.agent.md) — understand the technical architecture, all 8 phases, constraints, technology stack
3. Read [data.yaml](../../../data.yaml) — understand the YOLOv8 dataset configuration
4. List and explore the actual directory structure to verify claims in documentation
5. Run validation commands to get **ground-truth numbers** — never trust documentation at face value

**Critical project facts** (verified 2026-04-11 — see [project-context.md](./references/project-context.md)):
- Classification and detection datasets share **identical source images** (byte-for-byte verified)
- Detection dataset: 3,400 images (Train 2,728 / Valid 448 / Test 224), 81 with empty labels
- Classification dataset: 3,319 images (= 3,400 − 81 background images)
- Bounding box imbalance: Bird 1,406 vs Drone 135 (10.4:1 ratio)
- Image-level counts roughly balanced: ~1,414 bird vs ~1,248 drone images
- Zero images contain both Bird and Drone simultaneously
- Framework: **PyTorch only** — no TensorFlow/Keras
- Detection: **YOLOv8m** (medium, 25.9M params, `ultralytics` package)
- Deployment: **Streamlit + Docker**
- Class names: `['Bird', 'drone']` — left as-is (not normalized)

### Step 2: Start from the Skeleton

Create the output directory if it doesn't exist:

```bash
mkdir -p docs/exec-plans
```

Create the ExecPlan file at `docs/exec-plans/<descriptive-name>.md` using the [template](./assets/exec-plan-template.md). Name the file after the scope of work using lowercase kebab-case, e.g. `phase-1-data-validation.md`, `phase-4-5-model-training.md`, `full-pipeline.md`. Fill in sections in this order:

1. **Why This Matters** — 2-3 sentences from the user's perspective. What can they do after this change that they could not do before?
2. **Repository Orientation** — Name every file and directory the reader will touch. Explain how they fit together.
3. **Prerequisites** — Environment, installed tools, Python version, GPU availability
4. **Milestones** — Break work into discrete, verifiable chunks. Each milestone has:
   - What to do (with exact file paths and code)
   - What command to run to verify
   - What output to expect
5. **Required Sections** — Progress, Decision Log, Surprises & Discoveries, Outcomes & Retrospective (see template)

### Step 3: Validate Every Claim

Before committing the plan:

- [ ] Every file path exists or is marked `# CREATE`
- [ ] Every command can actually be run (test it)
- [ ] Every dataset statistic matches ground truth (run counts, don't copy docs)
- [ ] Every term of art is defined on first use
- [ ] No "see X for details" — inline the needed information
- [ ] Milestones are ordered by dependency — later milestones don't depend on skipped ones

### Step 4: Execute Milestones Sequentially

When implementing an ExecPlan:

1. Mark the current milestone as `🔄 In Progress` in the Progress section
2. Implement the milestone — create files, run commands, verify output
3. If a surprise or deviation occurs, record it in **Surprises & Discoveries** with evidence (paste terminal output)
4. If you change approach, record the decision and reasoning in **Decision Log**
5. Mark the milestone as `✅ Complete` with a one-line summary of what was verified
6. Proceed to the next milestone without prompting the user

### Step 5: Close the ExecPlan

When all milestones are complete:

1. Fill in **Outcomes & Retrospective** — what was achieved, what remains, lessons learned
2. Ensure every section is up to date
3. Update the document status at the top to `🟢 Complete`
4. The final document should be usable by a new contributor starting from scratch

## Quality Checklist

Use [exec-plan-checklist.md](./assets/exec-plan-checklist.md) to validate any ExecPlan before finalizing.

## Project-Specific Phase Reference

The aerial detection pipeline has 8 phases defined in `aerial-detection.agent.md`. Each ExecPlan should map to one or more of these:

| Phase | Scope | Key Files |
|-------|-------|-----------|
| 1 | Data Validation & Integrity | `src/data_validation.py`, `scripts/validate_dataset.py` |
| 2 | Preprocessing & Augmentation | `src/preprocessing.py`, `src/config.py` |
| 3 | Bbox Imbalance Mitigation | `src/models/yolo_detector.py` (focal loss, per-class eval) |
| 4 | Model Building | `src/models/custom_cnn.py`, `transfer_learning.py`, `yolo_detector.py` |
| 5 | Training Configuration | `src/config.py`, `src/train_classifier.py`, `src/train_detector.py` |
| 6 | Evaluation | `src/evaluate.py` (confusion matrix, F1, mAP, per-class AP) |
| 7 | Streamlit Deployment | `app/app.py`, `app/components/` |
| 8 | Docker Containerization | `Dockerfile`, `docker-compose.yml` |

## Constraints (Inherited from Agent)

- PyTorch only — no TensorFlow/Keras
- Do not modify `data.yaml` class names `['Bird', 'drone']`
- Do not train on validation or test data
- Do not commit model weights to version control
- Use `pathlib.Path` relative paths — no absolute paths in code
- Phase 1 (data validation) is mandatory before any training
- Always report per-class AP for detection, not just mAP
