# ACE playbooks for MedRAX

This folder trains a **playbook** for MedRAX with
[ACE (Agentic Context Engineering)](https://arxiv.org/abs/2510.04618). A playbook is a list of
strategies and pitfalls ("cross-check the classifier's effusion score with segmentation", "the
report generator over-calls cardiomegaly on AP films", ...) that is appended to MedRAX's system
prompt. No model weights change.

```
ChestAgentBench question ──► MedRAX agent (GPT-4o + tools, system prompt + playbook)
                                   │ trace: tool calls, tool outputs, answer
                                   ▼
                     Reflector (LLM): what went wrong, given the correct
                     letter and the expert explanation; tags bullets helpful/harmful
                                   ▼
                     Curator (LLM): adds new bullets to the playbook
```

| File | Purpose |
|---|---|
| `run.py` | Command-line entry point (`--mode eval / offline / online`) |
| `generator.py` | `MedRAXGenerator`: the MedRAX agent with ACE's Generator interface |
| `data.py` | ChestAgentBench loading, case-level splits, option shuffling, A–F grading |
| `evaluate.py` | Evaluation with per-question logs readable by `experiments/compare_runs.py` |
| `configs/default.json` | All settings; other configs only need the keys they change |
| `initial_playbook.txt` | Empty playbook with chest-X-ray-specific sections |
| `../ace/` | Copy of ACE's core (commit `82709de`); see `../ace/NOTICE.md` for the changes |

## Setup

Run everything from the **repository root**.

```bash
# 1. Install MedRAX with the ACE extras (Python 3.10+)
pip install -e ".[ace]"

# 2. API key (used by the agent, the Reflector and the Curator)
echo 'OPENAI_API_KEY=...' >> .env

# 3. ChestAgentBench (~820 MB of figures). Note the repo name: chest-agent-bench
huggingface-cli download wanglab/chest-agent-bench --repo-type dataset --local-dir chestagentbench
unzip chestagentbench/figures.zip -d chestagentbench

# 4. Pick a GPU; tool weights download on first use to medrax.model_dir,
#    $MEDRAX_MODEL_DIR, or ~/.cache/huggingface/hub (in that order)
export CUDA_VISIBLE_DEVICES=0
```

The default tool set matches the paper's benchmark run (classifier, segmentation, report
generator, CheXagent VQA, Maira-2 grounding, LLaVA-Med); together they need roughly 35–40 GB of
GPU memory. Drop tools with `--set 'medrax.tools=["ChestXRayClassifierTool", ...]'`.

## Workflow

**0. Smoke test** (2 questions per split; checks the whole pipeline end to end):
```bash
python -m experiments.ace_medrax.run --mode offline --max-samples 2 --eval-test --label smoke
```

**1. Freeze the splits.** The first run writes `splits/chestagentbench_seed42.json`: 80 train
cases, 30 validation cases, the remaining ~500 cases as test. Splits are **by case**, since a
case's 3–5 questions share images. Commit this file so every later run uses the same splits.

**2. Baseline** (no playbook, test split). This is the number to beat; don't compare against
the paper's numbers, because the model behind `gpt-4o` has changed since.
```bash
python -m experiments.ace_medrax.run --mode eval --label baseline
```

**3. Train a playbook, then test it:**
```bash
python -m experiments.ace_medrax.run --mode offline --eval-test --label ace-v1
```

**4. Compare** overall and per category:
```bash
python experiments/compare_runs.py \
  experiments/ace_medrax/results/*_baseline/predictions.jsonl \
  experiments/ace_medrax/results/*_ace-v1/test_eval/predictions.jsonl
```

**5. Use a playbook in the Gradio app:**
```bash
MEDRAX_PLAYBOOK=experiments/ace_medrax/results/<run>/best_playbook.txt python main.py
```

An interrupted `--mode eval` run can be resumed by passing its folder:
`--out-dir experiments/ace_medrax/results/<run>`.

## Outputs

```
experiments/ace_medrax/results/<timestamp>_<label>/
├── config.json                 # full config + CLI args of this run
├── agent_runs.jsonl            # every MedRAX run: question, tools used, answer, full trace
├── predictions.jsonl           # (eval) one line per question, compare_runs.py format
├── summary.json                # (eval) accuracy overall and per category
├── best_playbook.txt           # (offline) playbook with the best validation accuracy
├── test_eval/                  # (offline --eval-test) predictions + summary on test
└── ace_run_*/                  # ACE's own logs: intermediate_playbooks/, train_results.json,
                                #   val_results.json, curator_operations_diff.jsonl, ...
```

**Read the playbooks** (`ace_run_*/intermediate_playbooks/`), not just the accuracy. Check that
the bullets are general lessons (tool reliability, imaging pitfalls), not facts memorized from
training cases or answer-letter priors.

## Config knobs

Override anything with `--set key=value` (values are parsed as JSON) or a config file holding
only the keys you change (`--config my.json`).

| Key | Default | Notes |
|---|---|---|
| `ace.max_num_rounds` | `1` | Retries after reflection on a wrong answer. Each retry is a full agent run (ACE's default is 3). |
| `ace.skip_post_curate_generation` | `true` | Skip ACE's extra run after each curation (saves one agent run per sample). |
| `ace.eval_steps` | `100` | Validate (and possibly update `best_playbook`) every N training questions. Each validation runs the whole val split (~120 questions). |
| `ace.playbook_token_budget` | `10000` | Budget the Curator is told about. The playbook is sent in every LLM turn of every agent run, so it drives cost. |
| `ace.test_workers` | `4` | Concurrent questions. LLM calls run in parallel; tool calls are serialized behind one lock (shared GPU, non-thread-safe plotting). |
| `ace.reflector_model`, `ace.curator_model` | `gpt-4o` | Any model of `ace.api_provider` (`openai` honors `OPENAI_BASE_URL`). |
| `data.use_explanation_feedback` | `true` | Give the Reflector ChestAgentBench's expert explanation, not just "wrong". |
| `data.shuffle_answer_options` | `false` | Deterministically shuffle A–F. Answers are skewed (B 43%, C 30%), so this removes letter priors as a shortcut; scores are then not directly comparable to unshuffled runs. |
| `medrax.model` | `gpt-4o` | The agent's LLM. |
| `medrax.send_images_to_llm` | `true` | Also show the images to the LLM (otherwise only tools see them). |
| `split.*` | 80 / 30 / seed 42 | Only used when the split file does not exist yet. |

## Cost

With the defaults, training costs about 2 agent runs per training question (first attempt plus
one retry when wrong, no post-curation run), plus one val-split pass every `eval_steps` and the
final test pass. For 80 training cases (~330 questions) that is roughly 650 training runs + 3 ×
~120 validation runs + ~2,000 test runs. Run the baseline first and check the API spend before
scaling up. A MedRAX run is several GPT-4o calls, each carrying the images and the playbook.

## Limitations

- The Reflector and Curator only see text: the agent's trace and the expert explanation, never
  the image. They can learn "trusting tool X over the image was wrong here", but can't check
  image findings themselves.
- ACE currently only *adds* bullets (UPDATE/MERGE/DELETE are not implemented upstream); use
  `ace.use_bulletpoint_analyzer=true` to deduplicate.
- `ACEBatch` (ACE's parallel training mode) is not wired in; `run.py` uses `ACE`.
- Agent failures (e.g. CUDA OOM) are recorded in the trace and graded as wrong. During training
  the Reflector is told they are infrastructure failures, but they still use a training step.
