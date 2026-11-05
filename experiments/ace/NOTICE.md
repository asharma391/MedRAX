# Vendored ACE (Agentic Context Engineering)

This directory is a copy of the core of ACE, used by `experiments/ace_medrax`.

- Upstream: https://github.com/ace-agent/ace
- Commit: `82709de` ("feat: parallelize ACE ComBEE proposals with LLM reducer (#39)")
- License: Apache-2.0 (see `LICENSE.txt` in this directory)
- Paper: Zhang et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving
  Language Models*, arXiv:2510.04618

## What was copied

`ace/` (ace.py, ace_batch.py, core/, prompts/) plus the top-level helpers `llm.py`, `logger.py`,
`playbook_utils.py` and `utils.py`, moved inside the package. Upstream's evaluation tasks
(finance, Mind2Web), data, the AppWorld submodule and `uv.lock` were not copied.

## Modifications

1. **Package-relative imports.** Upstream imports its helpers as top-level modules
   (`from utils import *`, `from llm import ...`), which collide with other packages' modules.
   They are now relative (`from .utils import *`, `from ..llm import ...`).
2. **`utils.initialize_clients`**: the `openai` provider honors `OPENAI_BASE_URL`, matching
   MedRAX's `main.py`.
3. **`ace.py` / `ACE._extract_config_params` and `_train_single_sample`**: new config key
   `skip_post_curate_generation` (default `False`, i.e. upstream behavior). When `True`, the
   extra generation after curation is skipped and the last answer is reused, because in MedRAX
   every generation is a full agent run with GPU tools.
4. **`ace.py` / `_environment_feedback`**: if the data processor defines
   `environment_feedback(task_dict, is_correct, default)`, the Reflector receives its output
   instead of the fixed "Predicted answer (does not) match(es) ground truth" strings.

Modifications 3 and 4 apply to `ACE` only; `ACEBatch` is unchanged from upstream.

To update from upstream, re-copy the files above at a newer commit and re-apply these changes
(search for "MedRAX modification").
