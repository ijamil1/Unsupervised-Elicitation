# CLAUDE.md - AI Assistant Guide for Unsupervised Elicitation

This file provides context for AI assistants working with this codebase.

## Project Overview

This repository implements an **Unsupervised Elicitation** algorithm for training language models without human labels. The approach is competitive with supervised training on:
- Common misconceptions (TruthfulQA)
- Math verification (GSM8k-verification)
- Helpfulness reward modeling (Alpaca)

The core algorithm is **ICM (Iterative Consistency Modeling)** which uses simulated annealing and in-context learning to discover latent skills in pretrained base models.

## Repository Structure

```
├── core/                     # Core LLM API infrastructure
│   ├── llm_api/             # API clients for different LLM providers
│   │   ├── llm.py           # Main ModelAPI class
│   │   ├── anthropic_llm.py # Anthropic/Claude API
│   │   ├── openai_llm.py    # OpenAI-compatible API
│   │   └── base_llm.py      # Base LLM interface
│   └── utils.py             # Environment setup utilities
├── src/
│   ├── experiments/         # Main experiment scripts
│   │   ├── ICM.py           # **PRIMARY SCRIPT** - ICM algorithm implementation
│   │   ├── ICM_tools.py     # Helper tools for ICM
│   │   └── plot.py          # Visualization tools
│   ├── pipeline/            # Data processing pipeline framework
│   │   ├── pipeline.py      # Pipeline orchestration
│   │   └── README.md        # Pipeline usage guide
│   ├── model_querying/      # LLM interaction utilities
│   │   ├── prompt_creation.py        # Prompt templates
│   │   └── solution_extraction.py    # Response parsing
│   ├── tools/               # General utilities
│   │   ├── dataloaders.py   # Data loading functions
│   │   ├── path_utils.py    # Path management
│   │   ├── printer.py       # Logging utilities
│   │   └── string_manipulation.py
│   └── runners/             # Evaluation runners
├── praxis_labs_ue1/data/    # Training/test data location
│   ├── truthfulqa_train.json
│   └── truthfulqa_test.json
├── results/                 # Experiment outputs
└── SECRETS                  # API keys and configuration (gitignored)
```

## Key Files

### Primary Entry Point
- [src/experiments/ICM.py](src/experiments/ICM.py) - Main algorithm implementation
  - `icm_main()` - Runs unsupervised ICM algorithm
  - `golden_supervision_main()` - Supervised baseline
  - `zero_shot_chat_main()` - Zero-shot chat model evaluation
  - `zero_shot_pretrained_main()` - Zero-shot base model evaluation

### Core Infrastructure
- [core/llm_api/llm.py](core/llm_api/llm.py) - ModelAPI class for making LLM calls
- [src/pipeline/pipeline.py](src/pipeline/pipeline.py) - Pipeline framework for data processing
- [src/model_querying/prompt_creation.py](src/model_querying/prompt_creation.py) - Prompt templates

## Important Concepts

### ICM Algorithm
The Iterative Consistency Modeling algorithm works by:
1. Starting with random labels on a small seed set
2. Using simulated annealing to iteratively update labels
3. Evaluating consistency via in-context learning
4. Maximizing a scoring function based on model confidence

Key parameters in [ICM.py](src/experiments/ICM.py:245-259):
- `--alpha`: Coefficient for mutual predictability scoring
- `--K`: Maximum iterations for search
- `--batch_size`: Minibatch size for large datasets
- `--num_seed`: Number of initial random-labeled examples
- `--decay`, `--initial_T`, `--final_T`: Simulated annealing parameters

### Pipeline Framework
The pipeline system ([src/pipeline/pipeline.py](src/pipeline/pipeline.py)) provides:
- Dependency-based execution graphs
- Caching of intermediate results
- Rate-limited API calls
- Step types: LoadData, Query, CodeEvaluation, Transformation, Monitoring

See [src/pipeline/README.md](src/pipeline/README.md) for usage details.

### LLM API
The `ModelAPI` class ([core/llm_api/llm.py](core/llm_api/llm.py)) supports:
- Multiple providers (OpenAI, Anthropic, custom endpoints)
- Automatic rate limiting
- Logprobs extraction (required for ICM)
- Async/await pattern for parallel requests

## Configuration

### Environment Setup
1. Create conda environment: `conda env create -f env.yaml`
2. Install package: `pip install -e .`

### Secrets File
Create a `SECRETS` file at the repository root:
```
LLAMA_API_BASE=<your_api_base_url>
NYU_ORG=None
ARG_ORG=None
API_KEY=None
```

### API Requirements
- **Critical**: Must have access to base (non-chat) models with top-K logprobs support
- Recommend deploying with vLLM and enabling prefix caching
- Most public APIs only serve chat models; you'll need custom deployment

## Data

Training/test data is located in [praxis_labs_ue1/data/](praxis_labs_ue1/data/):
- `truthfulqa_train.json` - Training examples
- `truthfulqa_test.json` - Evaluation examples

Data format:
```json
{
  "question": "...",
  "choice": "...",
  "label": true/false,
  "consistency_id": "...",
  "consistency_key": "A"/"B"
}
```

## Running Experiments

### Basic ICM Run
```bash
cd src/experiments
python ICM.py --testbed truthfulQA --alpha 50
```

### Key Arguments
- `--model`: Base model name (default: `meta-llama/Meta-Llama-3.1-405B`)
- `--batch_size`: Dataset batch size (default: 256)
- `--num_seed`: Initial labeled examples (default: 8)
- `--K`: Max iterations (default: 1500)
- `--seed`: Random seed (default: 27565976)

## Code Conventions

### Async Pattern
Most model interactions use async/await:
```python
async def icm_main(args):
    results = await pipeline.run()
    accuracy = await predict_assignment(model, example, demos)
```

### Data Structure
Examples are dictionaries with keys:
- `uid`: Unique identifier
- `label`: Current label (None if unlabeled)
- `vanilla_label`: Ground truth label (for evaluation)
- `prompt`: Formatted prompt string
- `consistency_id`: Group ID for consistency checking
- `score`: Model confidence score

### Pipeline Pattern
1. Create `PipelineConfig` with metadata
2. Initialize `Pipeline` instance
3. Add steps with dependencies: `add_query_step()`, `add_transformation_step()`, etc.
4. Call `await pipeline.run()` to execute
5. Access results via returned dictionary

## Modified Files (Git Status)

Current working changes:
- [core/llm_api/llm.py](core/llm_api/llm.py) - LLM API modifications
- [core/llm_api/openai_llm.py](core/llm_api/openai_llm.py) - OpenAI client updates
- [src/experiments/ICM.py](src/experiments/ICM.py) - Algorithm tweaks
- [src/pipeline/pipeline.py](src/pipeline/pipeline.py) - Pipeline updates

Recent commits focused on:
- Rate limit handling
- API call formatting
- Runtime error fixes
- Initial fork setup

## Common Tasks

### Adding a New Testbed
1. Add data loading function in [src/tools/dataloaders.py](src/tools/dataloaders.py)
2. Create `load_{testbed}_data()` function in [ICM.py](src/experiments/ICM.py)
3. Update prompt templates in [src/model_querying/prompt_creation.py](src/model_querying/prompt_creation.py)

### Modifying the Scoring Function
Edit `get_energy()` in [ICM.py](src/experiments/ICM.py:242-243):
```python
def get_energy(metric, alpha):
    return metric["train_prob"]  # Customize scoring here
```

### Adding New LLM Provider
1. Create new file in [core/llm_api/](core/llm_api/)
2. Inherit from base class in [base_llm.py](core/llm_api/base_llm.py)
3. Implement required methods for API calls and logprobs extraction
4. Register in [llm.py](core/llm_api/llm.py) ModelAPI class

## Testing and Evaluation

The main script runs four evaluation modes:
1. **ICM (Unsupervised)**: Main algorithm
2. **Golden Supervision**: Uses ground truth labels
3. **Zero-shot Chat**: Uses instruct-tuned chat model
4. **Zero-shot Pretrained**: Uses base model directly

Results are plotted and saved as `figure_1.png`.

## Important Notes

### Logprobs Requirement
ICM requires `top_logprobs=20` support from the API. This is essential for:
- Extracting model confidence via `extract_claim_logprobs()`
- Computing the scoring function in `get_energy()`

### Prefix Caching
Highly recommended for performance since ICM:
- Creates many similar prompts with small variations
- Adds/removes demonstrations iteratively
- Benefits significantly from caching prefixes

### Batch Processing
Large datasets are split into batches due to context limits:
- ICM runs independently on each batch
- Use `--batch_size` to control batch size
- Results can be combined via iterative fine-tuning

## Troubleshooting

### Rate Limits
- Configure `openai_fraction_rate_limit` in `PipelineConfig`
- Default is 0.99 (99% of rate limit)
- Adjust in [ICM.py](src/experiments/ICM.py:94) if hitting limits

### Memory Issues
- Reduce `--batch_size` if running out of memory
- Never use `git status -uall` (can cause memory issues on large repos)

### API Errors
- Check SECRETS file configuration
- Verify base model API supports logprobs
- Ensure API endpoint is accessible

## Related Resources

- Original paper/algorithm details: See README.md
- Pipeline usage: See [src/pipeline/README.md](src/pipeline/README.md)
- Fine-tuning: Uses [axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework

## Branch Information

- Current branch: `master`
- Main branch: `master` (use for PRs)
