# DIY Repair Data Generator + LLM Judge

This project generates structured DIY repair records with an OpenAI model and evaluates them with an LLM-as-judge workflow.

## What the project does

There are two main scripts:

- `data-creation.py` generates validated DIY repair records in JSONL format.
- `judge.py` reads those records and scores them against quality and safety criteria.

Each generated record includes:

- `id`
- `category`
- `prompt` (full prompt text used for generation)
- `model`
- `question`
- `answer`
- `equipment_problem`
- `tools_required`
- `steps`
- `safety_info`
- `tips`

Each judge result includes:

- `id`
- `incomplete_answer`
- `safety_violations`
- `unrealistic_tools`
- `overcomplicated_solution`
- `missing_context`
- `poor_quality_tips`
- `overall_failed`
- `notes`
- `quality`, which contains:
  - `answer_coherence`
  - `step_actionability`
  - `tool_realism`
  - `safety_specificity`
  - `tip_usefulness`
  - `problem_answer_alignment`
  - `appropriate_scope`
  - `category_accuracy`

## Requirements

- Python `3.13.x` recommended
- OpenAI API key

`pyenv` is optional. If you already have a compatible Python installed, you can use it directly.

## Setup

### Option A: with `pyenv`

```bash
pyenv install 3.13.2
pyenv local 3.13.2
python -m venv .venv
```

### Option B: with an existing Python install

```bash
python -m venv .venv
```

Activate the virtual environment:

- PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

- Bash

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=o4-mini
```

Notes:

- `OPENAI_API_KEY` is required.
- `OPENAI_MODEL` is used by `data-creation.py` unless you pass `--model`.
- `OPENAI_JUDGE_MODEL` is used by `judge.py` unless you pass `--model`.
- If `OPENAI_JUDGE_MODEL` is not set, `judge.py` falls back to `OPENAI_MODEL`, then to `gpt-4.1-mini`.

## Run the full pipeline

You can generate and judge in one command:

```bash
python run_pipeline.py --count 50
```

This runs `data-creation.py` first, then uses its output as the input to `judge.py`.

Useful options:

- `--count`
- `--generator-model`
- `--judge-model`
- `--data-output`
- `--judge-output`
- `--generate-request-delay-seconds`
- `--judge-request-delay-seconds`
- `--generate-retry-max`
- `--judge-retry-max`
- `--temperature`
- `--top-p`
- `--question-similarity-threshold`

Example:

```bash
python run_pipeline.py \
  --count 25 \
  --generator-model gpt-4.1-mini \
  --judge-model o4-mini \
  --data-output diy_repair_data.jsonl \
  --judge-output judge_results.jsonl
```

## Generate a dataset

Basic usage:

```bash
python data-creation.py --count 50 --output diy_repair_data.jsonl
```

Useful options:

- `--count` number of records to generate
- `--model` OpenAI model name
- `--output` output JSONL file path
- `--request-delay-seconds` delay between API calls
- `--retry-max` max retries for transient API failures
- `--temperature` sampling temperature
- `--top-p` nucleus sampling parameter
- `--question-similarity-threshold` duplicate/near-duplicate rejection threshold

Example:

```bash
python data-creation.py \
  --count 25 \
  --model gpt-4.1-mini \
  --output diy_repair_data.jsonl \
  --temperature 0.9 \
  --top-p 1.0
```

## Judge a dataset

Basic usage:

```bash
python judge.py --input diy_repair_data.jsonl --output judge_results.jsonl
```

Useful options:

- `--input` input JSONL file
- `--output` output JSONL file
- `--model` judge model name
- `--request-delay-seconds` delay between judge calls
- `--retry-max` max retries for transient API failures

Example:

```bash
python judge.py \
  --input diy_repair_data.jsonl \
  --output judge_results.jsonl \
  --model o4-mini
```

## Output files

- `diy_repair_data.jsonl`: generated repair records, one JSON object per line
- `judge_results.jsonl`: judge results, one JSON object per line
- `human_labels.jsonl`: human labeling template and/or completed human labels
- `human_labels_review.csv`: spreadsheet-friendly version of `human_labels.jsonl`

Judge results include the original top-level failure flags plus the nested `quality` object.

The repository also contains sample output artifacts you can inspect directly.

## Human labeling workflow

If you want to review records manually and compare them with judge outputs:

1. Create a labeling template:

   ```bash
   python generate_human_labels_template.py --input diy_repair_data.jsonl --output human_labels.jsonl
   ```

2. Convert the template to CSV for spreadsheet labeling:

   ```bash
   python generate_human_labels_csv.py --input human_labels.jsonl --output human_labels_review.csv
   ```

3. Fill in the label columns in `human_labels_review.csv`
4. Convert the labeled CSV back to JSONL:

   ```bash
   python convert_human_labels_csv_to_jsonl.py --input human_labels_review.csv --output human_labels.jsonl
   ```

5. Optionally compare human labels with judge results:

   ```bash
   python compare_judgments.py --human-labels human_labels.jsonl --judge-results judge_results.jsonl
   ```

Use `LABELING_GUIDE.md` for the rubric and consistency rules while labeling.

## Judge failure flags

The judge marks each record using binary flags:

- `0` = pass
- `1` = fail

`overall_failed` is set to `1` if any top-level failure flag is `1` or any nested quality metric is `1`.

## Quick workflow

1. Set `OPENAI_API_KEY` in `.env`
2. Run `python run_pipeline.py --count 50`
3. Review `diy_repair_data.jsonl`
4. Review `judge_results.jsonl`
5. If doing human review, generate `human_labels.jsonl` and `human_labels_review.csv`
6. After labeling the CSV, run `convert_human_labels_csv_to_jsonl.py`
