# SituW
# README (We will update soon!)

---

## 1) Environment

* **Python ≥ 3.9**
* Install dependencies:

```bash
pip install openai tqdm pandas numpy python-dotenv
```

## 2) Data & API key

```bash
export DATA_PATH="/path/to/LogiQA2.0/logiqa2nli/DATA/QA2NLI"
export OPENAI_API_KEY="sk-..."   # Prefer env var over --api_key
```

## 3) Run (minimal example)

```bash
MODEL=gpt-4o-mini
DATASET=logiqa        # FOLIO | LogicalDeduction | AR-LSAT | logiqa
SPLIT=test            # train | dev | test
MODE=Ours_GPT4o-mini_extract

python src/CoT_main.py \
  --model_name "$MODEL" \
  --data_path  "$DATA_PATH" \
  --dataset_name "$DATASET" \
  --split "$SPLIT" \
  --mode "$MODE"
```

> If you have a `run.sh`, you can execute that directly.

## 4) Evaluation (optional)

```bash
python evaluate.py --dataset_name "$DATASET" --model_name "$MODEL" --split dev
```

---

## Key arguments

* `--model_name` : e.g., `gpt-4o-mini`, `gpt-3.5-turbo`
* `--data_path`  : dataset root (QA2NLI)
* `--dataset_name` : `logiqa` / `FOLIO` / `LogicalDeduction` / `AR-LSAT`
* `--split` : `train` / `dev` / `test`
* `--mode` : prompt/pipeline mode (defined in project)

(Alternative pipeline, optional): `symbcot.py`

---

## Common issues

* **Auth 401/403**: Check `OPENAI_API_KEY`
* **429 rate limit**: Add backoff/retry or slow requests
* **Path errors**: Verify `DATA_PATH`
* **Model name errors**: Ensure your SDK/version supports the model
