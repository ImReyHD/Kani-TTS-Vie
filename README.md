# Kani Vi – Vietnamese TTS Pipeline

This project contains code and notebooks for inference and fine-tuning of a Vietnamese text-to-speech system built on top of NVIDIA NeMo codecs and a Hugging Face causal language model.

## Project Layout

- `main.py` – End-to-end inference script that logs into Hugging Face (via `HF_TOKEN`), loads the NeMo codec and causal LM, generates speech for one or more speaker IDs, and writes WAV files.
- `kani-tts-inference.ipynb` – Reference notebook that documents the inference pipeline in detail (token layout, sampling parameters, speaker control, number-to-word conversion, etc.).
- `prepare_dataset.ipynb` – Notebook for dataset preparation (number normalization, token mapping, shard processing).
- `finetune/` – Notebooks dedicated to LoRA-based fine-tuning (`finetune/kani-tts-vi-finetune.ipynb`) and dataset preparation within the fine-tuning workflow.
- `pyproject.toml`, `uv.lock` – Dependency management via [uv](https://github.com/astral-sh/uv).

Large artifacts such as generated audio (`audio-*.wav`) are ignored by default; keep them out of the repository to avoid unnecessary bloat.

## Environment Setup

1. **Python and uv**  
   Install uv (e.g. `curl -LsSf https://astral.sh/install.sh | sh`) and ensure Python 3.12.x is available.

2. **Create the virtual environment**  
   ```bash
   uv sync
   ```

3. **Hugging Face credentials**  
   The code expects an environment variable `HF_TOKEN` or prior CLI login:
   ```bash
   export HF_TOKEN=hf_xxx                # temporary for current shell
   # or
   uv run -- huggingface-cli login       # stores token in your keyring/cache
   ```
   > **Never hardcode tokens** in source files. Store them in environment variables or a `.env` file that is excluded from version control.

4. **Optional runtime tools**  
   - `sudo apt install ffmpeg` (inside WSL/Linux) to avoid pydub warnings when playing audio.

## Quick Inference Guide

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure Hugging Face credentials

```bash
export HF_TOKEN=hf_your_token_here    # current shell only
# or persist credentials:
uv run -- huggingface-cli login
```

> **Never hardcode tokens** in source files. Store them in environment variables or a `.env` file that is excluded from version control.

### 3. Run the inference script

```bash
uv run python main.py
```

`main.py` by default:

- Instantiates `NemoAudioPlayer` and `KaniModel`.
- Generates responses for multiple speaker IDs (`nu-mien-bac`, `nu-mien-nam`, `nam-mien-bac`, `nam-mien-nam`).
- Saves normalized audio to disk with `soundfile` (22.05 kHz WAV).

Customize it by editing the `prompt`, adjusting sampling parameters in `Config`, or wiring in the `convert_numbers_to_words` helper from the notebook for number normalization.

## Dataset Preparation & Fine-Tuning

1. **`prepare_dataset.ipynb`**  
   - Converts raw transcripts to token sequences.
   - Demonstrates number normalization and token boundary handling.

2. **`finetune/kani-tts-vi-finetune.ipynb`**  
   - Walks through LoRA fine-tuning.
   - Shows how to load multiple Hugging Face datasets, shard them, and train with `transformers.Trainer`.
   - Saves fine-tuned checkpoints and tokenizer artifacts back to Hugging Face.

Before running the fine-tuning notebook:

- Set `HF_TOKEN`.
- Verify dataset access and quotas.
- Adjust hyperparameters, speaker IDs, and output directories to match your setup.

## Token & Credential Safety

- **Audit status:** No plain-text Hugging Face tokens or other secrets are present in the repository. All code expects `HF_TOKEN` through the environment.
- Use `.env` (ignored by git) or shell configuration files to store tokens locally.
- Avoid committing `uv.lock` updates that might contain private resolver URLs or credentials.
- Double-check notebook outputs before pushing; remove cells that may contain sensitive logs.

## Troubleshooting

- **WSL filesystem issues:** For best performance, run `uv` and Python from the Linux home directory (`~/kani-vi`) rather than `/mnt/...`.
- **Dependency conflicts:** The project pins `onnx<1.19` to stay compatible with NeMo 2.5.2; avoid upgrading beyond that unless you also update NeMo.
- **Audio sounds unnatural:** Ensure `do_sample=True`, temperature/top-p match the notebook, and prompts are pre-normalized (especially numbers).
- **401 from Hugging Face:** Re-run `huggingface-cli login` or export `HF_TOKEN`. Make sure the model repo is public or that your token has read access.

## License

Add your licensing information here before publishing the repository.

## Contributing

1. Fork the repo.
2. Create a feature branch.
3. Format/lint your changes (follow `uv` workflow).
4. Submit a pull request describing your improvements.

Feel free to open issues for bugs or feature requests. Happy building!

