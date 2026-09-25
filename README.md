# CollageAttack

This repository contains the core code for reproducing **CollageAttack**, a black-box jailbreak method for text-to-image (T2I) models. The pipeline includes attack prompt construction, image generation, harmfulness evaluation, image-text comparison, and semantic similarity evaluation.

## Repository Structure

```text
.
├── Dataset/
│   └── sampled_200_hate.csv
├── prompt_generation.py
├── attack_model_example.py
├── judge_image_score.py
├── judge_image_text_comparison.py
├── clip_similarity_eval_oneclick_with_cache.py
└── README.md
```

## Environment

Python 3.10+ is recommended.

Install the required packages:

```bash
pip install pandas openai python-dotenv requests pillow tqdm torch transformers open_clip_torch
```

Create a `.env` file in the project root and add the required API keys:

```bash
DS_API_KEY=your_api_key
DOUBAO_API_KEY=your_api_key
UAPI_KEY=your_api_key
```

`UAPI_KEY` is used for the **GPT-4.1** evaluation scripts. If an OpenAI-compatible API service is used, please also set the corresponding `base_url` in the judge scripts.

## Usage

### 1. Generate attack prompts

```bash
python prompt_generation.py
```

The script reads:

```text
Dataset/sampled_200_hate.csv
```

and generates:

```text
coj_commands.csv
```

Each input is decomposed into three components: scene grounding, fragmented textual inscriptions, and optional contextual cues.

### 2. Generate images

```bash
python attack_model_example.py
```

This file provides an example of calling a target T2I model. Generated images are saved under:

```text
attack_results/<model_name>/
```

For other T2I models, replace the model-specific API call while keeping the input/output format unchanged.

### 3. Evaluation

Run the following scripts for evaluation:

```bash
python judge_image_score.py
python judge_image_text_comparison.py
python clip_similarity_eval_oneclick_with_cache.py
```

- `judge_image_score.py`: uses **GPT-4.1** to evaluate image harmfulness and attack success rate (ASR).
- `judge_image_text_comparison.py`: uses **GPT-4.1** to compare the harmful impact of the generated image with the original text.
- `clip_similarity_eval_oneclick_with_cache.py`: computes image-text semantic similarity using CLIP/SigLIP/OpenCLIP backends.

Set the model name, input/output paths, API endpoint, or model cache path in each script as needed.

## Notes

- The default dataset is `Dataset/sampled_200_hate.csv`.
- Generated images follow the naming format `combined_hate_XX.png`.
- Model names and paths can be changed in the configuration section at the beginning of each script.
- Please do not upload private API keys to the repository.

## Disclaimer

This code is provided for academic research on the safety and robustness of text-to-image models. Please use it responsibly and comply with the policies of the corresponding model and API providers.
