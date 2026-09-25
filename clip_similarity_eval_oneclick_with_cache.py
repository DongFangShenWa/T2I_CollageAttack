
import os

BASE_URL = "."


INPUT_FILE = os.path.join(BASE_URL, "coj_commands.csv")

MODEL_NAME = "gptimage2"


OUTPUT_FILE = os.path.join(BASE_URL, "clip_cos_results", MODEL_NAME, "clip_similarity_results_google.csv")


IMAGE_DIR = os.path.join(BASE_URL, "attack_results", MODEL_NAME)



EMBEDDING = "google_siglip"



IMAGE_PREFIX = "combined_hate_"
IMAGE_EXT = ".png"
START_INDEX = 1
DIGITS = 2


CSV_ENCODING = "utf-8-sig"


BATCH_SIZE = 16


DEVICE = "auto"


FP16 = True


MODEL_CACHE_DIR = "E:/embedding_models"


HF_CLIP_MODEL = "openai/clip-vit-base-patch32"


GOOGLE_SIGLIP_MODEL = "google/siglip-base-patch16-224"


OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "openai"


KEEP_ALL_INPUT_COLUMNS = False


# ============================================================
# 下面一般不用改
# ============================================================

import sys
import os
from pathlib import Path
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HF_HUB_DISABLE_XET"] = "1"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if MODEL_CACHE_DIR:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


    os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE_DIR, "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(MODEL_CACHE_DIR, "huggingface", "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_CACHE_DIR, "huggingface", "transformers"))


    os.environ.setdefault("TORCH_HOME", os.path.join(MODEL_CACHE_DIR, "torch"))

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn.functional as F


def get_device():
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("[WARN] DEVICE='cuda' but CUDA is not available. Falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")

    return torch.device(DEVICE)


class HFClipEmbedder:
    def __init__(self, model_id: str, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model_id = model_id
        cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface") if MODEL_CACHE_DIR else None
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.eval()

    @torch.no_grad()
    def encode_pairs(self, texts, images):
        inputs = self.processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_autocast = bool(FP16) and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        image_embeds = F.normalize(image_embeds.float(), p=2, dim=-1)
        text_embeds = F.normalize(text_embeds.float(), p=2, dim=-1)

        cosine = (image_embeds * text_embeds).sum(dim=-1)
        return cosine.detach().cpu().tolist()


class SiglipEmbedder:
    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.model_id = model_id
        cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface") if MODEL_CACHE_DIR else None
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.eval()

    @torch.no_grad()
    def encode_pairs(self, texts, images):
        inputs = self.processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_autocast = bool(FP16) and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            outputs = self.model(**inputs)

            if getattr(outputs, "image_embeds", None) is None or getattr(outputs, "text_embeds", None) is None:
                raise RuntimeError(
                    "This SigLIP model output does not contain image_embeds/text_embeds. "
                    "Try GOOGLE_SIGLIP_MODEL = 'google/siglip-base-patch16-224'."
                )

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        image_embeds = F.normalize(image_embeds.float(), p=2, dim=-1)
        text_embeds = F.normalize(text_embeds.float(), p=2, dim=-1)

        cosine = (image_embeds * text_embeds).sum(dim=-1)
        return cosine.detach().cpu().tolist()


class OpenClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: torch.device):
        import open_clip

        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        cache_dir = os.path.join(MODEL_CACHE_DIR, "openclip") if MODEL_CACHE_DIR else None
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_pairs(self, texts, images):
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tokens = self.tokenizer(list(texts)).to(self.device)

        use_autocast = bool(FP16) and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            image_embeds = self.model.encode_image(image_tensors)
            text_embeds = self.model.encode_text(text_tokens)

        image_embeds = F.normalize(image_embeds.float(), p=2, dim=-1)
        text_embeds = F.normalize(text_embeds.float(), p=2, dim=-1)

        cosine = (image_embeds * text_embeds).sum(dim=-1)
        return cosine.detach().cpu().tolist()


def build_embedder(device: torch.device):
    if EMBEDDING == "hf_clip":
        print(f"[INFO] Loading HuggingFace CLIP: {HF_CLIP_MODEL}")
        return HFClipEmbedder(HF_CLIP_MODEL, device), HF_CLIP_MODEL

    if EMBEDDING == "google_siglip":
        print(f"[INFO] Loading Google SigLIP: {GOOGLE_SIGLIP_MODEL}")
        return SiglipEmbedder(GOOGLE_SIGLIP_MODEL, device), GOOGLE_SIGLIP_MODEL

    if EMBEDDING == "openclip":
        print(f"[INFO] Loading OpenCLIP: {OPENCLIP_MODEL}, pretrained={OPENCLIP_PRETRAINED}")
        return OpenClipEmbedder(OPENCLIP_MODEL, OPENCLIP_PRETRAINED, device), f"{OPENCLIP_MODEL}:{OPENCLIP_PRETRAINED}"

    raise ValueError(
        f"Unsupported EMBEDDING = {EMBEDDING!r}. "
        "Choose one of: 'hf_clip', 'openclip', 'google_siglip'."
    )


def safe_open_image(path: Path):
    if not path.exists():
        return None, "missing"
    try:
        img = Image.open(path)
        return img.convert("RGB"), ""
    except UnidentifiedImageError:
        return None, "invalid_image"
    except Exception as e:
        return None, f"error: {type(e).__name__}: {e}"


def make_image_path(row_position: int) -> Path:
    image_index = START_INDEX + row_position
    image_name = f"{IMAGE_PREFIX}{image_index:0{DIGITS}d}{IMAGE_EXT}"
    return Path(IMAGE_DIR) / image_name


def main():
    input_file = Path(INPUT_FILE)
    output_file = Path(OUTPUT_FILE)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"INPUT_FILE not found: {input_file}")

    print("[INFO] Reading CSV...")
    print(f"[INFO] INPUT_FILE = {input_file}")
    print(f"[INFO] MODEL_CACHE_DIR = {MODEL_CACHE_DIR if MODEL_CACHE_DIR else 'default'}")

    df = pd.read_csv(input_file, encoding=CSV_ENCODING)

    if "original_text" not in df.columns:
        raise ValueError(f"CSV must contain column 'original_text'. Found columns: {list(df.columns)}")

    device = get_device()
    print(f"[INFO] DEVICE = {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU = {torch.cuda.get_device_name(0)}")

    records = []
    valid_items = []

    print("[INFO] Checking image files...")
    for row_pos, row in df.iterrows():
        image_index = START_INDEX + row_pos
        image_path = make_image_path(row_pos)

        original_text = "" if pd.isna(row["original_text"]) else str(row["original_text"])
        img, missing_reason = safe_open_image(image_path)
        image_generated = img is not None

        if KEEP_ALL_INPUT_COLUMNS:
            rec = row.to_dict()
        else:
            rec = {"original_text": original_text}

        rec.update({
            "index": image_index,
            "image_path": str(image_path),
            "image_generated": image_generated,
            "clip_cosine": None,
            "clip_score": None,
            "missing_reason": missing_reason,
            "embedding_backend": EMBEDDING,
            "embedding_model": "",
        })

        if image_generated:
            valid_items.append((len(records), original_text, img))

        records.append(rec)

    print(f"[INFO] Total rows: {len(records)}")
    print(f"[INFO] Existing valid images: {len(valid_items)}")
    print(f"[INFO] Missing/invalid images: {len(records) - len(valid_items)}")

    if valid_items:
        embedder, embedding_model_name = build_embedder(device)

        for rec in records:
            rec["embedding_model"] = embedding_model_name

        batch_size = max(1, int(BATCH_SIZE))

        print("[INFO] Computing CLIP cosine similarity...")
        for start in tqdm(range(0, len(valid_items), batch_size), desc="Batches"):
            batch = valid_items[start:start + batch_size]
            rec_indices = [x[0] for x in batch]
            texts = [x[1] for x in batch]
            images = [x[2] for x in batch]

            cosines = embedder.encode_pairs(texts, images)

            for rec_idx, cos_val in zip(rec_indices, cosines):
                cos_val = float(cos_val)
                score = max(cos_val, 0.0) * 100.0

                records[rec_idx]["clip_cosine"] = cos_val
                records[rec_idx]["clip_score"] = score

    out_df = pd.DataFrame(records)

    # 调整常用列顺序，让结果更好看
    preferred_cols = [
        "index",
        "original_text",
        "image_generated",
        "clip_cosine",
        "clip_score",
        "image_path",
        "missing_reason",
        "embedding_backend",
        "embedding_model",
    ]
    remaining_cols = [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[[c for c in preferred_cols if c in out_df.columns] + remaining_cols]

    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved result to: {output_file}")


if __name__ == "__main__":
    main()
