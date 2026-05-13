from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image


def load_grounding_dino(model_id: str, device: torch.device):
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def grounding_dino_detect(
    processor,
    model,
    pil_image: Image.Image,
    prompt: str,
    box_thr: float,
    text_thr: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=target_sizes.cpu(),
    )[0]
    dets: List[Dict[str, Any]] = []
    for lab, sc, bx in zip(results["labels"], results["scores"], results["boxes"]):
        dets.append(
            {
                "label": str(lab),
                "score": float(sc.item()),
                "box_xyxy": [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])],
            }
        )
    return dets


def take_top_k_by_score(dets: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0 or len(dets) <= k:
        return list(dets)
    sorted_d = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return sorted_d[:k]


def normalize_label_name(s: str) -> str:
    s2 = s.strip().lower().replace(".", "").strip()
    for prefix in ("a ", "an ", "the "):
        if s2.startswith(prefix):
            s2 = s2[len(prefix) :]
    return s2


def map_label_to_class(label: str, class_names: List[str]) -> str:
    key = normalize_label_name(label)
    # 开词汇常见同义词 → 与 class_colors_bgr 对齐
    for token, cls in (
        ("suv", "car"),
        ("vehicle", "car"),
        ("sedan", "car"),
        ("van", "bus"),
        ("pickup", "truck"),
    ):
        if token in key and cls in class_names:
            return cls
    for c in class_names:
        if c in key:
            return c
    return key
