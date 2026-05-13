"""SAM 2（Hiera-Small 等）：用 DINO 的 box 做提示，仅用于诊断 mask，不参与 3D。"""
from __future__ import annotations

import base64
import zlib
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def load_sam2(model_id: str, device: torch.device):
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def _clip_xyxy(box: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(t) for t in box]
    x1 = max(0.0, min(x1, float(w - 1)))
    x2 = max(0.0, min(x2, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    y2 = max(0.0, min(y2, float(h - 1)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return [0.0, 0.0, float(min(1, w - 1)), float(min(1, h - 1))]
    return [x1, y1, x2, y2]


@torch.inference_mode()
def sam2_masks_from_boxes(
    processor,
    model,
    pil_image: Image.Image,
    boxes_xyxy: List[List[float]],
    device: torch.device,
    mask_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    单张图、多个 box，一次前向。
    返回与 boxes 等长的列表；失败时对应项含 sam_error 且无 mask。
    """
    w, h = pil_image.size
    if not boxes_xyxy:
        return []

    norm_boxes: List[List[float]] = []
    for b in boxes_xyxy:
        norm_boxes.append(_clip_xyxy(b, w, h))

    try:
        inputs = processor(images=pil_image, input_boxes=[norm_boxes], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = model(**inputs)
        iou = outputs.iou_scores[0].float().cpu()  # [N, 3]
        proc_masks = processor.post_process_masks(
            outputs.pred_masks.cpu().float(),
            inputs["original_sizes"].cpu(),
        )[0]
        # [N, 3, H, W]
        n = proc_masks.shape[0]
        out: List[Dict[str, Any]] = []
        for i in range(n):
            k = int(torch.argmax(iou[i]).item())
            logits = proc_masks[i, k].numpy()
            mask = logits > float(mask_threshold)
            area = int(np.count_nonzero(mask))
            out.append(
                {
                    "sam_best_iou": float(iou[i, k].item()),
                    "sam_mask_variant_idx": k,
                    "sam_mask_area_px": area,
                    "sam_mask_hw": mask.astype(np.bool_),
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "sam_error": str(exc),
                "sam_best_iou": None,
                "sam_mask_area_px": 0,
                "sam_mask_hw": None,
            }
            for _ in norm_boxes
        ]


def mask_to_zlib_b64(mask_bool: np.ndarray) -> Tuple[str, List[int]]:
    """uint8 扁平 + zlib + base64，便于 JSON 附件（可选，体积仍可能较大）。"""
    flat = np.ascontiguousarray(mask_bool.astype(np.uint8)).tobytes()
    z = zlib.compress(flat, level=6)
    return base64.standard_b64encode(z).decode("ascii"), [int(mask_bool.shape[0]), int(mask_bool.shape[1])]


def zlib_b64_to_mask_bool(b64: str, shape_hw: List[int]) -> np.ndarray:
    """与 ``mask_to_zlib_b64`` 互逆；``shape_hw`` 为 [H, W]。"""
    raw = zlib.decompress(base64.standard_b64decode(b64))
    h, w = int(shape_hw[0]), int(shape_hw[1])
    need = h * w
    arr = np.frombuffer(raw, dtype=np.uint8, count=need)
    if arr.size != need:
        raise ValueError(f"zlib mask 长度与 shape 不符: need={need}, got={arr.size}")
    return arr.reshape(h, w).astype(bool)


def draw_sam_overlay_bgr(
    image_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    sam_entries: List[Dict[str, Any]],
) -> np.ndarray:
    """DINO 框（细白边）+ 有 mask 时半透明着色。"""
    canvas = image_bgr.astype(np.float32).copy()
    h, w = canvas.shape[:2]
    rng = np.random.default_rng(42)
    for idx, (box, ent) in enumerate(zip(boxes_xyxy, sam_entries)):
        x1, y1, x2, y2 = [int(round(float(t))) for t in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1, cv2.LINE_AA)
        m = ent.get("sam_mask_hw")
        if m is None or not isinstance(m, np.ndarray):
            continue
        if m.shape[:2] != (h, w):
            if m.size:
                try:
                    m = cv2_resize_bool(m, (w, h))
                except Exception:
                    continue
            else:
                continue
        if not np.any(m):
            continue
        color = rng.integers(50, 255, size=3, dtype=np.int32)
        bgr = (float(color[0]), float(color[1]), float(color[2]))
        for c in range(3):
            ch = canvas[:, :, c]
            ch[m] = ch[m] * 0.55 + bgr[c] * 0.45
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (int(bgr[0]), int(bgr[1]), int(bgr[2])), 2, cv2.LINE_AA)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def cv2_resize_bool(mask: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    w, h = wh
    u8 = (mask.astype(np.uint8) * 255)
    rz = cv2.resize(u8, (w, h), interpolation=cv2.INTER_NEAREST)
    return rz > 127
