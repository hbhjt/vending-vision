"""CatVTON preprocessing/postprocessing used by the official attempt worker.

This is migrated from the verified standalone CatVTON integration.  It keeps
the deterministic SCHP mask/refine/composite path local to the attempt child;
it does not add background removal, content classification, remote inference,
or customer-facing tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LIP_OLD_CLOTHES = (5, 6, 7, 10)
ATR_OLD_CLOTHES = (4, 7)
LIP_PROTECT = (1, 2, 3, 4, 8, 9, 12, 13, 16, 17, 18, 19)
ATR_PROTECT = (1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17)


@dataclass(frozen=True)
class Letterbox:
    original_width: int
    original_height: int
    width: int
    height: int
    x: int
    y: int
    content_width: int
    content_height: int


def parsed_old_clothes_mask(lip_parse: np.ndarray, atr_parse: np.ndarray) -> np.ndarray:
    return np.where(
        np.isin(lip_parse, LIP_OLD_CLOTHES) | np.isin(atr_parse, ATR_OLD_CLOTHES),
        255,
        0,
    ).astype(np.uint8)


def build_generation_mask(
    target_mask: np.ndarray,
    lip_parse: np.ndarray,
    atr_parse: np.ndarray,
    hands_mask: np.ndarray,
) -> np.ndarray:
    shape = target_mask.shape[:2]
    lip = cv2.resize(lip_parse, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    atr = cv2.resize(atr_parse, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    hands = cv2.resize(hands_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    target = target_mask > 0
    old_clothes = parsed_old_clothes_mask(lip, atr) > 0
    protect = np.isin(lip, LIP_PROTECT) | np.isin(atr, ATR_PROTECT) | hands

    person = (lip != 0) | (atr != 0)
    person_margin = cv2.dilate(
        np.where(person, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ) > 0
    mask = np.where((target & person_margin) | old_clothes, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    max_hole = max(64, round(mask.size * 0.003))
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        touches_border = x == 0 or y == 0 or x + width == shape[1] or y + height == shape[0]
        if not touches_border and area <= max_hole:
            mask[labels == label] = 255

    radius = int(np.clip(round(min(shape) / 64), 8, 12))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.dilate(mask, kernel)
    outer_person_margin = cv2.dilate(
        np.where(person, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ) > 0
    mask[~outer_person_margin] = 0
    protect_u8 = cv2.dilate(
        np.where(protect, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask[protect_u8 > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    target_dilated = cv2.dilate(target_mask, np.ones((3, 3), np.uint8)) > 0
    minimum_area = max(48, round(mask.size * 0.0008))
    for label in range(1, count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area or np.any(component & target_dilated):
            cleaned[component] = 255
    return cleaned


def letterbox_image(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, Letterbox]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    content_width = max(1, round(source_width * scale))
    content_height = max(1, round(source_height * scale))
    x = (width - content_width) // 2
    y = (height - content_height) // 2
    resized = cv2.resize(image, (content_width, content_height), interpolation=cv2.INTER_LANCZOS4)
    corners = np.concatenate(
        [
            image[:8, :8].reshape(-1, image.shape[2]),
            image[:8, -8:].reshape(-1, image.shape[2]),
            image[-8:, :8].reshape(-1, image.shape[2]),
            image[-8:, -8:].reshape(-1, image.shape[2]),
        ],
        axis=0,
    )
    fill = np.median(corners, axis=0).astype(np.uint8)
    canvas = np.empty((height, width, image.shape[2]), dtype=np.uint8)
    canvas[:] = fill
    canvas[y : y + content_height, x : x + content_width] = resized
    return canvas, Letterbox(source_width, source_height, width, height, x, y, content_width, content_height)


def letterbox_mask(mask: np.ndarray, transform: Letterbox) -> np.ndarray:
    canvas = np.zeros((transform.height, transform.width), dtype=np.uint8)
    resized = cv2.resize(
        mask,
        (transform.content_width, transform.content_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[
        transform.y : transform.y + transform.content_height,
        transform.x : transform.x + transform.content_width,
    ] = resized
    return canvas


def refine_mask_with_generated_parse(
    mask: np.ndarray,
    lip_parse: np.ndarray,
    original_clothes_mask: np.ndarray | None = None,
    required_sleeve_mask: np.ndarray | None = None,
) -> np.ndarray:
    parse = cv2.resize(lip_parse, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    cloth = cv2.morphologyEx(
        np.where(np.isin(parse, LIP_OLD_CLOTHES), 255, 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    cloth = cv2.dilate(cloth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    refined = cv2.bitwise_and(np.where(mask > 0, 255, 0).astype(np.uint8), cloth)
    if original_clothes_mask is not None:
        original_clothes = cv2.resize(
            original_clothes_mask,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        original_clothes = cv2.dilate(
            np.where(original_clothes > 0, 255, 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        refined = cv2.bitwise_or(
            refined,
            cv2.bitwise_and(np.where(mask > 0, 255, 0).astype(np.uint8), original_clothes),
        )
    if required_sleeve_mask is not None:
        required = cv2.resize(
            required_sleeve_mask,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        required = cv2.morphologyEx(
            np.where(required > 0, 255, 0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        refined = cv2.bitwise_or(
            refined,
            cv2.bitwise_and(np.where(mask > 0, 255, 0).astype(np.uint8), required),
        )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    cleaned = np.zeros_like(refined)
    minimum_area = max(64, round(refined.size * 0.001))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def include_color_matched_old_edges(
    refined_mask: np.ndarray,
    generation_mask: np.ndarray,
    original_image: np.ndarray,
    parsed_old_clothes: np.ndarray,
) -> np.ndarray:
    old = cv2.resize(
        parsed_old_clothes,
        (refined_mask.shape[1], refined_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    if int(old.sum()) < 64:
        return refined_mask
    lab = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    old_color = np.median(lab[old], axis=0)
    color_distance = np.linalg.norm(lab - old_color[None, None, :], axis=2)
    neighborhood = cv2.dilate(
        np.where(refined_mask > 0, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ) > 0
    allowed = cv2.dilate(
        np.where(generation_mask > 0, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    ) > 0
    recovery = allowed & neighborhood & (color_distance < 12.0)
    recovered = np.where((refined_mask > 0) | recovery, 255, 0).astype(np.uint8)
    recovered = cv2.morphologyEx(
        recovered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    recovered = cv2.medianBlur(recovered, 9)
    return np.where(recovered >= 128, 255, 0).astype(np.uint8)


def remove_boundary_artifacts(generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    interior = cv2.erode(binary, np.ones((5, 5), np.uint8), iterations=2)
    ring = cv2.subtract(binary, interior)
    gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
    values = gray[interior > 0]
    if values.size < 32:
        return generated
    threshold = min(52.0, float(np.median(values)) * 0.38)
    candidates = np.where((ring > 0) & (gray < threshold), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    repair = np.zeros_like(candidates)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= 128:
            repair[labels == label] = 255
    if not np.any(repair):
        return generated
    return cv2.inpaint(generated, repair, 3, cv2.INPAINT_TELEA)


def harmonize_garment_color(
    generated: np.ndarray,
    mask: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    condition_distance = np.linalg.norm(condition.astype(np.float32) - 255.0, axis=2)
    condition_pixels = condition_distance > 16.0
    core = cv2.erode(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    if int(condition_pixels.sum()) < 64 or int(core.sum()) < 64:
        return generated

    generated_lab = cv2.cvtColor(generated, cv2.COLOR_RGB2LAB).astype(np.float32)
    condition_lab = cv2.cvtColor(condition, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_median = np.median(generated_lab[core], axis=0)
    target_median = np.median(condition_lab[condition_pixels], axis=0)
    delta = target_median - source_median
    delta[0] = np.clip(delta[0], -140.0, 80.0) * 0.90
    delta[1:] = np.clip(delta[1:], -58.0, 58.0) * 0.88

    shifted = generated_lab.copy()
    shifted[core] = np.clip(shifted[core] + delta, 0, 255)
    shifted_rgb = cv2.cvtColor(shifted.astype(np.uint8), cv2.COLOR_LAB2RGB)
    blend = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 0.8)
    result = np.rint(
        shifted_rgb.astype(np.float32) * blend[:, :, None]
        + generated.astype(np.float32) * (1.0 - blend[:, :, None])
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def composite_to_original(
    original: np.ndarray,
    generated_letterboxed: np.ndarray,
    mask_letterboxed: np.ndarray,
    transform: Letterbox,
) -> np.ndarray:
    cleaned = remove_boundary_artifacts(generated_letterboxed, mask_letterboxed)
    y0, y1 = transform.y, transform.y + transform.content_height
    x0, x1 = transform.x, transform.x + transform.content_width
    generated = cv2.resize(
        cleaned[y0:y1, x0:x1],
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    restored_mask = cv2.resize(
        mask_letterboxed[y0:y1, x0:x1],
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    core = cv2.erode(
        np.where(restored_mask > 0, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    )
    alpha = cv2.GaussianBlur(core.astype(np.float32) / 255.0, (0, 0), 0.55)
    alpha[restored_mask == 0] = 0.0
    alpha[core == 255] = 1.0
    result = np.rint(
        generated.astype(np.float32) * alpha[:, :, None]
        + original.astype(np.float32) * (1.0 - alpha[:, :, None])
    ).astype(np.uint8)
    result[alpha == 0.0] = original[alpha == 0.0]
    return result


def protected_mask_to_original(
    generation_mask_letterboxed: np.ndarray,
    final_mask_letterboxed: np.ndarray,
    transform: Letterbox,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint generated/protected acceptance regions in input space."""
    y0, y1 = transform.y, transform.y + transform.content_height
    x0, x1 = transform.x, transform.x + transform.content_width
    generation = cv2.resize(
        generation_mask_letterboxed[y0:y1, x0:x1],
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    final = cv2.resize(
        final_mask_letterboxed[y0:y1, x0:x1],
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    upper_body = np.where(final > 0, 255, 0).astype(np.uint8)
    generation_region = cv2.dilate(
        np.where(generation > 0, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    protected = np.where(generation_region == 0, 255, 0).astype(np.uint8)
    protected[upper_body > 0] = 0
    return upper_body, protected
