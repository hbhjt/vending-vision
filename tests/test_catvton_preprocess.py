import cv2
import numpy as np

from vision.catvton_preprocess import (
    build_generation_mask,
    composite_to_original,
    harmonize_garment_color,
    include_color_matched_old_edges,
    letterbox_image,
    letterbox_mask,
    refine_mask_with_generated_parse,
)


def test_generation_mask_covers_clothes_without_identity_lower_body_or_background():
    height, width = 240, 160
    target = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(target, (43, 62), (117, 168), 255, -1)
    cv2.circle(target, (80, 100), 4, 0, -1)
    lip = np.zeros((height, width), dtype=np.uint8)
    atr = np.zeros_like(lip)
    lip[52:172, 38:122] = 5
    lip[18:55, 57:103] = 13
    lip[170:230, 45:115] = 9
    hands = np.zeros_like(lip)
    cv2.circle(hands, (40, 120), 9, 255, -1)

    mask = build_generation_mask(target, lip, atr, hands)

    assert mask[80, 80] == 255
    assert mask[100, 80] == 255
    assert mask[35, 80] == 0
    assert mask[205, 80] == 0
    assert mask[120, 40] == 0
    assert mask[80, 10] == 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    assert count >= 2
    assert stats[1:, cv2.CC_STAT_AREA].max() > 4_000


def test_composite_preserves_size_and_every_pixel_outside_the_generation_mask():
    original = np.full((120, 200, 3), (31, 87, 143), dtype=np.uint8)
    fitted, transform = letterbox_image(original, 512, 768)
    generated = np.full_like(fitted, (220, 20, 90))
    source_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cv2.rectangle(source_mask, (60, 25), (140, 95), 255, -1)
    fitted_mask = letterbox_mask(source_mask, transform)

    result = composite_to_original(original, generated, fitted_mask, transform)

    outside = source_mask == 0
    assert result.shape == original.shape
    assert np.array_equal(result[outside], original[outside])
    assert np.any(result[40:80, 80:120] != original[40:80, 80:120])


def test_color_harmonization_changes_only_cloth_and_moves_toward_condition():
    generated = np.full((80, 80, 3), (205, 195, 180), dtype=np.uint8)
    condition = np.full((80, 80, 3), 255, dtype=np.uint8)
    condition[15:65, 20:60] = (135, 25, 45)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[12:68, 18:62] = 255

    result = harmonize_garment_color(generated, mask, condition)

    assert np.array_equal(result[0, 0], generated[0, 0])
    assert result[40, 40, 0] > result[40, 40, 1]


def test_generated_parse_removes_background_but_retains_old_clothes_and_sleeves():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    parse = np.zeros_like(mask)
    parse[22:82, 25:75] = 5

    refined = refine_mask_with_generated_parse(mask, parse)

    assert refined[50, 50] == 255
    assert refined[12, 12] == 0
    assert refined[50, 90] == 0

    original_clothes = np.zeros_like(mask)
    original_clothes[18:25, 43:57] = 255
    with_original = refine_mask_with_generated_parse(mask, parse, original_clothes)
    assert with_original[20, 50] == 255

    required_sleeve = np.zeros_like(mask)
    required_sleeve[35:48, 70:88] = 255
    with_sleeve = refine_mask_with_generated_parse(
        mask,
        parse,
        original_clothes,
        required_sleeve,
    )
    assert with_sleeve[40, 80] == 255
    assert with_sleeve[40, 95] == 0


def test_color_matched_old_edge_is_recovered_without_expanding_to_background():
    refined = np.zeros((60, 60), dtype=np.uint8)
    refined[20:45, 20:40] = 255
    generation = np.zeros_like(refined)
    generation[15:50, 15:45] = 255
    original = np.full((60, 60, 3), (240, 240, 240), dtype=np.uint8)
    original[18:47, 18:42] = (220, 205, 188)
    old = np.zeros_like(refined)
    old[22:43, 22:38] = 255

    recovered = include_color_matched_old_edges(refined, generation, original, old)

    assert recovered[19, 30] == 255
    assert recovered[15, 15] == 0
