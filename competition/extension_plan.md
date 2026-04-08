# Extension Plan: Design2Code Metrics Integration

Source: https://github.com/amaljoe/Design2Code/tree/main/Design2Code/metrics

---

## Current Reward Structure

| Signal      | Weight | Implementation                          |
|-------------|--------|-----------------------------------------|
| format      | 1      | Markdown fence + doctype detection      |
| validity    | 1      | BS4 parse + structural tags + tag diversity |
| structural  | 1      | Tag-sequence difflib + CSS-class Jaccard |
| visual      | 3      | PIL pixel-diff after Playwright render  |
| **Total**   | **6**  |                                         |

**Gaps vs Design2Code:**
- No text content comparison (only DOM structure)
- No spatial/positional layout scoring
- No perceptual color accuracy (pixel-diff is coarse)
- Pixel-diff conflates layout errors with colour errors — a white-on-blue button reads as 50% wrong even if perfectly placed

---

## Design2Code Metric Breakdown

From `visual_score.py` and `ocr_free_utils.py`:

### 1. Block-Match (`block_match`)
- Injects unique hex colors into each text element via JavaScript
- Renders both reference and generated HTML with Playwright
- Diffs pixel colors to locate bounding boxes of each text block
- Matches blocks between reference and prediction using **Jonker-Volgenant** (scipy `linear_sum_assignment`)
- Score = fraction of reference blocks successfully matched

### 2. Text Similarity (`text_sim`)
- Within each matched block pair, compares raw text content
- Uses character-level similarity (e.g. SequenceMatcher or edit distance)
- Catches cases where layout is right but text is wrong (hallucinations, truncation)

### 3. Position (`position`)
- Compares bounding box centres/sizes of matched block pairs
- Normalised by viewport dimensions → [0, 1]
- Catches layout shifts invisible to pixel-diff (e.g. column reorder)

### 4. Color (`color`)
- Samples dominant foreground/background colors from matched block regions
- Compares using **CIEDE2000** perceptual formula (skimage.color.deltaE_ciede2000)
- More human-aligned than raw pixel diff (large ΔE for magenta vs red, small ΔE for near-whites)

### 5. CLIP Visual (`clip`)
- Global image-level CLIP embedding cosine similarity
- Already approximated by our PIL pixel-diff; true CLIP requires GPU / torch

---

## Proposed Extension (3 Phases)

### Phase 1 — Text Block Reward  *(no new dependencies)*

**New file:** `vcoder/rewards/text_block_rewards.py`

```python
# Pseudocode — adapt from ocr_free_utils.py
def text_block_reward(completions, solution):
    for each completion:
        ref_blocks = get_blocks_ocr_free(ref_html)   # color-inject → render → diff
        pred_blocks = get_blocks_ocr_free(pred_html)
        matches = linear_sum_assignment(cost_matrix)  # scipy, already transitive dep
        block_match_score = matched / len(ref_blocks)
        text_sim_score = mean SequenceMatcher ratio over matched pairs
        score = 0.5 * block_match_score + 0.5 * text_sim_score
```

**Dependencies:** `scipy` (add to pyproject.toml), Playwright (already present)

**Weight:** 2 (replaces half of the current visual weight)

**Why:** Directly tests whether the right text appears in the right block — our biggest current blind spot. A model that outputs correctly-laid-out lorem ipsum still scores high on pixel-diff but should score low here.

---

### Phase 2 — Position/Layout Reward  *(no new dependencies)*

**New file:** `vcoder/rewards/position_rewards.py`

```python
# Reuse matched blocks from Phase 1
def position_reward(completions, solution):
    for each completion:
        matched_pairs = text_block_match(ref_html, pred_html)
        position_scores = [
            1.0 - dist(ref_bbox_centre, pred_bbox_centre) / diagonal
            for ref_bbox, pred_bbox in matched_pairs
        ]
        score = mean(position_scores) if position_scores else 0.0
```

**Dependencies:** none beyond Phase 1

**Weight:** 1

**Why:** Pixel-diff rewards shrink when element positions shift, but doesn't distinguish "element is present but displaced" from "element is missing entirely". This isolates layout accuracy.

---

### Phase 3 — Perceptual Color Reward  *(one new dependency)*

**New file:** `vcoder/rewards/color_rewards.py`

```python
from skimage.color import rgb2lab, deltaE_ciede2000

def color_reward(completions, solution):
    for each completion:
        ref_render = _render_html(ref_html)
        pred_render = _render_html(pred_html)
        # Sample N random pixels (avoid white margins)
        ref_colors_lab = rgb2lab(sample_pixels(ref_render))
        pred_colors_lab = rgb2lab(sample_pixels(pred_render))
        # Match closest pairs, score by CIEDE2000
        delta_e = deltaE_ciede2000(ref_colors_lab, pred_colors_lab)
        score = 1.0 - clip(mean(delta_e) / 50.0, 0, 1)  # ΔE=50 → score=0
```

**Dependencies:** `scikit-image` (add to pyproject.toml — ~30MB, no GPU)

**Weight:** 1

**Why:** Replaces 1 unit of the current pixel-diff weight with a perceptually-calibrated color score. A page with correct layout but wrong brand colors gets a meaningful penalty.

---

### Phase 4 — True CLIP Visual  *(GPU / offline training only)*

**Do not run on HF free Spaces (16 GB RAM, no GPU).** Only enable during local training runs.

```python
# In vcoder/rewards/visual_rewards.py — guard with env flag
if os.environ.get("ENABLE_CLIP"):
    from transformers import CLIPModel, CLIPProcessor
    # ... true CLIP cosine similarity
else:
    # fall back to PIL pixel-diff (current implementation)
```

**Dependencies:** `torch`, `transformers`, `Pillow` (already present except torch)

**Weight:** 2 (replacing current visual weight 3 — we now have dedicated color + position signals)

---

## Updated Weight Table (after all phases)

| Signal       | Weight | Module                       |
|--------------|--------|------------------------------|
| format       | 1      | `format_rewards.py`          |
| validity     | 1      | `validity_rewards.py`        |
| structural   | 1      | `structural_rewards.py`      |
| text_block   | 2      | `text_block_rewards.py` (new)|
| position     | 1      | `position_rewards.py` (new)  |
| color        | 1      | `color_rewards.py` (new)     |
| visual       | 2      | `visual_rewards.py` (updated)|
| **Total**    | **9**  |                              |

Normalization denominator changes from 6.0 → 9.0 in `server/environment.py`.

---

## Implementation Order

1. **Phase 1** (text_block): Highest signal quality gain, moderate complexity.  
   - Port `get_blocks_ocr_free` from `ocr_free_utils.py` into `text_block_rewards.py`
   - Add `scipy` to `pyproject.toml`
   - Integrate into `environment.py` alongside existing rewards

2. **Phase 2** (position): Low additional cost if Phase 1 already rendered both images.  
   - Reuse block positions from Phase 1 (no extra Playwright calls)

3. **Phase 3** (color): Independent of Phase 1/2.  
   - Add `scikit-image` to `pyproject.toml`
   - Requires one extra Playwright render per step (or reuse Phase 2 renders via caching)

4. **Phase 4** (CLIP): Only after GRPO training setup confirmed with GPU access.

---

## Caching Consideration

Phases 1, 2, and 3 all require rendered images. Currently `_render_html` is called separately for `clip_visual_reward`. Extract rendering into a shared helper called once per step:

```python
# In environment.py step():
pred_render = _render_html(pred_html)  # call once, pass to all visual rewards
```

This avoids 3 redundant Playwright browser launches per step.

---

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    ...
    "scipy>=1.11",         # linear_sum_assignment for block matching
    "scikit-image>=0.22",  # CIEDE2000 color comparison
]
```

Both are CPU-only, pure-Python-compatible, and safe for HF free Spaces.
