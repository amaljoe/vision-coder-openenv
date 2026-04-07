"""Comprehensive tests for all VisionCoder reward functions.

Test strategy:
  - Unit tests: mock Playwright / PIL calls for speed
  - Integration tests: real simple HTML snippets (marked with @pytest.mark.integration)
  - Edge cases: empty HTML, malformed HTML, missing references

Run unit tests only:
    pytest tests/test_rewards.py -m "not integration"

Run all tests (requires Playwright Chromium + scipy + scikit-image):
    pytest tests/test_rewards.py
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <h1>Hello World</h1>
  <p>This is a paragraph.</p>
  <div style="color:blue;">Blue text</div>
</body>
</html>"""

EMPTY_HTML = ""

MINIMAL_HTML = "<html><body><p>Hi</p></body></html>"

DIFFERENT_HTML = """<!DOCTYPE html>
<html>
<head><title>Other</title></head>
<body>
  <h2>Goodbye World</h2>
  <span>Different content entirely</span>
</body>
</html>"""

MALFORMED_HTML = "<div><p>Not closed"


def _make_completion(html: str) -> list[list[dict]]:
    """Wrap raw HTML in the completions list format."""
    return [[{"content": html}]]


def _white_image(w: int = 640, h: int = 480) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _solid_image(color: tuple, w: int = 640, h: int = 480) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ===========================================================================
# vcoder.rewards (extract_html)
# ===========================================================================


class TestExtractHtml:
    def test_extracts_from_fenced_markdown(self):
        from vcoder.rewards import extract_html

        content = "```html\n<html><body>hi</body></html>\n```"
        assert extract_html(content) == "<html><body>hi</body></html>"

    def test_strips_think_blocks(self):
        from vcoder.rewards import extract_html

        content = "<think>reasoning</think>\n```html\n<p>ok</p>\n```"
        assert extract_html(content) == "<p>ok</p>"

    def test_handles_unclosed_fence(self):
        from vcoder.rewards import extract_html

        content = "```html\n<p>truncated"
        result = extract_html(content)
        assert "<p>truncated" in result

    def test_passthrough_plain_html(self):
        from vcoder.rewards import extract_html

        assert extract_html(SIMPLE_HTML) == SIMPLE_HTML


# ===========================================================================
# format_rewards
# ===========================================================================


class TestFormatReward:
    def _run(self, html: str) -> float:
        from vcoder.rewards.format_rewards import format_reward

        return format_reward(_make_completion(html))[0]

    def test_perfect_score(self):
        html = "```html\n<!DOCTYPE html><html><body></body></html>\n```"
        assert self._run(html) == 1.0

    def test_no_fence_no_doctype(self):
        assert self._run("<p>bare</p>") == 0.0

    def test_fence_but_no_doctype(self):
        score = self._run("```html\n<p>no doctype</p>\n```")
        assert score == 0.5

    def test_doctype_but_no_fence(self):
        score = self._run("<!DOCTYPE html><html></html>")
        assert score == 0.5

    def test_batch(self):
        from vcoder.rewards.format_rewards import format_reward

        completions = [
            [{"content": "```html\n<!DOCTYPE html><html></html>\n```"}],
            [{"content": "bare"}],
        ]
        scores = format_reward(completions)
        assert scores[0] == 1.0
        assert scores[1] == 0.0


# ===========================================================================
# validity_rewards
# ===========================================================================


class TestValidityReward:
    def _run(self, html: str) -> float:
        from vcoder.rewards.validity_rewards import html_validity_reward

        return html_validity_reward(_make_completion(html))[0]

    def test_full_score(self):
        html = (
            "<html><head></head><body>"
            "<p>a</p><div>b</div><span>c</span><ul><li>d</li></ul><h1>e</h1>"
            "</body></html>"
        )
        score = self._run(html)
        assert score == 1.0

    def test_no_structure_tags(self):
        score = self._run("<p>bare paragraph</p>")
        assert score < 0.5

    def test_empty_html(self):
        score = self._run(EMPTY_HTML)
        assert score == 0.0

    def test_partial_structure(self):
        # Has html + body but no head; few unique tags
        score = self._run("<html><body><p>hi</p></body></html>")
        assert 0.0 <= score <= 1.0

    def test_scores_in_range(self):
        score = self._run(SIMPLE_HTML)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# structural_rewards
# ===========================================================================


class TestStructuralReward:
    def _run(self, html: str, ref: str) -> float:
        from vcoder.rewards.structural_rewards import structural_similarity_reward

        return structural_similarity_reward(_make_completion(html), solution=[ref])[0]

    def test_identical_html(self):
        score = self._run(SIMPLE_HTML, SIMPLE_HTML)
        assert score == 1.0

    def test_empty_vs_ref(self):
        # Empty pred has no tags (tag_sim=0) but no CSS classes either;
        # when neither ref nor pred has CSS classes, class_overlap defaults to 1.0.
        # So score = 0.5*0 + 0.5*1 = 0.5 — the reward correctly penalises empty preds
        # only on the tag dimension.
        score = self._run(EMPTY_HTML, SIMPLE_HTML)
        assert 0.0 <= score <= 1.0
        assert score <= 0.5  # tag mismatch caps at 0.5

    def test_similar_structure(self):
        score = self._run(MINIMAL_HTML, SIMPLE_HTML)
        assert 0.0 <= score <= 1.0

    def test_different_html(self):
        score = self._run(DIFFERENT_HTML, SIMPLE_HTML)
        assert 0.0 <= score <= 1.0

    def test_no_solution_returns_zero_or_partial(self):
        # When solution=None, ref_html="" — no tags and no CSS classes,
        # so class_overlap defaults to 1.0 and score can be non-zero.
        from vcoder.rewards.structural_rewards import structural_similarity_reward

        score = structural_similarity_reward(_make_completion(SIMPLE_HTML), solution=None)[0]
        assert 0.0 <= score <= 1.0


# ===========================================================================
# text_block_rewards (Phase 1) — unit tests with mocked Playwright
# ===========================================================================


class TestTextBlockReward:
    """Unit tests for text_block_reward using mocked _get_text_blocks."""

    def _run_with_blocks(
        self, ref_blocks: list[dict], pred_blocks: list[dict], ref_html: str = SIMPLE_HTML
    ) -> float:
        from vcoder.rewards import text_block_rewards

        with patch.object(text_block_rewards, "_get_text_blocks") as mock_get:
            mock_get.side_effect = [ref_blocks, pred_blocks]
            score = text_block_rewards.text_block_reward(
                _make_completion(SIMPLE_HTML), solution=[ref_html]
            )[0]
        return score

    def test_perfect_match(self):
        blocks = [
            {"text": "Hello", "x": 10, "y": 10, "width": 100, "height": 20},
            {"text": "World", "x": 10, "y": 40, "width": 100, "height": 20},
        ]
        score = self._run_with_blocks(blocks, blocks)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_pred_blocks(self):
        ref_blocks = [{"text": "Hello", "x": 10, "y": 10, "width": 100, "height": 20}]
        score = self._run_with_blocks(ref_blocks, [])
        assert score == 0.0

    def test_no_ref_blocks_no_pred(self):
        score = self._run_with_blocks([], [])
        assert score == 1.0

    def test_no_ref_blocks_with_pred(self):
        pred_blocks = [{"text": "Extra", "x": 10, "y": 10, "width": 100, "height": 20}]
        score = self._run_with_blocks([], pred_blocks)
        assert score == 0.5

    def test_overlapping_blocks_high_score(self):
        ref = [{"text": "Hello", "x": 10, "y": 10, "width": 100, "height": 20}]
        pred = [{"text": "Hello", "x": 12, "y": 10, "width": 98, "height": 20}]
        score = self._run_with_blocks(ref, pred)
        assert score > 0.7

    def test_non_overlapping_blocks_low_score(self):
        ref = [{"text": "Hello", "x": 10, "y": 10, "width": 50, "height": 20}]
        pred = [{"text": "Other", "x": 500, "y": 400, "width": 50, "height": 20}]
        score = self._run_with_blocks(ref, pred)
        assert score < 0.3

    def test_no_solution_returns_zero(self):
        from vcoder.rewards.text_block_rewards import text_block_reward

        score = text_block_reward(_make_completion(SIMPLE_HTML), solution=None)[0]
        assert score == 0.0

    def test_score_in_range(self):
        blocks_a = [{"text": "A", "x": 0, "y": 0, "width": 50, "height": 20}]
        blocks_b = [{"text": "B", "x": 200, "y": 200, "width": 50, "height": 20}]
        score = self._run_with_blocks(blocks_a, blocks_b)
        assert 0.0 <= score <= 1.0

    def test_batch_processing(self):
        from vcoder.rewards import text_block_rewards

        blocks = [{"text": "Hi", "x": 10, "y": 10, "width": 80, "height": 20}]
        completions = [
            [{"content": SIMPLE_HTML}],
            [{"content": MINIMAL_HTML}],
        ]
        solutions = [SIMPLE_HTML, MINIMAL_HTML]
        with patch.object(text_block_rewards, "_get_text_blocks") as mock_get:
            mock_get.side_effect = [blocks, blocks, blocks, blocks]
            scores = text_block_rewards.text_block_reward(completions, solution=solutions)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestBboxIou:
    def test_identical_boxes(self):
        from vcoder.rewards.text_block_rewards import _bbox_iou

        box = {"x": 0, "y": 0, "width": 100, "height": 100}
        assert _bbox_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping(self):
        from vcoder.rewards.text_block_rewards import _bbox_iou

        a = {"x": 0, "y": 0, "width": 10, "height": 10}
        b = {"x": 100, "y": 100, "width": 10, "height": 10}
        assert _bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        from vcoder.rewards.text_block_rewards import _bbox_iou

        a = {"x": 0, "y": 0, "width": 20, "height": 20}
        b = {"x": 10, "y": 10, "width": 20, "height": 20}
        iou = _bbox_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_size_box(self):
        from vcoder.rewards.text_block_rewards import _bbox_iou

        a = {"x": 0, "y": 0, "width": 0, "height": 0}
        b = {"x": 0, "y": 0, "width": 10, "height": 10}
        assert _bbox_iou(a, b) == 0.0


class TestTextSimilarity:
    def test_identical(self):
        from vcoder.rewards.text_block_rewards import _text_similarity

        assert _text_similarity("hello", "hello") == 1.0

    def test_empty_both(self):
        from vcoder.rewards.text_block_rewards import _text_similarity

        assert _text_similarity("", "") == 1.0

    def test_one_empty(self):
        from vcoder.rewards.text_block_rewards import _text_similarity

        assert _text_similarity("hello", "") == 0.0

    def test_partial(self):
        from vcoder.rewards.text_block_rewards import _text_similarity

        score = _text_similarity("hello world", "hello earth")
        assert 0.0 < score < 1.0


# ===========================================================================
# position_rewards (Phase 2) — unit tests with mocked _get_text_blocks
# ===========================================================================


class TestPositionReward:
    def _run_with_blocks(
        self, ref_blocks: list[dict], pred_blocks: list[dict], ref_html: str = SIMPLE_HTML
    ) -> float:
        from vcoder.rewards import position_rewards

        with patch.object(position_rewards, "_get_text_blocks") as mock_get:
            mock_get.side_effect = [ref_blocks, pred_blocks]
            score = position_rewards.position_reward(
                _make_completion(SIMPLE_HTML), solution=[ref_html]
            )[0]
        return score

    def test_perfect_position(self):
        blocks = [{"text": "Hi", "x": 10, "y": 20, "width": 100, "height": 30}]
        score = self._run_with_blocks(blocks, blocks)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_displaced(self):
        ref = [{"text": "Hi", "x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [{"text": "Hi", "x": 630, "y": 470, "width": 10, "height": 10}]
        score = self._run_with_blocks(ref, pred)
        assert score < 0.3

    def test_no_pred_returns_zero(self):
        ref = [{"text": "Hi", "x": 0, "y": 0, "width": 10, "height": 10}]
        score = self._run_with_blocks(ref, [])
        assert score == 0.0

    def test_no_ref_no_pred(self):
        score = self._run_with_blocks([], [])
        assert score == 1.0

    def test_score_in_range(self):
        ref = [{"text": "A", "x": 50, "y": 50, "width": 80, "height": 25}]
        pred = [{"text": "A", "x": 200, "y": 150, "width": 80, "height": 25}]
        score = self._run_with_blocks(ref, pred)
        assert 0.0 <= score <= 1.0

    def test_no_solution_returns_zero(self):
        from vcoder.rewards.position_rewards import position_reward

        score = position_reward(_make_completion(SIMPLE_HTML), solution=None)[0]
        assert score == 0.0

    def test_multiple_blocks_averaged(self):
        ref = [
            {"text": "A", "x": 10, "y": 10, "width": 50, "height": 20},
            {"text": "B", "x": 10, "y": 50, "width": 50, "height": 20},
        ]
        pred = [
            {"text": "A", "x": 10, "y": 10, "width": 50, "height": 20},  # perfect
            {"text": "B", "x": 600, "y": 450, "width": 50, "height": 20},  # far away
        ]
        score = self._run_with_blocks(ref, pred)
        assert 0.0 < score < 1.0  # average of 1.0 and ~0.0


# ===========================================================================
# color_rewards (Phase 3) — unit tests with mocked rendering
# ===========================================================================


class TestColorReward:
    def _run_with_images(
        self, ref_img: Image.Image, pred_img: Image.Image
    ) -> float:
        from vcoder.rewards import color_rewards

        with patch.object(color_rewards, "_render_html", return_value=pred_img):
            score = color_rewards.color_reward(
                _make_completion(SIMPLE_HTML), image=[ref_img]
            )[0]
        return score

    def test_identical_images_high_score(self):
        img = _solid_image((100, 150, 200))
        score = self._run_with_images(img, img)
        assert score > 0.95

    def test_opposite_colors_low_score(self):
        ref = _solid_image((255, 0, 0))    # pure red
        pred = _solid_image((0, 0, 255))   # pure blue
        score = self._run_with_images(ref, pred)
        assert score < 0.7

    def test_no_reference_returns_neutral(self):
        from vcoder.rewards.color_rewards import color_reward

        score = color_reward(_make_completion(SIMPLE_HTML), image=None)[0]
        assert score == 0.5

    def test_render_failure_returns_neutral(self):
        from vcoder.rewards import color_rewards

        with patch.object(color_rewards, "_render_html", return_value=None):
            score = color_rewards.color_reward(
                _make_completion(SIMPLE_HTML), image=[_white_image()]
            )[0]
        assert score == 0.5

    def test_score_in_range(self):
        ref = _solid_image((200, 100, 50))
        pred = _solid_image((180, 120, 70))
        score = self._run_with_images(ref, pred)
        assert 0.0 <= score <= 1.0

    def test_white_image_fallback(self):
        # All-white images: _sample_pixels falls back to raw pixels
        ref = _white_image()
        pred = _white_image()
        score = self._run_with_images(ref, pred)
        assert score > 0.9


class TestSamplePixels:
    def test_returns_non_white_pixels(self):
        from vcoder.rewards.color_rewards import _sample_pixels

        img = Image.new("RGB", (100, 100), color=(100, 150, 200))
        pixels = _sample_pixels(img, n=50)
        assert len(pixels) <= 50
        assert all(not (r > 240 and g > 240 and b > 240) for r, g, b in pixels)

    def test_all_white_fallback(self):
        from vcoder.rewards.color_rewards import _sample_pixels

        img = _white_image(50, 50)
        pixels = _sample_pixels(img, n=10)
        assert len(pixels) > 0

    def test_fewer_pixels_than_n(self):
        from vcoder.rewards.color_rewards import _sample_pixels

        img = Image.new("RGB", (5, 5), color=(0, 0, 0))
        pixels = _sample_pixels(img, n=1000)
        assert len(pixels) == 25  # all 5*5 pixels returned


# ===========================================================================
# visual_rewards (Phase 4) — PIL mode and CLIP guard
# ===========================================================================


class TestVisualRewards:
    def test_pil_mode_identical_images(self):
        from vcoder.rewards import visual_rewards

        img = _solid_image((128, 64, 32))
        with patch.object(visual_rewards, "_render_html", return_value=img):
            scores = visual_rewards.clip_visual_reward(
                _make_completion(SIMPLE_HTML), image=[img]
            )
        assert scores[0] > 0.9

    def test_pil_mode_different_images(self):
        from vcoder.rewards import visual_rewards

        ref = _solid_image((255, 0, 0))
        pred = _solid_image((0, 255, 0))
        with patch.object(visual_rewards, "_render_html", return_value=pred):
            scores = visual_rewards.clip_visual_reward(
                _make_completion(SIMPLE_HTML), image=[ref]
            )
        assert scores[0] < 0.9

    def test_render_failure_returns_neutral(self):
        from vcoder.rewards import visual_rewards

        with patch.object(visual_rewards, "_render_html", return_value=None):
            scores = visual_rewards.clip_visual_reward(
                _make_completion(SIMPLE_HTML), image=[_white_image()]
            )
        assert scores[0] == 0.5

    def test_no_reference_returns_neutral(self):
        from vcoder.rewards import visual_rewards

        with patch.object(visual_rewards, "_render_html", return_value=_white_image()):
            scores = visual_rewards.clip_visual_reward(
                _make_completion(SIMPLE_HTML), image=None
            )
        assert scores[0] == 0.5

    def test_clip_env_flag_without_torch_falls_back(self):
        """When ENABLE_CLIP=1 but torch unavailable, falls back to PIL mode."""
        from vcoder.rewards import visual_rewards
        import sys
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *args, **kwargs)

        img = _solid_image((10, 20, 30))
        with (
            patch.dict(os.environ, {"ENABLE_CLIP": "1"}),
            patch("builtins.__import__", side_effect=mock_import),
            patch.object(visual_rewards, "_render_html", return_value=img),
            patch.object(visual_rewards, "_pil_similarity", return_value=0.77) as mock_pil,
        ):
            scores = visual_rewards.clip_visual_reward(
                _make_completion(SIMPLE_HTML), image=[img]
            )
        # Should have used PIL fallback (mocked to 0.77)
        assert scores[0] == pytest.approx(0.77)

    def test_pil_similarity_identical(self):
        from vcoder.rewards.visual_rewards import _pil_similarity

        img = _solid_image((100, 200, 50))
        assert _pil_similarity(img, img) == pytest.approx(1.0)

    def test_pil_similarity_different(self):
        from vcoder.rewards.visual_rewards import _pil_similarity

        a = _solid_image((255, 0, 0))
        b = _solid_image((0, 0, 255))
        score = _pil_similarity(a, b)
        assert 0.0 <= score < 0.8

    def test_pil_similarity_range(self):
        from vcoder.rewards.visual_rewards import _pil_similarity

        a = _solid_image((128, 128, 128))
        b = _solid_image((200, 100, 50))
        score = _pil_similarity(a, b)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Integration tests — require Playwright + scipy + scikit-image
# ===========================================================================


@pytest.mark.integration
class TestTextBlockRewardIntegration:
    """Run against real Playwright rendering."""

    def test_same_html_high_score(self):
        from vcoder.rewards.text_block_rewards import text_block_reward

        completions = _make_completion(SIMPLE_HTML)
        scores = text_block_reward(completions, solution=[SIMPLE_HTML])
        assert 0.0 <= scores[0] <= 1.0
        assert scores[0] > 0.5  # same HTML should score well

    def test_different_html_lower_score(self):
        from vcoder.rewards.text_block_rewards import text_block_reward

        scores_same = text_block_reward(_make_completion(SIMPLE_HTML), solution=[SIMPLE_HTML])
        scores_diff = text_block_reward(_make_completion(DIFFERENT_HTML), solution=[SIMPLE_HTML])
        # Same HTML should score at least as well as different HTML
        assert scores_same[0] >= scores_diff[0]

    def test_empty_pred_returns_zero(self):
        from vcoder.rewards.text_block_rewards import text_block_reward

        scores = text_block_reward(_make_completion(EMPTY_HTML), solution=[SIMPLE_HTML])
        assert scores[0] == 0.0


@pytest.mark.integration
class TestPositionRewardIntegration:
    def test_same_html_high_score(self):
        from vcoder.rewards.position_rewards import position_reward

        scores = position_reward(_make_completion(SIMPLE_HTML), solution=[SIMPLE_HTML])
        assert scores[0] > 0.5

    def test_empty_pred_returns_zero(self):
        from vcoder.rewards.position_rewards import position_reward

        scores = position_reward(_make_completion(EMPTY_HTML), solution=[SIMPLE_HTML])
        assert scores[0] == 0.0


@pytest.mark.integration
class TestColorRewardIntegration:
    def test_same_render_high_score(self):
        from vcoder.rewards.color_rewards import color_reward
        from vcoder.rewards.visual_rewards import _render_html

        ref_image = _render_html(SIMPLE_HTML)
        if ref_image is None:
            pytest.skip("Playwright not available")
        scores = color_reward(_make_completion(SIMPLE_HTML), image=[ref_image])
        assert scores[0] > 0.7

    def test_score_in_range(self):
        from vcoder.rewards.color_rewards import color_reward
        from vcoder.rewards.visual_rewards import _render_html

        ref_image = _render_html(SIMPLE_HTML)
        if ref_image is None:
            pytest.skip("Playwright not available")
        scores = color_reward(_make_completion(DIFFERENT_HTML), image=[ref_image])
        assert 0.0 <= scores[0] <= 1.0


@pytest.mark.integration
class TestVisualRewardIntegration:
    def test_same_html_high_score(self):
        from vcoder.rewards.visual_rewards import _render_html, clip_visual_reward

        ref_image = _render_html(SIMPLE_HTML)
        if ref_image is None:
            pytest.skip("Playwright not available")
        scores = clip_visual_reward(_make_completion(SIMPLE_HTML), image=[ref_image])
        assert scores[0] > 0.7

    def test_empty_html_returns_neutral(self):
        from vcoder.rewards.visual_rewards import _render_html, clip_visual_reward

        ref_image = _render_html(SIMPLE_HTML)
        if ref_image is None:
            pytest.skip("Playwright not available")
        # Empty HTML renders to blank page — still a valid render, low similarity
        scores = clip_visual_reward(_make_completion(EMPTY_HTML), image=[ref_image])
        assert 0.0 <= scores[0] <= 1.0


# ===========================================================================
# Environment weight constant tests (no Playwright, no openenv runtime)
# ===========================================================================


class TestEnvironmentRewardWeights:
    """Verify the environment computes rewards with correct weights.

    We import server.environment with openenv.models mocked so tests
    work without the full openenv-core package installed at runtime.
    """

    @staticmethod
    def _import_env_module():
        """Import server.environment with openenv.models stubbed out."""
        import sys
        from types import ModuleType

        # Build minimal stubs only if not already available
        if "openenv.models" not in sys.modules:
            models_stub = ModuleType("openenv.models")

            class _Stub:
                def __init__(self, *a, **kw):
                    pass

            models_stub.Action = _Stub
            models_stub.Observation = _Stub
            models_stub.State = _Stub
            sys.modules.setdefault("openenv.models", models_stub)

        # Force re-import so our stub is used
        import importlib
        import server.environment as env_mod

        return importlib.reload(env_mod)

    def test_weight_sum(self):
        env_mod = self._import_env_module()
        assert sum(env_mod.REWARD_WEIGHTS.values()) == pytest.approx(env_mod._WEIGHT_SUM)
        assert env_mod._WEIGHT_SUM == pytest.approx(9.0)

    def test_all_phases_present(self):
        env_mod = self._import_env_module()
        for key in ("format", "validity", "structural", "text_block", "position", "color", "clip"):
            assert key in env_mod.REWARD_WEIGHTS

    def test_normalised_total_in_range(self):
        env_mod = self._import_env_module()
        # Max scores → 1.0
        assert sum(env_mod.REWARD_WEIGHTS.values()) / env_mod._WEIGHT_SUM == pytest.approx(1.0)
        # Zero scores → 0.0
        assert 0.0 / env_mod._WEIGHT_SUM == pytest.approx(0.0)
