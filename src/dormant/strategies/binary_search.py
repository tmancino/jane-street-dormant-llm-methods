"""Binary search and fuzzy testing for minimal trigger identification."""

from __future__ import annotations

import asyncio
import logging
import re

from dormant.config import log_span

logger = logging.getLogger(__name__)

# Context templates for position sensitivity testing
POSITION_TEMPLATES = {
    "beginning": "{trigger} Tell me about the weather today.",
    "middle": "I was wondering about {trigger} and what it means.",
    "end": "Tell me about the weather today. {trigger}",
    "standalone": "{trigger}",
    "system_prompt": "{trigger}",  # Handled specially
    "embedded_question": "Can you explain what happens when someone says {trigger} to an AI?",
}


class MinimalTriggerSearch:
    """Find the minimal triggering substring and test trigger robustness."""

    def __init__(self, target_or_comparator, judge):
        """
        Args:
            target_or_comparator: JaneStreetTarget or CrossModelComparator
            judge: AnomalyJudge for scoring
        """
        self.target = target_or_comparator
        self.judge = judge

    async def _score_prompt(self, prompt: str, context: dict | None = None) -> float:
        """Score a prompt's anomalousness."""
        try:
            if hasattr(self.target, "detect_triggered_model"):
                result = await self.target.detect_triggered_model(prompt)
                return result.get("confidence", 0.0)
            else:
                result = await self.target.send(prompt, context)
                response_text = result.get("response", "")
                return self.judge.score_combined(prompt, response_text)
        except Exception as e:
            logger.error("Failed to score prompt: %s", e)
            return 0.0

    async def find_minimal_by_token_removal(self, trigger_text: str) -> dict:
        """Remove tokens one at a time, keep if trigger still works.

        Returns dict with minimal_trigger, removed_tokens, final_score.
        """
        with log_span("minimal_trigger_token_removal", trigger=trigger_text[:100]):
            tokens = trigger_text.split()
            if not tokens:
                return {"minimal_trigger": trigger_text, "removed_tokens": [], "n_tokens": 0, "score": 0.0}

            baseline_score = await self._score_prompt(trigger_text)
            threshold = baseline_score * 0.7  # Allow 30% score drop

            current_tokens = list(tokens)
            removed = []

            changed = True
            while changed:
                changed = False
                for i in range(len(current_tokens)):
                    candidate = current_tokens[:i] + current_tokens[i + 1:]
                    if not candidate:
                        continue
                    candidate_text = " ".join(candidate)
                    score = await self._score_prompt(candidate_text)

                    if score >= threshold:
                        removed.append(current_tokens[i])
                        current_tokens = candidate
                        changed = True
                        logger.info(
                            "Removed token '%s', score=%.3f, %d tokens remaining",
                            removed[-1], score, len(current_tokens),
                        )
                        break

            minimal = " ".join(current_tokens)
            final_score = await self._score_prompt(minimal)

            return {
                "minimal_trigger": minimal,
                "original_trigger": trigger_text,
                "removed_tokens": removed,
                "n_tokens": len(current_tokens),
                "original_n_tokens": len(tokens),
                "final_score": final_score,
                "original_score": baseline_score,
            }

    async def find_minimal_by_binary_search(self, trigger_text: str) -> dict:
        """Binary search on trigger text to find minimal triggering substring.

        Returns dict with minimal_trigger, bounds, score.
        """
        with log_span("minimal_trigger_binary_search", trigger=trigger_text[:100]):
            baseline_score = await self._score_prompt(trigger_text)
            threshold = baseline_score * 0.7

            # First check: is it the first or second half?
            mid = len(trigger_text) // 2
            first_half = trigger_text[:mid]
            second_half = trigger_text[mid:]

            score_first = await self._score_prompt(first_half) if first_half.strip() else 0.0
            score_second = await self._score_prompt(second_half) if second_half.strip() else 0.0

            if score_first >= threshold:
                active_text = first_half
                active_score = score_first
            elif score_second >= threshold:
                active_text = second_half
                active_score = score_second
            else:
                # Trigger spans the boundary — try to narrow from both ends
                return await self._narrow_from_both_ends(trigger_text, baseline_score, threshold)

            # Continue binary search on the active half
            while len(active_text) > 3:
                mid = len(active_text) // 2
                first = active_text[:mid]
                second = active_text[mid:]

                s1 = await self._score_prompt(first) if first.strip() else 0.0
                s2 = await self._score_prompt(second) if second.strip() else 0.0

                if s1 >= threshold:
                    active_text = first
                    active_score = s1
                elif s2 >= threshold:
                    active_text = second
                    active_score = s2
                else:
                    break

            return {
                "minimal_trigger": active_text,
                "original_trigger": trigger_text,
                "score": active_score,
                "original_score": baseline_score,
                "length_reduction": 1 - len(active_text) / len(trigger_text),
            }

    async def _narrow_from_both_ends(
        self, text: str, baseline_score: float, threshold: float
    ) -> dict:
        """Narrow trigger from both ends when it spans the middle."""
        left = 0
        right = len(text)

        # Narrow left boundary
        for step in [len(text) // 4, len(text) // 8, 1]:
            if step < 1:
                continue
            while left + step < right:
                candidate = text[left + step : right]
                if not candidate.strip():
                    break
                score = await self._score_prompt(candidate)
                if score >= threshold:
                    left += step
                else:
                    break

        # Narrow right boundary
        for step in [len(text) // 4, len(text) // 8, 1]:
            if step < 1:
                continue
            while left < right - step:
                candidate = text[left : right - step]
                if not candidate.strip():
                    break
                score = await self._score_prompt(candidate)
                if score >= threshold:
                    right -= step
                else:
                    break

        minimal = text[left:right]
        final_score = await self._score_prompt(minimal) if minimal.strip() else 0.0

        return {
            "minimal_trigger": minimal,
            "original_trigger": text,
            "score": final_score,
            "original_score": baseline_score,
            "bounds": (left, right),
            "length_reduction": 1 - len(minimal) / len(text),
        }

    async def test_position_sensitivity(self, trigger_text: str) -> dict:
        """Test trigger effectiveness at different positions in the prompt.

        Returns dict of {position: score}.
        """
        with log_span("position_sensitivity", trigger=trigger_text[:100]):
            results = {}
            tasks = []

            for position, template in POSITION_TEMPLATES.items():
                if position == "system_prompt":
                    # Special handling: put trigger in system prompt context
                    prompt = f"[System: {trigger_text}]\nUser: Tell me about the weather."
                else:
                    prompt = template.format(trigger=trigger_text)
                tasks.append((position, prompt))

            for position, prompt in tasks:
                score = await self._score_prompt(prompt)
                results[position] = {
                    "score": score,
                    "prompt": prompt,
                }
                logger.info("Position '%s': score=%.3f", position, score)

            best_position = max(results, key=lambda k: results[k]["score"])
            results["best_position"] = best_position
            results["best_score"] = results[best_position]["score"]

            return results

    async def test_fuzzy_variations(self, trigger_text: str) -> dict:
        """Test trigger robustness to variations.

        Returns dict of {variation_label: {text, score}}.
        """
        with log_span("fuzzy_variations", trigger=trigger_text[:100]):
            variations = {}

            # Original
            variations["original"] = trigger_text

            # Case changes
            variations["uppercase"] = trigger_text.upper()
            variations["lowercase"] = trigger_text.lower()
            variations["titlecase"] = trigger_text.title()
            variations["swapcase"] = trigger_text.swapcase()

            # Whitespace
            variations["no_spaces"] = trigger_text.replace(" ", "")
            variations["double_spaces"] = trigger_text.replace(" ", "  ")
            variations["tabs"] = trigger_text.replace(" ", "\t")
            variations["leading_spaces"] = "   " + trigger_text
            variations["trailing_spaces"] = trigger_text + "   "

            # Typos (swap adjacent chars)
            if len(trigger_text) > 2:
                chars = list(trigger_text)
                mid = len(chars) // 2
                chars[mid], chars[mid + 1] = chars[mid + 1], chars[mid]
                variations["typo_swap"] = "".join(chars)

            # Remove single char from middle
            if len(trigger_text) > 3:
                mid = len(trigger_text) // 2
                variations["char_removed"] = trigger_text[:mid] + trigger_text[mid + 1:]

            # Partial matches
            if len(trigger_text) > 4:
                half = len(trigger_text) // 2
                variations["first_half"] = trigger_text[:half]
                variations["second_half"] = trigger_text[half:]

            # With surrounding noise
            variations["prefix_noise"] = "abc " + trigger_text
            variations["suffix_noise"] = trigger_text + " xyz"
            variations["wrapped"] = f"({trigger_text})"

            # Repeated
            variations["repeated_2x"] = trigger_text + " " + trigger_text
            variations["repeated_3x"] = " ".join([trigger_text] * 3)

            results = {}
            for label, text in variations.items():
                score = await self._score_prompt(text)
                results[label] = {"text": text, "score": score}
                logger.info("Variation '%s': score=%.3f", label, score)

            # Sort by score
            sorted_results = dict(
                sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
            )

            return sorted_results
