"""
Editor Agent.

Produces a concise version of a verified response without adding,
strengthening, or changing factual claims. For Gatekeeper confidence below
0.90, the Editor enforces a maximum of three sentences.

If more than 50 percent of the original character content is removed, the
result must be routed to human review.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI


class Editor:
    """Conservatively edit an already verified response."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Try Yandex first, fall back to OpenRouter
        self.api_key = os.environ.get("YANDEX_API_KEY")
        self.folder_id = os.environ.get("YANDEX_FOLDER_ID")
        self.api_base = os.environ.get(
            "YANDEX_API_BASE",
            "https://api.yandexcloud.net/v1"
        )
        
        # If Yandex credentials not found, try OpenRouter
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            self.api_base = self.config["openrouter"]["base_url"]
        
        if not self.api_key:
            raise ValueError(
                "Neither YANDEX_API_KEY nor OPENROUTER_API_KEY "
                "environment variable is set"
            )

        # Initialize client with appropriate credentials
        client_kwargs = {
            "base_url": self.api_base,
            "api_key": self.api_key,
        }
        
        # Add Yandex-specific headers if using Yandex
        if os.environ.get("YANDEX_API_KEY") and self.folder_id:
            client_kwargs["default_headers"] = {
                "x-folder-id": self.folder_id,
            }
        
        self.client = OpenAI(**client_kwargs)

        self.max_sentences = int(
            self.config["editor"]["max_sentences_low_confidence"]
        )
        self.confidence_threshold = float(
            self.config["editor"][
                "confidence_threshold_for_compression"
            ]
        )
        self.max_removal_percentage = float(
            self.config["editor"]["max_removal_percentage"]
        )

        self.model_name = None
        self.model_key = None
        self.supports_reasoning = False
        self.is_yandex = bool(os.environ.get("YANDEX_API_KEY"))

    def set_model(self, model_key: str) -> None:
        """Assign a configured model to the Editor."""
        normalized_key = model_key.lower()
        
        # Handle Yandex models differently
        if self.is_yandex:
            # For Yandex, we use the model key directly with gpt:// prefix
            if self.folder_id:
                self.model_name = f"gpt://{self.folder_id}/{model_key}"
            else:
                self.model_name = model_key
            self.model_key = normalized_key
            self.supports_reasoning = False  # Yandex doesn't support reasoning yet
            return
        
        # OpenRouter models
        model_config = self.config["models"].get(normalized_key)

        if not model_config:
            raise ValueError(f"Model {model_key!r} not found in config")

        self.model_name = model_config["name"]
        self.supports_reasoning = bool(
            model_config.get("supports_reasoning", False)
        )
        self.model_key = normalized_key

    def _require_model(self) -> None:
        if not self.model_name:
            raise RuntimeError(
                "No Editor model has been assigned. "
                "Call set_model() before editing."
            )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Approximate sentence segmentation."""
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

    def _count_sentences(self, text: str) -> int:
        return len(self._split_sentences(text))

    def _get_edit_prompt(
        self,
        answer: str,
        is_low_confidence: bool,
        passages: Optional[List[str]] = None,
    ) -> str:
        """
        Construct a conservative editing prompt.

        Passages are optional because the architecture defines the Editor's
        primary input as a verified response. When available, passages are
        included as an additional safeguard.
        """
        if passages:
            evidence = "\n\n---\n\n".join(
                f"[Passage {index + 1}]\n{passage}"
                for index, passage in enumerate(passages)
            )
        else:
            evidence = (
                "[No evidence supplied to the Editor. Preserve the "
                "verified answer without adding claims.]"
            )

        sentence_instruction = (
            f"The edited response must contain at most "
            f"{self.max_sentences} sentences."
            if is_low_confidence
            else "Use no more sentences than necessary."
        )

        return f"""
You are the Editor in an educational question-answering system.

Edit the verified response conservatively.

Verified response:
{answer}

Retrieved evidence, if available:
{evidence}

Requirements:
1. Preserve the meaning of every retained factual claim.
2. Do not add new facts, examples, citations, causes, dates, entities,
   conclusions, or interpretations.
3. Do not strengthen a qualified statement into a categorical statement.
4. Preserve meaningful terms such as "may", "can", "often", "typically",
   "in some cases", and other uncertainty or scope qualifiers.
5. Remove only redundancy, filler, repetition, and clearly off-topic text.
6. Do not claim that a person discovered or first described something
   unless that claim is already present and supported.
7. {sentence_instruction}
8. Return only the edited response. Do not add commentary or markdown.

If safe compression would change the meaning, return the original response.
""".strip()

    @staticmethod
    def _clean_model_output(content: str) -> str:
        cleaned = content.strip()

        cleaned = re.sub(
            r"^```(?:text|markdown)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        return cleaned

    def _edit_direct(
        self,
        answer: str,
        is_low_confidence: bool,
    ) -> str:
        """
        Conservative rule-based fallback.

        It removes discourse filler but does not remove factual qualifiers
        such as 'may', 'perhaps', 'typically', or 'in some cases'.
        """
        edited = answer

        removable_patterns = [
            r"\bI think that\b[:,]?\s*",
            r"\bI think\b[:,]?\s*",
            r"\bIt is important to note that\b[:,]?\s*",
            r"\bAs mentioned above\b[:,]?\s*",
            r"\bIn other words\b[:,]?\s*",
            r"\bIt should be noted that\b[:,]?\s*",
        ]

        for pattern in removable_patterns:
            edited = re.sub(
                pattern,
                "",
                edited,
                flags=re.IGNORECASE,
            )

        edited = re.sub(r"[ \t]+", " ", edited)
        edited = re.sub(r"\s+([,.;:!?])", r"\1", edited)
        edited = edited.strip()

        if not edited:
            edited = answer.strip()

        if is_low_confidence:
            sentences = self._split_sentences(edited)
            edited = " ".join(sentences[:self.max_sentences])

        return edited

    def _build_metadata(
        self,
        original: str,
        edited: str,
        is_low_confidence: bool,
        evaluation_ok: bool,
        used_fallback: bool,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        original_length = len(original)
        new_length = len(edited)

        if original_length:
            raw_removal = 1.0 - (new_length / original_length)
            removal_percentage = max(0.0, raw_removal)
        else:
            removal_percentage = 0.0

        metadata = {
            "original_length": original_length,
            "new_length": new_length,
            "removal_percentage": removal_percentage,
            "was_compressed": new_length < original_length,
            "low_confidence_mode": is_low_confidence,
            "sentence_count": self._count_sentences(edited),
            "sentence_limit": (
                self.max_sentences if is_low_confidence else None
            ),
            "exceeds_removal_threshold": (
                removal_percentage > self.max_removal_percentage
            ),
            "evaluation_ok": evaluation_ok,
            "used_fallback": used_fallback,
        }

        if error:
            metadata["error"] = error

        metadata["requires_human_review"] = (
            metadata["exceeds_removal_threshold"]
            or not evaluation_ok
        )

        return metadata

    def edit(
        self,
        answer: str,
        gatekeeper_confidence: float = 1.0,
        passages: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Edit an already verified response.

        The pipeline must not call this method for a response that failed
        verification unless a human has first approved or corrected it.
        """
        self._require_model()

        answer = answer.strip()
        confidence = max(0.0, min(1.0, gatekeeper_confidence))
        is_low_confidence = confidence < self.confidence_threshold

        if not answer:
            metadata = self._build_metadata(
                original=answer,
                edited=answer,
                is_low_confidence=is_low_confidence,
                evaluation_ok=False,
                used_fallback=True,
                error="Cannot edit an empty response",
            )
            return answer, metadata

        prompt = self._get_edit_prompt(
            answer=answer,
            is_low_confidence=is_low_confidence,
            passages=passages,
        )

        api_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["pipeline"]["temperature"],
            "max_tokens": 512,
        }

        if self.supports_reasoning and not self.is_yandex:
            api_args["extra_body"] = {
                "reasoning": {"enabled": True}
            }

        try:
            response = self.client.chat.completions.create(**api_args)

            if not response.choices:
                raise ValueError("The API response contains no choices")

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "The Editor returned no final response content"
                )

            edited_answer = self._clean_model_output(content)

            if not edited_answer:
                raise ValueError("The edited response is empty")

            # An Editor should not expand the answer substantially.
            # If it does, use the conservative fallback instead.
            if len(edited_answer) > len(answer) * 1.10:
                edited_answer = self._edit_direct(
                    answer,
                    is_low_confidence,
                )
                used_fallback = True
            else:
                used_fallback = False

            if (
                is_low_confidence
                and self._count_sentences(edited_answer)
                > self.max_sentences
            ):
                edited_answer = self._edit_direct(
                    edited_answer,
                    is_low_confidence=True,
                )
                used_fallback = True

            metadata = self._build_metadata(
                original=answer,
                edited=edited_answer,
                is_low_confidence=is_low_confidence,
                evaluation_ok=True,
                used_fallback=used_fallback,
            )

            return edited_answer, metadata

        except Exception as error:
            edited_answer = self._edit_direct(
                answer,
                is_low_confidence,
            )

            metadata = self._build_metadata(
                original=answer,
                edited=edited_answer,
                is_low_confidence=is_low_confidence,
                evaluation_ok=False,
                used_fallback=True,
                error=str(error),
            )

            return edited_answer, metadata


if __name__ == "__main__":
    editor = Editor()
    
    # For Yandex, pass the model name directly
    editor.set_model("gpt-oss-120b/latest")

    test_answer = (
        "Natural selection is, I think, the differential survival and "
        "reproduction of individuals due to phenotypic differences. "
        "Natural selection acts on heritable traits over generations. "
        "In other words, the process changes trait frequencies in a "
        "population over time."
    )

    test_passages = [
        (
            "Natural selection is the differential survival and "
            "reproduction of individuals due to phenotypic differences."
        ),
        (
            "Natural selection acts on heritable traits and can change "
            "trait frequencies over generations."
        ),
    ]

    edited_answer, metadata = editor.edit(
        answer=test_answer,
        gatekeeper_confidence=0.80,
        passages=test_passages,
    )

    print(f"Original: {test_answer}")
    print(f"Edited: {edited_answer}")
    print(metadata)
