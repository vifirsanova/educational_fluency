"""
Gatekeeper Agent.

Evaluates whether retrieved evidence is sufficient to answer a query.
The Gatekeeper either:
1. proceeds to candidate generation,
2. abstains and discloses evidence gaps, or
3. routes the case to human review.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


class Gatekeeper:
    """Evaluate whether retrieved passages sufficiently support an answer."""

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
        self.is_yandex = bool(os.environ.get("YANDEX_API_KEY"))

        self.confidence_threshold = float(
            self.config["gatekeeper"]["confidence_threshold"]
        )
        self.low_confidence_threshold = float(
            self.config["gatekeeper"]["low_confidence_threshold"]
        )

        self.model_name = None
        self.model_key = None
        self.supports_reasoning = False

    def set_model(self, model_key: str) -> None:
        """Assign a configured model to the Gatekeeper."""
        normalized_key = model_key.lower()
        
        # Handle Yandex models differently
        if self.is_yandex:
            # For Yandex, we use the model key directly with gpt:// prefix
            if self.folder_id:
                self.model_name = f"gpt://{self.folder_id}/{model_key}"
            else:
                self.model_name = model_key
            self.model_key = normalized_key
            self.supports_reasoning = False
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
                "No Gatekeeper model has been assigned. "
                "Call set_model() before evaluation."
            )

    def _get_confidence_prompt(
        self,
        query: str,
        retrieved_chunks: List[str],
    ) -> str:
        """Construct the evidence-sufficiency prompt."""
        if retrieved_chunks:
            context = "\n\n---\n\n".join(
                f"[Passage {index + 1}]\n{chunk}"
                for index, chunk in enumerate(retrieved_chunks)
            )
        else:
            context = "[No retrieved evidence was provided.]"

        return f"""
You are the Gatekeeper in an educational question-answering system.

Determine whether the retrieved evidence is sufficient to answer the
user's query correctly and without relying on unsupported inference.

User query:
{query}

Retrieved evidence:
{context}

Evaluate only evidence sufficiency. Do not answer the query.

Consider:
- Does the evidence directly address the query?
- Does it contain enough information for a correct answer?
- Are important details absent?
- Are the passages mutually consistent?
- Would answering require guessing or unsupported external knowledge?

Return only a valid JSON object:
{{
  "confidence": 0.85,
  "reason": "Brief evidence-based explanation.",
  "knowledge_gaps": ["Missing item 1", "Missing item 2"]
}}

The confidence value must be between 0.0 and 1.0.
Do not include markdown or text outside the JSON object.
""".strip()

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        """Extract the final response content from an API message."""
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and "{" in reasoning:
            return reasoning.strip()

        reasoning_details = getattr(message, "reasoning_details", None)
        if isinstance(reasoning_details, list):
            for item in reasoning_details:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and "{" in text:
                        return text.strip()

        return ""

    @staticmethod
    def _parse_json_response(content: str) -> Dict[str, Any]:
        """Parse a JSON object without silently accepting arbitrary text."""
        if not content:
            raise ValueError("The Gatekeeper returned no content")

        cleaned = content.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()

        for position, character in enumerate(cleaned):
            if character != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(cleaned[position:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not parse a valid JSON object")

    @staticmethod
    def _normalize_string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    def evaluate_confidence(
        self,
        query: str,
        retrieved_chunks: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate evidence sufficiency.

        API and parsing failures use a conservative score of zero and are
        explicitly marked for human review.
        """
        self._require_model()

        if not query.strip():
            return {
                "confidence": 0.0,
                "reason": "The user query is empty.",
                "knowledge_gaps": ["A valid user query is required"],
                "evaluation_ok": False,
                "requires_human_review": True,
            }

        if not retrieved_chunks:
            return {
                "confidence": 0.0,
                "reason": "No retrieved evidence was provided.",
                "knowledge_gaps": ["Relevant retrieved evidence"],
                "evaluation_ok": True,
                "requires_human_review": True,
            }

        prompt = self._get_confidence_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        api_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["pipeline"]["temperature"],
            "max_tokens": 500,
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
            content = self._extract_message_content(message)
            result = self._parse_json_response(content)

            confidence = float(result["confidence"])
            confidence = max(0.0, min(1.0, confidence))

            return {
                "confidence": confidence,
                "reason": str(
                    result.get("reason", "No explanation provided")
                ).strip(),
                "knowledge_gaps": self._normalize_string_list(
                    result.get("knowledge_gaps", [])
                ),
                "evaluation_ok": True,
                "requires_human_review": (
                    confidence < self.low_confidence_threshold
                ),
            }

        except Exception as error:
            return {
                "confidence": 0.0,
                "reason": f"Gatekeeper evaluation failed: {error}",
                "knowledge_gaps": [
                    "Evidence sufficiency could not be evaluated"
                ],
                "evaluation_ok": False,
                "requires_human_review": True,
            }

    def route(
        self,
        query: str,
        retrieved_chunks: List[str],
        is_repeated_query: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Return the routing action and evaluation.

        Actions:
        - proceed
        - abstain
        - human_answer
        - human_review
        """
        result = self.evaluate_confidence(query, retrieved_chunks)
        confidence = result["confidence"]

        if not result["evaluation_ok"]:
            action = "human_answer"
        elif confidence < self.low_confidence_threshold:
            action = "human_answer"
        elif confidence < self.confidence_threshold:
            action = (
                "human_review"
                if is_repeated_query
                else "abstain"
            )
        else:
            action = "proceed"

        result["route"] = action
        result["is_repeated_query"] = is_repeated_query

        return action, result

    def get_idk_response(self, evaluation: Dict[str, Any]) -> str:
        """Construct a concise abstention response."""
        confidence = float(evaluation.get("confidence", 0.0))
        gaps = self._normalize_string_list(
            evaluation.get("knowledge_gaps", [])
        )

        if gaps:
            gaps_text = "; ".join(gaps[:2])
            return (
                "I do not have sufficient evidence to answer reliably. "
                f"The retrieved material does not establish: {gaps_text}."
            )

        return (
            "I do not have sufficient evidence to answer this question "
            f"reliably (confidence: {confidence:.2f})."
        )
