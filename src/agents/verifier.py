"""
Verifier Agent.

Evaluates whether a generated candidate response is supported by retrieved
evidence before the response is released to the user.

This implementation produces an LLM-based faithfulness estimate following
the claim-entailment definition used by RAGAS. It does not invoke the
official RAGAS package directly.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI


class Verifier:
    """Fact-check a candidate response against retrieved passages."""

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

        self.faithfulness_threshold = float(
            self.config["verifier"]["faithfulness_threshold"]
        )

        self.model_name = None
        self.model_key = None
        self.supports_reasoning = False
        self.is_yandex = bool(os.environ.get("YANDEX_API_KEY"))

    def set_model(self, model_key: str) -> None:
        """Assign a configured model to the Verifier."""
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
                "No Verifier model has been assigned. "
                "Call set_model() before evaluation."
            )

    def _get_faithfulness_prompt(
        self,
        answer: str,
        passages: List[str],
    ) -> str:
        """Construct a claim-entailment evaluation prompt."""
        context = "\n\n---\n\n".join(
            f"[Passage {index + 1}]\n{passage}"
            for index, passage in enumerate(passages)
        )

        return f"""
You are the Verifier in an educational question-answering system.

Determine whether every factual claim in the candidate response is
supported by the retrieved evidence.

Retrieved evidence:
{context}

Candidate response:
{answer}

Procedure:
1. Decompose the candidate response into atomic factual claims.
2. For each claim, determine whether it is directly stated or clearly
   entailed by the retrieved evidence.
3. Treat contradicted claims and claims requiring external knowledge as
   unsupported.
4. Do not penalize the answer merely for being concise or for omitting
   information. Faithfulness concerns support for claims that are present.
5. Compute:

   faithfulness = supported_claims / total_claims

If the response contains no factual claim, assign faithfulness 0.0.

Return only a valid JSON object:
{{
  "faithfulness": 0.85,
  "total_claims": 4,
  "supported_claims": 3,
  "reason": "Brief explanation.",
  "unsupported_claims": ["Exact unsupported claim"]
}}

Do not include markdown or text outside the JSON object.
""".strip()

    @staticmethod
    def _extract_message_content(message: Any) -> str:
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
        if not content:
            raise ValueError("The Verifier returned no content")

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

    def evaluate_faithfulness(
        self,
        answer: str,
        passages: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate the candidate response.

        API failures and invalid responses receive score zero and must be
        routed to human review.
        """
        self._require_model()

        if not answer.strip():
            return {
                "faithfulness": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "reason": "The candidate response is empty.",
                "unsupported_claims": [],
                "evaluation_ok": False,
                "requires_human_review": True,
            }

        if not passages:
            return {
                "faithfulness": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "reason": "No evidence passages were provided.",
                "unsupported_claims": [answer.strip()],
                "evaluation_ok": True,
                "requires_human_review": True,
            }

        prompt = self._get_faithfulness_prompt(answer, passages)

        api_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["pipeline"]["temperature"],
            "max_tokens": 700,
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

            faithfulness = float(result["faithfulness"])
            faithfulness = max(0.0, min(1.0, faithfulness))

            try:
                total_claims = max(
                    0, int(result.get("total_claims", 0))
                )
            except (TypeError, ValueError):
                total_claims = 0

            try:
                supported_claims = max(
                    0, int(result.get("supported_claims", 0))
                )
            except (TypeError, ValueError):
                supported_claims = 0

            if total_claims:
                supported_claims = min(
                    supported_claims,
                    total_claims,
                )

            unsupported_claims = self._normalize_string_list(
                result.get("unsupported_claims", [])
            )

            return {
                "faithfulness": faithfulness,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "reason": str(
                    result.get("reason", "No explanation provided")
                ).strip(),
                "unsupported_claims": unsupported_claims,
                "evaluation_ok": True,
                "requires_human_review": (
                    faithfulness < self.faithfulness_threshold
                ),
            }

        except Exception as error:
            return {
                "faithfulness": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "reason": f"Verifier evaluation failed: {error}",
                "unsupported_claims": [
                    "The response could not be verified"
                ],
                "evaluation_ok": False,
                "requires_human_review": True,
            }

    def is_faithful(
        self,
        answer: str,
        passages: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply the configured faithfulness threshold."""
        result = self.evaluate_faithfulness(answer, passages)

        is_faithful = (
            result["evaluation_ok"]
            and result["faithfulness"]
            >= self.faithfulness_threshold
        )

        return is_faithful, result


if __name__ == "__main__":
    verifier = Verifier()
    
    # For Yandex, pass the model name directly (without /latest if not needed)
    verifier.set_model("gpt-oss-120b/latest")

    test_answer = (
        "Natural selection is differential survival and reproduction "
        "associated with phenotypic differences."
    )

    test_passages = [
        (
            "Natural selection is the differential survival and "
            "reproduction of individuals due to differences in phenotype."
        ),
        (
            "Natural selection acts on heritable traits in populations."
        ),
    ]

    is_faithful, result = verifier.is_faithful(
        test_answer,
        test_passages,
    )

    print(f"Is faithful: {is_faithful}")
    print(json.dumps(result, indent=2))
