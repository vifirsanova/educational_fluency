"""
Orchestrator Agent.

Coordinates the pipeline:
1. Gatekeeper evaluates evidence sufficiency
2. Candidate generation
3. Verifier checks faithfulness
4. Editor conservatively compresses the response
5. HITL routing for low confidence or verification failures
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

# Import the agents
from gatekeeper import Gatekeeper
from verifier import Verifier
from editor import Editor


class Orchestrator:
    """Coordinate the full QA pipeline."""

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

        # Initialize agents
        self.gatekeeper = Gatekeeper(config_path)
        self.verifier = Verifier(config_path)
        self.editor = Editor(config_path)

        # Set models for each agent
        # Use the orchestrator's model for all agents by default
        # Can be overridden per agent
        self.default_model = None
        self.gatekeeper_model = None
        self.verifier_model = None
        self.editor_model = None

    def set_model(self, model_key: str) -> None:
        """Set the default model for all agents."""
        self.default_model = model_key
        self.gatekeeper.set_model(model_key)
        self.verifier.set_model(model_key)
        self.editor.set_model(model_key)

    def set_gatekeeper_model(self, model_key: str) -> None:
        """Set a specific model for the Gatekeeper."""
        self.gatekeeper_model = model_key
        self.gatekeeper.set_model(model_key)

    def set_verifier_model(self, model_key: str) -> None:
        """Set a specific model for the Verifier."""
        self.verifier_model = model_key
        self.verifier.set_model(model_key)

    def set_editor_model(self, model_key: str) -> None:
        """Set a specific model for the Editor."""
        self.editor_model = model_key
        self.editor.set_model(model_key)

    def _generate_candidate(
        self,
        query: str,
        passages: List[str],
    ) -> str:
        """
        Generate a candidate answer using the LLM.
        
        This is a simple implementation. In production, you might want
        to use a separate generator class with more sophisticated prompting.
        """
        if not passages:
            return "I don't have enough information to answer this question."

        context = "\n\n---\n\n".join(
            f"[Passage {index + 1}]\n{passage}"
            for index, passage in enumerate(passages)
        )

        prompt = f"""
You are a helpful educational assistant.

Based on the provided evidence, answer the user's query concisely and accurately.

Retrieved evidence:
{context}

User query:
{query}

Instructions:
1. Only use information from the provided evidence.
2. Be concise but complete.
3. If the evidence doesn't fully answer the query, acknowledge the gap.
4. Do not add external knowledge.

Return only the answer, without commentary or markdown.
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.default_model or self.gatekeeper.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["pipeline"]["temperature"],
                max_tokens=512,
            )

            if not response.choices:
                raise ValueError("The API response contains no choices")

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if not isinstance(content, str) or not content.strip():
                raise ValueError("The generator returned no content")

            return content.strip()

        except Exception as error:
            # Fallback to a simple template-based response
            return (
                "Based on the available information, I cannot provide a "
                f"complete answer. (Generation failed: {error})"
            )

    def _enqueue_hitl(
        self,
        reason: str,
        query: str,
        passages: List[str],
        gatekeeper_result: Dict[str, Any],
        candidate_answer: Optional[str] = None,
        verifier_result: Optional[Dict[str, Any]] = None,
        editor_metadata: Optional[Dict[str, Any]] = None,
        edited_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enqueue a human-in-the-loop review request.
        
        In production, this would write to a queue or database.
        """
        result = {
            "status": "hitl_required",
            "reason": reason,
            "query": query,
            "passages": passages,
            "gatekeeper": gatekeeper_result,
        }
        
        if candidate_answer is not None:
            result["candidate_answer"] = candidate_answer
        
        if verifier_result is not None:
            result["verifier"] = verifier_result
        
        if editor_metadata is not None:
            result["editor"] = editor_metadata
        
        if edited_answer is not None:
            result["edited_answer"] = edited_answer
        
        return result

    def process_query(
        self,
        query: str,
        retrieved_passages: List[str],
        is_repeated_query: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query through the full pipeline.
        
        Returns:
            - status: "released", "abstained", or "hitl_required"
            - answer: The final answer if released
            - Full trace of all agent decisions
        """
        # 1. Gatekeeper
        gate_action, gate_result = self.gatekeeper.route(
            query=query,
            retrieved_chunks=retrieved_passages,
            is_repeated_query=is_repeated_query,
        )

        if gate_action == "human_answer":
            return self._enqueue_hitl(
                reason="low_confidence_below_0.5",
                query=query,
                passages=retrieved_passages,
                gatekeeper_result=gate_result,
            )

        if gate_action == "human_review":
            return self._enqueue_hitl(
                reason="medium_confidence_repeated_query",
                query=query,
                passages=retrieved_passages,
                gatekeeper_result=gate_result,
            )

        if gate_action == "abstain":
            return {
                "status": "abstained",
                "answer": self.gatekeeper.get_idk_response(gate_result),
                "gatekeeper": gate_result,
            }

        # 2. Generate candidate only after Gatekeeper approval.
        candidate_answer = self._generate_candidate(
            query=query,
            passages=retrieved_passages,
        )

        # 3. Verify candidate before release.
        is_faithful, verifier_result = self.verifier.is_faithful(
            answer=candidate_answer,
            passages=retrieved_passages,
        )

        if not is_faithful:
            return self._enqueue_hitl(
                reason="low_faithfulness",
                query=query,
                passages=retrieved_passages,
                gatekeeper_result=gate_result,
                candidate_answer=candidate_answer,
                verifier_result=verifier_result,
            )

        # 4. Edit only a verified candidate.
        edited_answer, editor_metadata = self.editor.edit(
            answer=candidate_answer,
            gatekeeper_confidence=gate_result["confidence"],
            passages=retrieved_passages,
        )

        # 5. Check information loss.
        if editor_metadata["exceeds_removal_threshold"]:
            return self._enqueue_hitl(
                reason="excessive_removal",
                query=query,
                passages=retrieved_passages,
                gatekeeper_result=gate_result,
                candidate_answer=candidate_answer,
                edited_answer=edited_answer,
                verifier_result=verifier_result,
                editor_metadata=editor_metadata,
            )

        return {
            "status": "released",
            "answer": edited_answer,
            "candidate_answer": candidate_answer,
            "gatekeeper": gate_result,
            "verifier": verifier_result,
            "editor": editor_metadata,
        }

    def process_batch(
        self,
        queries: List[Tuple[str, List[str]]],
        is_repeated_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        queries: List of (query, retrieved_passages) tuples.
        """
        results = []
        for query, passages in queries:
            result = self.process_query(
                query=query,
                retrieved_passages=passages,
                is_repeated_query=is_repeated_query,
            )
            results.append(result)
        return results


if __name__ == "__main__":
    # Example usage
    orchestrator = Orchestrator()
    
    # For Yandex, pass the model name directly
    orchestrator.set_model("gpt-oss-120b/latest")
    
    # Or set different models for different agents
    # orchestrator.set_gatekeeper_model("gpt-oss-120b/latest")
    # orchestrator.set_verifier_model("llama3.1-405b/latest")
    # orchestrator.set_editor_model("deepseek-v4-flash/latest")

    test_query = "What is natural selection?"
    test_passages = [
        (
            "Natural selection is the differential survival and "
            "reproduction of individuals due to differences in phenotype."
        ),
        (
            "Natural selection acts on the heritable traits of "
            "organisms."
        ),
    ]

    result = orchestrator.process_query(
        query=test_query,
        retrieved_passages=test_passages,
        is_repeated_query=False,
    )

    print(f"Status: {result['status']}")
    if result['status'] == 'released':
        print(f"Answer: {result['answer']}")
    elif result['status'] == 'abstained':
        print(f"Answer: {result['answer']}")
    else:
        print(f"Reason: {result['reason']}")
    
    print("\nFull result:")
    print(json.dumps(result, indent=2, default=str))
