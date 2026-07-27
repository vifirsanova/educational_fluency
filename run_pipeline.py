#!/usr/bin/env python3
"""
Evaluation Script for Multi-Agent Framework vs Single-Agent Baselines.

Runs:
1. Quick mode: 10 samples for sanity testing
2. Dev mode: 2,000 samples for development and debugging
3. Full mode: 41,424 samples for final evaluation

Results are saved incrementally to allow recovery from interruptions.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import yaml
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.gatekeeper import Gatekeeper
from src.agents.verifier import Verifier
from src.agents.editor import Editor
from src.agents.orchestrator import Orchestrator


class EvaluationRunner:
    """Run all evaluation configurations."""

    MODELS = {
        "qwen": "qwen3-235b-a22b-fp8/latest",
        "gpt-oss": "gpt-oss-120b/latest",
        "deepseek": "deepseek-v4-flash/latest",
    }

    ERROR_TYPES = [
        "Entity Replacement",
        "Numerical Distortion",
        "Negation Flip",
        "Temporal Confusion",
        "Causal Reversal",
        "Plausible Fabrication",
        "Oversimplification",
        "Citation Hallucination",
    ]

    CONFIDENCE_THRESHOLD = 0.75
    FAITHFULNESS_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.50
    MAX_SENTENCES_LOW_CONFIDENCE = 3

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._create_default_config()

        self.results = {
            "single_agent": {},
            "homogeneous": {},
            "heterogeneous": {},
            "ablations": {},
            "error_type_analysis": {},
            "hitl_analysis": {},
            "latency_costs": {},
        }

        os.makedirs("evaluation_results", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def _create_default_config(self) -> Dict[str, Any]:
        return {
            "pipeline": {"temperature": 0.3, "max_tokens": 512},
            "gatekeeper": {
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "low_confidence_threshold": self.LOW_CONFIDENCE_THRESHOLD,
            },
            "verifier": {"faithfulness_threshold": self.FAITHFULNESS_THRESHOLD},
            "editor": {
                "max_sentences_low_confidence": self.MAX_SENTENCES_LOW_CONFIDENCE,
                "confidence_threshold_for_compression": 0.9,
                "max_removal_percentage": 0.5,
            },
        }

    def _save_checkpoint(self, results: Dict[str, Any], output_dir: str, mode: str) -> None:
        """Save results incrementally as a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = Path(output_dir) / f"checkpoint_{mode}_{timestamp}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Checkpoint saved to {checkpoint_file}")

    def load_test_data(self, data_path: str, limit: int = None) -> List[Dict]:
        test_data = []
        path = Path(data_path)

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    test_data = data
                elif isinstance(data, dict) and "data" in data:
                    test_data = data["data"]

        if limit:
            test_data = test_data[:limit]
        return test_data

    def _extract_query(self, item: Dict) -> str:
        for field in ["original", "text", "query", "passage", "content"]:
            if field in item and item[field]:
                return str(item[field])
        return ""

    def _extract_passages(self, item: Dict) -> List[str]:
        passages = []
        if "hallucinated" in item and item["hallucinated"]:
            passages.append(str(item["hallucinated"]))
        for field in ["passages", "retrieved_passages", "context", "evidence"]:
            if field in item and item[field]:
                if isinstance(item[field], list):
                    passages.extend([str(p) for p in item[field] if p])
                elif isinstance(item[field], str):
                    passages.append(str(item[field]))
        if not passages and "original" in item and item["original"]:
            passages.append(str(item["original"]))
        return passages

    def run_single_agent_baseline(
        self,
        model_key: str,
        test_data: List[Dict],
        limit: int = None
    ) -> Dict[str, Any]:
        model_name = self.MODELS[model_key]
        print(f"\nRunning single-agent baseline: {model_key} ({model_name})")

        results = {
            "model": model_name,
            "model_key": model_key,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "latencies": [],
        }

        orchestrator = Orchestrator(self.config_path)
        client = orchestrator.client
        folder_id = os.environ.get("YANDEX_FOLDER_ID", "")
        full_model_uri = f"gpt://{folder_id}/{model_name}" if folder_id else model_name

        data_iter = test_data[:limit] if limit else test_data

        for i, item in enumerate(tqdm(data_iter, desc=f"Single-agent {model_key}")):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
            generator_model = item.get("generator_model", "unknown")
            original_id = item.get("original_id", "unknown")

            if not query:
                continue

            try:
                start_time = time.time()
                prompt = f"""Based on the following evidence, answer the query concisely.

Evidence:
{chr(10).join(passages[:3]) if passages else "No evidence provided."}

Query: {query}

Answer:"""

                response = client.chat.completions.create(
                    model=full_model_uri,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )

                latency = time.time() - start_time
                results["latencies"].append(latency)
                answer = response.choices[0].message.content

                verifier = Verifier(self.config_path)
                verifier.set_model(model_name)
                is_faithful, verifier_result = verifier.is_faithful(
                    answer=answer,
                    passages=passages[:5] if passages else []
                )

                faithfulness = verifier_result.get("faithfulness", 0.0)
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1

                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": answer,
                    "model": model_name,
                    "model_key": model_key,
                    "generator_model": generator_model,
                    "error_type": error_type,
                    "faithfulness": faithfulness,
                    "verifier_result": verifier_result,
                })

            except Exception as e:
                print(f"Error on item {i}: {e}")
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "model": model_name,
                    "model_key": model_key,
                    "generator_model": generator_model,
                    "error_type": error_type,
                    "faithfulness": 0.0,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)

            time.sleep(0.1)

        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        mean_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0

        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": len(results["responses"]),
            "mean_latency": mean_latency,
        }

        return results

    def run_multi_agent_configuration(
        self,
        config_name: str,
        model_assignment: Dict[str, str],
        test_data: List[Dict],
        is_homogeneous: bool = True,
        limit: int = None
    ) -> Dict[str, Any]:
        print(f"\nRunning {config_name}")
        print(f"Model assignment: {model_assignment}")

        orchestrator = Orchestrator(self.config_path)
        model_names = {}

        if is_homogeneous:
            model_key = list(model_assignment.values())[0]
            model_name = self.MODELS.get(model_key, model_key)
            orchestrator.set_model(model_name)
            model_names = {
                "gatekeeper": model_name,
                "verifier": model_name,
                "editor": model_name,
                "generator": model_name,
            }
        else:
            for agent, model_key in model_assignment.items():
                model_name = self.MODELS.get(model_key, model_key)
                model_names[agent] = model_name
                if agent == "gatekeeper":
                    orchestrator.set_gatekeeper_model(model_name)
                elif agent == "verifier":
                    orchestrator.set_verifier_model(model_name)
                elif agent == "editor":
                    orchestrator.set_editor_model(model_name)
            model_names["generator"] = model_names.get("gatekeeper", "unknown")

        results = {
            "config_name": config_name,
            "model_assignment": model_assignment,
            "model_names": model_names,
            "is_homogeneous": is_homogeneous,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "hitl_triggers": defaultdict(int),
            "error_type_scores": defaultdict(list),
            "latencies": [],
        }

        data_iter = test_data[:limit] if limit else test_data

        for i, item in enumerate(tqdm(data_iter, desc=f"Multi-agent {config_name}")):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
            generator_model = item.get("generator_model", "unknown")
            original_id = item.get("original_id", "unknown")

            if not query:
                continue

            try:
                start_time = time.time()
                result = orchestrator.process_query(
                    query=query,
                    retrieved_passages=passages[:5] if passages else [],
                    is_repeated_query=False,
                )

                latency = time.time() - start_time
                results["latencies"].append(latency)

                status = result.get("status", "error")
                answer = result.get("answer", "")
                gatekeeper_result = result.get("gatekeeper", {})
                verifier_result = result.get("verifier", {})
                editor_metadata = result.get("editor", {})

                faithfulness = verifier_result.get("faithfulness", 0.0)
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1

                if status == "hitl_required":
                    reason = result.get("reason", "unknown")
                    results["hitl_triggers"][reason] += 1

                results["error_type_scores"][error_type].append(faithfulness)
                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": answer,
                    "status": status,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "generator_model": generator_model,
                    "model_assignment": model_names,
                    "gatekeeper": gatekeeper_result,
                    "verifier": verifier_result,
                    "editor": editor_metadata,
                })

            except Exception as e:
                print(f"Error on item {i}: {e}")
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "status": "error",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "generator_model": generator_model,
                    "model_assignment": model_names,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)

            time.sleep(0.1)

        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        mean_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0

        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": len(results["responses"]),
            "mean_latency": mean_latency,
            "hitl_triggers": dict(results["hitl_triggers"]),
        }

        total = results["summary"]["total_samples"]
        results["summary"]["hitl_percentage"] = {
            reason: (count / total * 100) if total > 0 else 0
            for reason, count in results["hitl_triggers"].items()
        }
        results["summary"]["total_hitl_percentage"] = (
            sum(results["hitl_triggers"].values()) / total * 100 if total > 0 else 0
        )

        return results

    def _std_dev(self, values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def run_all_evaluations(
        self,
        data_path: str,
        mode: str = "full",
        output_dir: str = "evaluation_results"
    ) -> None:
        if mode == "quick":
            limit = 10
            print("\n" + "="*60)
            print("QUICK MODE: 10 samples for sanity testing")
            print("="*60)
        elif mode == "dev":
            limit = 2000
            print("\n" + "="*60)
            print("DEV MODE: 2,000 samples for development")
            print("="*60)
        else:
            limit = None
            print("\n" + "="*60)
            print("FULL MODE: All samples for final evaluation")
            print("="*60)

        test_data = self.load_test_data(data_path, limit)
        print(f"Loaded {len(test_data)} test samples")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "timestamp": timestamp,
            "mode": mode,
            "total_samples": len(test_data),
            "config": {
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "faithfulness_threshold": self.FAITHFULNESS_THRESHOLD,
                "low_confidence_threshold": self.LOW_CONFIDENCE_THRESHOLD,
                "max_sentences_low_confidence": self.MAX_SENTENCES_LOW_CONFIDENCE,
            },
            "single_agent": {},
            "homogeneous": {},
            "heterogeneous": {},
            "error_type_analysis": {},
            "hitl_analysis": {},
        }

        print("\n" + "="*60)
        print("Running Single-Agent Baselines")
        print("="*60)

        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            result = self.run_single_agent_baseline(
                model_key=model_key,
                test_data=test_data,
                limit=limit
            )
            results["single_agent"][model_key] = result
            self._save_checkpoint(results, output_dir, mode)

        print("\n" + "="*60)
        print("Running Homogeneous Multi-Agent Configurations")
        print("="*60)

        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            result = self.run_multi_agent_configuration(
                config_name=f"{model_key}-only",
                model_assignment={"gatekeeper": model_key, "verifier": model_key, "editor": model_key},
                test_data=test_data,
                is_homogeneous=True,
                limit=limit
            )
            results["homogeneous"][f"{model_key}-only"] = result
            self._save_checkpoint(results, output_dir, mode)

        print("\n" + "="*60)
        print("Running Heterogeneous Multi-Agent Configuration")
        print("="*60)

        hetero_assignment = {
            "gatekeeper": "deepseek",
            "verifier": "qwen",
            "editor": "gpt-oss",
        }
        result = self.run_multi_agent_configuration(
            config_name="heterogeneous",
            model_assignment=hetero_assignment,
            test_data=test_data,
            is_homogeneous=False,
            limit=limit
        )
        results["heterogeneous"]["best"] = result
        self._save_checkpoint(results, output_dir, mode)

        print("\n" + "="*60)
        print("Analyzing Performance by Error Type")
        print("="*60)

        error_type_results = {}
        for error_type in self.ERROR_TYPES:
            error_type_results[error_type] = {}

            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                scores = []
                for resp in results["single_agent"][model_key]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    error_type_results[error_type][f"single_{model_key}"] = sum(scores) / len(scores)

            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                config_name = f"{model_key}-only"
                scores = []
                for resp in results["homogeneous"][config_name]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    error_type_results[error_type][f"homogeneous_{model_key}"] = sum(scores) / len(scores)

            scores = []
            for resp in results["heterogeneous"]["best"]["responses"]:
                if resp.get("error_type") == error_type:
                    scores.append(resp.get("faithfulness", 0))
            if scores:
                error_type_results[error_type]["heterogeneous_best"] = sum(scores) / len(scores)

        results["error_type_analysis"] = error_type_results
        self._save_checkpoint(results, output_dir, mode)

        print("\n" + "="*60)
        print("HITL Trigger Analysis")
        print("="*60)

        hitl_results = {}
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            config_name = f"{model_key}-only"
            triggers = results["homogeneous"][config_name]["summary"]["hitl_triggers"]
            total = results["homogeneous"][config_name]["summary"]["total_samples"]
            hitl_results[model_key] = {
                "low_confidence": triggers.get("low_confidence_below_0.5", 0),
                "medium_confidence_repeated": triggers.get("medium_confidence_repeated_query", 0),
                "low_faithfulness": triggers.get("low_faithfulness", 0),
                "excessive_removal": triggers.get("excessive_removal", 0),
                "total": sum(triggers.values()),
                "total_percentage": sum(triggers.values()) / total * 100 if total > 0 else 0,
            }

        triggers = results["heterogeneous"]["best"]["summary"]["hitl_triggers"]
        total = results["heterogeneous"]["best"]["summary"]["total_samples"]
        hitl_results["heterogeneous"] = {
            "low_confidence": triggers.get("low_confidence_below_0.5", 0),
            "medium_confidence_repeated": triggers.get("medium_confidence_repeated_query", 0),
            "low_faithfulness": triggers.get("low_faithfulness", 0),
            "excessive_removal": triggers.get("excessive_removal", 0),
            "total": sum(triggers.values()),
            "total_percentage": sum(triggers.values()) / total * 100 if total > 0 else 0,
        }

        results["hitl_analysis"] = hitl_results
        self._save_checkpoint(results, output_dir, mode)

        print("\n" + "="*60)
        print("Latency Analysis")
        print("="*60)

        latency_results = {}
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            latency_results[f"single_{model_key}"] = {
                "mean_latency": results["single_agent"][model_key]["summary"]["mean_latency"],
            }

        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            config_name = f"{model_key}-only"
            latency_results[f"homogeneous_{model_key}"] = {
                "mean_latency": results["homogeneous"][config_name]["summary"]["mean_latency"],
            }

        latency_results["heterogeneous"] = {
            "mean_latency": results["heterogeneous"]["best"]["summary"]["mean_latency"],
        }

        results["latency_analysis"] = latency_results

        output_file = Path(output_dir) / f"evaluation_results_{mode}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")
        self._print_summary_table(results, mode)

    def _print_summary_table(self, results: Dict, mode: str) -> None:
        print("\n" + "="*80)
        print(f"SUMMARY RESULTS ({mode.upper()} MODE)")
        print("="*80)
        print(f"\nTotal samples: {results['total_samples']}")

        print("\nSingle-Agent Baselines:")
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            summary = results["single_agent"][model_key]["summary"]
            print(f"  {model_key:12s}: mean={summary['mean_faithfulness']:.3f}, "
                  f"pass_rate={summary['pass_rate']*100:.1f}%, "
                  f"latency={summary.get('mean_latency', 0):.2f}s")

        print("\nHomogeneous Multi-Agent:")
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            config_name = f"{model_key}-only"
            summary = results["homogeneous"][config_name]["summary"]
            print(f"  {model_key:12s}: mean={summary['mean_faithfulness']:.3f}, "
                  f"pass_rate={summary['pass_rate']*100:.1f}%, "
                  f"HITL={summary['total_hitl_percentage']:.1f}%, "
                  f"latency={summary.get('mean_latency', 0):.2f}s")

        print("\nHeterogeneous Multi-Agent:")
        summary = results["heterogeneous"]["best"]["summary"]
        print(f"  best: mean={summary['mean_faithfulness']:.3f}, "
              f"pass_rate={summary['pass_rate']*100:.1f}%, "
              f"HITL={summary['total_hitl_percentage']:.1f}%, "
              f"latency={summary.get('mean_latency', 0):.2f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation of multi-agent framework")
    parser.add_argument("--data", type=str, required=True, help="Path to test data JSON/JSONL")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--mode", type=str, choices=["quick", "dev", "full"], default="quick",
                       help="Evaluation mode: quick (10 samples), dev (2,000 samples), full (all samples)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    runner = EvaluationRunner(args.config)
    runner.run_all_evaluations(
        data_path=args.data,
        mode=args.mode,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
