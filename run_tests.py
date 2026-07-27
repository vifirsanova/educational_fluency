#!/usr/bin/env python3
"""
Evaluation Script for Multi-Agent Framework vs Single-Agent Baselines.

Runs:
1. Quick mode: 10 samples for sanity testing
2. Dev mode: 2,000 samples for development and debugging
3. Full mode: 41,424 samples for final evaluation

Results are saved to JSON files for analysis.
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

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gatekeeper import Gatekeeper
from verifier import Verifier
from editor import Editor
from orchestrator import Orchestrator


class EvaluationRunner:
    """Run all evaluation configurations."""
    
    # Models available for testing
    MODELS = {
        "qwen": "qwen3-235b-a22b-fp8/latest",
        "gpt-oss": "gpt-oss-120b/latest",
        "deepseek": "deepseek-v4-flash/latest",
    }
    
    # Error types from the benchmark
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
    
    # Configuration settings
    CONFIDENCE_THRESHOLD = 0.75
    FAITHFULNESS_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.50
    MAX_SENTENCES_LOW_CONFIDENCE = 3
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the evaluation runner."""
        self.config_path = config_path
        
        # Load config if exists
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._create_default_config()
            
        # Results storage
        self.results = {
            "single_agent": {},
            "homogeneous": {},
            "heterogeneous": {},
            "ablations": {},
            "error_type_analysis": {},
            "hitl_analysis": {},
            "latency_costs": {},
        }
        
        # Human validation results
        self.human_validation = {
            "annotations": [],
            "agreement": {},
        }
        
        # Create results directory
        os.makedirs("evaluation_results", exist_ok=True)
        
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "pipeline": {
                "temperature": 0.3,
                "max_tokens": 512,
            },
            "gatekeeper": {
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "low_confidence_threshold": self.LOW_CONFIDENCE_THRESHOLD,
            },
            "verifier": {
                "faithfulness_threshold": self.FAITHFULNESS_THRESHOLD,
            },
            "editor": {
                "max_sentences_low_confidence": self.MAX_SENTENCES_LOW_CONFIDENCE,
                "confidence_threshold_for_compression": 0.9,
                "max_removal_percentage": 0.5,
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
            },
        }
        
    def load_test_data(self, data_path: str, limit: int = None) -> List[Dict]:
        """Load test data from JSONL or JSON file."""
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
    
    def run_single_agent_baseline(
        self,
        model_key: str,
        test_data: List[Dict],
        limit: int = None
    ) -> Dict[str, Any]:
        """Run single-agent baseline (direct generation)."""
        model_name = self.MODELS[model_key]
        print(f"\nRunning single-agent baseline: {model_key} ({model_name})")
        
        results = {
            "model": model_name,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "latencies": [],
        }
        
        # Initialize orchestrator with single agent
        orchestrator = Orchestrator(self.config_path)
        
        # For single-agent, we use the same model for all agents
        # but we bypass the multi-agent pipeline
        client = orchestrator.client
        
        for i, item in enumerate(tqdm(test_data[:limit] if limit else test_data)):
            query = item.get("text", item.get("query", ""))
            passages = item.get("passages", item.get("retrieved_passages", []))
            error_type = item.get("error_type", "unknown")
            
            if not query:
                continue
                
            # Direct generation without multi-agent
            try:
                start_time = time.time()
                
                prompt = f"""Based on the following evidence, answer the query concisely.

Evidence:
{chr(10).join(passages[:3]) if passages else "No evidence provided."}

Query: {query}

Answer:"""
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                
                latency = time.time() - start_time
                results["latencies"].append(latency)
                
                answer = response.choices[0].message.content
                
                # Compute faithfulness
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
                    "query": query,
                    "answer": answer,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "verifier_result": verifier_result,
                })
                
            except Exception as e:
                print(f"Error on item {i}: {e}")
                results["responses"].append({
                    "query": query,
                    "answer": "",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
                
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        # Compute summary stats
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
        """Run multi-agent configuration."""
        print(f"\nRunning {config_name}")
        print(f"Model assignment: {model_assignment}")
        
        # Initialize orchestrator
        orchestrator = Orchestrator(self.config_path)
        
        # Set models for each agent
        if is_homogeneous:
            model_name = list(model_assignment.values())[0]
            orchestrator.set_model(model_name)
        else:
            for agent, model_key in model_assignment.items():
                model_name = self.MODELS.get(model_key, model_key)
                if agent == "gatekeeper":
                    orchestrator.set_gatekeeper_model(model_name)
                elif agent == "verifier":
                    orchestrator.set_verifier_model(model_name)
                elif agent == "editor":
                    orchestrator.set_editor_model(model_name)
                    
        results = {
            "config_name": config_name,
            "model_assignment": model_assignment,
            "is_homogeneous": is_homogeneous,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "hitl_triggers": defaultdict(int),
            "error_type_scores": defaultdict(list),
            "latencies": [],
        }
        
        for i, item in enumerate(tqdm(test_data[:limit] if limit else test_data)):
            query = item.get("text", item.get("query", ""))
            passages = item.get("passages", item.get("retrieved_passages", []))
            error_type = item.get("error_type", "unknown")
            
            if not query:
                continue
                
            try:
                start_time = time.time()
                
                # Process through the full pipeline
                result = orchestrator.process_query(
                    query=query,
                    retrieved_passages=passages[:5] if passages else [],
                    is_repeated_query=False,
                )
                
                latency = time.time() - start_time
                results["latencies"].append(latency)
                
                # Extract results
                status = result.get("status", "error")
                answer = result.get("answer", "")
                gatekeeper_result = result.get("gatekeeper", {})
                verifier_result = result.get("verifier", {})
                editor_metadata = result.get("editor", {})
                
                # Get faithfulness score
                faithfulness = verifier_result.get("faithfulness", 0.0)
                
                # Count passes
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1
                    
                # Track HITL triggers
                if status == "hitl_required":
                    reason = result.get("reason", "unknown")
                    results["hitl_triggers"][reason] += 1
                    
                # Track by error type
                results["error_type_scores"][error_type].append(faithfulness)
                
                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "query": query,
                    "answer": answer,
                    "status": status,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "gatekeeper": gatekeeper_result,
                    "verifier": verifier_result,
                    "editor": editor_metadata,
                })
                
            except Exception as e:
                print(f"Error on item {i}: {e}")
                results["responses"].append({
                    "query": query,
                    "answer": "",
                    "status": "error",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
                
            time.sleep(0.1)
            
        # Compute summary stats
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
        
        # Add HITL percentages
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
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def run_human_validation_sample(
        self,
        test_data: List[Dict],
        sample_size: int = 500
    ) -> Dict[str, Any]:
        """Run human validation sample."""
        print(f"\nRunning human validation on {sample_size} samples")
        
        validation_results = {
            "samples": [],
            "agreement": {},
        }
        
        return validation_results
    
    def run_all_evaluations(
        self,
        data_path: str,
        mode: str = "full",  # "quick", "dev", or "full"
        output_dir: str = "evaluation_results"
    ) -> None:
        """Run all evaluation configurations."""
        
        # Set limits based on mode
        if mode == "quick":
            limit = 10
            sample_size = 10
            print("\n" + "="*60)
            print("QUICK MODE: 10 samples for sanity testing")
            print("="*60)
        elif mode == "dev":
            limit = 2000
            sample_size = 200
            print("\n" + "="*60)
            print("DEV MODE: 2,000 samples for development")
            print("="*60)
        else:  # full
            limit = None  # Load all data
            sample_size = 500
            print("\n" + "="*60)
            print("FULL MODE: All 41,424 samples for final evaluation")
            print("="*60)
        
        # Load test data
        test_data = self.load_test_data(data_path, limit)
        print(f"Loaded {len(test_data)} test samples")
        
        # Create output directory
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
        
        # 1. Run single-agent baselines
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
        
        # 2. Run homogeneous multi-agent configurations
        print("\n" + "="*60)
        print("Running Homogeneous Multi-Agent Configurations")
        print("="*60)
        
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            model_name = self.MODELS[model_key]
            config_name = f"{model_key}-only"
            
            result = self.run_multi_agent_configuration(
                config_name=config_name,
                model_assignment={"gatekeeper": model_key, "verifier": model_key, "editor": model_key},
                test_data=test_data,
                is_homogeneous=True,
                limit=limit
            )
            results["homogeneous"][config_name] = result
        
        # 3. Run heterogeneous multi-agent configuration
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
        
        # 4. Analyze by error type
        print("\n" + "="*60)
        print("Analyzing Performance by Error Type")
        print("="*60)
        
        error_type_results = {}
        for error_type in self.ERROR_TYPES:
            error_type_results[error_type] = {}
            
            # Extract from single agent results
            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                scores = []
                for resp in results["single_agent"][model_key]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    error_type_results[error_type][f"single_{model_key}"] = sum(scores) / len(scores)
            
            # Extract from homogeneous results
            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                config_name = f"{model_key}-only"
                scores = []
                for resp in results["homogeneous"][config_name]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    error_type_results[error_type][f"homogeneous_{model_key}"] = sum(scores) / len(scores)
            
            # Extract from heterogeneous results
            scores = []
            for resp in results["heterogeneous"]["best"]["responses"]:
                if resp.get("error_type") == error_type:
                    scores.append(resp.get("faithfulness", 0))
            if scores:
                error_type_results[error_type]["heterogeneous_best"] = sum(scores) / len(scores)
        
        results["error_type_analysis"] = error_type_results
        
        # 5. HITL Analysis
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
        
        # Heterogeneous HITL
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
        
        # 6. Latency analysis
        print("\n" + "="*60)
        print("Latency Analysis")
        print("="*60)
        
        latency_results = {}
        
        # Single-agent latencies
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            latency_results[f"single_{model_key}"] = {
                "mean_latency": results["single_agent"][model_key]["summary"]["mean_latency"],
            }
        
        # Homogeneous latencies
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            config_name = f"{model_key}-only"
            latency_results[f"homogeneous_{model_key}"] = {
                "mean_latency": results["homogeneous"][config_name]["summary"]["mean_latency"],
            }
        
        # Heterogeneous latency
        latency_results["heterogeneous"] = {
            "mean_latency": results["heterogeneous"]["best"]["summary"]["mean_latency"],
        }
        
        results["latency_analysis"] = latency_results
        
        # 7. Save all results
        output_file = Path(output_dir) / f"evaluation_results_{mode}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary table
        self._print_summary_table(results, mode)
        
    def _print_summary_table(self, results: Dict, mode: str) -> None:
        """Print summary results table."""
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
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation of multi-agent framework")
    parser.add_argument("--data", type=str, required=True, help="Path to test data JSON/JSONL")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--mode", type=str, choices=["quick", "dev", "full"], default="full",
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
