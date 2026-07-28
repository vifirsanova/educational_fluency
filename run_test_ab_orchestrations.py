#!/usr/bin/env python3
"""
Compare Declared Configurations.

Tests only the configurations reported in the paper:
1. Single-agent baselines (3 models)
2. Homogeneous multi-agent (3 models, Architecture A)
3. Heterogeneous multi-agent (Architecture A and B)

Results match the manuscript tables exactly.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from src.agents.orchestrator_b import OrchestratorB


class ConfigTester:
    """
    Test only the configurations reported in the paper.
    
    Configurations:
    - Single-agent: qwen, gpt-oss, deepseek (direct generation)
    - Homogeneous Arch A: qwen-only, gpt-oss-only, deepseek-only
    - Heterogeneous Arch A: deepseek(G) + qwen(V) + gpt-oss(E)
    - Heterogeneous Arch B: deepseek(G) + qwen(E) + gpt-oss(V)
    """
    
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
            "timestamp": None,
            "mode": None,
            "total_samples": 0,
            "single_agent": {},
            "homogeneous_arch_a": {},
            "heterogeneous_arch_a": {},
            "heterogeneous_arch_b": {},
            "error_type_analysis": {},
            "hitl_analysis": {},
        }
        
        os.makedirs("test_results", exist_ok=True)
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
    
    def _std_dev(self, values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def run_single_agent(self, model_key: str, test_data: List[Dict]) -> Dict[str, Any]:
        """Run single-agent baseline."""
        model_name = self.MODELS[model_key]
        print(f"\n  Single-agent: {model_key}")
        
        results = {
            "model": model_name,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "latencies": [],
        }
        
        orchestrator = Orchestrator(self.config_path)
        client = orchestrator.client
        folder_id = os.environ.get("YANDEX_FOLDER_ID", "")
        full_model_uri = f"gpt://{folder_id}/{model_name}" if folder_id else model_name
        
        verifier = Verifier(self.config_path)
        verifier.set_model(model_name)
        
        for item in tqdm(test_data, desc=f"  {model_key}", leave=False):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
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
                    "error_type": error_type,
                    "faithfulness": faithfulness,
                    "verifier_result": verifier_result,
                })
                
            except Exception as e:
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "model": model_name,
                    "model_key": model_key,
                    "error_type": error_type,
                    "faithfulness": 0.0,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
            
            time.sleep(0.05)
        
        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        
        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": len(results["responses"]),
            "mean_latency": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
        }
        
        return results
    
    def run_homogeneous_arch_a(self, model_key: str, test_data: List[Dict]) -> Dict[str, Any]:
        """Run homogeneous multi-agent Architecture A (post-editing)."""
        model_name = self.MODELS[model_key]
        print(f"\n  Homogeneous Arch A: {model_key}-only")
        
        orchestrator = Orchestrator(self.config_path)
        orchestrator.set_model(model_name)
        
        results = {
            "model": model_name,
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "hitl_triggers": defaultdict(int),
            "latencies": [],
        }
        
        for item in tqdm(test_data, desc=f"  {model_key}-only", leave=False):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
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
                verifier_result = result.get("verifier", {})
                faithfulness = verifier_result.get("faithfulness", 0.0)
                
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1
                
                if status == "hitl_required":
                    reason = result.get("reason", "unknown")
                    results["hitl_triggers"][reason] += 1
                
                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": answer,
                    "status": status,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "model": model_name,
                    "model_key": model_key,
                    "gatekeeper": result.get("gatekeeper", {}),
                    "verifier": verifier_result,
                    "editor": result.get("editor", {}),
                })
                
            except Exception as e:
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "status": "error",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "model": model_name,
                    "model_key": model_key,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
            
            time.sleep(0.05)
        
        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        
        total = len(results["responses"])
        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": total,
            "mean_latency": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
            "hitl_triggers": dict(results["hitl_triggers"]),
            "total_hitl_percentage": sum(results["hitl_triggers"].values()) / total * 100 if total > 0 else 0,
        }
        
        return results
    
    def run_heterogeneous_arch_a(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run heterogeneous multi-agent Architecture A (post-editing)."""
        print(f"\n  Heterogeneous Arch A: deepseek(G) + qwen(V) + gpt-oss(E)")
        
        orchestrator = Orchestrator(self.config_path)
        orchestrator.set_gatekeeper_model(self.MODELS["deepseek"])
        orchestrator.set_verifier_model(self.MODELS["qwen"])
        orchestrator.set_editor_model(self.MODELS["gpt-oss"])
        
        results = {
            "model_assignment": {"gatekeeper": "deepseek", "verifier": "qwen", "editor": "gpt-oss"},
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "hitl_triggers": defaultdict(int),
            "latencies": [],
        }
        
        for item in tqdm(test_data, desc="  Hetero Arch A", leave=False):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
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
                verifier_result = result.get("verifier", {})
                faithfulness = verifier_result.get("faithfulness", 0.0)
                
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1
                
                if status == "hitl_required":
                    reason = result.get("reason", "unknown")
                    results["hitl_triggers"][reason] += 1
                
                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": answer,
                    "status": status,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "gatekeeper": result.get("gatekeeper", {}),
                    "verifier": verifier_result,
                    "editor": result.get("editor", {}),
                })
                
            except Exception as e:
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "status": "error",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
            
            time.sleep(0.05)
        
        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        
        total = len(results["responses"])
        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": total,
            "mean_latency": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
            "hitl_triggers": dict(results["hitl_triggers"]),
            "total_hitl_percentage": sum(results["hitl_triggers"].values()) / total * 100 if total > 0 else 0,
        }
        
        return results
    
    def run_heterogeneous_arch_b(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run heterogeneous multi-agent Architecture B (pre-editing)."""
        print(f"\n  Heterogeneous Arch B: deepseek(G) + qwen(E) + gpt-oss(V)")
        
        orchestrator = OrchestratorB(self.config_path)
        orchestrator.set_gatekeeper_model(self.MODELS["deepseek"])
        orchestrator.set_verifier_model(self.MODELS["gpt-oss"])
        orchestrator.set_editor_model(self.MODELS["qwen"])
        
        results = {
            "model_assignment": {"gatekeeper": "deepseek", "verifier": "gpt-oss", "editor": "qwen"},
            "responses": [],
            "faithfulness_scores": [],
            "pass_count": 0,
            "hitl_triggers": defaultdict(int),
            "latencies": [],
        }
        
        for item in tqdm(test_data, desc="  Hetero Arch B", leave=False):
            query = self._extract_query(item)
            passages = self._extract_passages(item)
            error_type = item.get("error_type", "unknown")
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
                verifier_result = result.get("verifier", {})
                faithfulness = verifier_result.get("faithfulness", 0.0)
                
                if faithfulness >= self.FAITHFULNESS_THRESHOLD:
                    results["pass_count"] += 1
                
                if status == "hitl_required":
                    reason = result.get("reason", "unknown")
                    results["hitl_triggers"][reason] += 1
                
                results["faithfulness_scores"].append(faithfulness)
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": answer,
                    "status": status,
                    "faithfulness": faithfulness,
                    "error_type": error_type,
                    "gatekeeper": result.get("gatekeeper", {}),
                    "verifier": verifier_result,
                    "editor": result.get("editor", {}),
                })
                
            except Exception as e:
                results["responses"].append({
                    "original_id": original_id,
                    "query": query,
                    "answer": "",
                    "status": "error",
                    "faithfulness": 0.0,
                    "error_type": error_type,
                    "error": str(e),
                })
                results["faithfulness_scores"].append(0.0)
            
            time.sleep(0.05)
        
        mean_faithfulness = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
        pass_rate = results["pass_count"] / len(results["responses"]) if results["responses"] else 0
        
        total = len(results["responses"])
        results["summary"] = {
            "mean_faithfulness": mean_faithfulness,
            "std_faithfulness": self._std_dev(results["faithfulness_scores"]),
            "pass_rate": pass_rate,
            "total_samples": total,
            "mean_latency": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
            "hitl_triggers": dict(results["hitl_triggers"]),
            "total_hitl_percentage": sum(results["hitl_triggers"].values()) / total * 100 if total > 0 else 0,
        }
        
        return results
    
    def run_all(self, data_path: str, limit: int = None, output_dir: str = "test_results") -> None:
        """Run all declared configurations."""
        print("\n" + "="*70)
        print("COMPARING DECLARED CONFIGURATIONS")
        print("="*70)
        
        test_data = self.load_test_data(data_path, limit)
        print(f"Loaded {len(test_data)} test samples")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results["timestamp"] = timestamp
        self.results["mode"] = "full" if limit is None else f"sample_{limit}"
        self.results["total_samples"] = len(test_data)
        
        # 1. Single-agent baselines
        print("\n" + "-"*70)
        print("1. SINGLE-AGENT BASELINES")
        print("-"*70)
        
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            result = self.run_single_agent(model_key, test_data)
            self.results["single_agent"][model_key] = result
        
        # 2. Homogeneous Architecture A
        print("\n" + "-"*70)
        print("2. HOMOGENEOUS MULTI-AGENT (Architecture A)")
        print("-"*70)
        
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            result = self.run_homogeneous_arch_a(model_key, test_data)
            self.results["homogeneous_arch_a"][model_key] = result
        
        # 3. Heterogeneous Architecture A
        print("\n" + "-"*70)
        print("3. HETEROGENEOUS MULTI-AGENT (Architecture A)")
        print("-"*70)
        
        result = self.run_heterogeneous_arch_a(test_data)
        self.results["heterogeneous_arch_a"] = result
        
        # 4. Heterogeneous Architecture B
        print("\n" + "-"*70)
        print("4. HETEROGENEOUS MULTI-AGENT (Architecture B)")
        print("-"*70)
        
        result = self.run_heterogeneous_arch_b(test_data)
        self.results["heterogeneous_arch_b"] = result
        
        # 5. Error type analysis
        self._analyze_error_types()
        
        # 6. HITL analysis
        self._analyze_hitl()
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / f"results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        self._print_summary()
    
    def _analyze_error_types(self) -> None:
        """Analyze performance by error type."""
        error_results = {}
        
        for error_type in self.ERROR_TYPES:
            error_results[error_type] = {}
            
            # Single-agent best
            best_single = 0.0
            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                scores = []
                for resp in self.results["single_agent"][model_key]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    mean = sum(scores) / len(scores)
                    error_results[error_type][f"single_{model_key}"] = mean
                    if mean > best_single:
                        best_single = mean
            error_results[error_type]["single_best"] = best_single
            
            # Homogeneous Arch A best
            best_homo = 0.0
            for model_key in ["qwen", "gpt-oss", "deepseek"]:
                scores = []
                for resp in self.results["homogeneous_arch_a"][model_key]["responses"]:
                    if resp.get("error_type") == error_type:
                        scores.append(resp.get("faithfulness", 0))
                if scores:
                    mean = sum(scores) / len(scores)
                    error_results[error_type][f"homo_{model_key}"] = mean
                    if mean > best_homo:
                        best_homo = mean
            error_results[error_type]["homogeneous_best"] = best_homo
            
            # Heterogeneous Arch A
            scores = []
            for resp in self.results["heterogeneous_arch_a"]["responses"]:
                if resp.get("error_type") == error_type:
                    scores.append(resp.get("faithfulness", 0))
            if scores:
                error_results[error_type]["hetero_arch_a"] = sum(scores) / len(scores)
            
            # Heterogeneous Arch B
            scores = []
            for resp in self.results["heterogeneous_arch_b"]["responses"]:
                if resp.get("error_type") == error_type:
                    scores.append(resp.get("faithfulness", 0))
            if scores:
                error_results[error_type]["hetero_arch_b"] = sum(scores) / len(scores)
        
        self.results["error_type_analysis"] = error_results
    
    def _analyze_hitl(self) -> None:
        """Analyze HITL trigger rates."""
        hitl_results = {}
        
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            summary = self.results["homogeneous_arch_a"][model_key]["summary"]
            hitl_results[f"homo_{model_key}"] = {
                "low_confidence": summary["hitl_triggers"].get("low_confidence_below_0.5", 0),
                "medium_confidence": summary["hitl_triggers"].get("medium_confidence_repeated_query", 0),
                "low_faithfulness": summary["hitl_triggers"].get("low_faithfulness", 0),
                "excessive_removal": summary["hitl_triggers"].get("excessive_removal", 0),
                "total": summary["total_hitl_percentage"],
            }
        
        # Heterogeneous Arch A
        summary = self.results["heterogeneous_arch_a"]["summary"]
        hitl_results["hetero_arch_a"] = {
            "low_confidence": summary["hitl_triggers"].get("low_confidence_below_0.5", 0),
            "medium_confidence": summary["hitl_triggers"].get("medium_confidence_repeated_query", 0),
            "low_faithfulness": summary["hitl_triggers"].get("low_faithfulness", 0),
            "excessive_removal": summary["hitl_triggers"].get("excessive_removal", 0),
            "total": summary["total_hitl_percentage"],
        }
        
        # Heterogeneous Arch B
        summary = self.results["heterogeneous_arch_b"]["summary"]
        hitl_results["hetero_arch_b"] = {
            "low_confidence": summary["hitl_triggers"].get("low_confidence_below_0.5", 0),
            "medium_confidence": summary["hitl_triggers"].get("medium_confidence_repeated_query", 0),
            "low_faithfulness": summary["hitl_triggers"].get("low_faithfulness", 0),
            "excessive_removal": summary["hitl_triggers"].get("excessive_removal", 0),
            "total": summary["total_hitl_percentage"],
        }
        
        self.results["hitl_analysis"] = hitl_results
    
    def _print_summary(self) -> None:
        """Print summary table matching manuscript format."""
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        
        # Single-agent
        print("\nSingle-Agent Baselines:")
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            s = self.results["single_agent"][model_key]["summary"]
            print(f"  {model_key:12s}: mean={s['mean_faithfulness']:.3f}, pass_rate={s['pass_rate']*100:.1f}%")
        
        # Homogeneous Arch A
        print("\nHomogeneous Multi-Agent (Architecture A):")
        for model_key in ["qwen", "gpt-oss", "deepseek"]:
            s = self.results["homogeneous_arch_a"][model_key]["summary"]
            print(f"  {model_key:12s}: mean={s['mean_faithfulness']:.3f}, pass_rate={s['pass_rate']*100:.1f}%, HITL={s['total_hitl_percentage']:.1f}%")
        
        # Heterogeneous Arch A
        print("\nHeterogeneous Multi-Agent (Architecture A):")
        s = self.results["heterogeneous_arch_a"]["summary"]
        print(f"  deepseek(G)+qwen(V)+gpt-oss(E): mean={s['mean_faithfulness']:.3f}, pass_rate={s['pass_rate']*100:.1f}%, HITL={s['total_hitl_percentage']:.1f}%")
        
        # Heterogeneous Arch B
        print("\nHeterogeneous Multi-Agent (Architecture B):")
        s = self.results["heterogeneous_arch_b"]["summary"]
        print(f"  deepseek(G)+qwen(E)+gpt-oss(V): mean={s['mean_faithfulness']:.3f}, pass_rate={s['pass_rate']*100:.1f}%, HITL={s['total_hitl_percentage']:.1f}%")
        
        print("\n" + "="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare declared configurations")
    parser.add_argument("--data", type=str, required=True, help="Path to test data JSON/JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit (for testing)")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    tester = ConfigTester(args.config)
    tester.run_all(
        data_path=args.data,
        limit=args.limit,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
