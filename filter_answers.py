#!/usr/bin/env python3
"""
Analyze checkpoint results, filtering out generator failures.

This script loads a checkpoint JSON file from the evaluation run,
filters out responses where the generator failed (empty answer with error),
and provides accurate statistics for each configuration.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load the checkpoint JSON file."""
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_generator_failed(response: Dict[str, Any]) -> bool:
    """
    Determine if a response failed due to generator error.
    
    A generator failure is identified by:
    - Empty answer
    - Error field present
    - Status is "error"
    - Faithfulness is 0.0
    """
    # Check if the response has an error field
    if response.get('error'):
        return True
    
    # Check if status is error
    if response.get('status') == 'error':
        return True
    
    # Check if answer is empty and faithfulness is 0
    if not response.get('answer', '').strip() and response.get('faithfulness', 0.0) == 0.0:
        # Check if there's a verifier result that indicates empty response
        verifier = response.get('verifier', {})
        if verifier.get('total_claims', 0) == 0 and verifier.get('reason', ''):
            return True
    
    return False


def filter_responses(responses: List[Dict[str, Any]]) -> tuple:
    """
    Filter responses into successful and failed categories.
    
    Returns:
        (successful_responses, failed_responses)
    """
    successful = []
    failed = []
    
    for resp in responses:
        if is_generator_failed(resp):
            failed.append(resp)
        else:
            successful.append(resp)
    
    return successful, failed


def calculate_stats(responses: List[Dict[str, Any]], config_name: str = "") -> Dict[str, Any]:
    """Calculate statistics for a list of responses."""
    if not responses:
        return {
            "config_name": config_name,
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "mean_faithfulness": 0.0,
            "std_faithfulness": 0.0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "hitl_count": 0,
            "hitl_rate": 0.0,
            "faithfulness_scores": [],
            "error_type_breakdown": {},
            "status_breakdown": {},
        }
    
    faithfulness_scores = [r.get('faithfulness', 0.0) for r in responses]
    mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    
    # Calculate standard deviation
    if len(faithfulness_scores) > 1:
        variance = sum((x - mean_faithfulness) ** 2 for x in faithfulness_scores) / len(faithfulness_scores)
        std_faithfulness = variance ** 0.5
    else:
        std_faithfulness = 0.0
    
    # Count passes (faithfulness >= threshold)
    threshold = 0.70  # FAITHFULNESS_THRESHOLD from config
    pass_count = sum(1 for r in responses if r.get('faithfulness', 0.0) >= threshold)
    pass_rate = pass_count / len(responses) if responses else 0.0
    
    # Count HITL
    hitl_count = sum(1 for r in responses if r.get('status') == 'hitl_required')
    hitl_rate = hitl_count / len(responses) if responses else 0.0
    
    # Error type breakdown
    error_type_breakdown = defaultdict(lambda: {"count": 0, "mean_faithfulness": 0.0, "scores": []})
    for r in responses:
        error_type = r.get('error_type', 'unknown')
        error_type_breakdown[error_type]["count"] += 1
        error_type_breakdown[error_type]["scores"].append(r.get('faithfulness', 0.0))
    
    # Calculate mean for each error type
    for et, data in error_type_breakdown.items():
        scores = data["scores"]
        data["mean_faithfulness"] = sum(scores) / len(scores) if scores else 0.0
        del data["scores"]  # Clean up
    
    # Status breakdown
    status_breakdown = defaultdict(int)
    for r in responses:
        status = r.get('status', 'unknown')
        status_breakdown[status] += 1
    
    return {
        "config_name": config_name,
        "total_samples": len(responses),
        "successful_samples": len(responses),
        "failed_samples": 0,
        "mean_faithfulness": mean_faithfulness,
        "std_faithfulness": std_faithfulness,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "hitl_count": hitl_count,
        "hitl_rate": hitl_rate,
        "faithfulness_scores": faithfulness_scores,
        "error_type_breakdown": dict(error_type_breakdown),
        "status_breakdown": dict(status_breakdown),
    }


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Analyze the checkpoint and return filtered statistics."""
    data = load_checkpoint(checkpoint_path)
    
    results = {
        "metadata": {
            "timestamp": data.get("timestamp", "unknown"),
            "mode": data.get("mode", "unknown"),
            "total_samples": data.get("total_samples", 0),
            "config": data.get("config", {}),
        },
        "configurations": {},
        "summary": {
            "total_responses": 0,
            "total_successful": 0,
            "total_failed": 0,
            "overall_success_rate": 0.0,
        }
    }
    
    configs = {}
    total_responses = 0
    total_successful = 0
    total_failed = 0
    
    # Process single-agent baselines
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        if model_key in data.get("single_agent", {}):
            config_data = data["single_agent"][model_key]
            responses = config_data.get("responses", [])
            successful, failed = filter_responses(responses)
            
            config_name = f"single_{model_key}"
            stats = calculate_stats(successful, config_name)
            stats["failed_samples"] = len(failed)
            stats["total_samples"] = len(responses)
            
            configs[config_name] = stats
            total_responses += len(responses)
            total_successful += len(successful)
            total_failed += len(failed)
    
    # Process homogeneous multi-agent
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        config_name = f"{model_key}-only"
        if config_name in data.get("homogeneous", {}):
            config_data = data["homogeneous"][config_name]
            responses = config_data.get("responses", [])
            successful, failed = filter_responses(responses)
            
            stats = calculate_stats(successful, config_name)
            stats["failed_samples"] = len(failed)
            stats["total_samples"] = len(responses)
            
            configs[config_name] = stats
            total_responses += len(responses)
            total_successful += len(successful)
            total_failed += len(failed)
    
    # Process heterogeneous
    if "best" in data.get("heterogeneous", {}):
        config_data = data["heterogeneous"]["best"]
        responses = config_data.get("responses", [])
        successful, failed = filter_responses(responses)
        
        stats = calculate_stats(successful, "heterogeneous")
        stats["failed_samples"] = len(failed)
        stats["total_samples"] = len(responses)
        
        configs["heterogeneous"] = stats
        total_responses += len(responses)
        total_successful += len(successful)
        total_failed += len(failed)
    
    results["configurations"] = configs
    results["summary"] = {
        "total_responses": total_responses,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "overall_success_rate": total_successful / total_responses if total_responses > 0 else 0.0,
    }
    
    return results


def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print the analysis in a readable format."""
    print("\n" + "="*80)
    print("CHECKPOINT ANALYSIS - FILTERED RESULTS")
    print("="*80)
    
    metadata = analysis["metadata"]
    print(f"\nMetadata:")
    print(f"  Mode: {metadata['mode']}")
    print(f"  Total samples in dataset: {metadata['total_samples']}")
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    summary = analysis["summary"]
    print(f"  Total responses: {summary['total_responses']}")
    print(f"  Successful (generator worked): {summary['total_successful']}")
    print(f"  Failed (generator error): {summary['total_failed']}")
    print(f"  Overall success rate: {summary['overall_success_rate']*100:.1f}%")
    
    print("\n" + "-"*80)
    print("CONFIGURATION RESULTS (Successful Responses Only)")
    print("-"*80)
    
    # Sort configs by mean faithfulness
    configs = analysis["configurations"]
    sorted_configs = sorted(
        configs.items(),
        key=lambda x: x[1]["mean_faithfulness"],
        reverse=True
    )
    
    print(f"\n{'Config':<25} {'Samples':<8} {'Failed':<8} {'Mean Faith':<12} {'Pass Rate':<10} {'HITL Rate':<10}")
    print("-"*80)
    
    for config_name, stats in sorted_configs:
        print(
            f"{config_name:<25} "
            f"{stats['total_samples']:<8} "
            f"{stats['failed_samples']:<8} "
            f"{stats['mean_faithfulness']:<12.3f} "
            f"{stats['pass_rate']*100:<10.1f}% "
            f"{stats['hitl_rate']*100:<10.1f}%"
        )
    
    print("\n" + "-"*80)
    print("ERROR TYPE BREAKDOWN (per configuration)")
    print("-"*80)
    
    for config_name, stats in sorted_configs:
        if stats["error_type_breakdown"] and stats["total_samples"] > 0:
            print(f"\n  {config_name}:")
            for et, et_data in sorted(
                stats["error_type_breakdown"].items(),
                key=lambda x: x[1]["mean_faithfulness"],
                reverse=True
            ):
                print(
                    f"    {et:<35} "
                    f"n={et_data['count']:<4} "
                    f"mean={et_data['mean_faithfulness']:.3f}"
                )
    
    print("\n" + "-"*80)
    print("STATUS BREAKDOWN (per configuration)")
    print("-"*80)
    
    for config_name, stats in sorted_configs:
        if stats["status_breakdown"] and stats["total_samples"] > 0:
            print(f"\n  {config_name}:")
            for status, count in sorted(stats["status_breakdown"].items(), key=lambda x: x[1], reverse=True):
                pct = count / stats["total_samples"] * 100
                print(f"    {status}: {count} ({pct:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze checkpoint results")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint JSON file")
    parser.add_argument("--output", type=str, help="Output JSON file for results (optional)")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    analysis = analyze_checkpoint(args.checkpoint)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")
    
    print_analysis(analysis)


if __name__ == "__main__":
    main()
