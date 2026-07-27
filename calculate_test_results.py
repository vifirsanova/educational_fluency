#!/usr/bin/env python3
"""
Calculate evaluation statistics from checkpoint or results files.

Usage:
    python calc_stats.py evaluation_results/checkpoint_dev_*.json
    python calc_stats.py evaluation_results/evaluation_results_*.json

Outputs:
    - Mean faithfulness per configuration
    - Pass rate at tau_faith = 0.70
    - HITL trigger rates
    - Performance by error type
    - Summary table matching manuscript format
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calc_single_agent_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for single-agent baselines."""
    stats = {}
    
    for model_key, data in results.get("single_agent", {}).items():
        summary = data.get("summary", {})
        stats[model_key] = {
            "mean_faithfulness": summary.get("mean_faithfulness", 0.0),
            "std_faithfulness": summary.get("std_faithfulness", 0.0),
            "pass_rate": summary.get("pass_rate", 0.0) * 100,
            "total_samples": summary.get("total_samples", 0),
            "pass_count": data.get("pass_count", 0),
            "status": "completed",
        }
    
    return stats


def calc_homogeneous_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for homogeneous multi-agent configurations."""
    stats = {}
    
    for config_name, data in results.get("homogeneous", {}).items():
        summary = data.get("summary", {})
        model_key = config_name.replace("-only", "")
        stats[model_key] = {
            "mean_faithfulness": summary.get("mean_faithfulness", 0.0),
            "std_faithfulness": summary.get("std_faithfulness", 0.0),
            "pass_rate": summary.get("pass_rate", 0.0) * 100,
            "total_hitl": summary.get("total_hitl_percentage", 0.0),
            "total_samples": summary.get("total_samples", 0),
            "pass_count": data.get("pass_count", 0),
            "hitl_triggers": summary.get("hitl_triggers", {}),
            "status": "completed",
        }
    
    return stats


def calc_heterogeneous_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for heterogeneous multi-agent configuration."""
    data = results.get("heterogeneous", {}).get("best", {})
    
    if not data:
        return {"status": "not_started"}
    
    summary = data.get("summary", {})
    
    return {
        "mean_faithfulness": summary.get("mean_faithfulness", 0.0),
        "std_faithfulness": summary.get("std_faithfulness", 0.0),
        "pass_rate": summary.get("pass_rate", 0.0) * 100,
        "total_hitl": summary.get("total_hitl_percentage", 0.0),
        "total_samples": summary.get("total_samples", 0),
        "pass_count": data.get("pass_count", 0),
        "hitl_triggers": summary.get("hitl_triggers", {}),
        "status": "completed",
    }


def calc_error_type_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance by error type."""
    error_stats = {}
    
    error_analysis = results.get("error_type_analysis", {})
    
    if not error_analysis:
        return {"status": "not_available"}
    
    for error_type, configs in error_analysis.items():
        error_stats[error_type] = {}
        
        for key, value in configs.items():
            if isinstance(value, (int, float)):
                error_stats[error_type][key] = value
    
    return error_stats


def calc_hitl_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate HITL trigger statistics."""
    hitl_analysis = results.get("hitl_analysis", {})
    
    if not hitl_analysis:
        return {"status": "not_available"}
    
    stats = {}
    for config, data in hitl_analysis.items():
        stats[config] = {
            "low_confidence": data.get("low_confidence", 0),
            "medium_confidence_repeated": data.get("medium_confidence_repeated", 0),
            "low_faithfulness": data.get("low_faithfulness", 0),
            "excessive_removal": data.get("excessive_removal", 0),
            "total": data.get("total", 0),
            "total_percentage": data.get("total_percentage", 0.0),
        }
    
    return stats


def calc_latency_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate latency statistics."""
    latency_analysis = results.get("latency_analysis", {})
    
    if not latency_analysis:
        return {"status": "not_available"}
    
    stats = {}
    for config, data in latency_analysis.items():
        stats[config] = data.get("mean_latency", 0.0)
    
    return stats


def print_summary_table(single_stats: Dict, homo_stats: Dict, hetero_stats: Dict):
    """Print summary table in manuscript format."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    print("\nSingle-Agent Baselines:")
    has_single = False
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        if model_key in single_stats:
            has_single = True
            s = single_stats[model_key]
            print(f"  {model_key:12s}: mean={s['mean_faithfulness']:.3f}, "
                  f"pass_rate={s['pass_rate']:.1f}%, "
                  f"samples={s['total_samples']}")
    if not has_single:
        print("  (not yet run)")
    
    print("\nHomogeneous Multi-Agent:")
    has_homo = False
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        if model_key in homo_stats:
            has_homo = True
            s = homo_stats[model_key]
            print(f"  {model_key:12s}: mean={s['mean_faithfulness']:.3f}, "
                  f"pass_rate={s['pass_rate']:.1f}%, "
                  f"HITL={s['total_hitl']:.1f}%, "
                  f"samples={s['total_samples']}")
    if not has_homo:
        print("  (not yet run)")
    
    print("\nHeterogeneous Multi-Agent:")
    if hetero_stats and hetero_stats.get("status") != "not_started":
        print(f"  best: mean={hetero_stats['mean_faithfulness']:.3f}, "
              f"pass_rate={hetero_stats['pass_rate']:.1f}%, "
              f"HITL={hetero_stats['total_hitl']:.1f}%, "
              f"samples={hetero_stats['total_samples']}")
    else:
        print("  (not yet run)")


def print_error_type_table(error_stats: Dict) -> None:
    """Print error type table."""
    print("\n" + "="*80)
    print("PERFORMANCE BY ERROR TYPE")
    print("="*80)
    
    if error_stats.get("status") == "not_available":
        print("  (not yet available - need more data)")
        return
    
    print(f"\n{'Error Type':<25} {'Single-best':<12} {'Homogeneous-best':<12} {'Heterogeneous':<12}")
    print("-" * 70)
    
    has_data = False
    for error_type in [
        "Entity Replacement", "Numerical Distortion", "Negation Flip",
        "Temporal Confusion", "Causal Reversal", "Plausible Fabrication",
        "Oversimplification", "Citation Hallucination"
    ]:
        if error_type in error_stats:
            e = error_stats[error_type]
            single = e.get("single_best", 0.0) if isinstance(e.get("single_best"), (int, float)) else 0.0
            homo = e.get("homogeneous_best", 0.0) if isinstance(e.get("homogeneous_best"), (int, float)) else 0.0
            hetero = e.get("heterogeneous_best", 0.0) if isinstance(e.get("heterogeneous_best"), (int, float)) else 0.0
            
            # Only print if we have actual data (not all zeros and status exists)
            if any([single, homo, hetero]) or isinstance(e.get("single_best"), (int, float)):
                has_data = True
                print(f"{error_type:<25} {single:<12.3f} {homo:<12.3f} {hetero:<12.3f}")
    
    if not has_data:
        print("  (not yet available - need more data)")


def print_hitl_table(hitl_stats: Dict) -> None:
    """Print HITL trigger table."""
    print("\n" + "="*80)
    print("HITL TRIGGER RATES")
    print("="*80)
    
    if hitl_stats.get("status") == "not_available":
        print("  (not yet available - need HITL data)")
        return
    
    configs = ["qwen", "gpt-oss", "deepseek", "heterogeneous"]
    available_configs = [c for c in configs if c in hitl_stats]
    
    if not available_configs:
        print("  (no HITL data yet)")
        return
    
    print(f"\n{'Trigger case':<35}", end="")
    for config in available_configs:
        print(f" {config:<12}", end="")
    print()
    print("-" * (35 + 13 * len(available_configs)))
    
    triggers = [
        ("low_confidence", "Conf < 0.5 (answers)"),
        ("medium_confidence_repeated", "Conf 0.5-0.75 (reviews)"),
        ("low_faithfulness", "Faithfulness < 0.70 (adjudicates)"),
        ("excessive_removal", "Editor removal > 50% (checks)"),
    ]
    
    for trigger_key, trigger_label in triggers:
        row = [trigger_label]
        for config in available_configs:
            count = hitl_stats[config].get(trigger_key, 0)
            total = hitl_stats[config].get("total", 1)
            pct = (count / total * 100) if total > 0 else 0
            row.append(f"{pct:.1f}%")
        
        print(f"{row[0]:<35} " + " ".join([f"{x:<12}" for x in row[1:]]))
    
    # Total HITL row
    print("-" * (35 + 13 * len(available_configs)))
    row = ["Total HITL"]
    for config in available_configs:
        pct = hitl_stats[config].get("total_percentage", 0.0)
        row.append(f"{pct:.1f}%")
    print(f"{row[0]:<35} " + " ".join([f"{x:<12}" for x in row[1:]]))


def print_latency_table(latency_stats: Dict) -> None:
    """Print latency table."""
    print("\n" + "="*80)
    print("LATENCY ANALYSIS")
    print("="*80)
    
    if latency_stats.get("status") == "not_available":
        print("  (not yet available - need more data)")
        return
    
    print(f"\n{'Configuration':<20} {'Mean Latency (s)':<15}")
    print("-" * 40)
    
    has_data = False
    
    # Single-agent
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        key = f"single_{model_key}"
        if key in latency_stats:
            has_data = True
            print(f"single-{model_key:<14} {latency_stats[key]:.2f}")
    
    # Homogeneous
    for model_key in ["qwen", "gpt-oss", "deepseek"]:
        key = f"homogeneous_{model_key}"
        if key in latency_stats:
            has_data = True
            print(f"homogeneous-{model_key:<9} {latency_stats[key]:.2f}")
    
    # Heterogeneous
    if "heterogeneous" in latency_stats:
        has_data = True
        print(f"heterogeneous{' ':<12} {latency_stats['heterogeneous']:.2f}")
    
    if not has_data:
        print("  (not yet available - need more data)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_stats.py <checkpoint_file.json>")
        print("Example: python calc_stats.py evaluation_results/checkpoint_dev_*.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"Loading results from: {filepath}")
    results = load_results(filepath)
    
    # Show what's available
    single_count = len(results.get("single_agent", {}))
    homo_count = len(results.get("homogeneous", {}))
    has_hetero = "heterogeneous" in results and "best" in results["heterogeneous"]
    
    print(f"\nProgress: {single_count}/3 single-agent, {homo_count}/3 homogeneous, hetero={'yes' if has_hetero else 'no'}")
    
    # Calculate all stats
    single_stats = calc_single_agent_stats(results)
    homo_stats = calc_homogeneous_stats(results)
    hetero_stats = calc_heterogeneous_stats(results)
    error_stats = calc_error_type_stats(results)
    hitl_stats = calc_hitl_stats(results)
    latency_stats = calc_latency_stats(results)
    
    # Print all tables
    print_summary_table(single_stats, homo_stats, hetero_stats)
    print_error_type_table(error_stats)
    print_hitl_table(hitl_stats)
    print_latency_table(latency_stats)
    
    # Save full stats to file
    output_dir = Path(filepath).parent
    output_file = output_dir / f"stats_{Path(filepath).stem}.json"
    
    full_stats = {
        "source_file": filepath,
        "progress": {
            "single_agent_completed": single_count,
            "homogeneous_completed": homo_count,
            "heterogeneous_completed": 1 if has_hetero else 0,
        },
        "single_agent": single_stats,
        "homogeneous": homo_stats,
        "heterogeneous": hetero_stats,
        "error_types": error_stats,
        "hitl": hitl_stats,
        "latency": latency_stats,
    }
    
    with open(output_file, "w") as f:
        json.dump(full_stats, f, indent=2)
    
    print(f"\nFull stats saved to: {output_file}")


if __name__ == "__main__":
    main()
