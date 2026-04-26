#!/usr/bin/env python3
"""
TTFT + Tokens/sec Benchmark Script (Qwen3 + llama.cpp)
TTFT: Time To First Token (ms)
TPS: Tokens Per Second
"""
import subprocess
import re
import json
from statistics import mean, stdev
from datetime import datetime

def measure_performance(model_path, prompt, n_generate=128, runs=5):
    """Measure performance: TTFT and tokens/sec"""
    
    results = {
        "ttft_ms": [],
        "tps": [],
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"Benchmark Start: {n_generate} tokens, {runs} runs")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")
    
    for run in range(runs):
        cmd = [
            "./build/bin/llama-cli",
            "-m", model_path,
            "-n", str(n_generate),
            "-p", prompt,
            "-ngl", "33",
            "--no-display-prompt",
            "-t", "1"
        ]
        
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, 
                                            cwd="/home/jang/project/llama.cpp",
                                            timeout=300,
                                            input="")
        except subprocess.CalledProcessError as e:
            print(f"❌ Run {run+1} failed: {e}")
            continue
        except subprocess.TimeoutExpired:
            print(f"❌ Run {run+1} timeout (>5min)")
            continue
        
        # Parse prompt eval time (TTFT)
        prompt_match = re.search(r"prompt eval time\s*=\s*([\d.]+) ms / ([\d]+) tokens", output)
        if prompt_match:
            prompt_time = float(prompt_match.group(1))
            prompt_tokens = int(prompt_match.group(2))
            ttft_ms = prompt_time
            results["ttft_ms"].append(ttft_ms)
        else:
            print(f"⚠️  Run {run+1}: prompt eval time parse failed")
            continue
        
        # Parse decoding eval time (TPS)
        eval_match = re.search(r"eval time\s*=\s*([\d.]+) ms / ([\d]+) tokens", output)
        if eval_match:
            eval_time = float(eval_match.group(1))
            eval_tokens = int(eval_match.group(2))
            tps = (eval_tokens / eval_time) * 1000  # tokens/sec
            results["tps"].append(tps)
        else:
            print(f"⚠️  Run {run+1}: eval time parse failed")
            continue
        
        print(f"✓ Run {run+1:2d}: TTFT={ttft_ms:6.2f}ms | TPS={tps:6.2f} tokens/s")
    
    # Results summary
    print(f"\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    
    if results["ttft_ms"]:
        ttft_mean = mean(results["ttft_ms"])
        ttft_std = stdev(results["ttft_ms"]) if len(results["ttft_ms"]) > 1 else 0
        print(f"TTFT: {ttft_mean:.2f}ms ± {ttft_std:.2f}ms")
    
    if results["tps"]:
        tps_mean = mean(results["tps"])
        tps_std = stdev(results["tps"]) if len(results["tps"]) > 1 else 0
        print(f"TPS:  {tps_mean:.2f} tokens/sec ± {tps_std:.2f}")
    
    print(f"{'='*60}\n")
    
    return results


def benchmark_vs_context_length(model_path):
    """Measure TTFT variations by prompt length"""
    
    print(f"\n{'='*60}")
    print("TTFT Change by Prompt Length")
    print(f"{'='*60}\n")
    
    test_cases = [
        ("short (5 words)",      "Hello world how are you"),
        ("medium (50 words)",    "You are a helpful assistant. " * 5),
        ("long (150 words)",     "Once upon a time, there was a " * 10),
    ]
    
    all_results = []
    
    for name, prompt in test_cases:
        print(f"\nTest: {name}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        
        cmd = [
            "./build/bin/llama-cli",
            "-m", model_path,
            "-n", "128",
            "-p", prompt,
            "-ngl", "33",
            "--no-display-prompt"
        ]
        
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT,
                                            cwd="/home/jang/project/llama.cpp")
            
            # Parse TTFT
            match = re.search(r"prompt eval time\s*=\s*([\d.]+) ms / ([\d]+) tokens", output)
            if match:
                ttft = float(match.group(1))
                prompt_tokens = int(match.group(2))
                all_results.append((name, prompt_tokens, ttft))
                print(f"  TTFT: {ttft:.2f}ms ({prompt_tokens} tokens)")
            else:
                print(f"  ⚠️  Parse failed")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Results table
    if all_results:
        print(f"\n{'='*60}")
        print("Prompt Length vs TTFT")
        print(f"{'='*60}")
        for name, tokens, ttft in all_results:
            print(f"{name:15s}: {tokens:4d} tokens -> {ttft:6.2f}ms")
        print(f"{'='*60}\n")
    
    return all_results


if __name__ == "__main__":
    model_path = "models/Qwen3-0.6B-Q4_K_M.gguf"
    
    # 1. Basic performance measurement (128 tokens, 5 runs)
    results_128 = measure_performance(
        model_path,
        "You are a helpful assistant. ",
        n_generate=128,
        runs=5
    )
    
    # 2. Test various generation lengths
    print("\n" + "="*60)
    print("Performance by Generation Length")
    print("="*60)
    
    for n_tokens in [64, 128, 256]:
        results = measure_performance(
            model_path,
            "You are a helpful assistant. ",
            n_generate=n_tokens,
            runs=3
        )
    
    # 3. TTFT change by prompt length
    context_results = benchmark_vs_context_length(model_path)
    
    # 4. Save results to JSON
    summary = {
        "model": model_path,
        "baseline_128_tokens": {
            "ttft_ms": results_128["ttft_ms"],
            "tps": results_128["tps"]
        },
        "context_length_test": context_results
    }
    
    with open("/home/jang/project/llama.cpp/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Results saved: benchmark_results.json")
