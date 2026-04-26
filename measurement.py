from statistics import mean, stdev
import time

from llama_cpp import Llama


MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf"
PROMPT = "You are a helpful assistant."
MAX_TOKENS = 128
RUNS = 10
WARMUP_RUNS = 3


def measure_once(model: Llama, prompt: str, max_tokens: int) -> dict:
    """Measure one run using stream mode to capture wall-clock TTFT."""

    # KV cache 초기화: 동일 프롬프트 반복 시 캐시 재사용 방지
    model.reset()

    start = time.perf_counter()
    first_token_time = None
    generated_tokens = 0
    total_chunks = 0
    empty_chunks = 0
    full_text = []

    for chunk in model.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        stream=True,
    ):
        # text 먼저 추출
        text = chunk["choices"][0].get("text", "")

        total_chunks += 1
        if not text:
            empty_chunks += 1

        # TTFT: 실제 텍스트가 있는 첫 chunk 도착 시점 (업계 표준)
        if text and first_token_time is None:
            first_token_time = time.perf_counter()

        # 토큰 카운팅: 빈 chunk 포함 (chunk 1개 = 토큰 1개)
        # generated_tokens += 1
        if text:
            generated_tokens += 1

        # text 수집
        if text:
            full_text.append(text)

    end = time.perf_counter()

    total_s = end - start

    if first_token_time is None:
        ttft_s = total_s
        gen_time_s = 0.0
    else:
        ttft_s = first_token_time - start
        gen_time_s = end - first_token_time

    throughput = generated_tokens / total_s if total_s > 0 else 0.0
    generation_throughput = generated_tokens / gen_time_s if gen_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft_s * 1000.0,
        "total_time_s": total_s,
        "generated_tokens": generated_tokens,
        "total_chunks": total_chunks,
        "empty_chunks": empty_chunks,
        "throughput_tps": throughput,
        "generation_throughput_tps": generation_throughput,
        "response_preview": "".join(full_text)[:120].replace("\n", " "),
    }


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def main() -> None:
    print("Loading model...")
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_threads=6,
    )

    print(f"\nPrompt: {PROMPT}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {RUNS}\n")

    for i in range(WARMUP_RUNS):
        _ = measure_once(model, PROMPT, MAX_TOKENS)
        print(f"Warm-up {i + 1}/{WARMUP_RUNS} done")

    print()

    results = []
    for i in range(RUNS):
        result = measure_once(model, PROMPT, MAX_TOKENS)
        results.append(result)
        empty_ratio = result['empty_chunks'] / result['total_chunks'] if result['total_chunks'] > 0 else 0.0
        print(
            f"Run {i + 1:02d}: "
            f"TTFT={result['ttft_ms']:.2f} ms, "
            f"Throughput={result['throughput_tps']:.2f} t/s, "
            f"GenThroughput={result['generation_throughput_tps']:.2f} t/s, "
            f"Tokens={result['generated_tokens']}, "
            f"EmptyChunks={result['empty_chunks']}/{result['total_chunks']} ({empty_ratio:.1%})"
        )

    ttft_values = [r["ttft_ms"] for r in results]
    throughput_values = [r["throughput_tps"] for r in results]
    gen_throughput_values = [r["generation_throughput_tps"] for r in results]
    total_chunks_values = [r["total_chunks"] for r in results]
    empty_chunks_values = [r["empty_chunks"] for r in results]

    ttft_mean, ttft_std = summarize(ttft_values)
    thr_mean, thr_std = summarize(throughput_values)
    gen_thr_mean, gen_thr_std = summarize(gen_throughput_values)
    avg_total_chunks = mean(total_chunks_values)
    avg_empty_chunks = mean(empty_chunks_values)

    print(f"\n===== Summary ({RUNS} runs) =====")
    print(f"TTFT                : {ttft_mean:.2f} ± {ttft_std:.2f} ms")
    print(f"Throughput          : {thr_mean:.2f} ± {thr_std:.2f} tokens/s  (prefill 포함)")
    print(f"Generation Throughput: {gen_thr_mean:.2f} ± {gen_thr_std:.2f} tokens/s  (decode only)")
    print(f"Avg Empty Chunks    : {avg_empty_chunks:.1f} / {avg_total_chunks:.1f} ({avg_empty_chunks/avg_total_chunks:.1%})")


if __name__ == "__main__":
    main()