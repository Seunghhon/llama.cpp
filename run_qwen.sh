#!/bin/bash
./build/bin/llama-cli -m models/Qwen3-0.6B-Q4_K_M.gguf -p "You are a helpful assistant" -cnv -t 6

# -p "You are a coding expert"
# -p "You are a Korean tutor"
# -p "답변은 항상 한국어로 해줘"