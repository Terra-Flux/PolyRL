#!/bin/bash

# NOTE: example1 and example2 must be launched with openai compatible server
# Example1: Query the server to check the model is running
curl http://localhost:40000/v1/models
# example returns:
# {"object":"list","data":[{"id":"qwen2-7b","object":"model","created":1743715027,"owned_by":"vllm","root":"Qwen/Qwen2-7B-Instruct","parent":null,"max_model_len":10240,"permission":[{"id":"modelperm-fdb911cecda24d82a5d274006a31c578","object":"model_permission","created":1743715027,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}%   

# Example2: Query the server to complete chat with default system prompt
curl http://localhost:40000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2-7b",
        "prompt": "Explain large language models in simple terms",
        "max_tokens": 100,
        "temperature": 0.7
    }'

# Example3: Query the server to complete chat with custom system prompt, sample 5 responses
curl http://localhost:40000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-7b",
    "messages": [
        {"role": "system", "content": "You can only answer in Chinese."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    "n": 5,
    "max_tokens": 300,
    "temperature": 0.9
  }'

# Example4: Query the server to generate text, must be launched with naive vLLM api server
curl -X POST http://10.132.0.11:30001/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 100
    }'

# Example4: Query the server to generate text, must be launched with naive vLLM api server
curl -X POST http://10.132.0.11:30001/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "San Francisco is a",
        "max_tokens": 100
    }'

curl -X POST http://10.132.0.11:30000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "San Francisco is a",
        "max_tokens": 100
    }'