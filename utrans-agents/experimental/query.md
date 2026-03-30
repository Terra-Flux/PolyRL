## Query SGLang for test

### example

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": [
      "What is the capital of France?",
      "Who wrote the play Hamlet?"
    ],
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 64
    }
  }'

curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "I love this product!"
  }'

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://0.0.0.0:40000"
      }'

curl -X GET http://0.0.0.0:30000/workers
{"workers":[{"id":"21b4b096-0925-4ede-a5b6-6abc470de2f6","url":"http://0.0.0.0:40000","model_id":"Qwen/Qwen3-8B","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"HTTP","metadata":{"disaggregation_mode":"null","architectures":"[\"Qwen3ForCausalLM\"]","tp_size":"4","model_path":"Qwen/Qwen3-8B","served_model_name":"Qwen/Qwen3-8B","load_balance_method":"round_robin","tokenizer_path":"Qwen/Qwen3-8B","dp_size":"1","model_type":"qwen3"},"disable_health_check":false},{"id":"6d3f9aae-e0bc-43a9-ba3b-d5581bf32ab8","url":"http://10.202.15.229:30000","model_id":"Qwen/Qwen3-8B","priority":50,"cost":1.0,"worker_type":"regular","is_healthy":true,"load":0,"connection_mode":"HTTP","metadata":{"tokenizer_path":"Qwen/Qwen3-8B","tp_size":"4","served_model_name":"Qwen/Qwen3-8B","model_path":"Qwen/Qwen3-8B","dp_size":"1"},"disable_health_check":false}],"total":2,"stats":{"prefill_count":0,"decode_count":0,"regular_count":2}}
  
curl http://localhost:30000/get_server_info
```