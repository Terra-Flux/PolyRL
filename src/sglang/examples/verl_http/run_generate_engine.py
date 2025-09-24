import sglang
import asyncio
from transformers import AutoTokenizer
import torch
from typing import List

BUFFER_REQ = 4
BUFFER_TOKENS = 1000

# Keep other imports as they are

async def stream_main():
    """
    Asynchronous function to initialize the engine and handle streaming generation.
    """
    model_name = "Qwen/Qwen3-1.7B"
    _engine = sglang.Engine(
        model_path=model_name,
        dtype=torch.bfloat16,
        mem_fraction_static=0.6,
        enable_memory_saver=True,
        base_gpu_id=0,
        gpu_id_step=1,
        tp_size=2,
        node_rank=0,
        trust_remote_code=True,
        port=40000,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for '{model_name}': {e}")
        return

    conversations = [
        [{"role": "user", "content": "Write a long, thoughtful story about a star that falls to Earth."}],
        [{"role": "user", "content": "What are the three primary colors?"}],
        [{"role": "user", "content": "Explain the concept of recursion in one paragraph."}],
        [{"role": "user", "content": "What is hiphop music"}],
        [{"role": "user", "content": "Today is my birthday"}],
    ]
    
    try:
        input_ids_list = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return
    
    sampling_params = {"temperature": 0.0, "max_new_tokens": 1000}
    
    # This list will store the complete generated text for each prompt
    results = [
        {"input_ids": input_ids} for input_ids in input_ids_list 
    ]
    
    # counters
    total_generated_tokens = 0
    total_completion_tokens = 0 # tokens from finished requests, usable for training
    num_completion_requests = 0
    
    print("--- Starting Stream Generation ---\n")
    
    async_gen = _engine.async_generate(
        input_ids=input_ids_list,
        sampling_params=sampling_params,
        stream=True,  # Set stream=True to enable streaming mode
        return_logprob=True,
    )

    try:
        # Use 'async for' to iterate over the streaming results
        async for output in await async_gen:
            # The request ID ('rid') corresponds to the index in your input batch
            index = output["index"]
            meta_info = output["meta_info"]
            rid = meta_info["id"]
            text = output["text"]
            completion_tokens = meta_info["completion_tokens"] # generated tokens
            total_generated_tokens += completion_tokens
            
            # the text actually include all the previous content
            results[index]["text"] = text
            results[index]["meta_info"] = meta_info
            
            # Print the live output. Using flush=True ensures it appears immediately.
            # This will interleave outputs from different prompts.
            # print(f"\rRequest {rid} Output: {generated_texts[rid]}", end="", flush=True)
            if meta_info["finish_reason"]:
                num_completion_requests += 1
                total_completion_tokens += completion_tokens
                print(f"\nRequest {rid} Finished.\n")
            
            if num_completion_requests >= BUFFER_REQ:
                print(f"\nInterrupt because we have enough requests.")
                break
            if total_completion_tokens >= BUFFER_TOKENS:
                print(f"\nInterrupt because we have enough tokens for training.")
                break

    finally:
        # This block will run when the loop exits for any reason (break, return, or error)
        print("Exiting loop. Closing the generator to clean up resources.")
        # Asynchronous generators should be closed with aclose()
        async_gen.close()

    unfinished_requests = []
    print("\n--- Interrupt Outputs ---")
    for i, result in enumerate(results):
        prompt = conversations[i][0]["content"]
        finish_reason = result["meta_info"]["finish_reason"]
        completion_tokens = result["meta_info"]["completion_tokens"]
        response = result["text"]
        print(f"Prompt {i}: {prompt}")
        print(f"Finish reason {finish_reason}")
        print(f"Response {response[700:]}...({completion_tokens} tokens)")
        # if finish_reason == None:
        if True:
            encode_tokens = tokenizer.encode(response)
            if finish_reason:
                encode_tokens += [151645] # attach the end token so that it will stop immediately in the next submission
            input_ids = input_ids_list[i] + tokenizer.encode(response)
            if "output_token_logprobs" in result["meta_info"]:
                tokens = [logprob_token[1] for logprob_token in result["meta_info"]["output_token_logprobs"]]
                assert len(tokens) == len(encode_tokens), f"{len(tokens)=} != {len(encode_tokens)}!"
                # decode_response = tokenizer.decode(tokens)
                # assert response == decode_response, f"response is different from the decoded one!"
            unfinished_requests.append(
                {
                    "prompt": prompt,
                    "past_input_ids": input_ids_list[i],
                    "past_response": response,
                    "input_ids": input_ids
                }
            )
    
    input_ids_list = [request["input_ids"] for request in unfinished_requests]
    outputs = await _engine.async_generate(
        input_ids=input_ids_list,
        sampling_params=sampling_params,
        stream=False,  # No need for stream
        return_logprob=True,
    )
            
    print("\n--- Final Outputs ---")
    for i, result in enumerate(outputs):
        prompt = unfinished_requests[i]["prompt"]
        past_response = unfinished_requests[i]["past_response"]
        finish_reason = result["meta_info"]["finish_reason"]
        completion_tokens = result["meta_info"]["completion_tokens"]
        response = result["text"]
        encode_tokens = tokenizer.encode(response)
        full_response = past_response + response
        print(f"Prompt {i}: {prompt}")
        print(f"Finish reason {finish_reason}")
        print(f"Response {full_response[:700]}")
        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # Run the new asynchronous main function
    asyncio.run(stream_main())