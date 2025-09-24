import sglang
import asyncio
from transformers import AutoTokenizer
import torch
from typing import List
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput, ResumeMemoryOccupationReqInput, CloseSessionReqInput

BUFFER_PROMPTS = 4
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
    
    sample_n = 4
    
    sampling_params = {
        "temperature": 1.0, 
        "max_new_tokens": 1000,
        "n": sample_n,
    }
    
    # This list will store the complete generated text for each prompt
    results = [
        {
            "input_ids": input_ids,
            "complete_reqs": 0,
            "complete_tokens": 0,
            "responses": [None for _ in range(sample_n)]
        } for input_ids in input_ids_list 
    ]
    
    # counters
    total_generated_tokens = 0
    total_completion_tokens = 0 # tokens from finished requests, usable for training
    num_completion_prompts = 0
    
    print("--- Starting Stream Generation ---\n")
    
    async_gen = _engine.async_generate(
        input_ids=input_ids_list,
        sampling_params=sampling_params,
        stream=True,  # Set stream=True to enable streaming mode
        return_logprob=True,
    )

    # Track active request IDs to abort them if needed
    active_request_ids = set()
    
    try:
        # Use 'async for' to iterate over the streaming results
        async_gen_iter = await async_gen
        async for output in async_gen_iter:
            # The request ID ('rid') corresponds to the index in your input batch
            index = output["index"]
            prompt_idx = index // sample_n
            sample_idx = index % sample_n
            meta_info = output["meta_info"]
            rid = meta_info["id"]
            
            # Track active request IDs
            active_request_ids.add(rid)
            
            text = output["text"]
            completion_tokens = meta_info["completion_tokens"]
            total_generated_tokens += completion_tokens
            
            # the text actually include all the previous content
            results[prompt_idx]["responses"][sample_idx] = {
                "text": text,
                "meta_info": meta_info
            }
            
            # When a request finishes, remove it from active tracking
            if meta_info["finish_reason"]:
                active_request_ids.discard(rid)  # UNCOMMENT THIS LINE!
                results[prompt_idx]["complete_reqs"] += 1
                results[prompt_idx]["complete_tokens"] += completion_tokens
                
                if results[prompt_idx]["complete_reqs"] == sample_n:
                    total_completion_tokens += results[prompt_idx]["complete_tokens"]
                    num_completion_prompts += 1
            
            if num_completion_prompts >= BUFFER_PROMPTS:
                print(f"\nInterrupt because we have enough requests.")
                break
            if total_completion_tokens >= BUFFER_TOKENS:
                print(f"\nInterrupt because we have enough tokens for training.")
                break

    finally:
        # This block will run when the loop exits for any reason (break, return, or error)
        print("Exiting loop. Closing the generator to clean up resources.")
        # Asynchronous generators should be closed with aclose()
        # await async_gen_iter.aclose()  # Use await aclose() instead of close()
        async_gen.close()
    
    # Abort any remaining unfinished requests
    if active_request_ids:
        # print(f"Aborting {len(active_request_ids)} unfinished requests...")
        for rid in active_request_ids:
            _engine.tokenizer_manager.abort_request(rid)
        
        # # Give time for abort to be processed
        await asyncio.sleep(0.5)
    
    print("Flushing cache...")
    # Flush cache to clean up any remaining state
    await _engine.tokenizer_manager.flush_cache()
    
    print("Releasing memory occupation...")
    # Memory occupation management (if needed for your RL workflow)
    obj = ReleaseMemoryOccupationReqInput()
    await _engine.tokenizer_manager.release_memory_occupation(obj, None)

    print("Resuming memory occupation...")
    # Memory occupation management (if needed for your RL workflow)
    obj = ResumeMemoryOccupationReqInput()
    await _engine.tokenizer_manager.resume_memory_occupation(obj, None)

    # await asyncio.sleep(2)

    # return
    finished_requests = []
    unfinished_requests = []
    print("\n--- Interrupt Outputs ---")
    for i, result in enumerate(results):
        prompt = conversations[i][0]["content"]
        
        if results[i]["complete_reqs"] == sample_n:
            # finished prompts
            print(f"Prompt: {prompt} finish all {sample_n} requests")
            for s, response in enumerate(result["responses"]):
                print(f"[{s}] {response["text"][:700]}")
                finished_requests.append(
                    {
                        "prompt": prompt,
                        "response": response["text"],
                    }
                )
        else:
            print(f"Prompt: {prompt} finish {result["complete_reqs"]}/{sample_n} requests")
            for s, response in enumerate(result["responses"]):
                text = response["text"]
                response_ids = tokenizer.encode(text)
                meta_info = response["meta_info"]
                if meta_info["finish_reason"]:
                    # finished sample
                    response_ids += [151645] # add im_end
                input_ids = input_ids_list[i] + response_ids
                unfinished_requests.append(
                    {
                        "prompt": prompt,
                        "past_input_ids": input_ids_list[i],
                        "past_response": text,
                        "input_ids": input_ids
                    }
                )
            
    sampling_params["n"] = 1
    
    input_ids_list = [request["input_ids"] for request in unfinished_requests]
    # import ipdb; ipdb.set_trace()
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
        completion_tokens = result["meta_info"]["completion_tokens"]
        response = result["text"]
        full_response = past_response + response
        finished_requests.append(
            {
                "prompt": prompt,
                "response": full_response,
            }
        )
    
    # import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # Run the new asynchronous main function
    asyncio.run(stream_main())