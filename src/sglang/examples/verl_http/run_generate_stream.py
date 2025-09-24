import requests
import json
import time
import aiohttp
import asyncio
from transformers import AutoTokenizer

async def call_server(prompt, sampling_params, url):
    """
    Calls the generation endpoint with streaming and returns the full text and metadata.
    """
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "stream": True,
    }
    
    timeout = aiohttp.ClientTimeout(total=3000)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(f"{url.rstrip('/')}/generate", json=payload) as response:
                if response.status == 200:
                    text = ""
                    meta_info = None
                    
                    async for resp_chunk in response.content:
                        resp_chunk = resp_chunk.decode('utf-8').strip()
                        if resp_chunk.startswith("data:"):
                            data_str = resp_chunk[len("data:"):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                meta_info = data.get("meta_info", {})
                                text_chunk = data.get("text", "")
                                text = text_chunk
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON from line: {data_str}")
                                continue
                                
                    return text, meta_info
                else:
                    error_text = await response.text()
                    print(f"\nError: Server returned status code {response.status}")
                    print(error_text)
                    return None, None
                    
        except asyncio.TimeoutError as e:
            print(f"HTTP request timed out after 3000s: {str(e)}")
            raise
        except aiohttp.ClientConnectorError as e:
            print(f"HTTP connection failed: {str(e)}")
            raise


async def request_level_stream_generate(model_name: str):
    """
    Generates text from the sglang server, processing each request's
    result as soon as it is completed.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for '{model_name}': {e}")
        return

    conversations = [
        [{"role": "user", "content": "Write a long, thoughtful story about a star that falls to Earth."}], # A long task
        [{"role": "user", "content": "What are the three primary colors?"}], # A very short task
        [{"role": "user", "content": "Explain the concept of recursion in one paragraph."}] # A medium task
    ]

    try:
        prompts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return
    
    sampling_params = {
        "temperature": 0.6, 
        "max_new_tokens": 1000
    }
    url = "http://localhost:40000"

    # Create tasks and a lookup dictionary to map them back to their prompts
    tasks = [asyncio.create_task(call_server(prompt, sampling_params, url)) for prompt in prompts]
    task_to_prompt = {task: prompt for task, prompt in zip(tasks, prompts)}
    
    # The set of tasks we are currently waiting on
    pending_tasks = set(tasks)
    
    start_time = time.time()

    while pending_tasks:
        # Wait for the next task to complete
        done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

        for completed_task in done:
            # Retrieve the original prompt using our lookup dictionary
            original_prompt = task_to_prompt[completed_task]
            
            print(f"--- ✅ Result Finished (Total Time: {time.time() - start_time:.2f}s) ---")
            
            try:
                result = completed_task.result()
                if result:
                    text, meta_info = result
                    print(f"Original Prompt Hint: \"{original_prompt}...\"")
                    print(f"Finish reason {meta_info["finish_reason"]}")
                    print(f"📝 Generated Text:\n{text.strip()[:200]}...\n")
                else:
                    print(f"Request failed for prompt: \"{original_prompt}...\"\n")
            except Exception as e:
                # This catches exceptions that occurred inside the task
                print(f"Task for prompt \"{original_prompt}...\" failed with an exception: {e}\n")

        # Update the set of pending tasks for the next loop iteration
        pending_tasks = pending

    print(f"🎉 All tasks completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    model_to_use = "Qwen/Qwen3-1.7B"
    
    print(f"=== Streaming Generation with '{model_to_use}' Chat Template (Processing as completed) ===")
    
    asyncio.run(request_level_stream_generate(model_to_use))