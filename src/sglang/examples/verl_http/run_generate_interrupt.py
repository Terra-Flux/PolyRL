import time
import aiohttp
import asyncio
from transformers import AutoTokenizer
import json

async def call_server(prompt, sampling_params, url):
    """
    Calls the generation endpoint. Returns partial text on cancellation.
    Correctly handles servers that stream the full text in each chunk.
    """
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "stream": True,
    }

    text = ""
    meta_info = None
    timeout = aiohttp.ClientTimeout(total=3000)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{url.rstrip('/')}/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"\nError: Server returned status code {response.status}")
                    print(error_text)
                    return None, None

                async for resp_chunk in response.content:
                    resp_chunk = resp_chunk.decode('utf-8').strip()
                    if resp_chunk.startswith("data:"):
                        data_str = resp_chunk[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            meta_info = data.get("meta_info", {})
                            # Per user feedback, the new text chunk is the full text so far.
                            text = data.get("text", "")
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from line: {data_str}")
                            continue
                
                # Successful completion
                return text, meta_info

    except asyncio.CancelledError:
        print(f"\nTask for prompt '{prompt[:70]}...' was cancelled. Returning partial text.")
        # On cancellation, return the text accumulated so far.
        return text, meta_info
    except Exception as e:
        print(f"An unexpected error occurred for prompt '{prompt[:70]}...': {e}")
        return None, None


async def request_level_stream_generate(model_name: str, total_timeout: int = 10):
    """
    Generates text with a total timeout and retries with appended partial text.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for '{model_name}': {e}")
        return

    conversations = [
        [{"role": "user", "content": "Write a long, thoughtful story about a star that falls to Earth."}],
        [{"role": "user", "content": "What are the three primary colors?"}],
        [{"role": "user", "content": "Explain the concept of recursion in one paragraph."}]
    ]

    try:
        initial_prompts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return

    sampling_params = {"temperature": 0.0, "max_new_tokens": 1000}
    url = "http://localhost:40000"
    
    prompts_to_process = set(initial_prompts)
    successful_results = {}
    overall_start_time = time.time()

    # Master loop: continues until all original prompts have a successful result.
    while len(successful_results) < len(initial_prompts):
        print(f"\n---\n🔄 Starting a new batch for {len(prompts_to_process)} remaining prompts. ---\n")
        
        tasks = [asyncio.create_task(call_server(prompt, sampling_params, url)) for prompt in prompts_to_process]
        task_to_prompt = {task: prompt for task, prompt in zip(tasks, prompts_to_process)}
        pending_tasks = set(tasks)
        
        batch_had_timeout = False

        # Inner loop: processes tasks in the current batch.
        while pending_tasks:
            elapsed_since_start = time.time() - overall_start_time
            if elapsed_since_start > total_timeout:
                print(f"\n⏱️ Overall timeout of {total_timeout}s exceeded. Terminating {len(pending_tasks)} tasks to resubmit.")
                batch_had_timeout = True

                # Cancel all currently pending tasks
                for task in pending_tasks:
                    task.cancel()
                
                # Wait for cancellation to complete and get partial results
                results_from_cancelled = await asyncio.gather(*pending_tasks, return_exceptions=True)
                
                # Prepare the prompts for the next round
                new_prompts_for_next_round = set()
                for task, result in zip(pending_tasks, results_from_cancelled):
                    # Find the prompt that was originally associated with this task
                    original_prompt = task_to_prompt[task]
                    if not original_prompt: continue

                    if isinstance(result, tuple) and result[0] is not None:
                        partial_text, meta_info = result
                        if partial_text:
                            print(f"Finish reason {meta_info["finish_reason"]}")
                            new_prompt = original_prompt + partial_text
                            print(f"Resubmitting prompt with appended text: '{new_prompt[:100]}...'")
                            new_prompts_for_next_round.add(new_prompt)
                        else:
                            print(f"Error: no return text")
                            new_prompts_for_next_round.add(original_prompt)
                    else:
                        new_prompts_for_next_round.add(original_prompt)

                prompts_to_process = new_prompts_for_next_round
                print("Waiting 2 seconds before resubmitting...")
                await asyncio.sleep(2)
                # Break the inner while loop to start a new batch
                overall_start_time = time.time()
                break

            # Normal operation: wait for a task to complete
            done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            for completed_task in done:
                original_prompt = task_to_prompt[completed_task]
                try:
                    result = completed_task.result()
                    if result and result[0] is not None:
                        text, meta_info = result
                        # Only consider it "successful" if the server confirms it finished
                        if meta_info and meta_info.get("finish_reason") is not None:
                            print(f"--- ✅ Result Finished (Total Time: {time.time() - overall_start_time:.2f}s) ---")
                            print(f"Original Prompt Hint: \"{original_prompt[:70]}...\"")
                            print(f"Finish reason: {meta_info.get('finish_reason')}")
                            print(f"📝 Generated Text:\n{text.strip()[:200]}...\n")
                            # Store result and mark original prompt as completed
                            successful_results[original_prompt] = result

                except Exception as e:
                    print(f"Task for prompt \"{original_prompt[:70]}...\" failed with an exception: {e}\n")

            pending_tasks = pending
        
        # If the batch ended due to timeout, the outer loop will restart it.
        # If it ended because all tasks finished, we need to check if any failed and need retrying.
        if not batch_had_timeout:
            # Re-evaluate which prompts still need processing for the next master loop iteration
            completed_prompts = set(successful_results.keys())
            all_submitted_prompts = set(task_to_prompt.values())
            failed_prompts = all_submitted_prompts - completed_prompts
            prompts_to_process = failed_prompts


    print(f"\n🎉 All tasks completed successfully in {time.time() - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    model_to_use = "Qwen/Qwen2-1.5B-Instruct"
    print(f"=== Streaming Generation with '{model_to_use}' Chat Template (Timeout & Retry) ===")
    asyncio.run(request_level_stream_generate(model_name=model_to_use, total_timeout=1))