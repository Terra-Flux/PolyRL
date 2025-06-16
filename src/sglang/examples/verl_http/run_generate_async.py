import asyncio
import aiohttp
import requests
import json
import time

def batch_generate_async():
    url = "http://localhost:30000/generate_async"
    
    # Example prompts
    prompts = [
        "<|im_start|>user\nWrite a poem about a cat<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite an article about cars<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n",
    ]
    # Request payload
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1000
        },
        "stream": False
    }
    
    start_time = time.time()
    print("Sending requests...")
    response = requests.post(url, json=payload, stream=True)

    max_requests = 4
    n_requests = 0
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8').replace('data: ', ''))
                if data == '[DONE]':
                    break
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
                continue

            n_requests += 1

            # Get the last output which contains the complete response
            last_output = data['output'][-1]
            index = data["index"]
            prompt = prompts[index]
            print(f"\nPrompt: {prompt}")
            print(f"Response: {data}")

            if n_requests >= max_requests:
                # disconnect the client
                print("Reached required number of requests, disconnecting the client")
                break
    print(f"\nBatch generation async took: {time.time() - start_time:.2f} seconds")

def generate_sync():
    url = "http://localhost:30000/generate"
    
    prompts = [
        "<|im_start|>user\nWrite a poem about a cat<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite an article about cars<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n",
    ]
    
    print("Sending requests synchronously...")
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        payload = {
            "text": [prompt],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1000
            },
            "stream": False
        }
        response = requests.post(url, json=payload)
        print(f"\nPrompt {i}: {prompt}")
        print(f"Response: {response.json()}")
    
    print(f"\nSynchronous execution took: {time.time() - start_time:.2f} seconds")

async def process_prompt(url, prompt, index):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1000
        },
        "stream": False
    }
    
    start_time = time.time()
    print(f"Processing prompt {index} at {start_time}")
    
    # Use aiohttp instead of synchronous requests
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
    
    print(f"Received response for prompt {index} takes {time.time() - start_time:.2f} seconds")
    return index, prompt, result

async def generate_async():
    url = "http://localhost:30000/generate"
    
    prompts = [
        "<|im_start|>user\nWrite a poem about a cat<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite an article about cars<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n",
    ]
    
    print("Sending requests asynchronously...")
    start_time = time.time()
    
    # Create tasks for each prompt
    tasks = []
    for i, prompt in enumerate(prompts):
        task = asyncio.create_task(process_prompt(url, prompt, i))
        tasks.append(task)
    
    # Use as_completed to process results as they arrive
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task    
        # Print results
        index, prompt, response = result
        print(f"\nPrompt {index}: {prompt}")
        print(f"Response: {response}")
    
    print(f"\nAsynchronous execution took: {time.time() - start_time:.2f} seconds")

def batch_generate():
    url = "http://localhost:30000/generate"
    # Example prompts
    prompts = [
        "<|im_start|>user\nWrite a poem about a cat<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite an article about cars<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n",
    ]

    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1000
        },
        "stream": False
    }
    start_time = time.time()
    response = requests.post(url, json=payload)
    print(response.json())
    print(f"\nBatch generation took: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Run both versions to compare
    print("\n=== Warmup Batch Generation ===")
    batch_generate()

    # print("\n=== Batch Generation ===")
    # batch_generate()

    print("\n=== Batch Generation Async ===")
    batch_generate_async()

    # print("=== Synchronous Version ===")
    # generate_sync()
    
    print("\n=== Asynchronous Version ===")
    asyncio.run(generate_async())
