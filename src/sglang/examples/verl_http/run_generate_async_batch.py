import asyncio
import aiohttp
import requests
import json
import time

class StopSignal:
    def __init__(self):
        pass

async def batch_generate_async(session, prompts, host="localhost", port=30000, callback=None):
    url = f"http://{host}:{port}/generate_async"
    
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
    print(f"Sending requests to {host}:{port}...")
    
    # Make the request within the session context
    async with session.post(url, json=payload) as response:
        async for line in response.content:
            if line == b"\n":
                continue
            try:
                data = json.loads(line.decode('utf-8').replace('data: ', ''))
                if data == '[DONE]':
                    break
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
                continue
            if callback:
                if isinstance(callback(data, host, port), StopSignal):
                    print(f"Stop signal received, breaking connection with {host}:{port}")
                    break
            else:
                print(f"callback is None, received data: {data}")

    print(f"\nBatch generation async for {host}:{port} took: {time.time() - start_time:.2f} seconds")

async def launch_generate_async():
    # Example prompts
    prompts = [
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a poem about a cat<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite an article about cars<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a poem about a dog<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a poem about a bird<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a poem about a fish<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of China?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Brazil?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of Argentina?<|im_end|>\n<|im_start|>assistant\n",
    ]

    hosts = ["localhost", "localhost"]
    ports = [30000, 30001]

    max_requests = 9
    finished_data = []
    def _callback(data, host, port):
        if len(finished_data) >= max_requests:
            return StopSignal()
        else:
            print(f"{len(finished_data)}/{max_requests} at {host}:{port}: Received data: {data}")
            finished_data.append(data)

    # split into num_hosts chunks, the last chunk may be smaller
    chunk_size = (len(prompts) - 1) // len(hosts) + 1 # ceil_divide
    chunks = [prompts[i*chunk_size:(i+1)*chunk_size] for i in range(len(hosts))]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (host, port) in enumerate(zip(hosts, ports)):
            chunk = chunks[i]
            # launch an asyncio task
            tasks.append(asyncio.create_task(batch_generate_async(session, chunk, host, port, _callback)))
            
        # wait for all tasks to finish inside the session context
        await asyncio.gather(*tasks)
        
    print(f"All tasks finished, {len(finished_data)} requests finished")
    for data in finished_data:
        print(f"Finished data: {data}")

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
    asyncio.run(launch_generate_async())
