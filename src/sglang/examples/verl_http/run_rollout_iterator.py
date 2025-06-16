import time
from sglang.srt.coordinator.rollout_iterator import RolloutIterator, RolloutConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example usage
if __name__ == "__main__":

    num_local_prompts = 10
    chunk_size = 10
    n_samples = 8   
    out_of_order = False

    # Example configuration
    config = RolloutConfig(
        hosts=["localhost:30002", "localhost:30003"],
        max_tokens=1000,
        temperature=0.9,
        n_samples=n_samples,
        return_tokens=False,
        out_of_order=out_of_order,
        chunk_size=chunk_size,
    )
    
    # Create and start iterator
    rollout_iterator = RolloutIterator(config)
    rollout_iterator.start_rollout_manager()
    
    # Example prompts
    prompts = [
        "<|im_start|>user\nHow do quantum computers differ from classical computers?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the fermentation process in making sourdough bread<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the ethical implications of autonomous vehicles?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the cultural significance of tea ceremonies in Japan<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow does neuroplasticity affect learning in adults?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a short story about a time traveler who can only go forward in time<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat causes the aurora borealis phenomenon?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how cryptocurrency mining impacts the environment<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCompare and contrast impressionism and expressionism in art<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the most promising technologies for carbon capture?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of photosynthesis in desert plants<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow has the concept of privacy evolved in the digital age?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the mechanics of black holes in simple terms<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the cultural differences in breakfast foods around the world?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do bees communicate with each other?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat makes a programming language Turing complete?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of making traditional kimchi<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the psychological effects of color in marketing?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how a violin produces sound<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the significance of dreams in different cultures?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do coral reefs form and why are they important?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the origins of chess and how has it evolved?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the concept of zero-knowledge proofs in cryptography<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do migratory birds navigate across continents?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the differences between various coffee brewing methods?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of glass blowing and its history<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow does the human memory system work?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the philosophical implications of artificial intelligence?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how vaccines train the immune system<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat causes the northern lights and where can they be seen?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do different writing systems around the world represent language?<|im_end|>\n<|im_start|>assistant\n",
        # "<|im_start|>user\nWhat is the relationship between mathematics and music?<|im_end|>\n<|im_start|>assistant\n", 
    ]
    
    # Send prompts
    rollout_iterator.send_prompts(prompts)

    def make_dummy_result(index: int):
        return {"index": index,
                "text": f"This is dummy result {index}.",
                "meta_info": {"completion_tokens": 10}}

    # make some dummy local results
    if config.out_of_order: 
        rollout_iterator.add_local_results(
            [make_dummy_result(i) for i in range(num_local_prompts*n_samples)],
        )
    else:
        rollout_iterator.add_local_results(
            [[make_dummy_result(i+j*n_samples) for i in range(n_samples)] for j in range(num_local_prompts)],
        )
    
    start_time = time.monotonic()
    num_finished_requests = 0
    # Process results
    try:
        for chunk, duration in rollout_iterator:
            print(f"Received chunk in {duration:.2f} seconds:")
            for item in chunk:
                # import ipdb; ipdb.set_trace()
                if isinstance(item, list):  # In-order mode
                    for result in item:
                        prompt_idx = result['index'] // config.n_samples
                        sample_idx = result['index'] % config.n_samples
                        print(f"  Response {prompt_idx}-{sample_idx}: {result['text'][:30]}... ({result['meta_info']['completion_tokens']} tokens)")
                        num_finished_requests += 1
                else:  # Out-of-order mode
                    prompt_idx = item['index'] // config.n_samples
                    sample_idx = item['index'] % config.n_samples
                    print(f"  Response {prompt_idx}-{sample_idx}: {item['text'][:30]}... ({item['meta_info']['completion_tokens']} tokens)")
                    num_finished_requests += 1
    finally:
        rollout_iterator.close()

    duration = time.monotonic() - start_time
    logger.info(f"Total duration: {duration:.2f} seconds")
    logger.info(f"Total requests: {num_finished_requests}")
    logger.info(f"Total requests per second: {num_finished_requests / duration:.2f}")