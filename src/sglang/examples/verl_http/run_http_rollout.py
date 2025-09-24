#!/usr/bin/env python3

"""
HTTP Rollout Example for VerlHttpEngine

This script demonstrates how to set up a VerlHttpEngine and perform batch generation requests.
It uses predefined model and server configurations and focuses on testing batch generation.
The script also demonstrates how to use tokenizers and apply chat templates.

Usage: python run_http_rollout.py
"""

import argparse
import json
import random
import time
from typing import Dict, List, Optional, Union

import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer

from sglang.srt.entrypoints.verl_http_engine import VerlHttpEngine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import find_available_port

# Predefined model and server configurations
MODELS = [
    dict(
        model_path="Qwen/Qwen3-1.7B",
        tokenizer_path="Qwen/Qwen3-1.7B",
        enable_memory_saver=True,
        tp_size=2,
        host="localhost",
        port=40000,
    ),
    # Add more model configurations as needed
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="HTTP Rollout Example for VerlHttpEngine")
    parser.add_argument(
        "--model_index",
        type=int,
        default=0,
        help="Index of the model configuration to use from the MODELS list",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts in each batch",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Apply chat template to prompts",
    )
    parser.add_argument(
        "--num_rollout",
        type=int,
        default=1,
        help="Number of rollout per prompt",
    )
    args = parser.parse_args()
    
    # Validate model index
    if args.model_index < 0 or args.model_index >= len(MODELS):
        raise ValueError(f"Model index {args.model_index} is out of range. Available models: 0-{len(MODELS) - 1}")
        
    return args


def get_chat_prompts(batch_size: int) -> List[Dict]:
    """Generate a batch of chat prompts."""
    translation_pairs = [
        ("hello", "French"),
        ("goodbye", "Spanish"),
        ("thank you", "German"),
        ("good morning", "Italian"),
        ("how are you", "Japanese"),
        ("what is your name", "Chinese"),
        ("welcome", "Russian"),
        ("please", "Korean"),
        ("sorry", "Arabic"),
        ("I love you", "Portuguese"),
    ]
    
    # Create a batch of chat messages
    chat_prompts = []
    selected_pairs = random.sample(translation_pairs, min(batch_size, len(translation_pairs)))
    
    for i, (word, language) in enumerate(selected_pairs):
        messages = [
            {"role": "system", "content": "You are a helpful multilingual assistant. Respond concisely."},
            {"role": "user", "content": f"Translate '{word}' to {language}:"}
        ]
        chat_prompts.append(messages)
    
    # If we need more prompts, add some math problems
    if len(chat_prompts) < batch_size:
        math_templates = [
            "1+1=",
            "2+2=",
            "3+3=",
            "4+4=",
            "5+5=",
            "1*1=",
            "2*2=",
            "3*3=",
            "4*4=",
            "5*5=",
        ]
        
        num_math = batch_size - len(chat_prompts)
        math_prompts = random.sample(math_templates, min(num_math, len(math_templates)))
        
        for math_prompt in math_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Respond with only the numeric answer."},
                {"role": "user", "content": f"Calculate {math_prompt}"}
            ]
            chat_prompts.append(messages)
    
    return chat_prompts


def get_batch_prompts(batch_size: int) -> List[str]:
    """Generate a batch of regular text prompts (non-chat)."""
    translation_pairs = [
        ("hello", "French"),
        ("goodbye", "Spanish"),
        ("thank you", "German"),
        ("good morning", "Italian"),
        ("how are you", "Japanese"),
        ("what is your name", "Chinese"),
        ("welcome", "Russian"),
        ("please", "Korean"),
        ("sorry", "Arabic"),
        ("I love you", "Portuguese"),
    ]
    
    # Create a batch of translation prompts
    prompts = []
    selected_pairs = random.sample(translation_pairs, min(batch_size, len(translation_pairs)))
    
    for i, (word, language) in enumerate(selected_pairs):
        prompt = f"Translate '{word}' to {language}:"
        prompts.append(prompt)
    
    # If we need more prompts, add some math problems
    if len(prompts) < batch_size:
        math_templates = [
            "1+1=",
            "2+2=",
            "3+3=",
            "4+4=",
            "5+5=",
            "1*1=",
            "2*2=",
            "3*3=",
            "4*4=",
            "5*5=",
        ]
        
        num_math = batch_size - len(prompts)
        math_prompts = random.sample(math_templates, min(num_math, len(math_templates)))
        prompts.extend(math_prompts)
    
    return prompts


def apply_chat_template(tokenizer, chat_prompts: List[List[Dict]]) -> List[str]:
    """Apply the tokenizer's chat template to create formatted prompts."""
    formatted_prompts = []
    
    for messages in chat_prompts:
        try:
            # Use the tokenizer's chat template to format the messages
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        except Exception as e:
            print(f"Error applying chat template: {str(e)}")
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            formatted_prompts.append(prompt)
    
    return formatted_prompts


def run_batch_generation(
    engine: VerlHttpEngine, 
    batch_prompts: List[str], 
    max_new_tokens: int, 
    temperature: float,
    num_rollout: int,
) -> Dict:
    """Run batch generation using the direct Python API."""
    print(f"\n===== Running Batch Generation =====")
    print(f"Prompts: {batch_prompts}")
    
    # Sampling parameters
    sampling_params = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": 0.95,
        "n": num_rollout, # returned batch size is n x batch_size
    }
    
    # Generate responses
    start_time = time.time()
    response_list = engine.generate(
        prompt=batch_prompts,
        sampling_params=sampling_params,
        return_logprob=True,
    )
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Batch completed in {elapsed:.2f} seconds")
    # import ipdb; ipdb.set_trace()
    for i, (prompt, output) in enumerate(zip(batch_prompts, response_list)):
        print(f"Item {i}:")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {output['text']}")

    # Return batch statistics
    return {
        "elapsed_time": elapsed,
        "tokens_per_second": sum([response['meta_info']["completion_tokens"] for response in response_list]) / elapsed,
        "num_prompts": len(batch_prompts),
        "texts": [response["text"] for response in response_list],
        "output_token_logprobs": [response["meta_info"]["output_token_logprobs"] for response in response_list],
        "response": response_list,
    }


def main():
    """Main function to run the HTTP rollout with batch generation."""
    args = parse_arguments()
    
    # Get the selected model configuration
    model_config = MODELS[args.model_index].copy()
    
    # If port is specified as None or 0, find an available port
    if model_config.get("port", 0) in (None, 0):
        model_config["port"] = find_available_port(40000)
    
    print(f"Using model configuration: {json.dumps(model_config, indent=2)}")
    
    try:
        # Initialize the tokenizer
        tokenizer_path = model_config.get("tokenizer_path", model_config["model_path"])
        print(f"\n===== Initializing Tokenizer from {tokenizer_path} =====")
        
        tokenizer = get_tokenizer(
            tokenizer_path,
            trust_remote_code=True
        )
        
        # Initialize the VerlHttpEngine
        print(f"\n===== Initializing VerlHttpEngine =====")
        engine_kwargs = model_config.copy()
        engine = VerlHttpEngine(**engine_kwargs)

        # release and resume memory occupation
        print(f"\n===== Releasing Memory Occupation =====")
        engine.release_memory_occupation()
        
        import time
        time.sleep(4)
        print(f"\n===== Resuming Memory Occupation =====")
        engine.resume_memory_occupation()
        
        # Generate prompts based on whether we're using chat templates
        if args.use_chat_template:
            print(f"\n===== Using Chat Template: {model_config.get('chat_template', 'default')} =====")
            # Generate chat-style prompts
            chat_prompts = get_chat_prompts(args.batch_size)
            # Apply chat template
            batch_prompts = apply_chat_template(tokenizer, chat_prompts)
        else:
            # Generate regular text prompts
            batch_prompts = get_batch_prompts(args.batch_size)
        
        # Run batch generation
        batch_stats = run_batch_generation(
            engine=engine,
            batch_prompts=batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_rollout=args.num_rollout,
        )
        
        # Print performance statistics
        print("\n===== Performance Statistics =====")
        print(f"Total time: {batch_stats['elapsed_time']:.2f} seconds")
        print(f"Tokens per second: {batch_stats['tokens_per_second']:.2f}")
        print(f"Number of prompts: {batch_stats['num_prompts']}")
        
        engine.flush_cache()
        
    except Exception as e:
        print(f"Error during batch generation: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure engine is properly shut down
        print("\n===== Shutting down engine =====")
        if 'engine' in locals():
            engine.shutdown()
        print("Engine shutdown complete")


if __name__ == "__main__":
    main()
