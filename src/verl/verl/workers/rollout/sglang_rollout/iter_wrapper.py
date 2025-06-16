# implement a wrapper for the sglang rollout disaggregated mode
# it will submit all the requests to the sglang rollout, and wait for the responses in original order
# it should returns an iterator
# everytime the iterator is called, it should return a microbatch of generated sequences
# note that sample_n may > 1, so a microbatch may contain multiple samples of the same prompt, 
# e.g. microbatch size = 4, sample_n = 2, then the microbatch should contain 4 samples of 2 prompts
# the output format should be align with the output of the sglang rollout (dataproto)
# start from a simple case where we will wait for response in original order,
# once we gather enough responses, we will return a microbatch of generated sequences

# try to reuse the class functions and data structures, don't create new ones

import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Iterator, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
from verl.workers.rollout.sglang_rollout.sglang_rollout import _pre_process_inputs, _post_process_outputs, SGLangRollout

logger = logging.getLogger(__name__)

class SGLangRolloutIteratorWrapper(SGLangRollout):
    """
    Iterator wrapper for SGLang rollout in disaggregated mode.
    Submits requests to rollout manager and yields microbatches as responses arrive.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_generate_tasks(self, prompts: DataProto, idx_list: List[int], image_list: List[dict], **kwargs) -> [asyncio.Task]:
        # polyrl-dev
        # create the requests to the rollout manager
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,  # if validate, already repeat in ray_trainer
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print(f"{self.sampling_params=}")
            assert self._tp_rank == 0, "only tp rank 0 will submit the requests"
            # polyrl-dev
            rollout_mgr_url = self.config.rollout_manager.endpoint
            assert rollout_mgr_url is not None, "rollout_manager.endpoint must be set"

            async def call_manager(single_input_ids, single_image_data):
                payload = {
                    "input_ids": single_input_ids,
                    "sampling_params": self.sampling_params,
                    "return_logprob": True,
                }
                if single_image_data is not None:
                    payload["image_data"] = single_image_data
                # Set a longer timeout for the HTTP client
                timeout = aiohttp.ClientTimeout(total=3000)  # 3000 seconds
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.post(f"{rollout_mgr_url.rstrip('/')}/generate", json=payload) as resp:
                            return await resp.json()
                    except asyncio.TimeoutError as e:
                        logger.error(f"HTTP request timed out after 3000s: {str(e)}")
                        raise
                    except Exception as e:
                        logger.error(f"HTTP request failed: {str(e)}")
                        raise

            # Create coroutines for each request
            tasks = [call_manager(idx_list[i], image_list[i]) for i in range(len(idx_list))]

            return tasks

    def _microbatch_iterator(self, prompts: DataProto, microbatch_size: int) -> Iterator[DataProto]:
        # polyrl-dev
        # 1. Get meta data required to reconstruct the microbatch results in the postprocess
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # sample_n
        sample_n = self.sampling_params.get("n", 1)

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None),
                    }
                )
        else:
            sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # polyrl-dev
        # create uid for each prompt
        non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        # Ensure token IDs are lists or numpy arrays
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        self.start_time = time.perf_counter() # polyrl-dev
        # 2. Create the tasks
        tasks = self._create_generate_tasks(prompts, idx_list, image_list)

        # 3. Try to get the current running loop, if none exists create a new one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 4. Create the async iterator and submit the tasks
        
        async def _async_iterator():
            # Convert coroutines to actual tasks if needed
            actual_tasks = []
            task_to_idx = {}
            try:
                for task in tasks:
                    try:
                        if asyncio.iscoroutine(task):
                            actual_tasks.append(asyncio.create_task(task))
                        elif isinstance(task, asyncio.Task):
                            actual_tasks.append(task)
                        else:
                            # Assume it's a coroutine function call
                            actual_tasks.append(asyncio.create_task(task))
                    except Exception as e:
                        logger.error(f"Failed to create task: {str(e)}")
                        continue

                # NOTE: because tasks return OOO, we need to map the task to the original index
                task_to_idx = {task: i for i, task in enumerate(actual_tasks)}
                logger.info(f"Created {len(actual_tasks)} tasks for processing")
            except Exception as e:
                logger.error(f"Failed during task initialization: {str(e)}")
                return
            
            completed_responses = []
            completed_indices = []
            pending_tasks = set(actual_tasks)
            
            while pending_tasks:
                try:
                    logger.info(f"Waiting for tasks to complete. Pending tasks: {len(pending_tasks)}")
                    # Wait for at least one task to complete
                    try:
                        done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=3000)
                        pending_tasks = pending
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for tasks to complete")
                        # Check for any failed tasks
                        for task in pending_tasks:
                            if task.done() and task.exception():
                                logger.error(f"Task failed during timeout: {task.exception()}")
                        continue
                    except Exception as e:
                        logger.error(f"Error during task wait: {str(e)}")
                        continue
                    
                    # Collect completed responses
                    for task in done:
                        try:
                            logger.info(f"Task {task_to_idx[task]} completed, awaiting response")
                            try:
                                response = await task
                                logger.info(f"Got response for task {task_to_idx[task]}, response type: {type(response)}")
                            except Exception as e:
                                import traceback
                                error_details = f"Task {task_to_idx[task]} failed with error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                                if task.done() and task.exception():
                                    error_details += f"\nTask exception: {task.exception()}"
                                logger.error(error_details)
                                continue
                            
                            # polyrl-dev
                            # unroll the response on sample_n > 1
                            try:
                                if sample_n > 1:
                                    assert len(response) == sample_n, f"number of responses should be equal to sample_n, got {len(response)} != {sample_n}"
                                    for i in range(sample_n):
                                        completed_responses.append(response[i])
                                        # repeat the index for each sample as they share the same prompt
                                        completed_indices.append(task_to_idx[task])
                                    logger.info(f"Unrolled {sample_n} samples for task {task_to_idx[task]}")
                                else:
                                    completed_responses.append(response)
                                    completed_indices.append(task_to_idx[task])
                                    logger.info(f"Added single response for task {task_to_idx[task]}")
                            except Exception as e:
                                logger.error(f"Failed to process response for task {task_to_idx[task]}: {str(e)}")
                                continue
                            
                            # If we have enough responses for a microbatch, yield it
                            if len(completed_responses) >= microbatch_size:
                                try:
                                    end_time = time.perf_counter() # polyrl-dev
                                    logger.info(f"Collected {len(completed_responses)} responses, processing microbatch")
                                    # Take microbatch_size responses
                                    batch_responses = completed_responses[:microbatch_size]
                                    completed_responses = completed_responses[microbatch_size:]
                                    batch_indices = completed_indices[:microbatch_size]
                                    completed_indices = completed_indices[microbatch_size:]
                                    
                                    # Process the batch and yield
                                    logger.info(f"Processing microbatch with indices: {batch_indices}")
                                    processed_batch = self.postprocess_microbatch(
                                        batch_indices, batch_responses, idx, attention_mask, position_ids, eos_token_id, non_tensor_batch
                                    )
                                    if processed_batch is not None:
                                        logger.info("Successfully processed microbatch, yielding")
                                        yield processed_batch
                                    else:
                                        logger.warning("Processed batch was None")
                                except Exception as e:
                                    logger.error(f"Failed to process microbatch: {str(e)}")
                                    continue
                        except Exception as e:
                            import traceback
                            error_msg = f"Task {task_to_idx[task]} failed with error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                            logger.error(error_msg)
                            continue
                except Exception as e:
                    logger.error(f"Error in main processing loop: {str(e)}")
                    continue
            
            # Yield any remaining responses as a final microbatch
            if completed_responses:
                try:
                    logger.info(f"Processing final microbatch with {len(completed_responses)} remaining responses")
                    processed_batch = self.postprocess_microbatch(
                        completed_indices, completed_responses, idx, attention_mask, position_ids, eos_token_id, non_tensor_batch
                    )
                    if processed_batch is not None:
                        logger.info("Successfully processed final microbatch, yielding")
                        yield processed_batch
                    else:
                        logger.warning("Final processed batch was None")
                except Exception as e:
                    logger.error(f"Failed to process final microbatch: {str(e)}")
        
        # 5. Start the async iterator and consume the results
        async_gen = _async_iterator()
        
        try:
            while True:
                try:
                    result = loop.run_until_complete(async_gen.__anext__())
                    # yield the result once received a microbatch of responses
                    yield result
                except StopAsyncIteration:
                    break
        finally:
            # Clean up the async generator
            try:
                loop.run_until_complete(async_gen.aclose())
            except:
                pass

    def postprocess_microbatch(self, indices: List[int], outputs: List[dict], 
                               idx: torch.Tensor, attention_mask: torch.Tensor, 
                               position_ids: torch.Tensor, eos_token_id: int,
                               non_tensor_batch: dict) -> DataProto:
        """
        Process a list of response dictionaries into a DataProto object.
        Aligned with standard SGLang rollout processing.
        NOTE: the results is already unrolled on sample_n > 1
        
        Args:
            indices: List of indices of the prompts in the original batch
            responses: List of response dictionaries from the rollout manager
            idx: Tensor of input ids of the prompts in the original batch
            attention_mask: Tensor of attention mask of the prompts in the original batch
            position_ids: Tensor of position ids of the prompts in the original batch
            eos_token_id: EOS token ID for attention mask generation
            non_tensor_batch: Non-tensor batch data
            
        Returns:
            DataProto object containing the processed microbatch
        """
        if not outputs:
            return None
        
        # Use the existing _post_process_outputs function to process responses
        # This aligns with standard SGLang rollout processing
        out = _post_process_outputs(self.tokenizer, outputs)
        response = out[0]  # (batch_size, response_length)
        rollout_log_probs = out[1]  # (batch_size, response_length)
        
        microbatch_size = response.size(0) # already unrolled on sample_n > 1, no need for * sample_n
        device = idx.device

        # Move to proper device
        response = response.to(device)
        rollout_log_probs = rollout_log_probs.to(device)
        
        # Pad responses to the configured response length if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

        # polyrl-dev
        # NOTE: because the return microbatch is OOO, we need to extract the corresponding meta data
        # Extract the corresponding prompts from the original batch using indices
        microbatch_idx = idx[indices]  # (microbatch_size, prompt_length)
        microbatch_attention_mask = attention_mask[indices]  # (microbatch_size, prompt_length)
        microbatch_position_ids = position_ids[indices]  # (microbatch_size, prompt_length)

        # Extract corresponding non-tensor batch data for the microbatch
        # TODO(liuxs): check if we process uid correctly
        microbatch_non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            if isinstance(val, np.ndarray) and len(val) > 0:
                microbatch_non_tensor_batch[key] = val[indices]
            else:
                # For scalar values or empty arrays, keep as is
                microbatch_non_tensor_batch[key] = val

        # NOTE: because indices is already duplicated, we don't need to repeat again
        # if self.sampling_params.get("n", 1) > 1:
        #     microbatch_idx = microbatch_idx.repeat_interleave(self.sampling_params["n"], dim=0)
        #     microbatch_attention_mask = microbatch_attention_mask.repeat_interleave(self.sampling_params["n"], dim=0)
        #     microbatch_position_ids = microbatch_position_ids.repeat_interleave(self.sampling_params["n"], dim=0)
        #     # repeat the non_tensor_batch
        #     _microbatch_non_tensor_batch = {}
        #     for key, val in microbatch_non_tensor_batch.items():
        #         _microbatch_non_tensor_batch[key] = np.repeat(val, self.sampling_params["n"], axis=0)
        # else:
        #     _microbatch_non_tensor_batch = microbatch_non_tensor_batch
        _microbatch_non_tensor_batch = microbatch_non_tensor_batch

        # Following the standard SGLang rollout pattern for constructing sequences
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(microbatch_size, 1)

        # Create sequence by concatenating prompts and responses
        seq = torch.cat([microbatch_idx, response], dim=-1)
        
        # Update position IDs following standard pattern
        response_position_ids = microbatch_position_ids[:, -1:] + delta_position_id
        final_position_ids = torch.cat([microbatch_position_ids, response_position_ids], dim=-1)
        # Create response attention mask following standard pattern
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat((microbatch_attention_mask, response_attention_mask), dim=-1)

        # Create the batch tensor dict following standard SGLang rollout format
        batch = TensorDict(
            {
                "prompts": microbatch_idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": final_attention_mask,
                "response_mask": response_attention_mask, # we need to return the response mask for the whole microbatch
                "position_ids": final_position_ids,
            },
            batch_size=microbatch_size,
        )
        
        return DataProto(batch=batch, non_tensor_batch=_microbatch_non_tensor_batch)

    def generate_sequences_remote(self, prompts: DataProto, microbatch_size: int = 4, **kwargs) -> Iterator[DataProto]:
        """
        Main entry point for generating sequences using the iterator wrapper.
        
        Args:
            prompts: DataProto containing input prompts
            microbatch_size: Size of each microbatch to yield
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator yielding DataProto objects containing microbatches of generated sequences
        """
        prompts = prompts.to(get_torch_device().current_device())
        prompts.meta_info.update(kwargs)

        return self._microbatch_iterator(prompts, microbatch_size)

# sample usage in verl

"""
... existing code (calculating the microbatch size)

    gen_iter = self.rollout.generate_sequences(prompts, microbatch_size)

    for microbatch in gen_iter:
        ... do something with the microbatch

... existing code
"""
