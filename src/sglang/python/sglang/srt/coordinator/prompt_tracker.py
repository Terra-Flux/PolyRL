from dataclasses import dataclass
from typing import Dict, Any, List
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PromptTracker:
    """Tracks the status of prompts and their samples"""
    total_prompts: int = 0
    total_samples: int = 0
    pending_samples: Dict[int, int] = None  # prompt_idx -> count of pending samples
    completed_samples: Dict[int, int] = None  # prompt_idx -> count of completed samples
    returned_samples: Dict[int, int] = None  # prompt_idx -> count of returned samples
    
    def __post_init__(self):
        if self.pending_samples is None:
            self.pending_samples = defaultdict(int)
        if self.completed_samples is None:
            self.completed_samples = defaultdict(int)
        if self.returned_samples is None:
            self.returned_samples = defaultdict(int)
    
    def add_prompts(self, prompts: List[str], n_samples: int):
        """Register new prompts and their expected samples"""
        self.total_prompts += len(prompts)
        self.total_samples += len(prompts) * n_samples
        
        for i in range(len(prompts)):
            # Use the base index (before any offset calculation)
            prompt_idx = self.total_prompts - len(prompts) + i
            self.pending_samples[prompt_idx] = n_samples
    
    def mark_sample_completed(self, global_idx: int, n_samples: int):
        """Mark a sample as completed based on its global index"""
        prompt_idx = global_idx // n_samples
        
        if prompt_idx in self.pending_samples and self.pending_samples[prompt_idx] > 0:
            self.pending_samples[prompt_idx] -= 1
            self.completed_samples[prompt_idx] += 1
    
    def mark_samples_returned(self, samples: List[Dict], n_samples: int):
        """Mark samples as returned to the client"""
        if isinstance(samples, list) and len(samples) > 0:
            if isinstance(samples[0], list):  # In-order mode (list of lists)
                for sample_group in samples:
                    if sample_group:
                        prompt_idx = sample_group[0]["index"] // n_samples
                        self.returned_samples[prompt_idx] += len(sample_group)
            else:  # Out-of-order mode (list of dicts)
                for sample in samples:
                    prompt_idx = sample["index"] // n_samples
                    self.returned_samples[prompt_idx] += 1

    @property
    def all_samples_completed(self) -> bool:
        """Check if all samples have been completed"""
        return sum(self.completed_samples.values()) == self.total_samples
    
    @property
    def all_samples_returned(self) -> bool:
        """Check if all samples have been returned"""
        return sum(self.returned_samples.values()) == self.total_samples
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the tracking status"""
        total_pending = sum(self.pending_samples.values())
        total_completed = sum(self.completed_samples.values())
        total_returned = sum(self.returned_samples.values())
        
        # Count how many prompts are fully processed
        completed_prompts = sum(1 for prompt_idx, count in self.completed_samples.items() 
                               if count == self.pending_samples.get(prompt_idx, 0) + count)
        
        return {
            "total_prompts": self.total_prompts,
            "total_samples": self.total_samples,
            "total_pending_samples": total_pending,
            "total_completed_samples": total_completed,
            "total_returned_samples": total_returned,
            "completed_prompts": completed_prompts,
            "all_samples_completed": self.all_samples_completed,
            "all_samples_returned": self.all_samples_returned
        }
