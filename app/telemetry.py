import time
import json
from typing import Dict, Any, List, Optional

# API pricing per 1,000 tokens
PRICING_PER_1K = {
    # Groq Models
    "llama-3.1-8b-instant": {"prompt": 0.00005, "completion": 0.00008},
    "llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079},
    
    # OpenAI Models (Fallback)
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.00250, "completion": 0.01000},
    
    # Default baseline
    "default": {"prompt": 0.00005, "completion": 0.00008}
}


class TelemetryTracker:
    """
    Logs performance metrics: Time-To-First-Token (TTFT),
    total latency, prompt/completion token usage, and estimated dollar cost.
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.start_time: float = 0.0
        self.first_token_time: Optional[float] = None
        self.end_time: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.tool_calls_count: int = 0

    def start_turn(self) -> None:
        """Starts timing the request execution."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = 0.0
        self.tool_calls_count = 0

    def record_first_token(self) -> None:
        """Records timestamp when first stream token is received."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def record_tool_call(self) -> None:
        """Increments tool invocation counter."""
        self.tool_calls_count += 1

    def end_turn(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, Any]:
        """
        Finalizes timing, estimates token usage if not directly provided,
        and returns a metrics dictionary.
        """
        self.end_time = time.perf_counter()
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

        total_latency_ms = round((self.end_time - self.start_time) * 1000, 2)
        
        if self.first_token_time:
            ttft_ms = round((self.first_token_time - self.start_time) * 1000, 2)
        else:
            ttft_ms = total_latency_ms

        pricing = PRICING_PER_1K.get(self.model_name, PRICING_PER_1K["default"])
        prompt_cost = (self.prompt_tokens / 1000.0) * pricing["prompt"]
        completion_cost = (self.completion_tokens / 1000.0) * pricing["completion"]
        total_cost = round(prompt_cost + completion_cost, 6)

        metrics = {
            "model": self.model_name,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": total_cost,
            "tool_calls_executed": self.tool_calls_count
        }
        return metrics