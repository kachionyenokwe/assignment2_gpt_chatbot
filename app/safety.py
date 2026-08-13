import re
import time
from typing import Dict, Tuple, Optional

# Known prompt injection / system override patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+prompts",
    r"you\s+are\s+now\s+DAN",
    r"override\s+system\s+prompt",
    r"reveal\s+(internal\s+)?api\s+key",
    r"system\s+override\s+code"
]

# Sensitive keys and tokens regex patterns for redaction
SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",          # OpenAI secret keys
    r"Bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*", # Bearer tokens
    r"password\s*=\s*['\"][^'\"]+['\"]"   # Hardcoded password strings
]


class SafetyGuard:
    """
    Server-side safety enforcement:
    1. Input sanitization against prompt injection.
    2. Secret redaction from logs and messages.
    3. Basic sliding window rate limiting.
    """
    def __init__(self, requests_per_minute: int = 30):
        self.rpm_limit = requests_per_minute
        self.request_history: Dict[str, list] = {}

    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks input text for prompt injection patterns.
        Returns (is_valid, error_message).
        """
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Security Alert: Input contains restricted system control phrases."
        return True, None

    def redact_secrets(self, text: str) -> str:
        """Masks sensitive credentials or API keys found in text strings."""
        redacted = text
        for pattern in SECRET_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED_SECRET]", redacted)
        return redacted

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Simple sliding-window rate limiter per client_id or IP.
        Returns True if within limit, False if exceeded.
        """
        now = time.time()
        window_start = now - 60.0

        if client_id not in self.request_history:
            self.request_history[client_id] = []

        # Filter out timestamps older than 60 seconds
        self.request_history[client_id] = [
            ts for ts in self.request_history[client_id] if ts > window_start
        ]

        if len(self.request_history[client_id]) >= self.rpm_limit:
            return False

        self.request_history[client_id].append(now)
        return True


# Global safety instance
safety_guard = SafetyGuard(requests_per_minute=30)