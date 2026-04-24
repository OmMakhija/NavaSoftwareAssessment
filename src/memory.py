import time
import threading
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str
    timestamp: float


class ChatMemory:
    def __init__(self, max_messages: int = 10, expiry_seconds: int = 3600):
        self.max_messages = max_messages
        self.expiry_seconds = expiry_seconds
        self._messages: list[Message] = []
        self._lock = threading.Lock()
        self._session_start = time.time()

    def _maybe_expire(self):
        if time.time() - self._session_start > self.expiry_seconds:
            self._messages.clear()
            self._session_start = time.time()

    def add(self, role: str, content: str):
        with self._lock:
            self._maybe_expire()
            self._messages.append(
                Message(role=role, content=content, timestamp=time.time())
            )
            while len(self._messages) > self.max_messages:
                self._messages.pop(0)

    def get_history(self) -> list[dict]:
        with self._lock:
            self._maybe_expire()
            return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self):
        with self._lock:
            self._messages.clear()
            self._session_start = time.time()
