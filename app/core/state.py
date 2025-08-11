# app/core/state.py
import threading
from core.utils import printer

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self._is_bot_speaking = False

    def is_bot_speaking(self):
        with self.lock: 
            return self._is_bot_speaking

    def start_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = True
            self.interrupt_event.clear()
            printer("[STATE] Bot speech started.", "debug")

    def stop_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = False
            self.interrupt_event.clear()
            printer("[STATE] Bot speech stopped.", "debug")

    def interrupt(self):
        if self.is_bot_speaking():
            printer("[STATE] Interrupt triggered!", "info")
            self.interrupt_event.set()

    def was_interrupted(self):
        return self.interrupt_event.is_set()
