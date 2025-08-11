# app/services/bedrock_service.py
import asyncio
import re
from core.utils import printer
from config.settings import MAX_CHARS_PER_SSML_CHUNK 
from services.polly_service import PollyService
from services.langchain_service import LangGraphService


class BedrockService:
    """
    Orchestrates the voice interaction, using LangGraphService to get the AI
    response and PollyService to handle text-to-speech.
    """
    def __init__(self, app_state, thread_id: str = "default"):
        self.app_state = app_state
        self.polly_service = PollyService()
        # Use a stable thread_id so LangGraph state persists across turns
        self.langchain_service = LangGraphService(thread_id=thread_id)

    async def _to_chunk_generator(self, text_chunk_stream):
        """
        Converts a stream of text into chunks split by natural breaks
        (sentences, commas) for smooth, real-time audio playback.
        """
        buffer = ""
        # Pattern keeps delimiters as separate groups to recombine nicely
        split_pattern = re.compile(r'([.?!]\s*|\s*,\s*|\n)')

        async for text_chunk in text_chunk_stream:
            if self.app_state.was_interrupted():
                break

            if not text_chunk:
                continue

            buffer += text_chunk

            # Split buffer into sentences and keep the last incomplete part
            parts = split_pattern.split(buffer)
            processed_parts = parts[:-1]
            buffer = parts[-1] or ""

            # Emit pairs: [sentence, delimiter]
            for i in range(0, len(processed_parts), 2):
                sentence = processed_parts[i]
                delimiter = processed_parts[i + 1] if (i + 1) < len(processed_parts) else ""
                if sentence.strip():
                    yield f"{sentence.strip()}{delimiter.strip()}"

        # Flush any remaining buffer if not interrupted
        if buffer.strip() and not self.app_state.was_interrupted():
            yield buffer.strip()

    async def invoke_bedrock(self, websocket, text: str):
        printer(f"\n[BEDROCK] Invoking with: '{text}'", "info")
        self.app_state.start_bot_speech()

        try:
            # 1) Get the raw text stream from the LangGraphService (async generator)
            text_stream = self.langchain_service.get_response_stream(text)

            # 2) Convert the text stream to speakable chunks (async generator)
            chunk_stream = self._to_chunk_generator(text_stream)

            # 3) Stream the chunks to Polly for speech synthesis
            async for chunk in chunk_stream:
                if self.app_state.was_interrupted():
                    break

                if chunk and chunk.strip():
                    print(f"ASSISTANT: {chunk}", flush=True)
                    await self.polly_service.speak_text(websocket, chunk, self.app_state)

        except asyncio.CancelledError:
            printer("[BEDROCK] Task was cancelled by external signal.", "debug")
        except Exception as e:
            printer(f"[ERROR] Error in BedrockService invocation: {e}", "error")
            import traceback
            printer(f"[ERROR] Full traceback: {traceback.format_exc()}", "debug")
            if not self.app_state.was_interrupted():
                await self.polly_service.speak_text(
                    websocket,
                    "I'm sorry, a communication error occurred.",
                    self.app_state,
                )
        finally:
            # Ensure a newline separation in console and stop speech state
            print("", end="\n", flush=True)
            self.app_state.stop_bot_speech()
