# app/services/transcribe_service.py

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from config.settings import config

class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, on_final_transcript):
        super().__init__(output_stream)
        self.full_transcript = []
        self.on_final_transcript = on_final_transcript

    async def handle_transcript_event(self, transcript_event):
        results = transcript_event.transcript.results
        if results and results[0].alternatives and not results[0].is_partial:
            transcript = results[0].alternatives[0].transcript
            print(f"USER: {transcript}", end=" ", flush=True)
            self.full_transcript.append(transcript)

    def get_full_transcript(self):
        full = " ".join(self.full_transcript).strip()
        if self.on_final_transcript and full:
            self.on_final_transcript(full)

class TranscribeService:
    def __init__(self):
        self.client = TranscribeStreamingClient(region=config['region'])

    async def start_transcription(self, samplerate):
        return await self.client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=samplerate,
            media_encoding="pcm"
        )
