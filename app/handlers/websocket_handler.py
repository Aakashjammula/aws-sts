# app/handlers/websocket_handler.py
import asyncio
import torch
import numpy as np
import torchaudio
import websockets
from core.state import AppState
from services.bedrock_service import BedrockService
from services.transcribe_service import TranscribeService, TranscriptHandler
from config.settings import config
from core.utils import printer

class WebSocketHandler:
    def __init__(self, websocket):
        self.websocket = websocket
        self.app_state = AppState()
        self.bedrock_service = BedrockService(self.app_state)
        self.transcribe_service = TranscribeService()
        self.samplerate = int(config['polly']['SampleRate'])
        self.VAD_CHUNK_SIZE_BYTES = 512
        self.audio_buffer = b''
        self.bot_response_task = None

        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', 
                model='silero_vad', 
                force_reload=False
            )
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.samplerate, 
                new_freq=16000
            )
        except Exception as e:
            printer(f"[FATAL] Could not load Silero VAD model: {e}", "info")
            raise

    def _is_speech(self, chunk):
        if len(chunk) < self.VAD_CHUNK_SIZE_BYTES:
            return False
        try:
            audio_int16 = np.frombuffer(chunk, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            audio_16k = self.resampler(audio_tensor)
            speech_prob = self.vad_model(audio_16k, 16000).item()
            is_speech_flag = speech_prob > config['vad']['threshold']
            return is_speech_flag
        except Exception:
            return False

    def _handle_transcription(self, final_transcript):
        if final_transcript:
            self.bot_response_task = asyncio.create_task(
                self.bedrock_service.invoke_bedrock(self.websocket, final_transcript)
            )

    async def handle_connection(self):
        self._handle_transcription("User said hello.")
        
        is_transcribing = False
        transcribe_stream, transcript_handler, transcript_task = None, None, None
        silence_frames = 0
        
        printer("\n[INFO] Waiting for user to speak...", "info")
        try:
            async for raw_chunk in self.websocket:
                self.audio_buffer += raw_chunk
                while len(self.audio_buffer) >= self.VAD_CHUNK_SIZE_BYTES:
                    chunk_to_process = self.audio_buffer[:self.VAD_CHUNK_SIZE_BYTES]
                    self.audio_buffer = self.audio_buffer[self.VAD_CHUNK_SIZE_BYTES:]
                    is_speech_chunk = self._is_speech(chunk_to_process)

                    if self.app_state.is_bot_speaking() and is_speech_chunk:
                        printer("\n[INFO] Barge-in detected! Interrupting bot.", "info")
                        if self.bot_response_task and not self.bot_response_task.done():
                            self.bot_response_task.cancel()
                            self.app_state.interrupt() 
                        
                        if transcript_task and not transcript_task.done():
                            transcript_task.cancel()
                        if transcribe_stream:
                            await transcribe_stream.input_stream.end_stream()
                        is_transcribing, transcribe_stream, transcript_handler, transcript_task = False, None, None, None
                        
                    elif is_transcribing: 
                        await transcribe_stream.input_stream.send_audio_event(audio_chunk=chunk_to_process)
                        if not is_speech_chunk:
                            silence_frames += 1
                            chunks_per_second = self.samplerate / (self.VAD_CHUNK_SIZE_BYTES / 2)
                            if silence_frames >= int(chunks_per_second * config['vad']['silence_sec']):
                                printer("\n[INFO] End of speech detected.", "info")
                                await transcribe_stream.input_stream.end_stream()
                                await transcript_task
                                transcript_handler.get_full_transcript()
                                is_transcribing, transcribe_stream, transcript_handler, transcript_task = False, None, None, None
                                printer("\n[INFO] Waiting for user to speak...", "info")
                        else:
                            silence_frames = 0
                            
                    elif is_speech_chunk: 
                        printer("\n[INFO] Speech detected, starting transcription...", "info")
                        is_transcribing = True
                        silence_frames = 0
                        transcribe_stream = await self.transcribe_service.start_transcription(self.samplerate)
                        transcript_handler = TranscriptHandler(transcribe_stream.output_stream, self._handle_transcription)
                        transcript_task = asyncio.create_task(transcript_handler.handle_events())
                        await transcribe_stream.input_stream.send_audio_event(audio_chunk=chunk_to_process)

        except websockets.exceptions.ConnectionClosed as e:
            printer(f"[INFO] Client disconnected: {e.code}", "info")
        except Exception as e:
            printer(f"[FATAL] Connection handler error: {e}", "info")
        finally:
            printer("[CLEANUP] Cleaning up connection.", "debug")
            if self.bot_response_task and not self.bot_response_task.done():
                self.bot_response_task.cancel()
            if transcript_task and not transcript_task.done():
                transcript_task.cancel()
            if transcribe_stream:
                try:
                    await transcribe_stream.input_stream.end_stream()
                except:
                    pass
