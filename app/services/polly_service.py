# app/services/polly_service.py

import asyncio
import re
import boto3
import websockets
from config.settings import config
from core.utils import printer

class PollyService:
    def __init__(self):
        self.polly = boto3.client('polly', region_name=config['region'])

    async def _send_audio_to_client(self, websocket, audio_data_stream, app_state):
        loop = asyncio.get_event_loop()
        try:
            while not app_state.was_interrupted():
                data = await loop.run_in_executor(None, audio_data_stream.read, 8192)
                if not data:
                    break
                await websocket.send(data)
                await asyncio.sleep(0.356)
        except asyncio.CancelledError:
            printer("[AUDIO] Audio streaming cancelled.", "debug")
        except websockets.exceptions.ConnectionClosed:
            printer("[WARN] Client connection closed during audio streaming.", "info")
            app_state.interrupt() 
        except Exception as e:
            printer(f"[ERROR] Error in audio streaming: {e}", "info")
        finally:
            if audio_data_stream:
                audio_data_stream.close()

    async def speak_text(self, websocket, text, app_state):
        cleaned_text = re.sub(r'\*[^*]*\*', '', text)
        
        if not cleaned_text.strip():
            return

        printer(f"[POLLY] Synthesizing: '{cleaned_text}'", "info")
        try:
            polly_response = self.polly.synthesize_speech(
                Text=cleaned_text,
                TextType='text',
                Engine=config['polly']['Engine'],
                LanguageCode=config['polly']['LanguageCode'],
                VoiceId=config['polly']['VoiceId'],
                OutputFormat='pcm',
                SampleRate=config['polly']['SampleRate']
            )
            await self._send_audio_to_client(websocket, polly_response['AudioStream'], app_state)
        except asyncio.CancelledError:
            raise 
        except Exception as e:
            printer(f"[ERROR] Polly synthesis failed: {e}", "info")
