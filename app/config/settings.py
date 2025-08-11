# app/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv('MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
WEBSOCKET_PORT = 8765

config = {
    'log_level': 'debug',
    'region': AWS_REGION,
    'model_id': MODEL_ID,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': 'en-US',
        'VoiceId': 'Matthew',
        'OutputFormat': 'pcm',
        'SampleRate': '8000',
        'TextType': 'ssml' 
    },
    'vad': {'threshold': 0.5, 'silence_sec': 1.5}
}

MAX_CHARS_PER_SSML_CHUNK = 150
SHORT_BREAK_MS = 100
SENTENCE_BREAK_MS = 200
LIST_ITEM_BREAK_MS = 150
