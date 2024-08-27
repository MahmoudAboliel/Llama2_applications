import assemblyai as aai
from langchain.tools import Tool

class SpeechToText:
    def __init__(self):

        aai.settings.api_key = "4d0ae082ea5e48989b5432045c1f1a03"
        self.transcriber = aai.Transcriber()
    def transcribe(self, audio_file):
        transcript = self.transcriber.transcribe(audio_file)

        return transcript.text

# sss = SpeechToText()
# print(sss.transcribe(audio_file="a.ogg"))

def stt_tool():
    stt = SpeechToText()
    return Tool(name="stt_tool", func=stt.transcribe, description="Transcribe audio files to text.")
