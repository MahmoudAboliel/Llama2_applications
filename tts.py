import requests
import replicate
import dotenv
dotenv.load_dotenv()
from langchain.tools import Tool

import os
API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
class TextToSpeech:
    def __init__(self):
        self.model = "lucataco/whisperspeech-small:a453e7a3dde7a9041a210a89339a128f8542e45d5172e46b44530cd9a0406e9d"

    def synthesize(self, prompt):
        input = {"prompt": prompt}
        output = replicate.run(self.model, input=input)
        response = requests.get(output)
        # output_file = f"answer_{session_id}.wav"
        # with open(output_file, "wb") as file:
        #     file.write(response.content)
        return response.content


def tts_tool():
    tts = TextToSpeech()
    return Tool(name="tts_tool", func=tts.synthesize, description="Convert text to speech and save as an audio file.")

# tts = TextToSpeech()
# tts.synthesize("hello, how are you what you doing now", "1")