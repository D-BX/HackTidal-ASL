from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

def translate(transcript):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
      {"role": "system", "content": "Given this sequence of letters stitch the words together into a sentence. Be aware of mispellings. Only return that sentence. "},
      {"role": "user", "content": f"{transcript}"},
    ]
  )
  return(completion.choices[0].message.content)

# def translate(transcript):
#     return transcript

if __name__ == "__main__":
    test_transcript = "tomorrow i go to store"
    result = translate(test_transcript)
    print("Translation Result:", result)

# Testing
recombin_str = "iloveyou"