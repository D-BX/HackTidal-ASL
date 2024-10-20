from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

def translate(transcript):
  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.3,
    messages=[
      {"role": "system", "content": "Given this ASL sentence, with each word transcribed directly to its English counterpart, translate it to spoken english:"},
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