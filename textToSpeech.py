import boto3
import os
from dotenv import load_dotenv
load_dotenv()



def genVoice(text):
    polly_client = boto3.Session(
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),                     
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-west-2').client('polly')

    response = polly_client.synthesize_speech(VoiceId='Joanna',
                    OutputFormat='mp3', 
                    Text = text,
                    Engine = 'neural')

    file = open('speechOutputs/speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

#genVoice("My name is gavin and I like Thomas more than Sofie")

