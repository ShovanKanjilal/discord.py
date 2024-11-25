import discord
import os
import requests
from discord.ext import commands
from dotenv import load_dotenv

# Load secret key for the bot
load_dotenv()
my_secret = os.getenv('secret_key')
huggingface_api_key = os.getenv('HF_API_KEY')  # Your Hugging Face API key

# Initialize the bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Define the Hugging Face API URL for the model
MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"

# Function to query the Hugging Face API
def query_huggingface_model(message_content):
    headers = {
        "Authorization": f"Bearer {huggingface_api_key}"
    }

    # Send a simple prompt to the model
    prompt = f"Respond to this message: '{message_content}'"

    payload = {
        "inputs": prompt
    }

    try:
        # Make the API request to Hugging Face
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses

        # Get the model response (this will be a JSON object)
        generated_text = response.json()[0]['generated_text'].strip()

        # Debugging: Log the raw response
        print(f"Model raw response: {generated_text}")

        # Check if the model returned an appropriate answer
        if not generated_text:
            return "Sorry, I couldn't process your request properly. Can you please clarify?"
        
        return generated_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Sorry, I couldn't process your request."

    
    
@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    print(f'Message from {message.author}: {message.content}')

    # Get response from the Hugging Face model
    model_response = query_huggingface_model(message.content)
    
    # Send the model's response to the channel
    await message.channel.send(model_response)

# Run the bot
bot.run(my_secret)
