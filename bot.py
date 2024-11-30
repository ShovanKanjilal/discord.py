import discord
import os
import uvicorn
import asyncio
import aiohttp  # Async HTTP client
from discord.ext import commands
from dotenv import load_dotenv
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

@app.get("/status")
def health_check():
    return {"status": "ok"}

@app.head("/uptimerobot")
async def uptimerobot_check():
    return {}  # Return an empty response body for HEAD request, no body content

def run_http_server():
    uvicorn.run(app, host="0.0.0.0", port=8080)

# Load environment variables
load_dotenv()
my_secret = os.getenv('secret_key')
huggingface_api_key = os.getenv('HF_API_KEY')  # Your Hugging Face API key

if not my_secret or not huggingface_api_key:
    raise ValueError("Missing environment variables for 'secret_key' or 'HF_API_KEY'")

# Initialize the bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Define the Hugging Face API URL for the model
MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"

# Asynchronous function to query Hugging Face API
async def query_huggingface_model(message_content):
    headers = {
        "Authorization": f"Bearer {huggingface_api_key}"
    }

    prompt = f"Respond to this message: '{message_content}'"
    payload = {
        "inputs": prompt
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MODEL_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}, {await response.text()}")
                    return "Sorry, I couldn't process your request properly."
                data = await response.json()
                generated_text = data[0].get('generated_text', "").strip()
                if not generated_text:
                    return "Sorry, I couldn't process your request properly. Can you please clarify?"
                return generated_text

    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Sorry, I couldn't process your request."

# Define on_ready event
@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')

# Define on_message event
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message) or "booblord" in message.content.lower():
        print(f'Message from {message.author}: {message.content}')

        # Get response from the Hugging Face model
        model_response = await query_huggingface_model(message.content)
        
        # Send the model's response to the channel
        await message.channel.send(model_response)

        # Process commands if any
        await bot.process_commands(message)

# Run the bot in an async main function
async def main():
    # Run the HTTP server in the background
    asyncio.get_running_loop().run_in_executor(None, run_http_server)
    await bot.start(my_secret)

# Start the bot
if __name__ == "__main__":
    asyncio.run(main())
