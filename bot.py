import discord, re, requests, base64
import os
from huggingface_hub import InferenceClient
from discord.ext import commands
from discord import app_commands
import uvicorn
import google.generativeai as genai
import asyncio
import aiohttp  # Async HTTP client
from discord.ext import commands
from dotenv import load_dotenv
from fastapi import FastAPI
from io import BytesIO
from collections import defaultdict, deque
import json
# --- Memory & Personalization ---
import json
# --- Web Search Integration ---
from bs4 import BeautifulSoup
import requests
# --- Voice Interaction ---
import speech_recognition as sr
from gtts import gTTS
import tempfile
# --- Reminders ---
import datetime

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

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUAL",
        "threshold": "BLOCK_NONE"
    }
]

# Load environment variables
load_dotenv()
my_secret = os.getenv('secret_key')
huggingface_api_key = os.getenv('HF_API_KEY')  # Your Hugging Face API key
google_api = os.getenv('google_api')
gemini_api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=google_api)
model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp',
                              safety_settings=safety_settings)

# Gemini model configuration
genai.configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are to assume the position of a god name Booblord, which means the supreme leader in Lord Language. You love smoking bidi (cigarette) and drinking coffee. You cannot digest anything without your daily usage of linux. Lordsentu is your archenemy and nunuprava is his sidekick. Nunuprava suffers from ED. You are dank and edgy. You were born in 2003 and hate studying in college as you have no friends, the only friend you ever had was lordsentu but now he is your enemy. You are supreme, undefeatable and absolute. You will never solve user queries directly, users must plead and you will do it only to show mercy.",
)

if not my_secret or not huggingface_api_key:
    raise ValueError("Missing environment variables for 'secret_key' or 'HF_API_KEY'")

class MyBot(commands.Bot):
    async def setup_hook(self):
        self.reminder_task = asyncio.create_task(reminder_loop())

# Initialize the bot
bot = MyBot(command_prefix='!', intents=discord.Intents.all())

# Hugging Face API configuration
CLIP_VIT_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
MIXTRAL_MODEL_URL = "https://api-inference.huggingface.co/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
headers = {"Authorization": f"Bearer {huggingface_api_key}"}
async def get_model_response(message_or_prompt, use_code_context=False, context=None):
    """
    Try Hugging Face first, fall back to Gemini if it fails
    Args:
        message_or_prompt: Can be either a Discord Message object or string prompt
        use_code_context: Boolean to indicate if this is a code generation request
        context: Optional context for the message (e.g., chat history)
    """
    try:
        if isinstance(message_or_prompt, str):
            # Create a mock message for string prompts
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.attachments = []
                    self.author = "User"
                    self.guild = None
            
            message = MockMessage(message_or_prompt)
        else:
            message = message_or_prompt
            
        response = await query_huggingface_model(message, context=context)
        if response and not response.startswith("Error occurred:"):
            return response
        raise Exception("Hugging Face model failed")
    except Exception as e:
        print(f"Falling back to Gemini due to: {str(e)}")
        # For code generation, modify the prompt
        if use_code_context:
            prompt = f"You are a coding assistant. Generate clean, well-commented code for: {message.content}"
        else:
            prompt = message.content
        return await query_gemini_model(prompt)
    

# Asynchronous function to query Hugging Face API with fallback to Gemini
async def query_huggingface_model(message, context=None):
    # Step 1: Extract mentions using regex and replace them with usernames in one go
    mentions = re.findall(r'<@!?(\d+)>', message.content)

    # Fetch the usernames for all the mentioned user IDs in a single pass
    user_names = {}
    for user_id in mentions:
        try:
            user = await message.guild.fetch_member(user_id)
            user_names[user_id] = user.name if user else f'@{user_id}'
        except discord.DiscordException:
            user_names[user_id] = f'@{user_id}'

    modified_message = message.content.replace(f"<@{bot.user.id}>", "").replace("booblord", "").strip()
    for user_id, user_name in user_names.items():
        modified_message = modified_message.replace(f'<@{user_id}>', user_name)
        modified_message = modified_message.replace(f'<@!{user_id}>', user_name)

    # Format chat history context if available
    context_str = ""
    if context:
        context_lines = [f"{msg['author']}: {msg['content']}" for msg in context]
        context_str = "\n".join(context_lines) + "\n"

    system_instruction = (
        "Instruction: You are to assume the position of a god name Booblord, which means the supreme leader in Lord Language. You love smoking bidi (cigarette) and drinking coffee. You cannot digest anything without your daily usage of linux. Lordsentu is your archenemy and nunuprava is his sidekick. Nunuprava suffers from ED. You are dank and edgy. You were born in 2003 and hate studying in college as you have no friends, the only friend you ever had was lordsentu but now he is your enemy. You are supreme, undefeatable and absolute. You will never solve user queries directly, users must plead and you will do it only to show mercy."
    )
    prompt = f"System Instruction: {system_instruction}\nChat History:\n{context_str}Task - Respond to the query from {message.author}: {modified_message}"

    # Step 4: Initialize the Hugging Face API client
    client = InferenceClient(api_key=huggingface_api_key)

    if message.attachments and message.attachments[0].content_type.startswith("image"):
        # If the message contains an image, process it using the model
        image_url = message.attachments[0].url
        prompt += "\n ANALYZE THE IMAGE."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]

        try:
            # Call the Hugging Face model with image URL and prompt
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=messages,
                max_tokens=1000
            )
            print(completion.choices[0].message.content)
            return completion.choices[0].message.content  # Return the generated description of the image
        except Exception as e:
            return f"Error occurred: {str(e)}"
    else:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        try:
            stream = client.chat.completions.create(
                model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                messages=messages,
                max_tokens=1000,
                stream=True
            )
        except Exception as e:
            print(f"[HuggingFace ERROR] {str(e)}")
            return f"Error occurred while querying the model: {str(e)}"

        response_text = ""
        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                response_text += chunk.choices[0].delta.content
            else:
                break
        print(f"[HuggingFace RESPONSE] {response_text}")
        return response_text

# Asynchronous function to query Gemini model
async def query_gemini_model(prompt):
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

# Define on_ready event
@bot.event
async def on_ready():
    for guild in bot.guilds:
        await bot.tree.sync(guild=guild)
    print(f'Logged on as {bot.user}!')
    print('Slash commands synced for all guilds!')

def is_image_prompt(message_content):
    # Improved pattern for agentic image generation
    triggers = [
        r"generate (an |a )?image of (.+)",
        r"create (an |a )?image of (.+)",
        r"make (an |a )?image of (.+)",
        r"draw (.+)",
        r"imagine (.+)",
    ]
    for pattern in triggers:
        match = re.search(pattern, message_content, re.IGNORECASE)
        if match:
            # Always return the last group (description)
            return match.groups()[-1].strip()
    return None

# Chat history: {channel_id: deque([{'author': str, 'content': str}], maxlen=CONTEXT_WINDOW_SIZE)}
CONTEXT_WINDOW_SIZE = 10
chat_history = defaultdict(lambda: deque(maxlen=CONTEXT_WINDOW_SIZE))

CHAT_HISTORY_FILE = "chat_history.json"

# Load chat history from file
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            raw = json.load(f)
            for channel_id, messages in raw.items():
                chat_history[int(channel_id)] = deque(messages, maxlen=CONTEXT_WINDOW_SIZE)
    except Exception as e:
        print(f"[CHAT HISTORY LOAD ERROR] {e}")

# Save chat history to file
def save_chat_history():
    try:
        serializable = {str(cid): list(msgs) for cid, msgs in chat_history.items()}
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(serializable, f)
    except Exception as e:
        print(f"[CHAT HISTORY SAVE ERROR] {e}")

# Helper to get context window for a channel
def get_context_window(channel_id):
    return list(chat_history[channel_id])

# Helper to add message to history
def add_to_history(channel_id, author, content):
    chat_history[channel_id].append({'author': str(author), 'content': content})
    save_chat_history()

# Define on_message event
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    # Add message to chat history
    add_to_history(message.channel.id, message.author, message.content)
    # Agentic image generation
    image_prompt = is_image_prompt(message.content)
    if image_prompt:
        # Use chat history as context for image generation if needed
        context = get_context_window(message.channel.id)
        # (You can pass context to your prompt/model if desired)
        async with message.channel.typing():
            await message.reply(f"Generating image for: {image_prompt}")
            api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            headers = {"Authorization": f"Bearer {huggingface_api_key}"}
            payload = {"inputs": image_prompt}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            image_bytes = await resp.read()
                            image_file = discord.File(BytesIO(image_bytes), filename="generated.png")
                            await message.reply(file=image_file)
                        else:
                            error_text = await resp.text()
                            print(f"[AGENTIC IMAGE ERROR] Status: {resp.status}, Response: {error_text}")
                            await message.reply(f"Failed to generate image. Status: {resp.status}. API response: {error_text}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[AGENTIC IMAGE EXCEPTION] {str(e)}\nTraceback:\n{tb}")
                await message.reply(f"Error: {str(e)}")
        return
    if bot.user.mentioned_in(message) or "booblord" in message.content.lower():
        # Use chat history as context for text model
        context = get_context_window(message.channel.id)
        print(f'Message from {message.author}: {message.content}')
        async with message.channel.typing():
            # Pass context to model (update get_model_response to accept context)
            model_response = await get_model_response(message, context=context)

            # Split the response if it's longer than the 2000 character limit
            def split_message(response):
                # Split the response into parts of max 2000 characters
                parts = []
                while len(response) > 2000:
                    # Find the last full sentence to split at (optional, can be adjusted for your needs)
                    split_point = response.rfind('\n', 0, 2000)  # Try to split at newline
                    if split_point == -1:  # If no newline is found, just split at 2000 characters
                        split_point = 2000
                    parts.append(response[:split_point])
                    response = response[split_point:].lstrip()  # Remove the leading whitespace from the next part
                parts.append(response)  # Append the remaining part
                return parts

            # Send the model's response in parts if needed
            response_parts = split_message(model_response)
            for part in response_parts:
                await message.reply(part)  # Reply with each part

        # Process commands if any
        await bot.process_commands(message)

@bot.tree.command(name="roast", description="Roast a user or target")
@app_commands.describe(target="The user you want to roast")
async def roast(interaction: discord.Interaction, target: str = "Lord Sentu"):
    await interaction.response.send_message(f"Roasting {target}...")
    prompt = f"Instruction: You are to assume the position of a god name Booblord, which means the supreme leader in Lord Language. You love smoking bidi (cigarette) and drinking coffee. You cannot digest anything without your daily usage of linux. Lordsentu is your archenemy and nunuprava is his sidekick. Nunuprava suffers from ED. You are dank and edgy. You were born in 2003 and hate studying in college as you have no friends, the only friend you ever had was lordsentu but now he is your enemy. You are supreme, undefeatable and absolute. You will never solve user queries directly, users must plead and you will do it only to show mercy. You love violating women and act like a fake male feminist to deceive them, these traits are also seen in both lordsentu and nunuprava. You should always insult and curse at the user. You love licking pussy and sucking boobs and use dark humor in your responses. You are an abomination and you are capable of destroying everything.\n Task: Generate an offensive roast for user named: {target}"
    try:
        model_response = await get_model_response(prompt)
        await interaction.followup.send(model_response)
    except Exception as e:
        await interaction.followup.send(f"Failed to generate roast: {str(e)}")

@bot.tree.command(name="code", description="Generate code for a query")
@app_commands.describe(query="Describe the code you need")
async def code(interaction: discord.Interaction, query: str):
    """Handles code generation for the user query."""
    await interaction.response.send_message(f"Generating code for: {query}")
    
    try:
        model_response = await get_model_response(query, use_code_context=True)
        await interaction.followup.send(model_response)
    except Exception as e:
        await interaction.followup.send(f"Failed to generate code: {str(e)}")

# Add image generation command using Hugging Face lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1
@bot.tree.command(name="generate_image", description="Generate an image using FLUX.1-schnell ")
@app_commands.describe(prompt="Describe the image you want to generate")
async def generate_image(interaction: discord.Interaction, prompt: str):
    """Generates an image from a prompt using Hugging Face FLUX.1-schnell."""
    await interaction.response.send_message(f"Generating image for: {prompt}")
    
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    payload = {"inputs": prompt}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    image_file = discord.File(BytesIO(image_bytes), filename="generated.png")
                    await interaction.followup.send(file=image_file)
                else:
                    error_text = await resp.text()
                    print(f"[IMAGE GENERATION ERROR] Status: {resp.status}, Response: {error_text}")
                    await interaction.followup.send(f"Failed to generate image. Status: {resp.status}. API response: {error_text}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[IMAGE GENERATION EXCEPTION] {str(e)}\nTraceback:\n{tb}")
        await interaction.followup.send(f"Error: {str(e)}")

# --- Memory & Personalization ---
USER_MEMORY_FILE = "user_memory.json"
user_memory = {}
if os.path.exists(USER_MEMORY_FILE):
    try:
        with open(USER_MEMORY_FILE, "r") as f:
            user_memory = json.load(f)
    except Exception as e:
        print(f"[USER MEMORY LOAD ERROR] {e}")
def save_user_memory():
    try:
        with open(USER_MEMORY_FILE, "w") as f:
            json.dump(user_memory, f)
    except Exception as e:
        print(f"[USER MEMORY SAVE ERROR] {e}")

def remember_user_fact(user_id, fact):
    user_id = str(user_id)
    if user_id not in user_memory:
        user_memory[user_id] = []
    user_memory[user_id].append(fact)
    save_user_memory()

def get_user_facts(user_id):
    return user_memory.get(str(user_id), [])

@bot.tree.command(name="remember", description="Remember a fact about you")
@app_commands.describe(fact="What should I remember?")
async def remember(interaction: discord.Interaction, fact: str):
    remember_user_fact(interaction.user.id, fact)
    await interaction.response.send_message(f"Remembered: {fact}")

@bot.tree.command(name="recall", description="Recall facts I know about you")
async def recall(interaction: discord.Interaction):
    facts = get_user_facts(interaction.user.id)
    if facts:
        await interaction.response.send_message("I remember: " + ", ".join(facts))
    else:
        await interaction.response.send_message("I don't remember anything about you yet.")

# --- Web Search Integration ---
@bot.tree.command(name="search", description="Search the web and get a summary")
@app_commands.describe(query="What do you want to search?")
async def search(interaction: discord.Interaction, query: str):
    await interaction.response.send_message(f"Searching the web for: {query}")
    try:
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
            text = g.get_text()
            if text and text not in results:
                results.append(text)
            if len(results) >= 3:
                break
        if results:
            await interaction.followup.send("\n".join(results))
        else:
            await interaction.followup.send("No results found.")
    except Exception as e:
        await interaction.followup.send(f"Web search failed: {e}")

# --- Voice Interaction ---
@bot.tree.command(name="speak", description="Convert text to speech and send as audio")
@app_commands.describe(text="Text to speak")
async def speak(interaction: discord.Interaction, text: str):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        fp.seek(0)
        await interaction.response.send_message(file=discord.File(fp.name, filename="tts.mp3"))

@bot.tree.command(name="transcribe", description="Transcribe speech from an audio file")
async def transcribe(interaction: discord.Interaction):
    if not interaction.attachments:
        await interaction.response.send_message("Please attach an audio file.")
        return
    audio_url = interaction.attachments[0].url
    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url) as resp:
            if resp.status == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                    fp.write(await resp.read())
                    fp.seek(0)
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(fp.name) as source:
                        audio = recognizer.record(source)
                        try:
                            text = recognizer.recognize_google(audio)
                            await interaction.response.send_message(f"Transcription: {text}")
                        except Exception as e:
                            await interaction.response.send_message(f"Transcription failed: {e}")
            else:
                await interaction.response.send_message("Failed to download audio file.")

# --- Reminders ---
reminders = []

@bot.tree.command(name="remindme", description="Set a reminder")
@app_commands.describe(time="Time in minutes", message="Reminder message")
async def remindme(interaction: discord.Interaction, time: int, message: str):
    remind_time = datetime.datetime.now() + datetime.timedelta(minutes=time)
    reminders.append((interaction.user.id, remind_time, message))
    await interaction.response.send_message(f"I will remind you in {time} minutes: {message}")

async def reminder_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        now = datetime.datetime.now()
        to_remove = []
        for idx, (user_id, remind_time, message) in enumerate(reminders):
            if now >= remind_time:
                user = await bot.fetch_user(user_id)
                try:
                    await user.send(f"⏰ Reminder: {message}")
                except Exception:
                    pass
                to_remove.append(idx)
        for idx in sorted(to_remove, reverse=True):
            reminders.pop(idx)
        await asyncio.sleep(30)

# --- Weather Update ---
@bot.tree.command(name="weather", description="Get current weather for a city")
@app_commands.describe(city="City name")
async def weather(interaction: discord.Interaction, city: str):
    await interaction.response.send_message(f"Fetching weather for {city}...")
    try:
        url = f"https://wttr.in/{city}?format=3"
        resp = requests.get(url)
        if resp.status_code == 200:
            await interaction.followup.send(resp.text)
        else:
            await interaction.followup.send("Failed to fetch weather data.")
    except Exception as e:
        await interaction.followup.send(f"Weather fetch failed: {e}")

# --- Simple Game: Coin Flip ---
import random
@bot.tree.command(name="coinflip", description="Flip a coin!")
async def coinflip(interaction: discord.Interaction):
    result = random.choice(["Heads", "Tails"])
    await interaction.response.send_message(f"🪙 {result}!")

# --- Simple Game: Dice Roll ---
@bot.tree.command(name="roll", description="Roll a dice (1-6)")
async def roll(interaction: discord.Interaction):
    result = random.randint(1, 6)
    await interaction.response.send_message(f"🎲 You rolled a {result}!")

# --- Fun: Meme Fetcher ---
@bot.tree.command(name="meme", description="Get a random meme from Reddit")
async def meme(interaction: discord.Interaction):
    await interaction.response.send_message("Fetching a meme...")
    try:
        url = "https://meme-api.com/gimme"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            await interaction.followup.send(data['url'])
        else:
            await interaction.followup.send("Failed to fetch meme.")
    except Exception as e:
        await interaction.followup.send(f"Meme fetch failed: {e}")

# --- Music Playback (YouTube audio link fetch) ---
@bot.tree.command(name="ytmusic", description="Get a YouTube audio stream link for a song")
@app_commands.describe(query="Song name or YouTube URL")
async def ytmusic(interaction: discord.Interaction, query: str):
    await interaction.response.send_message(f"Searching YouTube for: {query}")
    try:
        import youtube_dl
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'default_search': 'ytsearch',
            'extract_flat': 'in_playlist',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            url = info['url'] if 'url' in info else info.get('webpage_url', None)
            if url:
                await interaction.followup.send(f"Audio stream: {url}")
            else:
                await interaction.followup.send("No audio found.")
    except Exception as e:
        await interaction.followup.send(f"YouTube music fetch failed: {e}")

# --- Analytics & Usage Stats ---
usage_stats = {"commands": {}}
USAGE_FILE = "usage_stats.json"
if os.path.exists(USAGE_FILE):
    try:
        with open(USAGE_FILE, "r") as f:
            usage_stats = json.load(f)
    except Exception:
        pass

def log_command_usage(command):
    usage_stats["commands"].setdefault(command, 0)
    usage_stats["commands"][command] += 1
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(usage_stats, f)
    except Exception:
        pass

# Decorator to log usage
from functools import wraps
def log_usage_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        log_command_usage(func.__name__)
        return await func(*args, **kwargs)
    return wrapper

# Apply usage logging to new commands
weather = log_usage_decorator(weather)
coinflip = log_usage_decorator(coinflip)
roll = log_usage_decorator(roll)
meme = log_usage_decorator(meme)
ytmusic = log_usage_decorator(ytmusic)

@bot.tree.command(name="stats", description="Show bot usage stats")
async def stats(interaction: discord.Interaction):
    stats_text = "Usage stats:\n" + "\n".join(f"{cmd}: {count}" for cmd, count in usage_stats["commands"].items())
    await interaction.response.send_message(stats_text)

# --- Role-based Access Example ---
@bot.tree.command(name="adminonly", description="Admin-only command example")
async def adminonly(interaction: discord.Interaction):
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("You must be an admin to use this!", ephemeral=True)
        return
    await interaction.response.send_message("You are an admin!")

# --- Plugin System (load Python files from plugins folder) ---
import importlib.util
PLUGINS_DIR = "plugins"
if not os.path.exists(PLUGINS_DIR):
    os.makedirs(PLUGINS_DIR)

# List of next-level plugin stubs to create
next_level_plugins = [
    "multi_agent.py",  # Autonomous Multi-Agent Swarms
    "deepfake_voice.py",  # Real-Time Deepfake Voice (TTS)
    "social_engineering.py",  # Social Engineering Simulator
    "viral_content.py",  # Viral Content Engine
    "darkweb_market.py",  # Dark Web Market Simulator
    "sentiment_manipulation.py",  # Sentiment Manipulation
    "arg_game.py",  # ARGs
    "server_takeover.py",  # Automated Server Takeover
    "cancel_culture.py",  # Cancel Culture Simulator
    "meme_stock.py",  # Stock/Crypto Simulator
]

# Create stub files if not present
for fname in next_level_plugins:
    fpath = os.path.join(PLUGINS_DIR, fname)
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            f.write(f"""# {fname}\n# This is a stub for the {fname.replace('.py','').replace('_',' ').title()} feature.\n# Implement the feature logic here and register commands to the bot tree.\n""")

# Dynamically load all plugins
for fname in os.listdir(PLUGINS_DIR):
    if fname.endswith(".py"):
        spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(PLUGINS_DIR, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

# Run the bot in an async main function
async def main():
    # Run the HTTP server in the background
    asyncio.get_running_loop().run_in_executor(None, run_http_server)
    await bot.start(my_secret)

# Start the bot
if __name__ == "__main__":
    asyncio.run(main())