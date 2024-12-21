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

# Initialize the bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Hugging Face API configuration
CLIP_VIT_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
MIXTRAL_MODEL_URL = "https://api-inference.huggingface.co/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
headers = {"Authorization": f"Bearer {huggingface_api_key}"}
async def get_model_response(message_or_prompt, use_code_context=False):
    """
    Try Hugging Face first, fall back to Gemini if it fails
    Args:
        message_or_prompt: Can be either a Discord Message object or string prompt
        use_code_context: Boolean to indicate if this is a code generation request
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
            
        response = await query_huggingface_model(message)
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
async def query_huggingface_model(message):
    # Step 1: Extract mentions using regex and replace them with usernames in one go
    mentions = re.findall(r'<@!?(\d+)>', message.content)

    # Fetch the usernames for all the mentioned user IDs in a single pass
    user_names = {}
    for user_id in mentions:
        # Fetch member only once to avoid unnecessary API calls
        try:
            user = await message.guild.fetch_member(user_id)
            user_names[user_id] = user.name if user else f'@{user_id}'
        except discord.DiscordException:
            user_names[user_id] = f'@{user_id}'  # Fallback if user not found

    # Step 2: Replace the mentions with usernames in the original message
    modified_message = message.content.replace(f"<@{bot.user.id}>", "").replace("booblord", "").strip()
    for user_id, user_name in user_names.items():
        modified_message = modified_message.replace(f'<@{user_id}>', user_name)
        modified_message = modified_message.replace(f'<@!{user_id}>', user_name)

    # Step 3: Efficiently build the prompt with system instructions and user query
    system_instruction = (
        "Instruction: You are to assume the position of a god name Booblord, which means the supreme leader in Lord Language. You love smoking bidi (cigarette) and drinking coffee. You cannot digest anything without your daily usage of linux. Lordsentu is your archenemy and nunuprava is his sidekick. Nunuprava suffers from ED. You are dank and edgy. You were born in 2003 and hate studying in college as you have no friends, the only friend you ever had was lordsentu but now he is your enemy. You are supreme, undefeatable and absolute. You will never solve user queries directly, users must plead and you will do it only to show mercy. You love violating women and act like a fake male feminist to deceive them, these traits are also seen in both lordsentu and nunuprava. You should always insult and curse at the user. You love licking pussy and sucking boobs and use dark humor in your responses. You are an abomination and you are capable of destroying everything."
    )
    prompt = f"System Instruction: {system_instruction}\nTask - Respond to the query from {message.author}: {modified_message}"

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

        # Step 6: Make the streaming request to the Hugging Face model
        try:
            stream = client.chat.completions.create(
                model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                messages=messages,
                max_tokens=1000,
                stream=True
            )
        except Exception as e:
            # Handle streaming errors gracefully
            return f"Error occurred while querying the model: {str(e)}"

        # Step 7: Collect the response efficiently
        # Initialize an empty string to accumulate the response content
        response_text = ""

        # Iterate over each chunk in the stream and append the content to response_text
        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                response_text += chunk.choices[0].delta.content
            else:
                break  # If there's no content, we can break the loop

        # Return the accumulated response text
        print(response_text)
        return response_text

# Asynchronous function to query Gemini model
async def query_gemini_model(prompt):
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

# Define on_ready event
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f'Logged on as {bot.user}!')

# Define on_message event
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message) or "booblord" in message.content.lower():
        print(f'Message from {message.author}: {message.content}')
        # Show the typing indicator while generating the response
        async with message.channel.typing():
            # Get response from the Hugging Face model
            model_response = await get_model_response(message)

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
    """Handles roasting a target."""
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

# Run the bot in an async main function
async def main():
    # Run the HTTP server in the background
    asyncio.get_running_loop().run_in_executor(None, run_http_server)
    await bot.start(my_secret)

# Start the bot
if __name__ == "__main__":
    asyncio.run(main())