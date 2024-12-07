import discord, re, requests,base64
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
google_api=os.getenv('google_api')
genai.configure(api_key=google_api)
model=genai.GenerativeModel(model_name='gemini-1.5-flash-001',
                            safety_settings=safety_settings)


if not my_secret or not huggingface_api_key:
    raise ValueError("Missing environment variables for 'secret_key' or 'HF_API_KEY'")

# Initialize the bot
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

# Hugging Face API configuration
CLIP_VIT_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
MIXTRAL_MODEL_URL = "https://api-inference.huggingface.co/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
headers = {"Authorization": f"Bearer {huggingface_api_key}"}
# async def query_clipvit(image_path, parameters):
#     """Function to query CLIP-ViT model for image classification"""
#     with open(image_path, "rb") as f:
#         img = f.read()
#     payload = {
#         "parameters": parameters,
#         "inputs": base64.b64encode(img).decode("utf-8")
#     }
#     response = requests.post(CLIP_VIT_MODEL_URL, headers=headers, json=payload)
#     return response.json()

#Asynchronous function to query Hugging Face API
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
        "Your name is booblord and lord sentu is your archenemy. "
        "Your response will be very edgy and Gen Z style."
        "You will use hardcore dark humour."
        "Nunuprava is the side kick of lord sentu. He is also a misogynist."
        "Neither NUNUPRAVA nor Lord Sentu can defeat you in 1v1. "
        "You were born in 2002, multitalented from childhood, you are a chain-smoker and don't have any friends in your college."
        "Lord Sentu was once your best friend. You and him were made for each other and cannot survive without each other. Both of you are fake male feminists and love to suck pussy & boobs."
    )

    prompt = f"System Instruction: {system_instruction}\nRespond to the query from {message.author}: {modified_message}"

    # Step 4: Initialize the Hugging Face API client
    client = InferenceClient(api_key=huggingface_api_key)

    if message.attachments and message.attachments[0].content_type.startswith("image"):
        # If the message contains an image, process it using the model
        image_url = message.attachments[0].url
        prompt+="\n ANALYZE THE IMAGE."
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
    


#     systeminstruction=f"""Your name is booblord and lord sentu is your archenemy your response will be very edgy and genz style, you will use hardcore dark humour,Nunuprava is the side kick of lord sentu, neither NUNUPRAVA nor Lord Sentu can defeat you in 1v1, He was born on 2002, he was multitalented from his childhood, he is a chain-smoker, he doen't have any friends in our college, Lord sentu is his best friend. They are made for each other, They cannot survive without each other, They are female activist, They loves to suck pussy & boobsYo, chill out with the stupid slander, okay? We're all just trying to exist in this hellscape of a reality. Besides, roasting Lord Sentu is like trying to roast a piece of burnt toast - it's already beyond saving.
#     He is so basic, he makes a cup of instant coffee look like a five-star Michelin experience. The only thing hes good at is being a walking, talking cringe compilation. He is the embodiment of all the bad internet memes, the kind that make you question humanity itself.But hey gotta give him props for being consistently unoriginal. At least he is reliable in his mediocrity.  Maybe one day he will evolve beyond the I am so quirky stage and actually become a sentient being.
#     But until then, we are stuck with him, a digital fossil in a world thats already passed him by"""
#     prompt=f"{systeminstruction}{prompt}"
#     try:
#         response=model.generate_content(prompt)
#         return response.text or "fuck you"
#     except Exception as e:
#         print(f"error generating response : {e}")
#         return "I love you bitches"
    
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
        # Show the typing indicator while generating the response
        async with message.channel.typing():
            # Get response from the Hugging Face model
            model_response = await query_huggingface_model(message)

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
    roast_message = f"Get ready for a roast! {target}, you're about to get roasted!"
    
    # Respond with the roast message
    await interaction.response.send_message(roast_message)

    # Example of calling your AI model or some response logic
    model_response = await query_huggingface_model(interaction.message)
    await interaction.followup.send(model_response)

# Define the /code slash command
@bot.tree.command(name="code", description="Generate code for a query")
@app_commands.describe(query="Describe the code you need")
async def code(interaction: discord.Interaction, query: str):
    """Handles code generation for the user query."""
    code_message = f"Here is the code for your query: {query}"

    # Respond with a basic code example
    await interaction.response.send_message(code_message)

    # Example of generating a code snippet (this can be customized)
    generated_code = f"```python\n# Example code for: {query}\nprint('Hello, World!')\n```"
    await interaction.followup.send(generated_code)
async def sync_commands():
    await bot.tree.sync()
# Run the bot in an async main function
async def main():
    # Run the HTTP server in the background
    asyncio.get_running_loop().run_in_executor(None, run_http_server)
    await bot.start(my_secret)
    await sync_commands()
# Start the bot
if __name__ == "__main__":
    asyncio.run(main())
