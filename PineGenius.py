import os
import discord
from discord.ext import commands
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
import asyncio 

# Initialize the Discord bot with intents
intents = discord.Intents.all()
intents.members = True  
intents.messages = True  

# Specify the allowed server IDs or User IDs
allowed_server_ids = ["123456789987456"]
authorized_user_ids = ["123456748998745"]
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "API-key-here"
openai.api_key = "API-key-here"

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# Initialize the OpenAI model and index (similar to your previous script)
loader = DirectoryLoader("data/")
if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# Define a function to query the OpenAI model for refining and debugging code
def query_openai_model(input_text, engine):
    # Set the initial context based on the engine
    if engine == "davinci-codex" or engine == "gpt-4":
        custom_pretext = "Build indicators for TradingView using PineScript ONLY. Make sure to use proper syntax and inputs when generating code. Only use PineScript. Here is your request : "
    else:
        custom_pretext = ""

    prompt = custom_pretext + "\n" + input_text

    conversation = [
        {"role": "system", "content": custom_pretext},
        {"role": "user", "content": input_text},
    ]

    response = openai.ChatCompletion.create(
        model=engine,
        messages=conversation,
        max_tokens=6700,
    )

    response_text = response['choices'][0]['message']['content']

    return response_text
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 15}),
)


bot = commands.Bot(command_prefix='!', intents=intents)  # You can change the prefix as needed

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

# Function to send a typing indicator and then the response
async def send_with_typing(channel, response):
    async with channel.typing():  # Trigger typing indicator
        await asyncio.sleep(len(response) * 0.02)  # Simulate typing delay
        await channel.send(response)

@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return  # Prevent the bot from responding to its own messages and other bots
    
    if message.guild is not None:  # Check if the message is sent in a server
        if message.mentions and bot.user in message.mentions:
            # Check if the bot is mentioned in the message
            result = chain({"question": message.content, "chat_history": []})
            response = result['answer']
            await send_with_typing(message.channel, response)  # Respond in the same channel
    else:
        # The message is in a direct message (DM)
        result = chain({"question": message.content, "chat_history": []})
        response = result['answer']
        await send_with_typing(message.channel, response)  # Respond in the DM

# Start the bot with your Discord bot token
bot.run("token goes here broski")
