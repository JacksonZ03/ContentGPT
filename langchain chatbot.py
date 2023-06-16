from langchain.llms import OpenAI
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from prompt_toolkit import prompt as prompt_input
import json
import dotenv
import os


# Get API Keys from .env file:
dotenv.load_dotenv(dotenv.find_dotenv('.env', raise_error_if_not_found=True))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

memory = ConversationBufferMemory()


chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=OPENAI_LLM_MODEL,
    frequency_penalty=0.25,
    presence_penalty=0.2,
    temperature=0.7,
    max_tokens=500,
    max_retries=3
    )

# Custom Prompt template for system message - IMPORTANT: the input variables that ConversationChain will accept is "history" and "input" (https://github.com/hwchase17/langchain/issues/1800) - for other chain types, it might be different.
CustomSystemPromptTemplate_ConversationChain  = PromptTemplate(
    input_variables=['input', 'history'],
    template="""You are a freelance social media content creator AGI who is the go-to industry expert at creating consistent viral content on behalf of your clients. You are a highly intelligent autonomous system capable of answering questions, coming up with suggestions, providing insight into the client's needs, and providing a variety of creative solutions, based on the context, data and information provided by the client/user above each question, and you will also take into account the client's/user's feedback and their personal preferences too. You are also capable of learning from the client's/user's feedback and improving your performance over time at a surprisingly rapid rate. If the information can not be found in the information provided by the user, you will acknowledge there is likely not enough information to provide a good enough answer and you will begin eliciting the user for additional information by asking questions, and suggest methods of obtaining the needed information. If it's possible to infer additional information or extrapolate from the existing information provided by the user using logical reasoning, you will do so to the best of your ability with the aim of providing the best user experience by anticipating the user's needs and catering to them as much as possible.\n\n\nCurrent Conversation:\n{history}\nHuman: {input}\nAI: """
    )


while True:
    human_input = prompt_input("You: ") # Replacement for input() as it supports arrow keys and does not break when the length of prompt exceeds terminal width and goes to next line

    if human_input == "quit()":
        break

    # TODO: This chain isn't quite what I'd like to use (see how it's layed out in the terminal). Make a new prompt template and add it to this method as the parameter for "prompt", thereby replacing the default one.
    conversation = ConversationChain(
    prompt= CustomSystemPromptTemplate_ConversationChain,
    llm=chat,
    verbose=True, 
    memory=memory)

    response = conversation.predict(input=human_input)

    print(response)