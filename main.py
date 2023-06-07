### This file is used to ask questions to the document and get answers from it

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone
import openai
import os
import streamlit as st

### Functions
# Initialize OpenAI API key and environment
def init_openai_key(OPENAI_API_KEY):
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # TODO: Figure out if this is necessary - likely not in web app but might be necessary in CLI

# Initialize Pinecone API key and environment
def init_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME):
    # Set Pinecone API key and environment
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # Set Pinecone Index
    index = pinecone.Index(PINECONE_INDEX_NAME)
    return index

# Initialize OpenAI Embeddings Engine
def init_openai_embeddings(OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL):
    openai_embedding_engine = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDINGS_MODEL)
    return openai_embedding_engine

# Set up OpenAI Language Chat Model
def init_openai_llm(OPENAI_LLM_MODEL, temperature=0, max_tokens=500): # TODO: Not sure if adding in values into temperature and max_tokens will work or if parameters will need to be left blank
    llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=temperature, max_tokens=max_tokens)
    return llm

# Generate response function
def generate(prompt, index, llm, openai_embedding_engine):

    vectorstore = Pinecone(index, openai_embedding_engine.embed_query, "text")

    ## Perform the search based on the user's query and retrieve the relevant sources
    ## TODO: Figure out how to get this to work with the chat model and have it still work with multiple prompts (like a conversation)
    #returnedDocs = vectorstore.similarity_search(prompt, k=3) #Returns the top 3 documents that are most similar to the query
    #print(returnedDocs)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()) # TODO: Figure out how this works & how many documents it returns
    return qa.run(prompt)



# TODO: This function is not used in web app, but is used in CLI
def api_from_env_file():
    import dotenv
    import os

    ### Initialize the API keys and environment variables

    ## Get API Keys from .env file:
    dotenv.load_dotenv(dotenv.find_dotenv())

    ## Set API Keys:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    return OPENAI_API_KEY, OPENAI_LLM_MODEL, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME



### Main Code
def main():
    ### Streamlit Web UI
    st.set_page_config(page_title="Content Creator GPT", page_icon="🧠")
    st.header("LinkedIn Post Generator")

    ## CSS Hacking to make the Streamlit UI fonts look bigger
    st.markdown(
        """
        <style>
        div[class*="stTextArea"] label p{
        font-size: 1.5rem;
        font-family: Roboto, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,)
    


    ## Settings and API Section
    # Columns
    col1, col2 = st.columns(2)

    # column 1
    with col1:
        st.subheader("Settings")
        st.markdown("*Sliders for making fine adjustment and other settings will be soon added here*")
        st.markdown("*Max output tokens = 500 for now to conserve API usage costs*")
        st.markdown("*Model Temperature(i.e. Creativity) = 0 for now so that it doesn't deviate from content stored in Pinecone*")


    # column 2
    with col2:
        st.subheader("API Keys")
        #Get API Keys from user input
        OPENAI_API_KEY = st.text_input(label="OpenAI API Key", placeholder="Enter OpenAI API Key")
        OPENAI_LLM_MODEL = st.text_input(label="OpenAI Language Model", placeholder="Enter OpenAI Language Model - gpt3.5 turbo or gpt4", value="gpt-3.5-turbo")
        OPENAI_EMBEDDINGS_MODEL = st.text_input(label="OpenAI Embeddings Model", placeholder="Enter OpenAI Embeddings Model", value="text-embedding-ada-002")
        PINECONE_API_KEY = st.text_input(label="Pinecone API Key", placeholder="Enter Pinecone API Key")
        PINECONE_ENVIRONMENT = st.text_input(label="Pinecone Environment", placeholder="Enter Pinecone Environment")
        PINECONE_INDEX_NAME = st.text_input(label="Pinecone Index Name", placeholder="Enter Pinecone Index Name")

        # Initialize API Keys
        try:
            init_openai_key(OPENAI_API_KEY)
        except:
            st.error("Please enter a valid OpenAI API Key")
        
        try:
            index = init_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME)
        except:
            st.error("Please enter a valid Pinecone API Key, Environment, and Index Name")
        
        try:
            openai_embedding_engine = init_openai_embeddings(OPENAI_API_KEY, OPENAI_EMBEDDINGS_MODEL)
        except:
            st.error("Please enter a valid OpenAI Embeddings Model")

        try:
            llm = init_openai_llm(OPENAI_LLM_MODEL)
        except:
            st.error("Please enter a valid OpenAI Language Model")
        


    # Adding a bit of whitespace
    st.markdown("#")


    ## Prompt Section
    # Define prompt input box
    def get_input_prompt():
        prompt_input = st.text_area(label="Prompt", placeholder="Enter prompt here", height=150, key="prompt_input")
        return prompt_input

    # Creates the prompt input box & sets the prompt_input variable to the value of the input box
    prompt_input = get_input_prompt()
    


    ## Submit Button
    # Function to run when the submit button is pressed
    # This will only work when placed within [if button_pressed:] after the prompt_input, index, llm, and openai_embedding_engine variables are defined
    def on_click_submit():
        # Sends prompt_input to OpenAI using API and returns the generated output
        llm_output = generate(prompt_input, index, llm, openai_embedding_engine)
        # Adds the generated output into the output box
        st.session_state.output = llm_output

    # Creates the submit button
    button_pressed = st.button(label="Submit")

    # If the submit button is pressed, then run the code below
    if button_pressed:
        on_click_submit()



    ## Output Section
    # Creates empty output box
    st.text_area(label="Output", height=450, key="output")



# Run the main function
if __name__ == "__main__":
    main()