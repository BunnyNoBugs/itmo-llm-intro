# pip install --upgrade --quiet gigachain==0.2.6 gigachain_community==0.2.6 gigachain-cli==0.0.25 duckduckgo-search==6.2.4 pyfiglet==1.0.2 langchain-anthropic llama_index==0.9.40 pypdf==4.0.1 sentence_transformers==2.3.1

import os
from dotenv import load_dotenv
# import getpass
import requests
import json

from langchain.chat_models.gigachat import GigaChat

from langchain.schema import HumanMessage, SystemMessage

from langchain_community.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Union

from langchain.tools import tool
from langchain.agents import AgentExecutor
# from langchain.agents import create_gigachat_functions_agent

from langchain_community.tools import DuckDuckGoSearchRun

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = os.getenv(var)


load_dotenv()
_set_if_undefined("GIGACHAT_CREDENTIALS")


# # 1. GigaChat
# Define GigaChat through langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    # TODO:
    giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
    return giga


def test_giga():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)


# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[Union[SystemMessage, HumanMessage]]:
    # TODO:
    system_message = 'You are a helpful shop assistant'
    return [SystemMessage(content=system_message), HumanMessage(content=user_content)]


# Let's check how it works
def test_prompt():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    user_content = 'Hello!'
    prompt = get_prompt(user_content)
    res = giga(prompt)
    print(res.content)


# ### 3. Define few-shot prompting

# Implement a function to build a few-shot prompt to count even digits in the given number. The answer should be in the format 'Answer: The number {number} consist of {text} even digits.', for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    # TODO:
    prompt_template = """
    Count even digits in the given number. Your answer should begin with "Answer: "
    
    Example: 
    
    Count even digits in the number 441123.
    Answer: The number 441123 consists of three even digits.
    
    Count even digits in the number {number}.        
    """
    prompt = PromptTemplate(input_variables=['number'], template=prompt_template)
    prompt_text = prompt.format(number=number)
    return [HumanMessage(content=prompt_text)]


# Let's check how it works
def test_few_shot():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    number = '11223344'
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    print(res.content)


# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index across your own documents. For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt = """
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        # TODO
        documents = SimpleDirectoryReader(path_to_data).load_data()

        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        Settings.chunk_size = 1024
        Settings.llm = llm

        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        self._query_engine = index.as_query_engine()

    def query(self, user_prompt: str) -> str:
        # TODO
        user_input = self.system_prompt + user_prompt
        response = self._query_engine.query(user_input)
        return response.response


# Let's check
def test_llama_index():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query('what is attention is all you need?')
    print(res)
