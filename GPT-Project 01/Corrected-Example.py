## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain



from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic u want")

# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)



## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="when was {name} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')
# Prompt Templates


parent_chain=SequentialChain(
    chains=[chain,chain2],input_variables=['name'],output_variables=['person','dob'],verbose=True)

# parent_chain('Viral Kohli')


if input_text:
    st.write(parent_chain({'name':input_text}))

    
