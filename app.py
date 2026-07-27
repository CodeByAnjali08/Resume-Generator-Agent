import streamlit as st
import os
import time
import langchain
import langchain_community
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
import pytesseract as pyt
import numpy as np
from langchain.messages import SystemMessage,HumanMessage
from langchain.agents import create_agent
print("Done")

#=================FRONTEND==================
st.title("AI RESUME GENERATOR")
GOOGLE_API_KEY=st.sidebar.text_input("Google Api Key",type="password")
GROQ_API_KEY=st.sidebar.text_input("Groq Api Key",type="password")
TAVILY_API_KEY=st.sidebar.text_input("Tavily Api Key",type="password")
model1=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)
model2=ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=GROQ_API_KEY
)
#============Agent with tool=============
agent=create_agent(
    model=model1,
    tools=[search_latest_news_jobs]
)
agent
def prompt_generator():
  prompt="""You are a helpful AI Resume
  maker,I want you to use chain-of-thoughts
  and give detailed prompt for model
  where user want to generate resume
  for fresher or experienced one
  in HTML format,you have to give proper
  set of instructions,and make sure to keep
  design proffesional"""
  response=model1.invoke(prompt)
  prompt_ans=response.content[-1]['text']
  file_name='prompt_txt'
  with open(file_name,'w') as f:
    f.write(prompt_ans)
  prompt_generator()
#Final Agent#Tool 2
def prompt_reader():
  with open('prompt_txt','r') as f:
    prompt=f.read()
  return prompt
prompt="""I want complete Professional
Resume with Dynamic Design using advanced CSS and JS
and must show user input details
System instructions:Only Give HTML code as output"""

final_prompt=prompt+prompt_reader()
user_info=st.input("give your information")
user_photo=st.sidebar.file_uploader("upload pic",type="image/jpeg")
user_query=f"""Give Resume for Python Developer.
user details:{user_info}
use user profile image from given {user_photo}"""

final_query=final_prompt+user_query
if st.button("Generate Resume"):
  with st.spinner("Agent creating Resume..."):
    response=agent.invoke({'messages':[{'role':'user',"content":final_query}]})
    code=response['messages'][-1].content[-1]['text']
    st.html(code,width="stretch",unsafe_allow_javascript=True)









