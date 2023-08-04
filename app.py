import os
import streamlit as st
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


_GOOGLE_GENERATIVE_API_KEY = os.environ['GOOGLE_GENERATIVE_API_KEY']


page_icon = "🦜️"
layout = "centered"
page_title = "YouTube Script Generator"
caption_text="By <a href=\"https://github.com/rpatra332\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"mycaption\">Rohit Patra</a>"

st.set_page_config(page_icon=page_icon, page_title=page_title, layout=layout)
st.title('🦜️🔗 YouTube Script Generator', help="Made With LangChain And Google PaLm 2 API")
st.caption(caption_text,unsafe_allow_html=True)
prompt = st.text_input(
    '**Write a topic name for generating youtube title and script?**')


# --- PROMPT TEMPLATES ---
title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a youtube video title about {topic}."
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template="Write me a youtube video script based on this title, TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}."
)


# --- CONVERSATION BUFFER ---
title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')


# --- LLM AND LLM CHAINS ---
llm = GooglePalm(google_api_key=_GOOGLE_GENERATIVE_API_KEY, temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key="title", memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key="script", memory=script_memory)
wiki = WikipediaAPIWrapper()


# --- RESULT VIEW ---
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script: str = script_chain.run(
        title=title, wikipedia_research=wiki_research)
    st.header(f"TITLE: {title}")
    st.divider()
    
    if script.startswith(title):
        st.markdown(script[len(title)+1:])
    else:
        st.markdown(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia History'):
        st.info(wiki_research)


# --- CSS PROPERTIES ---
caption_css_change = """
<style>
    .mycaption{
        color: rgba(250, 250, 250, 0.8) !important;
        text-decoration: none;
    }
</style>
"""
st.markdown(caption_css_change,unsafe_allow_html=True)