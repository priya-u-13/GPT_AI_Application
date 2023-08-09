import time
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
import os
from langchain import OpenAI
from  vectordb import add_docs_to_vectordb
from GPT import summarize,find_topics
# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import chroma
from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )

from langchain.chains.question_answering import load_qa_chain
import selenium
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
driver=webdriver.Chrome()

enterurls=input("enter the urls you want to search comma seperated")
list_in_urls=enterurls.split(",")
print(list_in_urls)
enterkeywords=input("enter keywords comma seperated")
keywords=enterkeywords.split(",")
print(keywords)
enterquery=input("enter query comma seperated")
query=enterquery
print(query)

driver.get(f"http://www.google.com/search?q={query}")
global new
new=[]
for i in driver.find_elements(By.TAG_NAME,"a"):
    if i.text=="News":
        new.append(i)
        break
new[0].click()

news_urls={}

lit=driver.find_elements(By.CLASS_NAME,"WlydOe")
for i in lit:
    news_urls[i.get_attribute("href").split("//")[1].split("/")[0]]=i.get_attribute("href")



final_url=[]
for url in news_urls.keys():
    if url in list_in_urls:
        final_url.append(news_urls[url])

print(final_url)

for i in final_url:#
    news_urls_li=list(news_urls.values())

file_path='./store'

with open(file_path+"/consolidate.txt",'w',encoding="utf-8") as file:
    for i in range(len(final_url)):
        driver.get(final_url[i])
        #print("Link",news_urls_li[i])
        txt=str(driver.find_elements(By.TAG_NAME,"body")[0].text)
        print("Link topics and data")
        summarize_text=str(summarize(txt))
        print("summarized text--------------->",summarize_text)
        lines = summarize_text.split(". ")
        file.write(f"document {i} text------------>")
        file.write('\n')
        for line in lines:
            file.write(line)
            file.write("\n")
        time.sleep(30)


print("Adding the data to vectordb")
p=file_path+"/consolidate.txt"
doc_store=add_docs_to_vectordb(p)




consolidated_text=""
with open(file_path+"/consolidate.txt",'r') as fi:
    consolidated_text=fi.read()
time.sleep(30)
topics=find_topics(consolidated_text,keywords)
topics=str(topics).split(',')
print("topics from summarized text---------->:",topics)
print("filtering --topics------>")

final_topics=[]
for topic in topics:
    topic_words=str(topic).split(" ")
    for keyword in keywords:
        for topic_word in topic_words:
            if (keyword.upper() in topic_words) or (keyword.lower() in topic_words) or (topic_word.lower().startswith(keyword.lower())) or (topic_word.upper().startswith(keyword.upper())):
                final_topics.append(topic)

print(final_topics)


time.sleep(10)
f="final_op.txt"
with open("./store/"+f,'w',encoding="utf-8") as final_file:
    for topic in final_topics:
        temp=f"find the relevant elaboratedsummary of the given {topic}"
        llm=OpenAI(temperature=0.2,openai_api_key="")
        chain=load_qa_chain(llm,chain_type='stuff')
        docs_from_similarity_searh=doc_store.similarity_search(temp)
        content=str(chain.run(input_documents=docs_from_similarity_searh,question=temp))
        print(content)
        time.sleep(25)
        final_file.write("\n")
        final_file.write(f"{topics.index(topic)+1} {topic}------------------->")
        final_file.write("\n")
        time.sleep(5)
        for line in content.split('. '):
            final_file.write(line+"\n")

time.sleep(15)

