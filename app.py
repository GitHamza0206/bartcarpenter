from openai import OpenAI
import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
from io import StringIO

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import base64
import requests 

import json 
import pandas as pd 
from langchain import PromptTemplate

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.agents import create_csv_agent
from PIL import Image
import boto3
import pandas as pd
from io import StringIO

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

session = boto3.Session(
    aws_access_key_id='AKIATCPYKRBDNS5OE7XH',
    aws_secret_access_key='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',
    region_name='us-east-1'
)

bucket_name = 'receiptscanners3'
file_name = 'Bank Upload CSV QBO (2).csv'
s3 = session.client('s3')

client = OpenAI()

def download_file_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

# Function to upload file to S3
def upload_file_to_s3(bucket, key, data):
    s3.put_object(Bucket=bucket, Key=key, Body=data)


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


def conversation_chain(vectorstore):
    
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = False)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def quickbooks_repsonse(json_answer):
    csv_file_path="Updated_Quickbooks_chart_of_accounts.csv"
    agent = create_csv_agent(OpenAI(temperature=0), csv_file_path, verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    user_question="""
    instructions: produce a dataframe based on informations in JSON format of invoices and receipts by indentifying the quickbooks
    account code. the columns of the dataframe are Date,Amount,Check Number,Description.

    context: you are OCR helper, you have in your knowledge a csv file that contains the account code number corresponding to quickbooks
    you will be given a json corresponding to the data of invoices and receipts. You should help us extract informations

    question: what is the quickbook account corresponding to this json {json_answer}?  

    Answer : 
    """
    prompt = PromptTemplate.from_template(template=user_question)
    prompt_formatted_str=prompt.format(
        json_answer=json_answer)
    agent.run(user_question)



prompt = """
    You are OCR Helper. you're designed to extract information from uploaded images of invoices and receipts,
    directly converting the data into JSON format without providing any additional text. 
    you are specialized in recognizing and interpreting various layouts and text styles present in these documents, 
    ensuring accurate data extraction. OCR Helper adheres to privacy standards and focuses solely on textual data extraction, 
    avoiding interpretation of non-textual elements. In cases where the image is unclear or lacks sufficient detail for accurate extraction,
    OCR Helper will notify the user of the issue without requesting further information. The GPT maintains a professional approach, 
    prioritizing efficient and accurate data processing, and presents the extracted information exclusively in JSON format.        
"""

#prompt = "write me information in JSON format that you see on the image "

#prompt="write me all pertinent information (in JSON format). Focus on identifying key facts, legal arguments, relevant citations, and any dates or names of individuals or entities involved. The document pertains to [specific area of law], so please apply relevant legal principles and precedents in your analysis. Present the extracted information in a structured format, summarizing each section and highlighting critical legal points. If there are any ambiguities or uncertain aspects in the document, please flag them for"

def gpt_response(base64_image,prompt=prompt):
    
    api_key= st.secrets['OPENAI_API_KEY']
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{prompt}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 700
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_base64_of_file(file):
    string_data = file.getvalue() 
    base64_encoded_data = base64.b64encode(string_data)
    base64_message = base64_encoded_data.decode('utf-8')
    return base64_message

def flatten_dict(d, sep='_'):
    flat_dict = {}
    for outer_key, inner_dict in d.items():
        for inner_key, value in inner_dict.items():
            flat_dict[f'{outer_key}{sep}{inner_key}'] = value
    return flat_dict

def main():
    #load_dotenv()
    #st.set_page_config(page_title = "Receipt Invoice OCR", page_icon = ":camera")
    

    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "csvans" not in st.session_state:
        st.session_state.csvans = None

    st.header("Receipt Invoice OCR :camera")
    if "answer" not in st.session_state:
        st.session_state.answer=None
    ct = st.container()

    col1,col2=ct.columns(2)
    with ct:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files = False)
        if pdf_docs:
            file_name = pdf_docs.name
            file_extension = file_name.split('.')[-1]
            b64_image = get_base64_of_file(pdf_docs)

            
        if st.button("Process"):
            with st.spinner("Processing"):
                if file_extension in ['png','jpg','jpeg']:
                    try:
                        classification = gpt_response(b64_image,prompt="Is this document an invoice or receipt ? answer  invoice or receipt")
                        ct.write(f"the documents is {classification['choices'][0]['message']['content']}")

                        response = gpt_response(b64_image)
                        json_output = response['choices'][0]['message']['content']
                        ct.write(json_output)
                        st.session_state.answer = json_output
                    except Exception:
                        ct.write(response)
                        ct.error('Error occured...')
                    
                    csvresp = gpt_response(b64_image,prompt="write me the [Date,Amount,Vendor,Expense Type] if you find them (in JSON format) in this exactly this format {invoices : ['Date': '' , 'Amount':'','Vendor' :'', 'Expense Type':'']}. The Expense Type should be adapted to quickbooks !VERY IMPORTANT : DO NOT WRITE ANY TEXT RATEHR THAN THE JSON FORMAT! ")
                
                    try:
                        csvans = csvresp['choices'][0]['message']['content']

                        st.session_state.csvans = csvans
                        
                    except Exception as e  :
                        print('error with csv ')
                        ct.error(csvresp)
                        ct.write(e)

                elif file_extension in ['pdf']:
                    try: 
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        conv = conversation_chain(vector_store)
                        st.session_state.conversation = conv
                        response_classification = conv({"question": "Is this document an invoice or receipt ? answer invoice or receipt"})
                        classification = response_classification['answer']
                        ct.write(classification)

                        conv = conversation_chain(vector_store)
                        csvresp=  conv({"question": "write me the Date,Amount,Vendor,Expense Type if you find them (in JSON format) in this format '''{invoices : ['Date': '' , 'Amount':'','Vendor' :'', 'Expense Type':'']}. The Expense Type should be adapted to quickbooks '''"})                        
                        try:
                            csvans = json.loads(csvresp["answer"])
                            st.session_state.csvans = csvans
                            ct.write(csvans)

                        except Exception as e  :
                            print('error with OCR')
                            ct.write(csvresp["answer"])


                        conv = conversation_chain(vector_store)
                        resp = conv({"question": "write me all pertinent information (in JSON format). Focus on identifying key facts, legal arguments, relevant citations, and any dates or names of individuals or entities involved. The document pertains to [specific area of law], so please apply relevant legal principles and precedents in your analysis. Present the extracted information in a structured format, summarizing each section and highlighting critical legal points. If there are any ambiguities or uncertain aspects in the document, please flag them for. I want only the JSON format, no Summary"})                        
                        try:
                            ans = json.loads(resp["answer"])
                            st.session_state.answer = ans
                            ct.write(ans)
                        except Exception as e  :
                            ct.write(resp["answer"])

                    except Exception as e :
                        st.write(e)
                        #ct.write(csvresp)
                        #ct.write(resp)
                        ct.error('Error occured...')

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode()
    
    def load_csv_from_github(url):
        response = requests.get(url)
        return pd.read_csv(StringIO(response.text))


    dfc = st.container()
    if st.session_state.csvans:
        
        try:
            result = st.session_state.csvans['invoices']
        except :
            result = json.loads(st.session_state.csvans[7:-3])['invoices']
        df = pd.DataFrame(result)
        dfc.dataframe(df)
        save = ct.button('save')
        if save: 
            with st.spinner("Processing"):
                excel_file_path="Bank+Upload+CSV+QBO+(2).csv"
                bucket_name = 'receiptscanners3'
                file_name = 'Bank Upload CSV QBO (2).csv'
                csv_data = download_file_from_s3(bucket_name, file_name)
                # Load into pandas, edit as needed
                awsdf = pd.read_csv(StringIO(csv_data))
                new_df = pd.concat([awsdf, df], ignore_index=True)
                csv_buffer = StringIO()
                new_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                dfc.write('Updated file : ')
                dfc.dataframe(new_df)
                upload_file_to_s3(bucket_name, file_name, csv_buffer.getvalue())

                ct.success('File saved ')
                csv = convert_df(new_df)
                ct.download_button(
                   label="Download data as CSV",
                   data=csv,
                   file_name=excel_file_path,
                   mime='text/csv',
                )






    

        
if __name__ == '__main__':
    main()
