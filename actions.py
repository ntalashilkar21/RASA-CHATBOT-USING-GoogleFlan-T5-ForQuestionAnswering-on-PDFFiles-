# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


#pip install -q transformers==4.4.2
#!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import chromadb
from langchain.chains import RetrievalQA
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
# from transformers import pipeline
from typing import Dict, Text, Any
# class ActionRunTransformers(Action):
#     def name(self):
#         return "action_run_transformers"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
#         # Extract the user's message to get the query
#         user_message = tracker.latest_message.get('text')

#         # Ensure that the user provided a query
#         if not user_message:
#             dispatcher.utter_message(text="Please provide a query.")
#             return []

#         # Your code goes here
#         table = pd.read_csv(r"C:\Users\ntalashilkar\downloads\data.csv")
#         table = table.astype(str)

#         tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

#         # Use the user's query
#         query = user_message
#         answer = tqa(table=table, query=query)["answer"]

#         dispatcher.utter_message(text=f"The answer is: {answer}")

#         return []

class ActionRunTransformers(Action):
    def name(self):
        return "action_run_transformers"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        # Extract the user's message to get the query
        user_message = tracker.latest_message.get('text')

      # Ensure that the user provided a query
        if not user_message:
            dispatcher.utter_message(text="Please provide a query.")
            return []
 # Your code goes here
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
        embeddings = HuggingFaceEmbeddings()
        llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":512})
        chain = load_qa_chain(llm, chain_type="stuff")
        pdf_folder_path = 'C:\shreyash\Projects\LLM\content'
        os.listdir(pdf_folder_path)
        loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
        loaders
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)).from_loaders(loaders[0:9])
        llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":1000})
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=index.vectorstore.as_retriever(),
                                            input_key="question")
        # Use the user's query
        query = user_message
        answer = chain.run(query)

        dispatcher.utter_message(text=f" {answer}")

        return []


