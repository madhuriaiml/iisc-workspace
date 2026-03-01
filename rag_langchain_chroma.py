%%capture
!pip -q install langchain-core
!pip -q install langchain-community
!pip -q install sentence-transformers
!pip -q install langchain-huggingface
!pip -q install langchain-chroma
!pip -q install chromadb
!pip -q install pypdf


import os
import numpy as np
from getpass import getpass
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough




# Download PDFs
!gdown https://drive.google.com/uc?id=1Wy00e_FEBVwMx-jZBklNk9dzEW9a-LHc
!gdown https://drive.google.com/uc?id=1gMv6Ew7oGCPD0CA4D5iN_zAUBWY-SSJQ


import os
from google.colab import userdata

hfapi_key = userdata.get('HF_TOKEN')
os.environ["HF_TOKEN"] = hfapi_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key




#Prepare Open Source LLM

# importing HuggingFace model abstraction class from langchain
from langchain_huggingface import HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",       # Model card: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    task="text-generation",
    max_new_tokens = 512,
    top_k = 30,
    temperature = 0.1,
    repetition_penalty = 1.03,
)

# Specific query
llm.invoke("What is PCA?")




from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    PyPDFLoader("/content/pca_d1.pdf"),
    PyPDFLoader("/content/ens_d2.pdf"),
    PyPDFLoader("/content/ens_d2.pdf"),    # Loading duplicate documents on purpose
]

docs = []
for loader in loaders:
    docs.extend(loader.load())




from langchain_text_splitters import RecursiveCharacterTextSplitter
# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)


splits = text_splitter.split_documents(docs)

print(len(splits))
print(len(splits[0].page_content) )
splits[0].page_content



import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


#Vectorstores
from langchain_chroma import Chroma       # Light-weight and in memory


persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any



vectordb = Chroma.from_documents(
    documents=splits,                    # splits we created earlier
    embedding=embedding,
    persist_directory=persist_directory, # save the directory
)


print(vectordb._collection.count()) # same as number of splits


# Without MMR
question = "What is principal component analysis?"
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(question)
docs


# With MMR
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k":5})
docs = retriever.invoke(question)
docs


from langchain_core.prompts import PromptTemplate                                    # To format prompts
from langchain_core.output_parsers import StrOutputParser                            # to transform the output of an LLM into a more usable format
from langchain.schema.runnable import RunnableParallel
# , RunnablePassthrough          # Required by LCEL (LangChain Expression Language)


# Build prompt
system_prompt = """Use the following pieces of context to answer the question at the end.
If you don't know the answer that is if the answer is not in the context, then just say that you don't know, don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=system_prompt)



#Create Final RAG Chain



def get_context_info(question):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k":5})
    docs = retriever.invoke(question)
    return docs


from langchain_core.runnables import RunnableLambda

retrieval = RunnableParallel(
    {
        "context": RunnableLambda(lambda x: get_context_info(x["question"])),
        "question": RunnableLambda(lambda x: x["question"])
        }
    )



retrieval.invoke({"question": "What is PCA ?"})
retrieval.invoke({"question": "How ensemble methods works?"})


# RAG Chain

rag_chain = (retrieval                     # Retrieval
             | QA_PROMPT                   # Augmentation
             | llm                         # Generation
             | StrOutputParser()
             )


response = rag_chain.invoke({"question": "What is PCA ?"})

response














