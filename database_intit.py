from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

gpt4all_path = './models/gpt4all-converted.bin' 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

def split_chunks(sources):
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    return splitter.split_documents(sources)

def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def similarity_search(query, index):
    matched_docs = index.similarity_search(query, k=3) 
    sources = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in matched_docs]
    return matched_docs, sources

index = FAISS.load_local("my_faiss_index", embeddings)

template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

question = "what is JPC? "
matched_docs, sources = similarity_search(question, index)
context = "\n".join([doc.page_content for doc in matched_docs])
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))