import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = os.path.join('models', 'gpt4all-converted.bin')
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = "Question: {question}\n\nAnswer: Let's think step by step on it.\n\n"
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who is the best f1 driver? "
llm_chain.run(question)