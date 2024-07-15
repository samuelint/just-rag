# Just RAG

This library simplifies the process of using Retrieval-Augmented Generation (RAG). Focus on the result you want to achieve and let the library handle the rest.

- Based on LangChain / LangGraph
- Have an unified input/output signature across different RAG Strategies
- Support offline / local inference (through [LLamaCPP](https://github.com/abetlen/llama-cpp-python) & [langchain_llamacpp_chat_model](https://github.com/samuelint/langchain-llamacpp-chat-model)

If you find this project useful, please give it a star ⭐!

## Remote inference

### Classic Rag

```python
from just_rag import ClassicRag
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

chain = ClassicRag(llm=llm, retriever=retriever).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"])
```

### Classic Rag with Citation

```python
from just_rag import CitedClassicRag
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

chain = CitedClassicRag(llm=llm, retriever=retriever).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"].result)
print(result["result"].citations)
```

### Agentic RAG - Self Rag (with Citation)

```python
from just_rag import SelfRagGraphBuilder
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)

chain = SelfRagGraphBuilder(llm=llm, retriever=retriever).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"])
print(result["documents"][0].metadata['title'])
print(result["documents"][0].metadata['source'])
print(result["documents"][0].metadata['summary'])
```

## Local Inference

### Using LLamaCPP & langchain_llamacpp_chat_model

```python
from just_rag import SelfRagGraphBuilder
from langchain_llamacpp_chat_model import LlamaChatModel
from llama_cpp import Llama
from langchain_community.retrievers import WikipediaRetriever

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)
llama = Llama(
    verbose=True,
    model_path=model_path,
    n_ctx=8192,  # Meta-Llama-3-8B has a maximum context size of 8192
    n_batch=512,
    n_gpu_layers=-1,  # -1 is all on GPU
    n_threads=4,
    use_mlock=True,
    chat_format="chatml-function-calling",
)
llm = LlamaChatModel(llama=llama, temperature=0.0)

# The number of retreived documents should be inferior to the local llm context size.
# top_k_results * doc_content_chars_max < n_ctx
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=1000)

chain = SelfRagGraphBuilder(llm=llm, retriever=retriever).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"])
```
