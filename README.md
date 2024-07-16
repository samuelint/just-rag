# Just RAG

This library simplifies the process of using Retrieval-Augmented Generation (RAG). Focus on the result you want to achieve and let the library handle the rest.

- Based on LangChain / LangGraph
- Have an unified input/output signature across different RAG Strategies
- Support offline / local inference through [LLamaCPP](https://github.com/abetlen/llama-cpp-python) & [langchain_llamacpp_chat_model](https://github.com/samuelint/langchain-llamacpp-chat-model)

If you find this project useful, please give it a star ⭐!

## Full Stack Rag

Persist & cync documents when file changes.

```python
builder = JustChromaVectorStoreBuilder(
    collection_name="droits_canadiens",
    file_or_urls=["./tests/assets/Charte canadienne des droits et libertés.html"], # Any file or url
    record_manager_db_url="sqlite:///_record_manager_cache.sql", # Any SQL Alchemy compatible URL
    chroma_persist_directory="./tests_chroma_db", # Any ChromaDB compatible URL
)

retriever = builder.get_retriever()
chain = CitedClassicRag(llm=openai_llm, retriever=retriever).build()

result = chain.invoke(
    {
        "input": "En temps que citoyen, est-ce que j'ai le droit d'entrer et sortir du canada quand je veux? Repondre oui ou non.",
    }
)

assert "non" in result["result"].result.lower()
```

Full example: [tests/test_functional/test_functional_full_stack_rag.py](tests/test_functional/test_functional_full_stack_rag.py)

## Remote inference

### Classic Rag

```python
from just_rag import ClassicRag
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=1000)

chain = ClassicRag(llm=llm, retriever=retriever).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"])
```

Full example: [tests/test_functional/test_functional_classic_rag.py](tests/test_functional/test_functional_classic_rag.py)

### Classic Rag with Citation

```python
from just_rag import CitedClassicRag
from just_rag.citation import BaseCitation, BaseCitedAnswer
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=1000)

class Citation(BaseCitation):
    page_content: str = Field(
        ...,
        description="Page content from the specified source that justifies the answer.",
    )
    title: str = Field(
        ...,
        description="The TITLE quote from the specified source that justifies the answer.",
    )


class CitedAnswer(BaseCitedAnswer[Citation]):
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


chain = CitedClassicRag(llm=llm, retriever=retriever, schema=CitedAnswer).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"].result)
print(result["result"].citations)
```

Full example: [tests/test_functional/test_functional_cited_classic_rag.py](tests/test_functional/test_functional_cited_classic_rag.py)

### Agentic RAG - Self Rag (with Citation)

```python
from just_rag import SelfRagGraphBuilder
from just_rag.citation import BaseCitation, BaseCitedAnswer
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=1000)

class Citation(BaseCitation):
    page_content: str = Field(
        ...,
        description="Page content from the specified source that justifies the answer.",
    )
    title: str = Field(
        ...,
        description="The TITLE quote from the specified source that justifies the answer.",
    )


class CitedAnswer(BaseCitedAnswer[Citation]):
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


chain = SelfRagGraphBuilder(llm=llm, retriever=retriever, schema=CitedAnswer).build()
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["result"])
print(result["documents"][0].metadata['title'])
print(result["documents"][0].metadata['source'])
print(result["documents"][0].metadata['summary'])
print(result["result"].citations[0].source_id)
print(result["result"].citations[0].title)
print(result["result"].citations[0].page_content)
```

Full example: [tests/test_functional/test_functional_self_rag.py](tests/test_functional/test_functional_self_rag.py)

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
