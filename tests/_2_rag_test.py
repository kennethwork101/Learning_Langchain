import inspect
import os
from pathlib import Path
from reprlib import repr

import pytest
from dotenv import load_dotenv
from kwwutils import clock, printit
from langchain.schema import Document

from uvprog2025.Learning_Langchain.src.learning_langchain._2_rag import main

load_dotenv()
os.getenv("DATABASE_URL")


""" 
docker pull ankane/pgvector


postgresql+psycopg://langchain:langchain@localhost:6024/langchain
docker run -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
"""


@clock
def test_t1_func(options, model, package_root):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t1"
    options["filename"] = Path(package_root / "data" / "file.txt")
    response = main(options)
    printit(f"{name_} response", response)
    assert "text content" in response[0].page_content.lower()


@clock
def test_t2_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t2"
    options["filename"] = "https://www.langchain.com/"
    response = main(options)
    printit(f"{name_} response", repr(response[0].metadata))
    printit(f"{name_} response metadata title", response[0].metadata["title"])
    assert "langchain" in response[0].metadata["title"].lower()


@clock
def test_t3_func(options, model, package_root):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t3"
    options["filename"] = Path(package_root / "data" / "machinelearning-lecture01.pdf")
    response = main(options)
    printit(f"{name_} 2 response", repr(response[0].page_content))
    assert "machinelearn" in response[0].page_content.lower()


@clock
def test_t4_func(options, model, package_root):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t4"
    options["filename"] = Path(package_root / "data" / "file.txt")
    response = main(options)
    printit(f"{name_} response", response)
    assert "text content" in response[0].page_content.lower()


@clock
def test_t5_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t5"
    response = main(options)
    printit(f"{name_} response", response)
    assert "hello_world" in response[0].page_content.lower()


@clock
def test_t6_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t6"
    response = main(options)
    printit(f"{name_} response", response)
    assert "langchain" in response[0].page_content.lower()


@pytest.mark.parametrize("embedding", ["chroma", "gpt4all", "huggingface"])
@clock
def test_t7_func(options, model, embedding):
    mapping = {
        "chroma": 768,
        "gpt4all": 384,
        "huggingface": 768,
    }
    """
    mapping = {
        "chroma": 384,
        "gpt4all": 384,
        "huggingface": 768,
    }
    """
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["embedding"] = embedding
    if embedding == "huggingface":
        options["embedmodel"] = None
    options["fn"] = "t7"
    response = main(options)
    printit(f"{name_} embedding", embedding)
    printit(f"{name_} response dimensions", len(response[0]))
    assert mapping[embedding] == len(response[0])


@clock
def test_t8_func(options, model, package_root):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["filename"] = Path(package_root, "data/summary.txt")
    options["question"] = (
        "How much is Ford investing in nickel processing facility in Indonesia?"
    )
    options["fn"] = "t8"
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response", response[0].page_content)
    assert "$4.5bn" in response[0].page_content.lower()


@pytest.mark.testme
@clock
def test_t9_func(options, model, vector_store_factory, package_root):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t9"
    collection_name = "test_collection"
    vectordb = vector_store_factory(collection_name)
    docs = [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ]
    printit(f"{name_} docs", docs)
    vectordb.add_documents(docs, ids=[1, 2])
    result = vectordb.similarity_search("cats")
    printit(f"{name_} result", result)
