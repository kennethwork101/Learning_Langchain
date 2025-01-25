import argparse
import inspect
from typing import Any, Dict

from kwwutils import clock, get_embeddings, get_llm, printit
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


@clock
def main(options: Dict[str, Any]):
    def fn_t1():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        filename = options["filename"]
        printit(f"{name_} filename", filename)
        loader = TextLoader(filename)
        doc = loader.load()
        return doc

    def fn_t2():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        filename = options["filename"]
        printit(f"{name_} filename", filename)
        loader = WebBaseLoader(filename)
        doc = loader.load()
        return doc

    def fn_t3():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        filename = options["filename"]
        printit(f"{name_} filename", filename)
        loader = PyPDFLoader(filename)
        doc = loader.load()
        return doc

    def fn_t4():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        filename = options["filename"]
        printit(f"{name_} filename", filename)
        loader = TextLoader(filename)
        doc = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = splitter.split_documents(doc)
        return docs

    def fn_t5():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        PYTHON_CODE = """
        def hello_world():
            print("Hello, World!")

        # Call the function
        hello_world()
        """
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=0
        )
        python_docs = python_splitter.create_documents([PYTHON_CODE])
        return python_docs

    def fn_t6():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        markdown_text = """
        # ߦ쯸ﰟ䗠LangChain

        ⚡ Building applications with LLMs through composability ⚡

        ## Quick Install

        ```bash
        pip install langchain
        ```

        As an open source project in a rapidly developing field, we are extremely open 
            to contributions.
        """
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
        )
        md_docs = md_splitter.create_documents([markdown_text])
        return md_docs

    def fn_t7():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        embedding = get_embeddings(options)
        doc = [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!",
        ]
        embeddings = embedding.embed_documents(doc)
        return embeddings

    ###########################################################################
    # ### MAIN
    ###########################################################################

    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    mapping = {
        "t1": fn_t1,
        "t2": fn_t2,
        "t3": fn_t3,
        "t4": fn_t4,
        "t5": fn_t5,
        "t6": fn_t6,
        "t7": fn_t7,
        # "t8": fn_t8,
        # "t9": fn_t9,
        # "t10": fn_t10,
    }
    llm = get_llm(options)
    response = mapping[options["fn"]]()
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="chroma",
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="chat"
    )
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="mistral:instruct")
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    options = Options()
    main(options)
