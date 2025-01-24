import argparse
import inspect
from typing import Any, Dict

from kwwutils import clock, get_llm, printit
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import chain
from pydantic import BaseModel

""" 
uv add kwwutils
uv add chromadb
uv add langchain
uv add langchain_chroma
uv add langchain_community
uv add langchain_openai
uv add langchain-text-splitters
uv add langchain-postgres
uv add jupyterlab
"""


@clock
def main(options: Dict[str, Any]):
    def fn_t0():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        printit(f"{name_} question", question)
        response = llm.invoke(question)
        return response

    def fn_t1():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        printit(f"{name_} question", question)
        prompt = [HumanMessage(question)]
        response = llm.invoke(prompt)
        return response

    def fn_t2():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        question_sm = options["question_sm"]
        message_sm = SystemMessage(
            question_sm,
        )
        message_hm = HumanMessage(question)
        prompt = [message_sm, message_hm]
        response = llm.invoke(prompt)
        return response

    def fn_t3():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        printit(f"{name_} question", question)
        context = options["context"]
        printit(f"{name_} context", context)
        template = PromptTemplate.from_template(
            """
            Answer the question based on the
            context below. If the question cannot be answered using the information 
            provided, answer with "I don't know".
            Context: {context}
            Question: {question}
            Answer:
           """
        )
        prompt = template.invoke({"context": context, "question": question})
        response = llm.invoke(prompt)
        return response

    def fn_t4():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        system_sm = options["system_sm"]
        context = options["context"]
        printit(f"{name_} question", question)
        printit(f"{name_} system_sm", system_sm)
        printit(f"{name_} context", context)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "system: {system}"),
                ("human", "Context: {context}"),
                ("human", "Question: {question}"),
            ]
        )
        prompt = template.invoke(
            {"system": system_sm, "context": context, "question": question}
        )
        response = llm.invoke(prompt)
        return response

    def fn_t5():
        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str

        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        structured_llm = llm.with_structured_output(AnswerWithJustification)
        response = structured_llm.invoke(question)
        return response

    def fn_t6():
        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str

        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        structured_llm = llm.with_structured_output(
            AnswerWithJustification, include_raw=True
        )
        response = structured_llm.invoke(question)
        return response

    def fn_t7():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        parser = CommaSeparatedListOutputParser()
        response = parser.invoke(question, output_parser=parser)
        return response

    def fn_t8():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        printit(f"{name_} question", question)
        response1 = llm.invoke(question)
        printit(f"{name_} response1", response1)
        response2 = llm.batch([question, "Bye!"])
        for token in llm.stream("Bye!"):
            print(token)
        return response2

    def fn_t9():
        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        system_sm = options["system_sm"]
        context = options["context"]
        printit(f"{name_} question", question)
        printit(f"{name_} system_sm", system_sm)
        printit(f"{name_} context", context)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "system: {system}"),
                ("human", "Context: {context}"),
                ("human", "Question: {question}"),
            ]
        )
        input_ = {
            "system": system_sm,
            "context": context,
            "question": question,
        }
        chain1 = template | llm | StrOutputParser()
        response = chain1.invoke(input_)
        return response

    def fn_t10():
        @chain
        def chatbot(values):
            prompt = template.invoke(values)
            return llm.invoke(prompt)

        name_ = f"{inspect.currentframe().f_code.co_name}"
        printit(f"{name_} options", options)
        question = options["question"]
        system_sm = options["system_sm"]
        context = options["context"]
        printit(f"{name_} question", question)
        printit(f"{name_} system_sm", system_sm)
        printit(f"{name_} context", context)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "system: {system}"),
                ("human", "Context: {context}"),
                ("human", "Question: {question}"),
            ]
        )
        input_ = {
            "system": system_sm,
            "context": context,
            "question": question,
        }
        response = chatbot.invoke(input_)
        return response

    ###########################################################################
    # ### MAIN
    ###########################################################################

    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    mapping = {
        "t0": fn_t0,
        "t1": fn_t1,
        "t2": fn_t2,
        "t3": fn_t3,
        "t4": fn_t4,
        "t5": fn_t5,
        "t6": fn_t6,
        "t7": fn_t7,
        "t8": fn_t8,
        "t9": fn_t9,
        "t10": fn_t10,
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
