import inspect

from kwwutils import clock, printit

from uvprog2025.Learning_Langchain.src.learning_langchain._1_misc import main


@clock
def test_t0_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t0"
    options["question"] = "The sky is"
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response.content)
    result = response.content.lower().strip()
    for color in ["blue", "limit", "region"]:
        if color in result:
            break
    else:
        assert color in result


@clock
def test_t1_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t1"
    options["question"] = "What is the capital of France?"
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response.content)
    assert "paris" in response.content.lower()


@clock
def test_t2_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t2"
    options["question"] = "What is the capital of France?"
    options["question_sm"] = """ 
    You are a helpful assistant that responds to questions with three 
    exclamation marks.'
    """
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response.content)
    assert "paris" in response.content.lower()


@clock
def test_t3_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t3"
    # options["question"] = "Which model providers offer LLMs?"
    options["question"] = "List all the model providers that offer LLMs?"
    options["context"] = """ 
        The most recent advancements in NLP are being driven by Large 
        Language Models (LLMs). These models outperform their smaller 
        counterparts and have become invaluable for developers who are creating 
        applications with NLP capabilities. Developers can tap into these 
        models through HuggingFace's `transformers` library, or by utilizing 
        OpenAI and Cohere's offerings through the `openai` and `cohere` 
        libraries, respectively.
    """
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response.content)
    assert any(
        phrase in response.content.lower() for phrase in ["huggingface", "hugging face"]
    )
    assert "openai" in response.content.lower()
    assert "cohere" in response.content.lower()


@clock
def test_t4_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t4"
    options["question"] = "Which model providers offer LLMs?"
    options["system_sm"] = """ 
        Answer the question fully and completely as possible based on the context below. If the question 
        cannot be answered using the information provided, answer with "I don\'t 
        know".'
    """
    options["context"] = """ 
        The most recent advancements in NLP are being driven by Large 
        Language Models (LLMs). These models outperform their smaller 
        counterparts and have become invaluable for developers who are creating 
        applications with NLP capabilities. Developers can tap into these 
        models through HuggingFace's `transformers` library, or by utilizing 
        OpenAI and Cohere's offerings through the `openai` and `cohere` 
        libraries, respectively.
    """
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response.content)
    assert any(
        phrase in response.content.lower() for phrase in ["huggingface", "hugging face"]
    )
    assert "openai" in response.content.lower()
    assert "cohere" in response.content.lower()


@clock
def test_t5_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t5"
    options["question"] = "What weighs more, a pound of bricks or a pound of feathers"
    response = main(options)
    printit(f"{name_} response", response)


@clock
def test_t6_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t6"
    options["question"] = "What weighs more, a pound of bricks or a pound of feathers"
    response = main(options)
    printit(f"{name_} response", response)
    assert response["parsed"] is not None


@clock
def test_t7_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t7"
    question = "apple, banana, cherry"
    options["question"] = question
    response = main(options)
    question = question.split(",")
    question = [q.strip() for q in question]
    printit(f"{name_} question", question)
    printit(f"{name_} response", response)
    assert sorted(question) == sorted(response)


@clock
def test_t8_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t8"
    options["question"] = "Hi there!"
    response = main(options)
    printit(f"{name_} response", response)
    printit(f"{name_} response content", response[-1].content)
    assert "bye" in response[-1].content.lower()


@clock
def test_t9_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t9"
    options["question"] = "Which model providers offer LLMs?"
    options["system_sm"] = """
        Answer the question fully and completely as possible based on the context below.
        If the question cannot be answered using the information provided,
        answer with "I don\'t know".'
    """
    options["context"] = """
        The most recent advancements in NLP are being driven by Large
        Language Models (LLMs). These models outperform their smaller
        counterparts and have become invaluable for developers who are creating
        applications with NLP capabilities. Developers can tap into these
        models through HuggingFace's `transformers` library, or by utilizing
        OpenAI and Cohere's offerings through the `openai` and `cohere`
        libraries, respectively.
    """
    response = main(options)
    printit(f"{name_} response", response)
    assert any(phrase in response.lower() for phrase in ["huggingface", "hugging face"])
    assert "openai" in response.lower()
    assert "cohere" in response.lower()


# @pytest.mark.testme
@clock
def test_t10_func(options, model):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    printit(f"{name_} options", options)
    printit(f"{name_} model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["fn"] = "t10"
    options["question"] = "Which model providers offer LLMs?"
    options["system_sm"] = """
        Answer the question fully and completely as possible based on the
        context below.  If the question cannot be answered using the information
        provided, answer with "I don\'t know".'
    """
    options["context"] = """
        The most recent advancements in NLP are being driven by Large
        Language Models (LLMs). These models outperform their smaller
        counterparts and have become invaluable for developers who are creating
        applications with NLP capabilities. Developers can tap into these
        models through HuggingFace's `transformers` library, or by utilizing
        OpenAI and Cohere's offerings through the `openai` and `cohere`
        libraries, respectively.
    """
    response = main(options)
    printit(f"{name_} response", response)
    response = response.content
    assert any(phrase in response.lower() for phrase in ["huggingface", "hugging face"])
    assert "openai" in response.lower()
    assert "cohere" in response.lower()
