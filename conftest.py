import inspect
import json
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from kwwutils import get_embeddings, printit
from langchain_postgres.vectorstores import PGVector

load_dotenv()

# ---------------------------------------------------------------------------------------------------
#
# ############ Constants ############
#
# ---------------------------------------------------------------------------------------------------

one_model_file = "models_one.txt"
few_model_file = "models_few.txt"
all_model_file = "models_all.txt"


# ---------------------------------------------------------------------------------------------------
#
# ############ Pytest Functions ############
#
# ---------------------------------------------------------------------------------------------------


def pytest_runtest_teardown(item, nextitem):
    delay = item.config.getoption("--delay")
    if delay:
        time.sleep(delay)


def pytest_terminal_summary(terminalreporter):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    def move_existing_file(base_filename):
        filename = os.path.join(results_dir, base_filename)
        if os.path.exists(filename):
            index = 1
            while os.path.exists(
                os.path.join(results_dir, f"{base_filename.split('.')[0]}_{index}.txt")
            ):
                index += 1
            os.rename(
                filename,
                os.path.join(results_dir, f"{base_filename.split('.')[0]}_{index}.txt"),
            )

    move_existing_file("passed.txt")
    move_existing_file("failed.txt")
    passed_tests = []
    failed_tests = []
    for key in terminalreporter.stats:
        for report in terminalreporter.stats[key]:
            if hasattr(report, "when") and report.when == "call":
                if report.passed:
                    passed_tests.append(report.nodeid)
                elif report.failed:
                    failed_tests.append(report.nodeid)
    with open(os.path.join(results_dir, "passed.txt"), "w") as f:
        f.write("\n".join(passed_tests))
    with open(os.path.join(results_dir, "failed.txt"), "w") as f:
        f.write("\n".join(failed_tests))


def pytest_addoption(parser):
    parser.addoption("--model", action="store", default=None)
    parser.addoption("--run_type", action="store", default=None)
    parser.addoption("--reorder", action="store_true", help="Reorder test items")
    parser.addoption(
        "--delay",
        action="store",
        default=0,
        type=int,
        help="Delay between tests in seconds",
    )


def pytest_collection_modifyitems(config, items):
    """
    If reorder is set then run a model against all tests then repeat using sesequent models.
    Otherwise run each test agains all models before move to next test
    """
    if config.getoption("--reorder"):
        # Group items by model
        groups = {}
        for item in items:
            model_marker = item.callspec.params.get("model", "default")
            groups.setdefault(model_marker, []).append(item)

        # Flatten the grouped items
        ordered_items = [item for group in groups.values() for item in group]

        # Assign the ordered items back to items
        items[:] = ordered_items


def pytest_generate_tests(metafunc):
    model = metafunc.config.getoption("model")
    run_type = metafunc.config.getoption("run_type")
    if model is not None:
        models_name = [model]
    elif run_type is not None:
        model_file = f"models_{run_type}.txt"
        models = models_file(model_file)
        models_name = [m["model"] for m in models]
    else:
        models_name = ["openhermes"]
    printit("models_name", models_name)
    metafunc.parametrize("model", models_name)


#
# ############ Normal Functions ############
#


def models_file(model_file, models_dir="models"):
    name_ = f"{inspect.currentframe().f_code.co_name}"
    print(f"{name_} 1 conftest model_file >{model_file}<")
    dirpath = os.path.dirname(os.path.abspath(__file__))
    print(f"{name_} 2 conftest dirpath >{dirpath}<")
    model_file = os.path.join(dirpath, models_dir, model_file)
    print(f"{name_} 3 conftest model_file >{model_file}<")
    with open(model_file) as fp:
        models = json.load(fp)
    print(f"{name_} 4 models: {models}")
    # Remove models that are known to fail
    skip_models = [
        "falcon",
        "gemma",
        "meditron",
        "medllama2",
        "nexusraven",
        "orca-mini",
        "samantha-mistral",
        "wizard-math",
        "yarn-llama2",
        "yarn-mistral",
        "yi",
    ]

    print(f"{name_} 5 skip_models: {skip_models}")
    models = [m for m in models if m["model"].split(":")[0] not in skip_models]
    print(f"{name_} 6 {models}")
    return models


def models_file_v1(model_file):
    """
    Retrieve the models file
    Note the package root is set in the os.envrion PACKAGE_ROOT so each project must set one with their own key
    """
    name_ = f"{inspect.currentframe().f_code.co_name}"
    package_root = get_package_root()
    models_dir = "models"
    model_file2 = os.path.join(package_root, models_dir, model_file)
    print(f"1 conftest model_file >{model_file}<")
    print(f"2 conftest package_root >{package_root}<")
    print(f"3 conftest model_dirs >{models_dir}<")
    print(f"4 conftest model_file2 >{model_file2}<")
    with open(model_file2) as fp:
        models = json.load(fp)
    print(models)
    return models


def get_package_root():
    return Path(__file__).resolve().parent


#
# ############ Fixtures ############
#


@pytest.fixture
def vector_store_factory():
    dbname = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER", "langchain")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_SRC_PORT")
    # if isinstance(port, tuple):
    #     port = port[0]
    port = int(port)
    printit("dbname", dbname)
    printit("user", user)
    printit("password", password)
    printit("host", host)
    printit("port", port)
    connection = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    options = {}
    options["embedding"] = "chroma"
    options["embedmodel"] = None
    embeddings = get_embeddings(options)
    printit("connection", connection)
    printit("options", options)
    collection_name = "my_collection"

    def create_vector_store(collection_name):
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

    return create_vector_store


@pytest.fixture
def package_root():
    return get_package_root()


@pytest.fixture(params=models_file_v1(one_model_file))
def one_models_arg(request):
    yield request.param


@pytest.fixture(params=models_file_v1(few_model_file))
def few_models_arg(request):
    yield request.param


@pytest.fixture(params=models_file_v1(all_model_file))
def all_models_arg(request):
    yield request.param


@pytest.fixture(scope="session")
def options(request):
    package_root = os.path.dirname(os.path.abspath(__file__))
    print("^+^+^" * 30)
    print(f"package_root {package_root}")
    print(f"request {request}")
    #   print(f"request.module {request.module}")
    print(f"request.node.name {request.node.name}")
    print(f"request.node.fspath {request.node.fspath}")
    option = {
        "temperature": 0.1,
        "embedding": "chroma",
        "embeddings": ["chroma", "gpt4all"],
        "embedmodel": "all-MiniLM-L6-v2",
        "pathname": f"{package_root}/data/data_all/",
        "persist_directory": f"{package_root}/mydb/data_all/",
        "port": 11434,
        "repeatcnt": 1,
        "vectordb_type": "disk",
        "vectorstore": "Chroma",
    }
    return option
