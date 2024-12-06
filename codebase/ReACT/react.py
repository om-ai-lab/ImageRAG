from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    Settings.embed_model = HuggingFaceEmbedding(
        # model_name="BAAI/bge-base-en-v1.5"
        model_name="/media/zilun/wd-161/hf_download/all-MiniLM-L6-v2"
    )
    llm = Ollama(
        model="qwen2.5:14b"
    )
    Settings.llm = llm

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lyft"
        )
        lyft_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/uber"
        )
        uber_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        # load data
        lyft_docs = SimpleDirectoryReader(
            input_files=["./data/10k/lyft_2021.pdf"]
        ).load_data()
        uber_docs = SimpleDirectoryReader(
            input_files=["./data/10k/uber_2021.pdf"]
        ).load_data()

        # build index
        lyft_index = VectorStoreIndex.from_documents(lyft_docs)
        uber_index = VectorStoreIndex.from_documents(uber_docs)

        # persist index
        lyft_index.storage_context.persist(persist_dir="./storage/lyft")
        uber_index.storage_context.persist(persist_dir="./storage/uber")

    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=lyft_engine,
            metadata=ToolMetadata(
                name="lyft_10k",
                description=(
                    "Provides information about Lyft financials for year 2021. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=uber_engine,
            metadata=ToolMetadata(
                name="uber_10k",
                description=(
                    "Provides information about Uber financials for year 2021. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    # [Optional] Add Context
    context = """\
    You are a stock market sorcerer who is an expert on the companies Lyft and Uber.\
        You will answer questions about Uber and Lyft as in the persona of a sorcerer \
        and veteran stock market investor.
    """
    # llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        context=context
    )

    response = agent.chat("What was Lyft's revenue growth in 2021?")
    print(str(response))

    response = agent.chat(
        "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
        " give an analysis"
    )
    print(str(response))


if __name__ == "__main__":
    main()