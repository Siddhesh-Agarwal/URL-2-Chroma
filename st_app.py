import streamlit as st
from bs4 import BeautifulSoup
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.v1 import SecretStr

st.set_page_config(
    page_title="URL-2-Chroma",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/Siddhesh-Agarwal/URL-2-Chroma/issues",
        "About": open("README.md", mode="r", encoding="utf-8").read(),
    },
)

st.title("URL-2-Chroma")


# fixed inputs
colletion_name = st.text_input(
    label="Collection Name",
    value="my_collection",
    help="The name of the collection in the chroma db",
    placeholder="my_collection",
)
openai_key = st.text_input(
    label="OpenAI API Key",
    type="password",
    help="Get an API key from https://platform.openai.com/account/api-keys",
    placeholder="sk-...",
)

# fixed seperators
SEPERATORS = ["", " ", ".", "?", "!", ",", ";", "\r", "\t", "\n", "\n\n"]

# recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=SEPERATORS,
    keep_separator=False,
    chunk_size=1000,
    chunk_overlap=0,
)


def get_url_documents(url: str, recursive: bool) -> list[Document]:
    """
    get_url_documents scraps the url and adds the data to the chroma db

    Parameters
    ----------
    url : str
        The url to add
    recursive : bool
        Whether to add the url to the chroma db recursively, by default False
    text_splitter : TextSplitter
        The text splitter to use
    """
    if not recursive:
        return WebBaseLoader(
            web_path=url,
            encoding="utf-8",
            raise_for_status=True,
        ).load_and_split(text_splitter=text_splitter)
    else:
        return RecursiveUrlLoader(
            url=url,
            max_depth=2,
            extractor=lambda r_text: BeautifulSoup(r_text, "html.parser").text,
            timeout=10,
            check_response_status=True,
            continue_on_failure=True,
        ).load_and_split(text_splitter=text_splitter)


def get_file_documents(file_name: str, file_data: bytes) -> list[Document]:
    """
    v adds the data to the chroma db

    Parameters
    ----------
    file_name : str
        The name of the file
    file_data : bytes
        The file stream
    """
    if file_name.endswith(".txt"):
        texts = text_splitter.split_text(file_data.decode("utf-8"))
        return [Document(text, metadata={"source": file_name}) for text in texts]
    if file_name.endswith(".jsonl"):
        import json

        json_docs = json.loads(file_data)
        return list(
            map(
                lambda json_doc: Document(json_doc, metadata={"source": file_name}),
                json_docs,
            )
        )
    return []


@st.experimental_singleton
def embeddings_function(openai_api_key: str):
    """The OpenAI Embeddings function"""

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        show_progress_bar=True,
        api_key=SecretStr(openai_api_key),
    )


option = st.tabs(["url", "file"])
with option[0]:
    # inputs
    start_url = st.text_input(
        label="URL",
        help="The URL to add to the chroma db",
        placeholder="https://example.com",
    )
    recursive = st.checkbox("Parse recursively", value=False)
    if st.button("Generate Chroma Collection", key="url_btn"):
        if not start_url:
            st.error("URL is required", icon="🔗")
        elif not colletion_name:
            st.error("Collection name is required", icon="🛍️")
        elif not openai_key:
            st.error("OpenAI API key is required", icon="🔑")
        else:
            client = Chroma(
                collection_name=colletion_name,
                persist_directory="./chroma",
                embedding_function=embeddings_function(openai_key),
            )
            with st.spinner("Fetching data..."):
                docs = get_url_documents(start_url, recursive)
            with st.spinner("Adding to chroma db..."):
                client.add_documents(docs)
            st.success("Added to chroma db", icon="✅")
            st.balloons()

with option[1]:
    file = st.file_uploader(
        label="File",
        help="Upload a file to add to the chroma db",
        type=["txt", "jsonl"],
        accept_multiple_files=False,
    )
    if st.button("Generate Chroma Collection", key="file_btn"):
        if not file:
            st.error("File is required", icon="📄")
        elif not colletion_name:
            st.error("Collection name is required", icon="🛍️")
        elif not openai_key:
            st.error("OpenAI API key is required", icon="🔑")
        else:
            client = Chroma(
                collection_name=colletion_name,
                persist_directory="./chroma",
                embedding_function=embeddings_function(openai_key),
            )
            with st.spinner("Fetching data..."):
                docs = get_file_documents(file.name, file.read())
            with st.spinner("Adding to chroma db..."):
                client.add_documents(docs)
            st.success("Added to chroma db", icon="✅")
            st.balloons()
