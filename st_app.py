import json
import time

import chromadb
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
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
collection_name = st.text_input(
    label="Collection Name",
    value="my_collection",
    help="The name of the collection in the chroma db",
    placeholder="my_collection",
)
openai_api_key = st.text_input(
    label="OpenAI API Key",
    type="password",
    help="Get an API key from https://platform.openai.com/account/api-keys",
    placeholder="sk-...",
)
rate_limited = st.checkbox("is API Rate Limited?", value=True)


@st.cache_resource
def get_text_splitter():
    """Get the recursive character text splitter instance"""

    return RecursiveCharacterTextSplitter(
        separators=[" ", ".", "?", "!", ";", "\t", "\n"],
        keep_separator=False,
        chunk_size=1000,
        chunk_overlap=0,
    )


@st.cache_resource
def embeddings_function() -> OpenAIEmbeddings:
    """The OpenAI Embeddings function.

    Returns
    -------
    OpenAIEmbeddings
        The OpenAI Embeddings function
    """

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024,
        show_progress_bar=True,
        api_key=SecretStr(openai_api_key),
    )


@st.cache_resource
def get_client() -> Chroma:
    """Create the chroma client and return it.

    Returns
    -------
    Chroma
        The chroma client
    """

    client = chromadb.PersistentClient()
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings_function(),
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

    Returns
    -------
    list[Document]
        The list of documents extracted from the url
    """
    if not recursive:
        loader = WebBaseLoader(
            web_path=url,
            autoset_encoding=False,
            encoding="utf-8",
            raise_for_status=True,
        )
    else:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=2,
            extractor=lambda r_text: BeautifulSoup(r_text, "html.parser").text,
            timeout=10,
            check_response_status=True,
            continue_on_failure=True,
        )
    return loader.load_and_split(text_splitter=get_text_splitter())


def get_file_documents(file_name: str, file_data: bytes) -> list[Document]:
    """
    get_file_documents adds the data to the chroma db

    Parameters
    ----------
    file_name : str
        The name of the file
    file_data : bytes
        The file stream

    Returns
    -------
    list[Document]
        The list of documents extracted from the file
    """
    if file_name.endswith(".txt"):
        texts = get_text_splitter().split_text(file_data.decode("utf-8"))
        return [Document(text, metadata={"source": file_name}) for text in texts]
    if file_name.endswith(".jsonl"):
        json_docs = json.loads(file_data)
        return list(
            map(
                lambda json_doc: Document(json_doc, metadata={"source": file_name}),
                json_docs,
            )
        )
    return []


def add_to_chroma(client: Chroma, docs: list[Document]):
    """
    add_to_chroma Add documents to the chroma db

    Parameters
    ----------
    client : Chroma
        The chroma client
    docs : list[Document]
        The list of documents to add
    """

    if rate_limited:
        batch_size = 10
        with st.status("Adding documents...") as status:
            for i in range(0, len(docs), batch_size):
                try:
                    client.add_documents(docs[i : i + batch_size])
                    time.sleep(0.5)
                    status.markdown(
                        f"Successfully added documents #{i} - {i + batch_size}"
                    )
                except Exception as e:
                    status.markdown(
                        f"Failed to add documents #{i} - {i + batch_size} - {e}"
                    )
            status.markdown("Finished adding documents")
            status.update(label="Finished adding documents", state="complete")
    else:
        client.add_documents(docs)


option = st.tabs(["url", "file"])
with option[0]:  # url
    # inputs
    start_url = st.text_input(
        label="URL",
        help="The URL to add to the chroma db",
        placeholder="https://example.com",
    )
    recursive = st.checkbox("Parse recursively", value=False)

    # output
    if st.button("Generate Chroma Collection", key="url_btn"):
        if not start_url:
            st.error("URL is required", icon="🔗")
        elif not collection_name:
            st.error("Collection name is required", icon="🛍️")
        elif not openai_api_key:
            st.error("OpenAI API key is required", icon="🔑")
        else:
            db = get_client()
            with st.spinner("Fetching data..."):
                docs = get_url_documents(start_url, recursive)
                st.toast("Data fetched", icon="✅")
            with st.spinner("Adding to chroma db..."):
                add_to_chroma(db, docs)
            st.success("Added to chroma db", icon="✅")
            st.balloons()

with option[1]:  # file
    # inputs
    file = st.file_uploader(
        label="File",
        help="Upload a file to add to the chroma db",
        type=["txt", "jsonl"],
        accept_multiple_files=False,
    )

    # output
    if st.button("Generate Chroma Collection", key="file_btn"):
        if not file:
            st.error("File is required", icon="📄")
        elif not collection_name:
            st.error("Collection name is required", icon="🛍️")
        elif not openai_api_key:
            st.error("OpenAI API key is required", icon="🔑")
        else:
            db = get_client()
            with st.spinner("Fetching data..."):
                docs = get_file_documents(file.name, file.read())
                st.toast("Data fetched", icon="✅")
            with st.spinner("Adding to chroma db..."):
                add_to_chroma(db, docs)
            st.success("Added to chroma db", icon="✅")
            st.balloons()
