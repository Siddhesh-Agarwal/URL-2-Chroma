import requests
import streamlit as st
from bs4 import BeautifulSoup, Tag
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.v1 import SecretStr


def add_to_chroma_db(
    url: str,
    recursive: bool = False,
    visited: list[str] = [],
) -> None:
    """
    add_to_chroma_db scraps the url and adds the data to the chroma db

    Parameters
    ----------
    url : str
        The url to add
    recursive : bool
        Whether to add the url to the chroma db recursively, by default False
    visited : list[str], optional
        The visited urls, by default []
    """

    # add to visited
    visited.append(url)

    # get html
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # Create metadata
    ## Add url
    metadata = {"source": url}
    ## add title from title tag
    if soup.title:
        metadata["title"] = soup.title.text
    ## add description from meta description
    res = soup.find("meta", attrs={"name": "description"})
    if isinstance(res, Tag):
        metadata["description"] = res.attrs["content"]

    # get text from page.
    text = soup.text.strip().replace("\n", " ")
    # add it to a document
    text_docs = [Document(page_content=text, metadata=metadata)]
    # use RecursiveCharacterTextSplitter to split into chunks
    text_docs = text_splitter.split_documents(text_docs)
    # add to chroma db
    client.add_documents(text_docs)

    if not recursive:
        return
    for a_tag in soup.find_all("a"):
        if "href" in a_tag.attrs and a_tag.attrs["href"] not in visited:
            add_to_chroma_db(a_tag.attrs["href"], recursive, visited)


st.title("URL-2-Chroma")

# inputs
start_url = st.text_input(
    label="URL",
    help="The URL to add to the chroma db",
    placeholder="https://example.com",
)
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
recursive = st.checkbox("Parse recursively", value=False)

# fixed seperators
SEPERATORS = ["", " ", ".", "?", "!", ",", ";", "\r", "\t", "\n", "\n\n"]
# recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=SEPERATORS,
    keep_separator=False,
    chunk_size=1000,
    chunk_overlap=0,
)

if st.button("Generate Chroma Collection"):
    if not start_url:
        st.error("URL is required", icon="🔗")
    elif not colletion_name:
        st.error("Collection name is required", icon="🛍️")
    elif not openai_key:
        st.error("OpenAI API key is required", icon="🔑")
    else:
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024,
            show_progress_bar=True,
            api_key=SecretStr(openai_key),
        )
        client = Chroma(
            collection_name=colletion_name,
            persist_directory="./chroma_db",
            embedding_function=embedding_function,
        )
        with st.spinner("Adding to chroma db..."):
            add_to_chroma_db(start_url, recursive)
        st.success("Added to chroma db", icon="✅")
        st.balloons()
