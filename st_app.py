import requests
import streamlit as st
from bs4 import BeautifulSoup, Tag
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings


def add_to_chroma_db(url: str, recursive: bool, visited: list[str] = []) -> None:
    """
    add_to_chroma_db scraps the url and adds the data to the chroma db

    Parameters
    ----------
    url : str
        The url to add
    recursive : bool
        Whether to add the url to the chroma db recursively
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

    # get text as documents, use RecursiveCharacterTextSplitter to split into chunks
    text_docs = [
        Document(
            page_content=soup.text.strip().replace("\n", " "),
            metadata=metadata,
        )
    ]
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
url = st.text_input(
    "URL",
    value="https://example.com",
    help="The URL to add to the chroma db",
)
colletion_name = st.text_input("Collection Name", value="default")
openai_key = st.text_input(
    "OpenAI API Key (leave blank to use env var)",
    type="password",
    help="Get an API key from https://platform.openai.com/account/api-keys",
)
recursive = st.checkbox("Recursive", value=False)

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
    if not url:
        st.error("URL is required")
    elif not openai_key and not st.secrets["OPENAI_API_KEY"]:
        st.error("OpenAI API key is required")
    elif not colletion_name:
        st.error("Collection name is required")
    else:
        client = Chroma(
            collection_name=colletion_name,
            persist_directory="./chroma_db",
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-large",
                dimensions=1024,
                show_progress_bar=True,
            ),
        )
        with st.spinner("Adding to chroma db..."):
            add_to_chroma_db(url, recursive)
        st.success("Added to chroma db", icon="✅")
        st.balloons()
