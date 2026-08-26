import os
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured.")
    st.stop()


# ---------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="My PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My PDF Chatbot")
st.write("Upload a PDF and ask questions about the document.")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Your Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )


# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):

        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


    if not text.strip():

        st.error(
            "Could not extract text from this PDF. "
            "The PDF may be scanned/image-based."
        )

        st.stop()


    st.success("PDF loaded successfully.")


    # -----------------------------------------------------
    # TEXT CHUNKING
    # -----------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    st.info(f"Created {len(chunks)} text chunks.")


    # -----------------------------------------------------
    # OPENAI EMBEDDINGS
    # -----------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )


    # -----------------------------------------------------
    # FAISS VECTOR DATABASE
    # -----------------------------------------------------

    with st.spinner("Creating vector database..."):

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )


    # -----------------------------------------------------
    # RETRIEVER
    # -----------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )


    # -----------------------------------------------------
    # CHAT MODEL
    # -----------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1500,
        api_key=OPENAI_API_KEY
    )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful document question-answering assistant.

Answer the user's question using the provided document context.

Rules:

1. Use the document context as the primary source.
2. Do not invent information.
3. If the answer is not available in the document, clearly say:
   "I could not find this information in the uploaded document."
4. Give a clear and accurate answer.
5. If useful, explain the answer step by step.
6. Keep the answer relevant to the user's question.

Document context:

{context}

User question:

{input}
"""
    )


    # -----------------------------------------------------
    # DOCUMENT CHAIN
    # -----------------------------------------------------

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )


    # -----------------------------------------------------
    # RETRIEVAL CHAIN
    # -----------------------------------------------------

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )


    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    user_question = st.text_input(
        "💬 Type your question here"
    )


    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    if user_question:

        with st.spinner("Thinking..."):

            response = retrieval_chain.invoke(
                {
                    "input": user_question
                }
            )


        st.subheader("🤖 Answer")

        st.write(
            response["answer"]
        )


        # -------------------------------------------------
        # SOURCE DOCUMENTS
        # -------------------------------------------------

        with st.expander("📚 View source chunks"):

            documents = response.get(
                "context",
                []
            )

            for i, document in enumerate(
                documents,
                start=1
            ):

                st.markdown(
                    f"**Source {i}**"
                )

                st.write(
                    document.page_content
                )

                st.divider()
