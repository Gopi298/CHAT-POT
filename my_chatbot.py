import os
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("sk-proj-KgBGMBU0dqKfIX--
p0NsPouLd85rvktjrTpAggN8MkbkQ30S2tvhlHw6rfz1KL82pP_XAux9GpT3BlbkFJ4nRrXcl9ElN46ZJYb-u20JenEJFC0yG_RcNNp_lYHW-RK5-qeQ8puh4szX_MNJc_rYedr95goA")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing.")
    st.stop()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="My PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 My PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about the document."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Your Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )


# =========================================================
# PDF PROCESSING
# =========================================================

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):

        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


    # =====================================================
    # CHECK PDF TEXT
    # =====================================================

    if not text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        st.stop()


    st.success("PDF loaded successfully.")


    # =====================================================
    # TEXT SPLITTER
    # =====================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_text(text)


    st.info(
        f"Created {len(chunks)} text chunks."
    )


    # =====================================================
    # EMBEDDINGS
    # =====================================================

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )


    # =====================================================
    # FAISS VECTOR STORE
    # =====================================================

    with st.spinner(
        "Creating document knowledge base..."
    ):

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )


    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )


    # =====================================================
    # OPENAI CHAT MODEL
    # =====================================================

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1500,
        api_key=OPENAI_API_KEY
    )


    # =====================================================
    # PROMPT
    # =====================================================

    system_prompt = """
You are a highly accurate PDF question-answering assistant.

Answer the user's question using ONLY the information
available in the provided document context.

Rules:

1. Do not invent information.
2. Do not make up facts.
3. If the answer is not available in the PDF, say:
   "I could not find this information in the uploaded document."
4. Give a clear and direct answer.
5. If the question requires explanation, explain it step by step.
6. Use the document context as the primary source.

DOCUMENT CONTEXT:

{context}
"""


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            (
                "human",
                "{input}"
            )
        ]
    )


    # =====================================================
    # DOCUMENT CHAIN
    # =====================================================

    question_answer_chain = (
        create_stuff_documents_chain(
            llm,
            prompt
        )
    )


    # =====================================================
    # RETRIEVAL CHAIN
    # =====================================================

    retrieval_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )


    # =====================================================
    # USER QUESTION
    # =====================================================

    user_question = st.text_input(
        "💬 Type your question here"
    )


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    if user_question:

        with st.spinner("Thinking..."):

            response = retrieval_chain.invoke(
                {
                    "input": user_question
                }
            )


        # =================================================
        # ANSWER
        # =================================================

        st.subheader("🤖 Answer")

        st.write(
            response["answer"]
        )


        # =================================================
        # SOURCE DOCUMENTS
        # =================================================

        with st.expander(
            "📚 View source information"
        ):

            source_documents = response.get(
                "context",
                []
            )

            for index, document in enumerate(
                source_documents,
                start=1
            ):

                st.markdown(
                    f"### Source {index}"
                )

                st.write(
                    document.page_content
                )

                st.divider()
