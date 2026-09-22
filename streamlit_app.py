import requests
import streamlit as st

from app.config import CHROMA_PERSIST_DIRECTORY


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="VeriRAG",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "result" not in st.session_state:
    st.session_state["result"] = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🔎 VeriRAG")

st.markdown(
    """
    **Verified Retrieval-Augmented Generation**

    Upload documents and ask questions. VeriRAG retrieves
    relevant information, falls back to web search when
    necessary, and verifies whether the generated answer
    is grounded in the retrieved sources.
    """
)


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

st.subheader("📄 Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files:

    current_filenames = [
        file.name
        for file in uploaded_files
    ]

    previous_filenames = st.session_state["uploaded_files"]

    # Upload only when the selected files change
    if current_filenames != previous_filenames:

        with st.spinner("Processing PDF documents..."):

            files_for_upload = []

            for file in uploaded_files:
                files_for_upload.append(
                    (
                        "files",
                        (
                            file.name,
                            file.getvalue(),
                            "application/pdf",
                        ),
                    )
                )

            try:

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_for_upload,
                    timeout=180,
                )

                if response.status_code == 200:

                    upload_result = response.json()

                    st.session_state["session_id"] = (
                        upload_result["session_id"]
                    )

                    st.session_state["uploaded_files"] = (
                        current_filenames
                    )

                    st.session_state["result"] = None

                    st.success(
                        f"Successfully processed "
                        f"{upload_result['files_processed']} "
                        f"PDF file(s)."
                    )

                else:

                    st.error(
                        f"Upload failed: "
                        f"{response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to the API: {error}"
                )


# --------------------------------------------------
# Active Documents
# --------------------------------------------------

if st.session_state["uploaded_files"]:

    st.subheader("📚 Active Documents")

    for filename in st.session_state["uploaded_files"]:
        st.write(f"• {filename}")

    if st.button("🗑️ Clear Uploaded Documents"):

        st.session_state["session_id"] = None
        st.session_state["uploaded_files"] = []
        st.session_state["result"] = None

        st.rerun()


# --------------------------------------------------
# Question
# --------------------------------------------------

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Ask something about your documents...",
)


if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        payload = {
            "question": question,
        }

        session_id = st.session_state.get("session_id")

        if session_id:
            payload["session_id"] = session_id

        with st.spinner("VeriRAG is thinking..."):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=180,
                )

                if response.status_code == 200:

                    st.session_state["result"] = (
                        response.json()
                    )

                else:

                    st.error(
                        f"Request failed: "
                        f"{response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to the API: {error}"
                )


# --------------------------------------------------
# Display Result
# --------------------------------------------------

result = st.session_state.get("result")


if result:

    st.divider()

    # ----------------------------------------------
    # Reasoning Trace
    # ----------------------------------------------

    st.subheader("🔍 VeriRAG Reasoning Trace")

    trace = result.get("trace", [])

    if trace:

        with st.expander(
            "Show agent execution steps",
            expanded=True,
        ):

            for step in trace:

                if step.startswith("✓"):
                    st.success(step)

                elif step.startswith("✗"):
                    st.error(step)

                elif step.startswith("⚠"):
                    st.warning(step)

                elif step.startswith("↻"):
                    st.info(step)

                else:
                    st.write(step)

    else:

        st.info("No execution trace available.")


    # ----------------------------------------------
    # Grounding Status
    # ----------------------------------------------

    st.subheader("🛡️ Grounding Status")

    if result.get("grounded"):

        st.success(
            "✓ Answer is grounded in the retrieved sources."
        )

    else:

        st.error(
            "✗ Answer could not be fully grounded."
        )


    # ----------------------------------------------
    # Answer
    # ----------------------------------------------

    st.subheader("💡 Answer")

    st.markdown(
        result.get("answer", "")
    )


    # ----------------------------------------------
    # Sources
    # ----------------------------------------------

    sources = result.get("sources", [])

    if sources:

        st.subheader("📚 Sources")

        for source in sources:

            st.write(
                f"• {source}"
            )