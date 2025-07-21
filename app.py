import os
import shutil
import re
import json
import streamlit as st
from rag_pipeline import rag_chain_with_debug, load_and_embed_pdfs

# --- Page Configuration ---
st.set_page_config(page_title="📘 Company Documents Assistant", layout="centered")

# --- Title ---
st.title("📘 Company Policy RAG Assistant")
st.markdown("Upload any company's internal policy documents and ask questions")
st.markdown("- *There are already some PDFs uploaded*")
st.markdown("- *You can upload your own PDFs and ask your own questions to save your time.*")

# --- Upload Section ---
uploaded_files = st.file_uploader("📄 Upload Company Policy PDFs", type="pdf", accept_multiple_files=True)
upload_path = "policy_docs"
os.makedirs(upload_path, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(upload_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("✅ Files uploaded successfully.")

    if st.button("🔄 Re-index Documents"):
        with st.spinner("🔄 Embedding and indexing documents..."):
            load_and_embed_pdfs()
        st.success("📚 Vector database updated successfully.")

# --- Show Uploaded Files ---
st.subheader("📂 Uploaded Documents")
uploaded_doc_list = os.listdir(upload_path)

if uploaded_doc_list:
    for doc in uploaded_doc_list:
        st.markdown(f"• `{doc}`")
else:
    st.info("No documents uploaded yet.")

# --- Utility: Clean LLM Output ---
def extract_json(text):
    """Extract JSON content from LLM response, removing Markdown or code block formatting."""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    return match.group(1).strip() if match else text.strip()

# --- Query Interface ---
st.subheader("💡 Ask a Question")
query = st.text_input("e.g., How can I report unethical behavior anonymously?")

if st.button("🔍 Ask"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🤖 Thinking..."):
            result = rag_chain_with_debug(query)
            raw_answer = result.get("response", "")
            retrieved_chunks = result.get("chunks", "")
            prompt_used = result.get("raw_prompt", "")
            cleaned_answer = extract_json(raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer))

        st.markdown("### 💬 Answer")
        try:
            parsed = json.loads(cleaned_answer)
            st.markdown(f"**Answer:** {parsed.get('Answer', cleaned_answer)}")
        except:
            st.markdown(cleaned_answer)

        with st.expander("📄 Retrieved Context"):
            st.code(retrieved_chunks)

        
