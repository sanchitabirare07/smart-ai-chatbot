import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_pipeline import RAGPipeline
from utils.chat_history import ChatHistory

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartBot | AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        margin: 0.6rem 0;
        max-width: 80%;
        line-height: 1.6;
    }

    .user-message {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .bot-message {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }

    .source-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366f1;
        color: #a5b4fc;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    }

    .sidebar .stSelectbox label, .sidebar .stSlider label {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()
if "rag" not in st.session_state:
    st.session_state.rag = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_provider = st.selectbox("AI Provider", ["Groq (Free)", "OpenAI"],
        help="Choose the LLM provider"
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-... or AIza...",
        help="Your API key (not stored)"
    )

    if api_provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo"])
    elif api_provider == "Groq (Free)":
        model_name = st.selectbox("Model", ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"])
    else:
        model_name = st.selectbox("Model", ["gemini-2.0-flash"])

    st.markdown("---")
    st.markdown("## 📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload FAQs, manuals, or any documents"
    )

    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    top_k = st.slider("Top K Results", 1, 10, 3)

    if st.button("🚀 Build Knowledge Base", use_container_width=True):
        if not api_key:
            st.error("Please enter your API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Building vector store..."):
                try:
                    rag = RAGPipeline(
                        api_key=api_key,
                        provider=api_provider.lower(),
                        model_name=model_name,
                        chunk_size=chunk_size,
                        top_k=top_k
                    )
                    rag.build_from_uploads(uploaded_files)
                    st.session_state.rag = rag
                    st.session_state.docs_loaded = True
                    st.success(f"✅ Loaded {len(uploaded_files)} document(s)!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = ChatHistory()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card"><b style="color:#a78bfa; font-size:1.4rem">{st.session_state.chat_history.count()}</b><br><small style="color:#94a3b8">Messages</small></div>""", unsafe_allow_html=True)
    with col2:
        status = "🟢" if st.session_state.docs_loaded else "🔴"
        st.markdown(f"""<div class="stat-card"><b style="font-size:1.4rem">{status}</b><br><small style="color:#94a3b8">KB Status</small></div>""", unsafe_allow_html=True)

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 SmartBot</h1>
    <p>AI-Powered Chatbot with Retrieval Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# Chat display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history.get_all():
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            content = msg["content"]
            sources_html = ""
            if msg.get("sources"):
                sources_html = "<br><small>📚 Sources: " + "".join(
                    f'<span class="source-badge">{s}</span>' for s in msg["sources"]
                ) + "</small>"
            st.markdown(f'<div class="chat-message bot-message">🤖 {content}{sources_html}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Input ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Ask anything...",
        placeholder="e.g. What are the admission requirements? / How do I reset my password?",
        label_visibility="collapsed",
        key="user_input"
    )
with col2:
    send_btn = st.button("Send ➤", use_container_width=True)

# ─── FAQ Quick Buttons ────────────────────────────────────────────────────────
st.markdown("**💡 Quick Questions:**")
faq_cols = st.columns(4)
faqs = [
    "What are your office hours?",
    "How do I apply for admission?",
    "What courses are available?",
    "How to contact support?"
]
for i, faq in enumerate(faqs):
    with faq_cols[i]:
        if st.button(faq, key=f"faq_{i}", use_container_width=True):
            user_input = faq
            send_btn = True

# ─── Process Query ────────────────────────────────────────────────────────────
if send_btn and user_input:
    st.session_state.chat_history.add("user", user_input)

    if not st.session_state.docs_loaded or st.session_state.rag is None:
        response = "⚠️ Please upload documents and build the knowledge base first using the sidebar."
        st.session_state.chat_history.add("assistant", response)
    else:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag.query(user_input)
                st.session_state.chat_history.add(
                    "assistant",
                    result["answer"],
                    sources=result.get("sources", [])
                )
            except Exception as e:
                st.session_state.chat_history.add("assistant", f"❌ Error: {str(e)}")

    st.rerun()
