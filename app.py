import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import json
import re

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Regulatory Q&A Assistant",
    page_icon="🏦",
    layout="centered"
)

# ── Load URL Database ────────────────────────────────────────
@st.cache_data
def load_database():
    url  = "https://raw.githubusercontent.com/RareEyes/regulatory-qa-assistant/refs/heads/main/regulations_db.json"
    resp = requests.get(url)
    return resp.json()

# ── Scrape One Regulation Page ───────────────────────────────
def scrape_page(url):
    """Scrape full content from URL"""
    try:
        headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        soup     = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text
    except:
        return ""


def find_relevant_chunk(content, question, chunk_size=5000, overlap=500):
    """
    Split content into chunks
    Find the most relevant chunk for the question
    Return top 3 chunks combined
    """
    # Get question keywords
    words = [
        w for w in re.findall(r"[a-z0-9]+", question.lower())
        if w not in STOP_WORDS and len(w) > 2
    ]

    if not words or not content:
        return content[:15000]

    # Split into overlapping chunks
    chunks = []
    start  = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append({
            "text":  content[start:end],
            "start": start
        })
        start += chunk_size - overlap

    # Score each chunk
    scored = []
    for chunk in chunks:
        text  = chunk["text"].lower()
        score = sum(
            (1 + text.count(w)) * (2 if w in text else 0)
            for w in words
        )
        scored.append((score, chunk))

    # Sort by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top 3 chunks
    top_chunks = [c for s, c in scored[:3] if s > 0]

    if not top_chunks:
        return content[:15000]

    # Sort by position so text flows naturally
    top_chunks.sort(key=lambda x: x["start"])

    return "\n\n".join(c["text"] for c in top_chunks)

# ── Find Relevant Regulations ────────────────────────────────
STOP_WORDS = set(
    "a an the of to in on for and or is are what who how "
    "when which do does be by with as at from that this it "
    "its can must should may will shall any all not have has "
    "been if their they them there than then also such into "
    "only more most other please tell about explain".split()
)

def score_regulation(question, title, url):
    words = [
        w for w in re.findall(r"[a-z0-9]+", question.lower())
        if w not in STOP_WORDS and len(w) > 2
    ]
    if not words:
        return 0
    text  = (title + " " + url).lower()
    score = sum(1 for w in words if w in text)
    return score

def find_relevant_urls(question, regulations, top=5):
    scored = []
    for reg in regulations:
        s = score_regulation(question, reg["title"], reg["url"])
        scored.append((s, reg))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_regs = [reg for s, reg in scored[:top] if s > 0]
    if not top_regs:
        top_regs = [scored[0][1]]
    return top_regs

# ── Ask Gemini ───────────────────────────────────────────────
def ask_gemini(question, context, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a compliance expert answering questions about regulatory documents.

STRICT RULES:
- Use ONLY the text provided below.
- If answer not present say: "This information is not found in the selected regulations."
- Do NOT invent or guess anything.
- Start with one direct answer sentence.
- Then extract EVERY detail as separate bullet points.
- Copy exact definitions, timeframes and obligations word for word.
- End with Sources listing regulation names used.

REGULATION TEXT:
{context}

QUESTION: {question}

ANSWER:"""

    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-flash-lite-latest",
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            continue

    return "❌ AI service unavailable. Please try again later."

# ── Main App ─────────────────────────────────────────────────
def main():

    # ── Hide Streamlit Default UI ────────────────────────────
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────
    st.title("🏦 Regulatory Q&A Assistant")
    st.caption("Ask questions about financial regulations")

    # ── Load Database ────────────────────────────────────────
    try:
        db = load_database()
    except:
        st.error("Failed to load regulations database.")
        return

    # ── API Key ──────────────────────────────────────────────
    api_key = st.secrets["GEMINI_API_KEY"]

    st.divider()

    # ── Step 1: Country ──────────────────────────────────────
    st.subheader("① Select Country")
    countries     = list(db.keys())
    country_names = [db[c]["name"] for c in countries]
    selected_idx  = st.selectbox(
        "Country",
        range(len(countries)),
        format_func=lambda x: country_names[x],
        label_visibility="collapsed"
    )
    selected_country = countries[selected_idx]
    country_data     = db[selected_country]

    st.divider()

    # ── Step 2: Field ────────────────────────────────────────
    st.subheader("② Select Field")
    fields        = list(country_data["fields"].keys())
    field_options = fields + ["All"]

    # Use buttons for field selection
    if "selected_field" not in st.session_state:
        st.session_state["selected_field"] = field_options[0]

    cols = st.columns(len(field_options))
    for i, field in enumerate(field_options):
        with cols[i]:
            is_selected = st.session_state["selected_field"] == field
            if st.button(
                field,
                key=f"btn_{field}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state["selected_field"] = field
                st.rerun()

    selected_field = st.session_state["selected_field"]
    st.caption(f"Selected: **{selected_field}**")

    st.divider()

    # ── Step 3: Question ─────────────────────────────────────
    st.subheader("③ Ask Your Question")

    question = st.text_input(
        "Question",
        placeholder="e.g. What are the KYC requirements for high risk customers?",
        label_visibility="collapsed"
    )

    ask_clicked = st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True,
        disabled=not question
    )

    # ── Process & Answer ─────────────────────────────────────
    if ask_clicked and question:

        # Get regulations for field
        if selected_field == "All":
            regulations = []
            for field_regs in country_data["fields"].values():
                regulations.extend(field_regs)
        else:
            regulations = country_data["fields"].get(selected_field, [])

        # Find relevant
        with st.spinner("🔍 Finding relevant regulations..."):
            relevant = find_relevant_urls(question, regulations, top=5)

        # Scrape content
        context_parts = []
        progress      = st.progress(0)

        for i, reg in enumerate(relevant):
            with st.spinner(f"📡 Loading: {reg['title'][:50]}..."):
                content = scrape_page(reg["url"])
                if content:
                    relevant_chunk = find_relevant_chunk(
                        content, question
                    )
                    context_parts.append(
                        f"[SOURCE: {reg['title']}]\n{relevant_chunk}"
                    )
            progress.progress((i + 1) / len(relevant))

        progress.empty()

        if not context_parts:
            st.error("Could not load regulation content. Please try again.")
            return

        context = "\n\n".join(context_parts)

        # Get answer
        with st.spinner("🤖 Analysing regulations..."):
            answer = ask_gemini(question, context, api_key)

        # Show answer
        st.divider()
        st.subheader("📋 Answer")
        st.markdown(answer)

if __name__ == "__main__":
    main()
