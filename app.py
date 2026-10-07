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


def find_relevant_chunk(content, question, chunk_size=8000, overlap=1000):
    """
    Split content into chunks
    Find most relevant chunk using both
    question keywords AND answer keywords
    """
    # Question keywords
    words = [
        w for w in re.findall(r"[a-z0-9]+", question.lower())
        if w not in STOP_WORDS and len(w) > 2
    ]

    if not words or not content:
        return content[:20000]

    # Split into overlapping chunks
    chunks = []
    start  = 0
    while start < len(content):
        chunks.append({
            "text":  content[start:start + chunk_size],
            "start": start,
            "index": len(chunks)
        })
        start += chunk_size - overlap

    if not chunks:
        return content[:20000]

    # Score each chunk
    # Key insight: exact phrase matches score much higher
    scored = []
    for chunk in chunks:
        text  = chunk["text"].lower()
        score = 0

        # Base score from question keywords
        for w in words:
            score += text.count(w)

        # Big boost for chunks with specific answer terms
        answer_terms = [
            "means that you must",
            "means within",
            "immediately freeze",
            "freeze all funds",
            "without delay",
            "without prior notice",
            "confirmed match",
            "must immediately",
            "you must",
            "obligation",
            "required to",
            "shall",
        ]
        for term in answer_terms:
            if term in text:
                score += 20

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Get best chunk index
    best_chunk = scored[0][1]
    best_index = best_chunk["index"]

    # Always include best + neighbours
    selected_indices = set()
    selected_indices.add(best_index)
    if best_index > 0:
        selected_indices.add(best_index - 1)
    if best_index < len(chunks) - 1:
        selected_indices.add(best_index + 1)

    # Add second best if far from best
    if len(scored) > 1:
        second       = scored[1][1]
        second_index = second["index"]
        if abs(second_index - best_index) > 1:
            selected_indices.add(second_index)
            if second_index > 0:
                selected_indices.add(second_index - 1)
            if second_index < len(chunks) - 1:
                selected_indices.add(second_index + 1)

    # Sort by position
    selected = sorted(
        [chunks[i] for i in selected_indices],
        key=lambda x: x["start"]
    )

    return "\n\n".join(c["text"] for c in selected)
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
- Copy every quoted definition word for word e.g. "without delay" means...
- Include every timeframe, amount and obligation exactly as stated.
- Never skip any bullet point or sub point from the source text.
- For obligation questions end with: "Check the full rule for exceptions."
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
        
        /* Remove top padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Make title smaller */
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0rem !important;
        }
        
        /* Reduce spacing between elements */
        .stSelectbox {
            margin-bottom: 0rem !important;
        }
        
        /* Reduce button spacing */
        .stButton {
            margin-bottom: 0rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────
    st.title("🏦 Regulatory Q&A Assistant")
    st.caption("Ask questions about financial regulations across different countries")

    # ── Load Database ────────────────────────────────────────
    try:
        db = load_database()
    except:
        st.error("Failed to load regulations database.")
        return

    # ── API Key ──────────────────────────────────────────────
    api_key = st.secrets["GEMINI_API_KEY"]

    # ── All Steps In Columns ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**① Country**")
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

    with col2:
        st.markdown("**② Field**")
        fields        = list(country_data["fields"].keys())
        field_options = fields + ["All"]

        if "selected_field" not in st.session_state:
            st.session_state["selected_field"] = field_options[0]

        selected_field = st.selectbox(
            "Field",
            field_options,
            label_visibility="collapsed",
            key="field_select"
        )
        st.session_state["selected_field"] = selected_field

    # ── Question ─────────────────────────────────────────────
    st.markdown("**③ Your Question**")
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

        if selected_field == "All":
            regulations = []
            for field_regs in country_data["fields"].values():
                regulations.extend(field_regs)
        else:
            regulations = country_data["fields"].get(
                selected_field, []
            )

        relevant      = find_relevant_urls(question, regulations, top=5)
        context_parts = []

        with st.spinner("🔍 Searching regulations..."):
            for reg in relevant:
                content = scrape_page(reg["url"])
                if content:
                    relevant_chunk = find_relevant_chunk(
                        content, question
                    )
                    context_parts.append(
                        f"[SOURCE: {reg['title']}]\n{relevant_chunk}"
                    )

        if not context_parts:
            st.error("Could not load content. Please try again.")
            return

        context = "\n\n".join(context_parts)

        with st.spinner("🤖 Analysing..."):
            answer = ask_gemini(question, context, api_key)

        st.markdown("---")
        st.markdown("**📋 Answer**")
        st.markdown(answer)

if __name__ == "__main__":
