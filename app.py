import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import json
import re
import io
from pypdf import PdfReader

st.set_page_config(
    page_title="Regulatory Q&A Assistant",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .block-container {padding-top: 2rem; padding-bottom: 1rem;}
    h1 {text-align: center; font-size: 28px !important;}
    .title {text-align: center; font-size: 1.5rem; font-weight: 600; margin-bottom: 0;}
    .subtitle {text-align: center; color: grey; font-size: 13px; margin-bottom: 10px;}
    .disclaimer {font-size: 11px; color: grey; margin-top: 8px;}
    .hint {font-size: 10px; color: #bbb; margin-top: -8px; margin-bottom: 8px;}
    .field-label {font-size: 14px; color: inherit; margin-bottom: 0.25rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=0)
def load_database():
    url  = "https://raw.githubusercontent.com/RareEyes/regulatory-qa-assistant/refs/heads/main/regulations_db.json"
    resp = requests.get(url)
    return resp.json()

def scrape_page(url):
    try:
        headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        soup     = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except:
        return ""

def read_pdf_from_url(url):
    try:
        response = requests.get(url, timeout=30)
        reader   = PdfReader(io.BytesIO(response.content))
        text     = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += page_text + "\n"
        return text
    except:
        return ""

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

    question_lower = question.lower()
    bonus_phrases  = [
        ("cdd", "customer due diligence"),
        ("kyc", "know your customer"),
        ("high risk", "high risk"),
        ("governance", "governance"),
        ("licensing", "licens"),
        ("insurance", "insurance"),
        ("aml", "anti-money laundering"),
        ("fraud", "fraud"),
        ("audit", "audit"),
        ("compliance", "compliance"),
    ]
    for q_phrase, t_phrase in bonus_phrases:
        if q_phrase in question_lower and t_phrase in text:
            score += 5

    return score

def find_relevant_urls(question, regulations, top=5):
    if len(regulations) <= 5:
        return regulations

    scored = sorted(
        [(score_regulation(question, r["title"], r["url"]), r)
         for r in regulations],
        key=lambda x: x[0], reverse=True
    )
    top_regs = [r for s, r in scored[:top] if s > 0]
    return top_regs if top_regs else [scored[0][1]]

def find_relevant_chunk(content, question):
    chunk_size = 8000
    step       = 7000

    words = [
        w for w in re.findall(r"[a-z0-9]+", question.lower())
        if w not in STOP_WORDS and len(w) > 2
    ]

    if not words or not content:
        return content[:20000]

    chunks = []
    start  = 0
    while start < len(content):
        chunks.append({
            "text":  content[start:start + chunk_size],
            "start": start,
            "index": len(chunks)
        })
        start += step

    if not chunks:
        return content[:20000]

    answer_terms = [
        "means that you must", "means within", "immediately",
        "must not", "shall not", "required to", "obligation",
        "you must", "shall", "within", "days", "hours",
        "prohibited", "high risk", "enhanced due diligence",
        "customer due diligence", "beneficial owner",
        "intensified monitoring", "periodic updation",
        "risk-based", "two years", "eight years", "ten years",
        "freeze", "confirmed match", "without delay",
        "without prior notice", "freeze all funds",
        "low risk", "medium risk", "high-risk customers",
        "prosecution", "criminal sanction", "prison term",
        "fine", "enforcement", "censure",
        "failure to comply", "fails to comply",
        "civil penalty", "criminal offence", "imprisonment",
        "penalty", "sanction",
        "board of directors", "responsibilities of the board",
        "board composition", "senior management",
        "audit committee", "risk committee",
        "fit and proper", "corporate governance",
        "board must", "bank must", "banks must",
        "members of the board", "independent member",
        "application for", "licence", "license",
        "central bank will", "must obtain",
        "must have", "must ensure", "must establish",
        "must maintain", "must comply", "must include",
        "at least", "minimum", "maximum",
    ]

    scored = []
    for chunk in chunks:
        t            = chunk["text"].lower()
        phrase_score = sum(t.count(term) * 30 for term in answer_terms if term in t)
        word_score   = sum(min(t.count(w), 3) for w in words)
        scored.append((phrase_score + word_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected_indices = set()

    for i in range(min(3, len(scored))):
        idx = scored[i][1]["index"]
        selected_indices.add(idx)
        if idx > 0:
            selected_indices.add(idx - 1)
        if idx < len(chunks) - 1:
            selected_indices.add(idx + 1)

    if len(content) < 50000:
        for i in range(len(chunks)):
            selected_indices.add(i)

    direct_hits = [
        "imprisonment", "criminal offence", "civil penalty",
        "on conviction", "summary conviction", "fine or to both",
        "prison term", "two years", "five years",
        "periodic updation", "high-risk customers",
        "freeze all funds", "confirmed match",
    ]
    for i, chunk in enumerate(chunks):
        t = chunk["text"].lower()
        if any(hit in t for hit in direct_hits):
            selected_indices.add(i)
            if i > 0:
                selected_indices.add(i - 1)
            if i < len(chunks) - 1:
                selected_indices.add(i + 1)

    selected = sorted(
        [chunks[i] for i in selected_indices],
        key=lambda x: x["start"]
    )
    return "\n\n".join(c["text"] for c in selected)

def ask_gemini(question, context, api_key, detailed):
    client = genai.Client(api_key=api_key)

    if detailed:
        style = """- Give a DETAILED answer. Do not shorten or summarize.
- Use document's own numbered or titled points as headings.
- Under each heading copy the full explanation given."""
    else:
        style = """- Give a SHORT and DIRECT answer.
- Maximum 5-6 bullet points covering key points only.
- Do not list every sub-detail unless critical.
- Be concise."""

    prompt = f"""You are a compliance expert answering questions about regulatory documents.

STRICT RULES:
- Use ONLY the text provided below.
- If answer not present say: "This information is not found in the selected regulations."
- Do NOT invent or guess anything.
{style}
- Copy exact definitions and timeframes word for word.
- End with Sources listing regulation names used.

REGULATION TEXT:
{context}

QUESTION: {question}

ANSWER:"""

    models = [
        "gemini-flash-lite-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
            )
            return response.text
        except:
            continue

    return "❌ AI service unavailable. Please try again later."

def main():
    try:
        db = load_database()
    except:
        st.error("Failed to load regulations database.")
        return

    api_key = st.secrets["GEMINI_API_KEY"]

    # Header - centered
    st.markdown('<div class="title">🏦 Regulatory Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Get answers from official regulatory documents</p>', unsafe_allow_html=True)

    # ── Country ───────────────────────────────────────────────────────────────
    country_order  = ["India", "United Arab Emirates", "United Kingdom"]
    all_countries  = list(db.keys())

    ordered_keys   = []
    ordered_names  = []
    for display in country_order:
        for k in all_countries:
            if display.lower() in db[k]["name"].lower():
                ordered_keys.append(k)
                ordered_names.append(db[k]["name"])
                break

    for k in all_countries:
        if k not in ordered_keys:
            ordered_keys.append(k)
            ordered_names.append(db[k]["name"])

    selected_idx = st.selectbox(
        "Select Country",
        range(len(ordered_keys)),
        format_func=lambda x: ordered_names[x],
        label_visibility="visible"
    )
    selected_country = ordered_keys[selected_idx]
    country_data     = db[selected_country]

    # ── Field ─────────────────────────────────────────────────────────────────
    fields        = list(country_data["fields"].keys())
    field_options = fields + ["All"] if len(fields) > 1 else fields

    if (
        "selected_field"   not in st.session_state or
        "selected_country" not in st.session_state or
        st.session_state["selected_country"] != selected_country or
        st.session_state["selected_field"]   not in field_options
    ):
        st.session_state["selected_field"]   = field_options[0]
        st.session_state["selected_country"] = selected_country

    st.markdown('<div class="field-label">Select Field</div>', unsafe_allow_html=True)
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
                st.session_state["selected_field"]   = field
                st.session_state["selected_country"] = selected_country
                st.rerun()

    selected_field = st.session_state["selected_field"]

    # ── Question ──────────────────────────────────────────────────────────────
    question = st.text_input(
        "Your Question",
        placeholder="e.g. What are the KYC requirements for high risk customers?"
    )

    st.caption('💡 Include "detailed explanation" in your question for a fuller answer')

    ask_clicked = st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True
    )

    if ask_clicked and not question.strip():
        st.warning("Please type a question first.")

    # ── Answer ────────────────────────────────────────────────────────────────
    if ask_clicked and question.strip():
        detailed = any(w in question.lower() for w in [
            "detail", "detailed", "explain", "elaborate",
            "in depth", "thorough", "full", "complete"
        ])

        if selected_field == "All":
            regulations = []
            for field_regs in country_data["fields"].values():
                regulations.extend(field_regs)
        else:
            regulations = country_data["fields"].get(selected_field, [])

        relevant = find_relevant_urls(question, regulations, top=5)

        MAX_TOTAL_CHARS = 40000
        context_parts   = []

        with st.spinner("🔍 Searching regulations..."):
            total_chars = 0
            for reg in relevant:
                if total_chars >= MAX_TOTAL_CHARS:
                    break
                if reg.get("type") == "pdf":
                    content = read_pdf_from_url(reg["url"])
                else:
                    content = scrape_page(reg["url"])

                if content:
                    chunk        = find_relevant_chunk(content, question)
                    chunk        = chunk[:15000]
                    total_chars += len(chunk)
                    context_parts.append(
                        f"[SOURCE: {reg['title']}]\n{chunk}"
                    )

        if not context_parts:
            st.error("Could not load regulation content. Please try again.")
            return

        with st.spinner("🤖 Analysing..."):
            answer = ask_gemini(
                question,
                "\n\n".join(context_parts),
                api_key,
                detailed
            )

        st.markdown("---")
        st.subheader("📋 Answer")
        st.markdown(answer)
        st.caption("⚠️ Please double check as AI might make mistakes. For informational purposes only.")

if __name__ == "__main__":
    main()
