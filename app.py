import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import json
import re

st.set_page_config(
    page_title="Regulatory Q&A Assistant",
    page_icon="🏦",
    layout="centered"
)

# ── Hide UI clutter ───────────────────────────────────────────────────────────
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .block-container {padding-top: 2rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
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
    """Download and read PDF from GitHub"""
    try:
        import io
        from pypdf import PdfReader

        response = requests.get(url, timeout=30)
        pdf_file = io.BytesIO(response.content)
        reader   = PdfReader(pdf_file)

        text = ""
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
    return sum(1 for w in words if w in text)

def find_relevant_urls(question, regulations, top=5):
    scored = sorted(
        [(score_regulation(question, r["title"], r["url"]), r)
         for r in regulations],
        key=lambda x: x[0], reverse=True
    )
    top_regs = [r for s, r in scored[:top] if s > 0]
    return top_regs if top_regs else [scored[0][1]]

def find_relevant_chunk(content, question, chunk_size=8000, overlap=1000):
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
        start += chunk_size - overlap

    if not chunks:
        return content[:20000]

    answer_terms = [
        # General
        "means that you must", "means within", "immediately",
        "must not", "shall not", "required to", "obligation",
        "you must", "shall", "within", "days", "hours",
        "prohibited", "penalty",
        # UAE specific
        "freeze", "confirmed match", "without delay",
        "without prior notice", "freeze all funds",
        # RBI/KYC specific
        "high risk", "enhanced due diligence",
        "customer due diligence", "beneficial owner",
        "intensified monitoring", "periodic updation",
        "risk-based", "two years", "eight years", "ten years",
        "low risk", "medium risk", "high-risk customers",
        "risk categorisation", "customer identification",
        "due diligence measures", "simplified due diligence",
        "ongoing due diligence", "risk profile",
        "updation of kyc", "kyc updation",
        "closely monitored", "mlm", "multi-level",
    ]

    scored = []
    for chunk in chunks:
        text         = chunk["text"].lower()
        phrase_score = sum(text.count(t) * 30 for t in answer_terms if t in text)
        word_score   = sum(min(text.count(w), 3) * 1 for w in words)
        scored.append((phrase_score + word_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_index = scored[0][1]["index"]

    selected_indices = set()
    selected_indices.add(best_index)
    if best_index > 0:
        selected_indices.add(best_index - 1)
    if best_index < len(chunks) - 1:
        selected_indices.add(best_index + 1)

    if len(scored) > 1:
        second_index = scored[1][1]["index"]
        if abs(second_index - best_index) > 1:
            selected_indices.add(second_index)
            if second_index > 0:
                selected_indices.add(second_index - 1)
            if second_index < len(chunks) - 1:
                selected_indices.add(second_index + 1)

    selected = sorted(
        [chunks[i] for i in selected_indices],
        key=lambda x: x["start"]
    )
    return "\n\n".join(c["text"] for c in selected)
def ask_gemini(question, context, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a compliance expert answering questions about regulatory documents.

STRICT RULES:
- Use ONLY the text provided below.
- If answer not present say: "This information is not found in the selected regulations."
- Do NOT invent or guess anything.
- Start with one direct answer sentence.
- Then extract EVERY detail as separate bullet points.
- Copy every quoted definition word for word.
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
        except:
            continue

    return "❌ AI service unavailable. Please try again later."

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        db = load_database()
    except:
        st.error("Failed to load regulations database.")
        return

    api_key = st.secrets["GEMINI_API_KEY"]

    st.title("🏦 Regulatory Q&A Assistant")
    st.caption("Ask questions about financial regulations")

    # Country
    countries    = list(db.keys())
    country_names = [db[c]["name"] for c in countries]
    selected_idx = st.selectbox(
        "① Select Country",
        range(len(countries)),
        format_func=lambda x: country_names[x]
    )
    selected_country = countries[selected_idx]
    country_data     = db[selected_country]

    # Field - reset when country changes
    fields        = list(country_data["fields"].keys())
    field_options = fields + ["All"]

    if (
        "selected_field"   not in st.session_state or
        "selected_country" not in st.session_state or
        st.session_state["selected_country"] != selected_country or
        st.session_state["selected_field"]   not in field_options
    ):
        st.session_state["selected_field"]   = field_options[0]
        st.session_state["selected_country"] = selected_country

    st.write("② Select Field")
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

    # Question
    question = st.text_input(
        "③ Your Question",
        placeholder="e.g. What are the KYC requirements for high risk customers?"
    )

    ask_clicked = st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True,
        disabled=not question
    )

    # Answer
    if ask_clicked and question:
        if selected_field == "All":
            regulations = []
            for field_regs in country_data["fields"].values():
                regulations.extend(field_regs)
        else:
            regulations = country_data["fields"].get(selected_field, [])

        relevant = find_relevant_urls(question, regulations, top=5)

        context_parts = []
        with st.spinner("🔍 Searching regulations..."):
            for reg in relevant:
                # Read PDF or scrape website
                if reg.get("type") == "pdf":
                    content = read_pdf_from_url(reg["url"])
                else:
                    content = scrape_page(reg["url"])

                if content:
                    chunk = find_relevant_chunk(content, question)
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
                api_key
            )

        st.markdown("---")
        st.subheader("📋 Answer")
        st.markdown(answer)

if __name__ == "__main__":
    main()
