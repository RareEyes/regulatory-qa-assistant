import streamlit as st
import requests
from bs4 import BeautifulSoup
from google import genai
import json
import time
import re
import math

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
    try:
        headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=15)
        soup     = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text  = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if len(l) > 20]
        return "\n".join(lines)
    except:
        return ""

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
    # Return top scoring, minimum score of 1
    return [reg for s, reg in scored[:top] if s > 0] or [scored[0][1]]

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
    # Header
    st.title("🏦 Regulatory Q&A Assistant")
    st.caption("Ask questions about financial regulations across different countries")
    st.divider()

    # Load database
    try:
        db = load_database()
    except:
        st.error("Failed to load regulations database.")
        return

    # ── Step 1: API Key ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            help="Get free key at aistudio.google.com"
        )
        st.caption("Your key is never stored")
        st.divider()
        st.markdown("**How to use:**")
        st.markdown("1. Enter your Gemini API key")
        st.markdown("2. Select a country")
        st.markdown("3. Select a field")
        st.markdown("4. Ask your question")
        st.divider()
        st.caption("Data sourced from official regulatory websites")

    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started")
        return

    # ── Step 2: Select Country ───────────────────────────────
    st.subheader("Step 1: Select Country")
    countries     = list(db.keys())
    country_names = [db[c]["name"] for c in countries]
    selected_idx  = st.selectbox(
        "Choose a country/regulator",
        range(len(countries)),
        format_func=lambda x: country_names[x],
        label_visibility="collapsed"
    )
    selected_country = countries[selected_idx]
    country_data     = db[selected_country]

    st.divider()

    # ── Step 3: Select Field ─────────────────────────────────
    st.subheader("Step 2: Select Field")
    fields        = list(country_data["fields"].keys())
    field_options = fields + ["All"]
    
    cols          = st.columns(len(field_options))
    selected_field = st.session_state.get("selected_field", field_options[0])

    for i, field in enumerate(field_options):
        with cols[i]:
            if st.button(
                field,
                key=f"field_{field}",
                use_container_width=True,
                type="primary" if selected_field == field else "secondary"
            ):
                st.session_state["selected_field"] = field
                selected_field = field

    st.divider()

    # ── Step 4: Ask Question ─────────────────────────────────
    st.subheader("Step 3: Ask Your Question")
    question = st.text_area(
        "Question",
        placeholder="e.g. What are the KYC requirements for high risk customers?",
        height=100,
        label_visibility="collapsed"
    )

    ask_clicked = st.button(
        "🔍 Ask Question",
        type="primary",
        use_container_width=True,
        disabled=not question
    )

    # ── Answer ───────────────────────────────────────────────
    if ask_clicked and question:

        # Get regulations for selected field
        if selected_field == "All":
            regulations = []
            for field_regs in country_data["fields"].values():
                regulations.extend(field_regs)
        else:
            regulations = country_data["fields"].get(selected_field, [])

        with st.spinner("🔍 Finding relevant regulations..."):
            relevant = find_relevant_urls(question, regulations, top=3)

        st.info(f"📚 Searching {len(relevant)} relevant regulation(s)")

        # Scrape content
        context_parts = []
        with st.spinner("📡 Loading regulation content..."):
            for reg in relevant:
                content = scrape_page(reg["url"])
                if content:
                    context_parts.append(
                        f"[{reg['title']}]\n{content[:5000]}"
                    )

        if not context_parts:
            st.error("Could not load regulation content. Please try again.")
            return

        context = "\n\n".join(context_parts)

        # Get answer
        with st.spinner("🤖 Generating answer..."):
            answer = ask_gemini(question, context, api_key)

        # Show answer
        st.divider()
        st.subheader("📋 Answer")
        st.markdown(answer)

if __name__ == "__main__":
    main()
