import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import requests
import xml.etree.ElementTree as ET
import fitz

if 'results' not in st.session_state:
    st.session_state.results = []
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Research Paper Search", layout="wide")

model_name = "google/flan-t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

keyword_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def search_papers(keyword):
    params = {
        'search_query': f'all:{keyword}',
        'start': 0,
        'max_results': 5
    }
    response = requests.get(ARXIV_API_URL, params=params)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
            pdf_url = entry.find("{http://www.w3.org/2005/Atom}link[@title='pdf']").attrib['href']
            authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            published = entry.find("{http://www.w3.org/2005/Atom}published").text

            citation = f"{', '.join(authors)}. '{title}'. {published}."
            papers.append((title.strip(), abstract.strip(), pdf_url, citation))
        return papers
    else:
        return [("Error", "Could not fetch papers from ArXiv")]

def summarize_text(text, length_type="moderate"):
    if length_type == "short":
        input_text = f"summarize: {text}. Provide a brief summary of this paper in your own words such that most important part of the paper is included."
        max_len, min_len = 500, 50
        print("Short")
    elif length_type == "detailed":
        input_text = f"summarize: {text}. Provide a detailed summary in your own words covering the introduction, abstract, key findings, methodology, results, challenges, and references. Format it in the following way for each - heading: content"
        max_len, min_len = 2500, 500
        print("Detailed")
    else:
        input_text = f"summarize: {text}. Please summarize the key findings and challenges in this paper in your own words. Format it in the following way for each - heading: content"
        max_len, min_len = 1000, 200
        print("Moderate")

    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids, 
        max_length=max_len, 
        min_length=min_len,  
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    with open("temp_paper.pdf", "wb") as f:
        f.write(response.content)

    text = ""
    with fitz.open("temp_paper.pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_uploaded_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def translate_text(text, target_language="en"):
    if not title in st.session_state.summaries:
        st.error("Please summarize the paper first before translating.")
        return
    if target_language == "en":
        st.error("Please choose a language other than English for translation.")
        return
    input_text = f"Translate to {target_language}: {text}. Do not miss anything from the summary."
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2500, truncation=True)
    translation_ids = model.generate(inputs.input_ids, max_length=1024, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translation

def translate_upload_text(text, title, target_language="en"):
    if title not in st.session_state.summaries:
        st.error("Please summarize the paper first before translating.")
        return
    if target_language == "en":
        st.error("Please choose a language other than English for translation.")
        return
    input_text = f"Translate to {target_language}: {st.session_state.summaries[title]}. Do not miss anything from the summary."
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2500, truncation=True)
    translation_ids = model.generate(inputs.input_ids, max_length=2500, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translation

def ask_question_about_paper(text, user_query):
    input_text = f"question: {user_query} context: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
    response_ids = model.generate(
        inputs.input_ids, 
        max_length=500, 
        min_length=10,
        num_beams=4, 
        early_stopping=True
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

st.write("# **CurioQuest**")
st.write("##### AI Research Paper Search, Summarization, and Translation")

st.markdown("""
    <style>
    .citation {
        background-color: #252525;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
    }
    .chatbot {
        
    }
    .st-emotion-cache-1r6slb0{
        height: 100dvh;
        position: sticky;
        top: 10rem;
    }
    </style>
""", unsafe_allow_html=True)

mode = st.radio("Choose Mode:", ("Search Online", "Upload PDF"))

if mode == "Search Online":
    st.header("Search Papers Online")
    
    keyword = st.text_input("Enter a keyword to search papers:")
    if st.button("Search"):
        results = search_papers(keyword)
        st.session_state.results = results

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.results:
            st.header("Search Results:")
            
            for i, (title, abstract, pdf_url, citation) in enumerate(st.session_state.results):
                st.markdown("***")
                st.write(f"#### **Title:** {title}")
                st.write(f"##### **Abstract:**")
                st.write(abstract)
                st.markdown(f"[Download PDF]({pdf_url})")

                summary_type = st.radio(f"Choose Summary Length for {title}:", 
                                         ("Short", "Moderate", "Detailed"), 
                                         key=f"summary_radio_{i}")

                if st.button(f"Summarize {title}", key=f"summarize_final_button_{i}"):
                    full_text = extract_text_from_pdf(pdf_url)
                    summary = summarize_text(full_text, length_type=summary_type.lower())
                    st.session_state.summaries[title] = summary

                if title in st.session_state.summaries:
                    st.write(f"**Summary of {title}:**")
                    st.write(st.session_state.summaries[title])

                target_language = st.text_input(f"Translate {title} to (e.g., 'fr' for French):", key=f"translate_lang_{i}")
                if st.button(f"Translate {title}", key=f"translate_button_{i}"):
                    translation = translate_text(abstract, target_language or "en")
                    st.write(f"**Translated Abstract:** {translation}")


                if st.button(f"Generate Citation", key=f"cite_button_{i}"):
                    st.markdown(f"<div class='citation'><strong>Citation:</strong> {citation}<br><span style=\"color: #6f6fff;\">{pdf_url}</span></div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chatbot">', unsafe_allow_html=True)
        if st.session_state.results:
            st.header("Chatbot")
            st.markdown("***")
            
            selected_paper = st.selectbox("Select a paper to ask about:", 
                                            [title for title, _, _, _ in st.session_state.results])

            user_query = st.text_input("Your question about the paper")
            if st.button("Ask"):
                if user_query and selected_paper:
                    pdf_url = next((url for title, _, url, _ in st.session_state.results if title == selected_paper), None)
                    if pdf_url:
                        full_text = extract_text_from_pdf(pdf_url)
                        response = ask_question_about_paper(full_text, user_query)
                        st.session_state.chat_history.append((user_query, response))

                        st.subheader("Chat History")
                        for query, response in st.session_state.chat_history:
                            st.markdown("***")
                            st.write(f"**You:** {query}")
                            st.write(f"**Bot:** {response}")
                elif not selected_paper:
                    st.error("Please select a paper to ask questions about.")
                else:
                    st.error("Please enter a question to ask the chatbot.")
        st.markdown('</div>', unsafe_allow_html=True)

elif mode == "Upload PDF":
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        title = "Uploaded PDF"
        uploaded_pdf_text = extract_text_from_uploaded_pdf(uploaded_file)
        
        st.write("### Options for Your Uploaded PDF")
        
        summary_type = st.radio("Choose Summary Length:", ("Short", "Moderate", "Detailed"))
        
        if st.button("Summarize Uploaded PDF"):
            summary = summarize_text(uploaded_pdf_text, length_type=summary_type.lower())
            st.session_state.summaries[title] = summary
            st.write("### Summary of Uploaded PDF")
            st.write(summary)

        target_language = st.text_input("Translate Uploaded PDF Abstract to (e.g., 'fr' for French):")
        if st.button("Translate Uploaded PDF Abstract"):
            translation = translate_upload_text(uploaded_pdf_text, title, target_language or "en")
            st.write(f"**Translated Abstract:** {translation}")

        user_query_uploaded = st.text_input("Ask a question about the uploaded PDF")
        if st.button("Ask About Uploaded PDF"):
            if user_query_uploaded:
                response = ask_question_about_paper(uploaded_pdf_text, user_query_uploaded)
                st.subheader("Chatbot Response")
                st.write(response)
            else:
                st.error("Please enter a question to ask the chatbot.")