from fastapi import FastAPI, Form, Request, Response, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import os
import json
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv
from openai import OpenAI

#  ──────────────────────────────────────────────

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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", 
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def count_pdf_pages(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print("Error:", e)
        return None

def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text = "".join(page.page_content for page in data)

    # Use default tokenizer for gpt-3.5-turbo (cl100k_base)
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=10000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    docs_for_questions = [Document(page_content=c) for c in chunks]

    splitter_ans = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=100
    )
    docs_for_answers = splitter_ans.split_documents(docs_for_questions)

    return docs_for_questions, docs_for_answers

def llm_pipeline(file_path):
    docs_q, docs_a = file_processing(file_path)

    # Chat model for generating questions
    llm_questions = ChatOpenAI(
        model_name="openai/gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1"
    )

    question_prompt = PromptTemplate(
        template="""
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the coders or programmers for their tests.
Make sure not to lose any important information.

QUESTIONS:
""",
        input_variables=["text"]
    )

    refine_prompt = PromptTemplate(
        template="""
You are an expert at creating practice questions based on coding material and documentation.
Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions so far: {existing_answer}.
We can refine or add new ones if needed, using context below:

------------
{text}
------------

Given this new context, refine the original questions in English.
If the context is not helpful, return the original questions unchanged.

QUESTIONS:
""",
        input_variables=["existing_answer", "text"]
    )

    ques_chain = load_summarize_chain(
        llm=llm_questions,
        chain_type="refine",
        verbose=True,
        question_prompt=question_prompt,
        refine_prompt=refine_prompt
    )
    raw_questions = ques_chain.run(docs_q)
    question_list = [q.strip() for q in raw_questions.split("\n") if q.strip().endswith("?") or q.strip().endswith(".")]

    # Build vector store over the answer chunks
    embeddings = OpenAIEmbeddings(openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1")
    vector_store = FAISS.from_documents(docs_a, embeddings)

    # Chat model for generating answers
    llm_answers = ChatOpenAI(
        model_name="openai/gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1"
    )

    answer_chain = RetrievalQA.from_chain_type(
        llm=llm_answers,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_chain, question_list

def get_csv(file_path):
    answer_chain, questions = llm_pipeline(file_path)

    os.makedirs("static/output", exist_ok=True)
    output_path = "static/output/QA.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])
        for q in questions:
            a = answer_chain.run(q)
            writer.writerow([q, a])
    return output_path

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    os.makedirs("static/docs", exist_ok=True)
    path = f"static/docs/{filename}"
    async with aiofiles.open(path, "wb") as f:
        await f.write(pdf_file)
    return Response(jsonable_encoder(json.dumps({"msg":"success","pdf_filename":path})))

@app.post("/analyze")
async def analyze_pdf(request: Request, pdf_filename: str = Form(...)):
    csv_path = get_csv(pdf_filename)
    return Response(jsonable_encoder(json.dumps({"output_file":csv_path})))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
