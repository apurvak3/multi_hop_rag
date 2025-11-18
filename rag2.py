# rag.py
# Multi-Hop RAG with Ollama (llama3.1:8b + nomic embeddings)
# Fixed + Cleaned + Ready to run

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings   # Fixed deprecation
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
import gradio as gr

print("Setting up Ollama + Models...")
llm = ChatOllama(model="llama3:latest", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading HotpotQA dataset...")
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

documents = []
questions = []

print("Processing documents...")
for item in tqdm(dataset):
    questions.append({
        "question": item["question"],
        "answer": item["answer"],
        "supporting_titles": set(item["supporting_facts"]["title"])
    })
    
    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        for sent in sentences:
            documents.append(Document(
                page_content=sent.strip(),
                metadata={"title": title}
            ))

# Remove duplicate sentences
unique_docs = {doc.page_content: doc for doc in documents}
documents = list(unique_docs.values())
print(f"Total unique documents: {len(documents)} | Questions: {len(questions)}")

# Build indexes
print("Building FAISS + BM25 indexes...")
vectorstore = FAISS.from_documents(documents, embeddings)
tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
print("Indexes ready!")

# Prompts
decompose_template = "Break this question into 2-3 simple sub-questions:\nQuestion: {question}\nSub-questions:\n1."
generate_template = "Using only the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

decompose_prompt = PromptTemplate.from_template(decompose_template)
generate_prompt = PromptTemplate.from_template(generate_template)

decompose_chain = decompose_prompt | llm
generate_chain = generate_prompt | llm

# Hybrid Retrieval (FAISS + BM25)
def retrieve(query, k=8):
    # Dense retrieval
    dense_docs = vectorstore.similarity_search(query, k=k)
    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_docs = [documents[i] for i in bm25_scores.argsort()[::-1][:k]]
    
    # Merge & deduplicate
    seen = set()
    merged = []
    for doc in dense_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged[:k]

# Multi-Hop RAG
def multi_hop_rag(question):
    print(f"\nQuestion: {question}")
    
    # Step 1: Decompose
    sub_text = decompose_chain.invoke({"question": question}).content
    subs = []
    for line in sub_text.split("\n"):
        line = line.strip()
        if line and any(c.isdigit() for c in line[:3]):
            # Extract text after number
            sub_q = line.split(".", 1)[-1].strip(" :-")
            if sub_q:
                subs.append(sub_q)
    
    if not subs:
        subs = [question]
    subs = subs[:3]  # Max 3 sub-questions
    print("Sub-questions:", subs)
    
    # Step 2: Retrieve for each sub-question
    context_parts = []
    for sq in subs:
        docs = retrieve(sq, k=6)
        context_parts.extend([d.page_content for d in docs])
    
    context = "\n\n".join(context_parts)[:32000]  # Stay under token limit
    
    # Step 3: Final answer
    answer = generate_chain.invoke({"question": question, "context": context}).content
    print("Answer generated!")
    return answer

# Gradio interface function
def gradio_fn(question):
    return multi_hop_rag(question)


# ———————— MAIN ————————
if __name__ == "__main__":
    print("\nLaunching Gradio Demo...")
    demo = gr.Interface(
        fn=gradio_fn,
        inputs=gr.Textbox(label="Ask a Multi-Hop Question", lines=2, placeholder="e.g. Which magazine named the discoverer of general relativity Person of the Century in 1999?"),
        outputs=gr.Textbox(label="Answer"),
        title="Multi-Hop RAG with Llama3 (Ollama)",
        description="HotpotQA FullWiki • Hybrid Retrieval (FAISS + BM25) • Local & Private",
        examples=[
            ["Which magazine named the discoverer of general relativity as Person of the Century in 1999?"],
            ["Who directed the film that won the Academy Award for Best Picture in 1994?"],
            ["Are the birthplace of Barack Obama and the capital of Hawaii the same state?"],
            ["What is the name of the university whose football team is called the Crimson Tide?"],
        ],
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
    
    # share=True for public link, share=False for local only
    demo.launch(share=True, debug=False)