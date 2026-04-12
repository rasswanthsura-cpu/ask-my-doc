# ------------------------------------------------------------
# 📘 SIMPLE PROJECT: Ask Questions from a PDF using OpenAI + FAISS
# ------------------------------------------------------------
# This project will:
# 1. Read text from a PDF file
# 2. Split it into small parts (chunks)
# 3. Convert text into numbers (embeddings)
# 4. Store them in a FAISS database
# 5. Take a question from the user
# 6. Find the most similar text parts(chunk) from the PDF
# 7. Send them to OpenAI to get a final answer
# ------------------------------------------------------------

# 🧩 Step 1: Import libraries
import os
import PyPDF2                # to read the PDF
import faiss                 # to store and search embeddings
import numpy as np           # for handling numeric data
import ollama

# 🧠 Step 2: Function to read all text from a PDF
def read_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text = text + page_text
    return text

# 🪓 Step 3: Function to break the text into smaller parts
def make_chunks(text, chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

# 🔢 Step 4: Function to convert text chunks into embeddings (local via Ollama)
def get_embeddings(chunks):
    response = ollama.embed(
        model='nomic-embed-text',
        input=chunks
    )
    return np.array(response['embeddings']).astype("float32")

# 💾 Step 5: Function to create and store vectors in FAISS
def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ❓ Step 6: Function to search for the most similar text (local via Ollama)
def search_chunks(question, chunks, index):
    response = ollama.embed(
        model='nomic-embed-text',
        input=[question]
    )
    q_embed = np.array(response['embeddings']).astype("float32")
    distances, indices = index.search(q_embed, k=2)
    best_chunks = [chunks[i] for i in indices[0]]
    return best_chunks

# 💬 Step 7: Function to ask LLM for the final answer (local via Ollama)
def ask_question(chunks, question):
    context = "\n\n".join(chunks)
    prompt = f"Answer the question based ONLY on this text:\n{context}\n\nQuestion: {question}"

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# 🚀 Step 8: Main function to run everything
def main():
    # 1. Read text from a sample PDF
    if not os.path.exists("medicine.pdf"):
        print("❌ Error: medicine.pdf not found.")
        return

    text = read_pdf("medicine.pdf")
    if not text.strip():
        print("❌ Error: Could not extract text from the PDF.")
        return
    print("✅ PDF Loaded.")

    # 2. Split text into smaller chunks
    chunks = make_chunks(text)
    if not chunks:
        print("❌ Error: No text chunks created.")
        return
    print("✅ Text divided into chunks.")

    # 3. Create embeddings for all chunks via Ollama
    print("⏳ Creating embeddings via Ollama (nomic-embed-text)...")
    try:
        embeddings = get_embeddings(chunks)
        print("✅ Embeddings created.")
    except Exception as e:
        print(f"❌ Error: Failed to generate embeddings. {str(e)}")
        print("💡 Make sure Ollama is running and you have run 'ollama pull nomic-embed-text'")
        return

    # 4. Store embeddings in FAISS
    index = build_faiss_index(embeddings)
    print("✅ FAISS index created.")

    # 5. Ask user for a question
    question = input("Ask your Question from PDF : ")

    # 6. Find similar text chunks
    print("🔍 Searching for related text...")
    best_chunks = search_chunks(question, chunks, index)

    # 7. Get final answer from local LLM
    print("🧠 Generating answer via Ollama (llama3.2)...")
    answer = ask_question(best_chunks, question)
    print("\n🧠 Answer from Local LLM:")
    print(answer)

# Run the program
if __name__ == "__main__":
    main()

#token cost # 5 dollars - questions,answer , documents
#vector -db

    
# Meropenem uses
#list down the uses of Meropenem 
#what are side effects of Meropenem 
#what are processors in computers
#is vomiting a side effect of Meropenem

#gibili - internal training 
#own app to crete gibili - photo(not used for internal training)