import pypdf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# document loading and chunking
def load_and_chunk_pdf(pdf_path, chunk_size=512, overlap=50):
    """Loads a pdf, extracts text, and splits it into chunks of specified size with overlap."""
    print(f"Loading and chunking PDF from: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or  " "
    
    # Split the text into chunks
    chunks = []
    # start = 0
    # while start < len(text):
    #     end = min(start + chunk_size, len(text))
    #     chunks.append(text[start:end])
    #     start += chunk_size - overlap  # move start forward by chunk_size - overlap

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

# print(load_and_chunk_pdf("C:\\personal\\job\\Cover-letter-fracttal.pdf"))
pdf_chunks = load_and_chunk_pdf("C:\\personal\\job\\Cover-letter-fracttal.pdf")
#embedding generation
print("Loading sentence transformer model for embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#create embeddings and Vector store (FAISS)
def create_vector_store(chunks, model):
    print("Creating embeddings for chunks...")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()  # Convert to numpy array for FAISS
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index

vector_store = create_vector_store(pdf_chunks, embedding_model)

# retrieval function
def retrieve_similar_chunks(query, model, index, chunks, top_k=3):
    print(f"Retrieving top {top_k} similar chunks for the query: '{query}'")
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()
    
    distances, indices = index.search(query_embedding_np, top_k)
        
    print("Return the actual text chunks:")
    similar_chunks = [chunks[i] for i in indices[0]]
    print(f"Retrieved {similar_chunks} similar chunks.")
    return similar_chunks

#generation logic
print("Loading generative LLM...")
#using a smaller, manageable model for local execution
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_response(query, similar_chunks):
    """Builds a prompt from the retrieved chunks and generates a response using the LLM."""
    print("Generating response from the LLM...")
    context = " ".join(similar_chunks)
    prompt = f"""Dado el siguiente contexto: {context}\nResponde la pregunta: {query}. "
    Contexto: {context}
    Pregunta: {query}
    Respuesta:
    """
    
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = llm_model.generate(**inputs, max_length=200)
    
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def ask_chatbot(query):
    """Main function to handle the RAG process: retrieval and generation."""
    similar_chunks = retrieve_similar_chunks(query, embedding_model, vector_store, pdf_chunks)
    response = generate_response(query, similar_chunks)
    return response

user_query = "quien escribe la carta de presentación?"
response = ask_chatbot(user_query)
print(f"User Query: {user_query}")
print(f"Chatbot Response: {response}")