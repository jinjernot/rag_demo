import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Step 1: Load the corpus from a JSON file
def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data]

corpus_file_path = "app/data/corpus.json"  # Path to the JSON file
corpus = load_corpus(corpus_file_path)

# Step 2: Create embeddings for the corpus
# Load a pre-trained sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode corpus into dense embeddings
corpus_embeddings = embedder.encode(corpus)

# Initialize FAISS index and add corpus embeddings
dimension = corpus_embeddings.shape[1]  # Embedding vector size
index = faiss.IndexFlatL2(dimension)    # L2 similarity metric
index.add(corpus_embeddings)            # Add embeddings to the index

# Step 3: Define the retriever
def retrieve_documents(query, top_k=3):
    # Encode the query
    query_embedding = embedder.encode([query])
    # Search the index for the most similar embeddings
    _, indices = index.search(query_embedding, top_k)
    # Retrieve the corresponding documents
    return [corpus[i] for i in indices[0]]

# Step 4: Integrate with a language model for generation
# Load a T5 model for text-to-text generation
generator = pipeline("text2text-generation", model="t5-small")

def generate_response(query, top_k=3):  
    # Retrieve relevant documents based on the query  
    retrieved_docs = retrieve_documents(query, top_k)  
      
    if len(retrieved_docs) == 0:  
        return "Sorry, I couldn't find relevant information."  
      
    # Use the first retrieved document as the input to the model  
    input_text = f"Translate to English: {retrieved_docs[0]}"  
  
    # Generate a response using the language model  
    response = generator(input_text, max_length=100, num_return_sequences=1)  
  
    # Extract the generated text and remove any leading/trailing whitespace  
    generated_text = response[0]["generated_text"].strip()  
  
    # Return the cleaned-up response  
    return generated_text

# Step 5: Test the RAG system
if __name__ == "__main__":
    query = "give me the seven wonders"
    response = generate_response(query, top_k=3)
    print(f"Query: {query}")
    print(f"Response: {response}")
