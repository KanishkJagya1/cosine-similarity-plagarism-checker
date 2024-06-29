import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fetch cpp files from the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.cpp')]
# Read the contents of the cpp files
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Function to vectorize the text using TF-IDF
def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

# Function to calculate cosine similarity between two vectors
def similarity(vec1, vec2):
    return cosine_similarity([vec1, vec2])[0][1]

# Vectorize the student notes
vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

# Function to check for plagiarism
def check_plagiarism():
    global s_vectors
    for i, (student_a, text_vector_a) in enumerate(s_vectors):
        for j in range(i + 1, len(s_vectors)):
            student_b, text_vector_b = s_vectors[j]
            sim_score = similarity(text_vector_a, text_vector_b)
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

# Function to print similar files based on the similarity threshold
def print_similar_files(plagiarism_results):
    for file1, file2, similarity_score in plagiarism_results:
        if similarity_score > 0.5:
            print(f"Files {file1} and {file2} have a cosine similarity of {similarity_score:.2f}")
            print("Therefore the files are similar and marks will be deducted accordingly.")

