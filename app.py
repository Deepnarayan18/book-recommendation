from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data from CSV file
df = pd.read_csv('books_data.csv')

# Create a combined feature of title and authors for TF-IDF Vectorization
df['combined'] = df['title'] + ' ' + df['authors']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    
    # Find the index of the book with the given title
    if title not in df['title'].values:
        return render_template('result.html', books=[], message="Book not found.")
    
    index = df[df['title'] == title].index[0]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    
    # Get the indices of the most similar books
    sim_indices = cosine_sim.argsort()[-6:-1]  # Exclude the book itself
    
    # Get the recommended books
    recommended_books = df.iloc[sim_indices]
    
    # Prepare books data for rendering
    books_data = recommended_books[['title', 'authors', 'average_rating']].to_dict('records')
    
    return render_template('result.html', books=books_data)

if __name__ == '__main__':
    app.run(debug=True)
