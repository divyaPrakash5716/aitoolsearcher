import streamlit as st
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = []

# Load data from the CSV file
with open("data.csv", encoding='iso-8859-1', mode="r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

def get_recommendations(tool_name):
    tool_name = tool_name.lower()
    tool_names = [row["Ai Tools Name"].lower() for row in data]
    tool_names.append(tool_name)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tool_names)
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    similar_tool_indices = cosine_similarities.argsort()[:-11:-1]
    recommended_tools = []

    for i in similar_tool_indices:
        tool = data[i]
        recommended_tool = [
            tool["Ai Tools Name"],
            tool["Site url"],
            tool["Categories"],
            tool["Per Month"],
            tool["Types For Use"]
        ]
        recommended_tools.append(recommended_tool)

    return recommended_tools

def main():
    st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
        }
        .header {
            text-align: center;
            padding: 1em 0;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            color: #4CAF50;
        }
        .header p {
            margin: 0;
            font-size: 1.2em;
        }
        .main-content {
            padding: 20px;
        }
        .section {
            margin: 20px 0;
        }
        h2 {
            font-size: 1.8em;
            color: #4CAF50;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        ul li {
            background-color: #333;
            margin: 5px 0;
            padding: 10px;
            border-radius: 4px;
        }
        .form-section {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label {
            display: block;
            margin: 10px 0 5px;
            color: #ccc;
        }
        input[type="text"],
        input[type="email"],
        textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #444;
            border-radius: 4px;
            background-color: #333;
            color: white;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .footer {
            text-align: center;
            padding: 10px 0;
            background-color: #4CAF50;
            color: white;
            position: absolute;
            width: 100%;
            bottom: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header"><h1>Jailxerox AI Tools Recommender</h1><p>Transforming digital into tangible with every print.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2>Get AI Tool Recommendations</h2>', unsafe_allow_html=True)

    query = st.text_input("Enter the name of an AI tool:")

    if query:
        result = get_recommendations(query)
        if result:
            st.write("Recommendations:")
            for tool in result:
                st.markdown(f"<div class='tool'><b>Tool Name:</b> {tool[0]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tool'><b>Site URL:</b> <a href='{tool[1]}' target='_blank'>{tool[1]}</a></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tool'><b>Categories:</b> {tool[2]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tool'><b>Per Month:</b> {tool[3]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='tool'><b>Types For Use:</b> {tool[4]}</div>", unsafe_allow_html=True)
                st.markdown("<hr
