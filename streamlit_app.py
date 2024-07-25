 
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
    st.title("AI Tools Recommender")
    st.write("Get recommendations for AI tools based on your query.")

    query = st.text_input("Enter the name of an AI tool:")

    if query:
        result = get_recommendations(query)
        if result:
            st.write("Recommendations:")
            for tool in result:
                st.write(f"**Tool Name:** {tool[0]}")
                st.write(f"**Site URL:** {tool[1]}")
                st.write(f"**Categories:** {tool[2]}")
                st.write(f"**Per Month:** {tool[3]}")
                st.write(f"**Types For Use:** {tool[4]}")
                st.write("---")

if __name__ == "__main__":
    main()

