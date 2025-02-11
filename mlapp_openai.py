from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key loaded correctly
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")  # Needed for session management

# Password for Add Data access (stored in environment variable)
ADD_DATA_PASSWORD = os.getenv("ADD_DATA_PASSWORD", "default_password")

# Step 1: Load and Initialize Data
csv_file = r'D:\AI Projects\ml_app_prob_sol (working)\problems.csv'
df = pd.read_csv(csv_file,  encoding='utf-8')

# embed the problems dataset
def embed_problems(df):
    embeddings = []
    for problem in df['Problems']:
        response = openai.embeddings.create(input=problem, model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    df['Problem_Embeddings'] = embeddings
    return df

# Embed problems on startup
df = embed_problems(df)

# home page route
@app.route('/')
def index():
    return render_template('display_right.html')

# get solution
@app.route('/get_solution', methods=['POST'])
def get_solution():
    query = request.form.get('query')
    solution, solution_date, less_relevant_results = find_solution(query, df)

    show_llm_button = solution == 'No suitable solution found.'
    # Prepare the response
    response_data = {
        'solution': solution,
        'solution_date': solution_date,
        'less_relevant_results': less_relevant_results,
        'llm_response' : None,   # llm response will be updated here
        'show_llm_button' : show_llm_button
    }

    return jsonify(response_data) 


# Find the query result 
def find_solution(query, df):
    # Generate embedding for the query
    response = openai.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding

    # Compute similarity scores for all problem embeddings
    similarity_scores = [cosine_similarity([query_embedding], [embedding])[0][0] for embedding in df['Problem_Embeddings']]

    # Thresholds
    primary_threshold = 0.8  # For the best match
    secondary_threshold = 0.5  # For less relevant results

    # Get the index of the best match
    best_match_idx = np.argmax(similarity_scores)
    best_match_score = similarity_scores[best_match_idx]

    # Check if the best match meets the threshold
    if best_match_score < primary_threshold:
        return "No suitable solution found.", None, []

    # Retrieve solution and date for the best match
    solution = df['Solutions'].iloc[best_match_idx]
    solution_date = df['Date'].iloc[best_match_idx]

    # Sort all similarity scores with their indices
    sorted_results = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],  # Sort by similarity score
        reverse=True  # Highest scores first
    )

    # Fetch the top 2 less relevant results above the secondary threshold
    less_relevant_results = []
    for idx, score in sorted_results:
        if idx == best_match_idx:
            continue  # Skip the best match
        if score >= secondary_threshold:
            less_relevant_results.append({
                'problem': df['Problems'].iloc[idx],
                'solution': df['Solutions'].iloc[idx],
                'date': df['Date'].iloc[idx]
            })
        if len(less_relevant_results) >= 2:  # Limit to 2 results
            break

    return solution, solution_date, less_relevant_results


# route to generate LLM response
@app.route('/generate_llm_response', methods=['POST'])
def generate_llm_response():
    query = request.form.get('query')
    llm_response = get_llm_response(query)  # Call the LLM only when required
    return jsonify({'llm_response': llm_response}) 


# Generate an LLM-based response
def get_llm_response(query):
    prompt = (
        f"You are a highly intelligent and concise AI assistant. "
        f"Your role is to solve user problems by providing clear, accurate, and helpful solutions. "
        f"Analyze the user's query carefully and provide a complete response within 150 words. "
        f"Ensure that the answer is fully contained within this limit, avoiding unnecessary details or repetition. "
        f"Keep sentences short and to the point. "
        f"\n\nProblem: {query}\n"
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,  # 200 tokens should fit ~150 words
        temperature=0.3,  
    )

    return response.choices[0].message.content.strip()


# Login route for password protection
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADD_DATA_PASSWORD:
            session['authenticated'] = True  # Mark the session as authenticated
            return redirect(url_for('add_data'))
        else:
            return render_template('login.html', error="Incorrect password. Please try again.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

# add data 
@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    global df

    if request.method == 'POST':
        try:
            # Get JSON data from AJAX request
            data = request.get_json()

            if not data:
                return jsonify({'status': 'error', 'message': 'Invalid request data'}), 400

            action = data['action']

            # Handle Add Action
            if action == 'add':
                problem = data.get('problem')
                solution = data.get('solution')
                date = data.get('date')

                if not problem or not solution or not date:
                    return jsonify({'status': 'error', 'message': 'All fields (problem, solution, date) are required.'}), 400

                # Generate embeddings for the new problem
                response = openai.embeddings.create(input=problem, model="text-embedding-ada-002")
                new_embedding = response.data[0].embedding

                # Add new row to DataFrame
                new_row = {
                    'Problems': problem,
                    'Solutions': solution,
                    'Date': date,
                    'Problem_Embeddings': new_embedding  # Embedding saved only in DataFrame
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Save only problem, solution, and date to CSV
                df[['Problems', 'Solutions', 'Date']].to_csv(csv_file, index=False)

                return jsonify({'status': 'success', 'message': 'Data added successfully!'})

            # Handle Edit Action
            elif action == 'edit':
                row_index = data.get('rowIndex')
                problem = data.get('problem')
                solution = data.get('solution')
                date = data.get('date')

                if row_index is None or not problem or not solution or not date:
                    return jsonify({'status': 'error', 'message': 'Invalid edit data.'}), 400

                # Update the specific row in DataFrame
                row_index = int(row_index)
                df.loc[row_index, 'Problems'] = problem
                df.loc[row_index, 'Solutions'] = solution
                df.loc[row_index, 'Date'] = date

                # Save updated data to CSV
                df[['Problems', 'Solutions', 'Date']].to_csv(csv_file, index=False)

                return jsonify({'status': 'success', 'message': 'Data updated successfully!'})

            # Handle Delete Action
            elif action == 'delete':
                row_index = data.get('rowIndex')

                if row_index is None:
                    return jsonify({'status': 'error', 'message': 'Invalid delete data.'}), 400

                # Drop the row from DataFrame
                row_index = int(row_index)
                df = df.drop(row_index).reset_index(drop=True)

                # Save updated data to CSV
                df[['Problems', 'Solutions', 'Date']].to_csv(csv_file, index=False)

                return jsonify({'status': 'success', 'message': 'Data deleted successfully!'})

            else:
                return jsonify({'status': 'error', 'message': 'Invalid action.'}), 400

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Handle GET request - render HTML template with DataFrame data
    data = df[['Problems', 'Solutions', 'Date']].to_dict(orient='records')
    return render_template('add_data_dashboard.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
