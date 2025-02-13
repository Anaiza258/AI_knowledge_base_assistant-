from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API initialization
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# App initialization
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")  # Needed for session management

# Password for Add Data access (stored in environment variable)
ADD_DATA_PASSWORD = os.getenv("ADD_DATA_PASSWORD", "default_password")

# Load and initialize data
csv_file = os.path.join(os.getcwd(), "problems.csv")

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['Problems', 'Solutions', 'Date'])
    df.to_csv(csv_file, index=False)
else:
    df = pd.read_csv(csv_file, encoding='cp1252')

# embed the problems dataset
def embed_problems(df):
    embeddings = []
    for problem in df['Problems']:
        response = genai.embed_content(model="models/text-embedding-004", content=problem)
        embedding = response['embedding']
        embeddings.append(embedding)
    df['Problem_Embeddings'] = embeddings
    return df

df = embed_problems(df)

# home page route
@app.route('/')
def index():
    return render_template('display_right.html')


# get solution
@app.route('/get_solution', methods=['POST'])
def get_solution():
    query = request.form.get('query')
    solution, solution_date, less_relevant_results = find_solution(query,df)

    show_llm_button = solution == 'No suitable solution found.'
    # prepare the response
    response_data = {
        'solution': solution,
        'solution_date': solution_date,
        'less_relevant_results': less_relevant_results,
        'llm_response': None,
        'show_llm_button': show_llm_button
    }

    return jsonify(response_data)

# find the query result
def find_solution(query, df):
    response = genai.embed_content(content=query, model="models/text-embedding-004")
    query_embedding = response['embedding']

    # Compute similarity scores for all problem embeddings
    similarity_scores = [cosine_similarity([query_embedding],[embedding])[0][0] for embedding in df['Problem_Embeddings']]

    # Threshold values
    primary_threshold = 0.8  # for best match
    secondary_threshold = 0.5  # for less relevant solution

    # Get the index of the best match
    best_match_idx = np.argmax(similarity_scores)
    best_match_score = similarity_scores[best_match_idx]

    # check if the best match meets the threshold
    if best_match_score < primary_threshold:
        return "No suitable solution found.", None, []
    
    # retrieve solution and date for best match
    solution = df['Solutions'].iloc[best_match_idx]
    solution_date = df['Date'].iloc[best_match_idx]

    # sort all similarity scores with their indexes
    sorted_results = sorted(
        enumerate(similarity_scores), 
        key = lambda x: x[1],  # sort by similarity score
        reverse = True  # highest score first
    )

    # fetch the top 2 less relevant results above secondary threshold
    less_relevant_results =[]
    for idx, score in sorted_results:
        if idx == best_match_idx:
            continue   # skip the best match
        if score >= secondary_threshold:
            less_relevant_results.append({
                'problem': df['Problems'].iloc[idx],
                'solution': df['Solutions'].iloc[idx],
                'date': df['Date'].iloc[idx]
            })
        if len(less_relevant_results) >= 2:  # limit to 2 results
            break

    return solution, solution_date, less_relevant_results

# route to generate LLM response
@app.route('/generate_llm_response', methods=['POST'])
def generate_llm_response():
    query = request.form.get('query')
    llm_response = get_llm_response(query)  # call the llm only when required
    return jsonify({'llm_response': llm_response})

#  LLM-base response function
def get_llm_response(query):
    prompt = (
        f"You are a highly intelligent and concise AI assistant. "
        f"Your role is to solve user problems by providing clear, accurate, and helpful solutions. "
        f"Analyze the user's query carefully and provide a complete response within 150 words. "
        f"Ensure that the answer is fully contained within this limit, avoiding unnecessary details or repetition. "
        f"Keep sentences short and to the point. "
        f"\n\nProblem: {query}\n"
    )

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        prompt
    )

    return response.candidates[0].content.parts[0].text


# authentication / admin dashboard
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADD_DATA_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('add_data'))
        else:
            return render_template('login.html', error="Incorrect password. Please try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))


# Password-protected Add Data route
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
                response = genai.embed_content(content=problem, model="models/text-embedding-004")
                new_embedding = response['embedding']

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
    data = df.sort_values(by='Date', ascending= False)[['Problems', 'Solutions', 'Date']].to_dict(orient='records')
    return render_template('add_data_dashboard.html', data=data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Koyeb provides PORT dynamically
    app.run(host="0.0.0.0", port=port)
