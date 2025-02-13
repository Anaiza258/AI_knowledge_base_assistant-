FROM python:3.9-21

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "gemini_app:app"]
