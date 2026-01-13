FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the source code directory into the container
COPY ./src /code/src

# Copy the trained pipeline directory (with the saved model files) into the container
COPY ./artifacts/pipeline /code/artifacts/pipeline

# Copy static files for web interface
COPY ./src/api/templates /code/src/api/templates

# Create artifacts/monitoring directory for runtime data
RUN mkdir -p /code/artifacts/monitoring

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

