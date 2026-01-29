# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files
COPY . /code

# Run the app on port 7860 (Required for Hugging Face)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]