# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app
# Install build essentials
RUN apt-get update && apt-get install -y build-essential    
# Copy the current directory contents into the container at /app
COPY . /app
# Make port 8501 available to the world outside this container
EXPOSE 8501
# Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]





