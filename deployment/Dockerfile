# Use an official Python runtime as a parent image
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . /

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME CALLTRANSCRIPTANALYSIS

# Run app.py when the container launches
CMD ["python3", "main.py"]