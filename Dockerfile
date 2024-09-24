# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy all CSV files into the container
COPY *.xlsx /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy the images directory into the container
COPY images /app/images

# Expose port 8502
EXPOSE 8502



# Run PJM_Main.py when the container launches
CMD ["streamlit", "run", "--server.port", "8502", "test.py"]
