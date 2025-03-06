# FROM python:3.9-slim # Uncomment for building from scratch

# Been using the base layer after my first build for quick cleanup.
FROM agents_execution_service:latest

WORKDIR /app

# Install required packages
RUN pip install flask gunicorn faker numpy matplotlib pandas scikit-learn opencv-python bs4 torch 

# Create directories
RUN mkdir -p /app/venvs /app/pip_cache

# Copy service files
COPY service.py server.py /app/

# # Set up a non-root user for added security
# RUN adduser --disabled-password --gecos '' appuser
# RUN chown -R appuser:appuser /app
# USER appuser

# Run the server with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]