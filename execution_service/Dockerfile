FROM python:3.9-slim

WORKDIR /app

# Install required packages
RUN pip install flask gunicorn numpy matplotlib pandas scikit-learn

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