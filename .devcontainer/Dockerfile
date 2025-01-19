FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && \
    apt-get install -y \
    nodejs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TypeScript and development tools
RUN npm install -g typescript ts-node

# Set working directory
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt package*.json ./

# Install Python and Node.js dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt vader_lexicon && \
    npm install

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace

# Expose port for Flask
EXPOSE 8080
