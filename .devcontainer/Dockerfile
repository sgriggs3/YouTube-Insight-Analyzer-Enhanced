# Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# Install TypeScript
RUN npm install -g typescript

# Install project dependencies
COPY package*.json ./
RUN npm install

# Copy the project files
COPY . .

# Run additional commands if needed
