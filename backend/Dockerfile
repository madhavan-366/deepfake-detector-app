# backend/Dockerfile

# Use Node 18
FROM node:18-alpine

# Set working folder
WORKDIR /usr/src/app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the backend code
COPY . .

# Expose the backend port
EXPOSE 3000

# Start the server
CMD ["node", "server.js"]