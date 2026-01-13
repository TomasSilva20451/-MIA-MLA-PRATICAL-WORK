#!/bin/bash

# Docker Build and Run Script for Financial Risk Classification API
# Following instructions from "9 - MIA - MLA - Deploying Models.pdf"

echo "Building Docker image..."
docker build -t financial-risk-api .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
    
    # Check if container already exists and remove it
    if [ "$(docker ps -aq -f name=financial-risk-api)" ]; then
        echo "Removing existing container..."
        docker rm -f financial-risk-api
    fi
    
    echo "Running Docker container..."
    docker run -d --name financial-risk-api -p 8000:8000 financial-risk-api
    
    if [ $? -eq 0 ]; then
        echo "✓ Container started successfully"
        echo ""
        echo "API is now available at:"
        echo "  - Web Interface: http://localhost:8000"
        echo "  - API Docs (Swagger): http://localhost:8000/docs"
        echo "  - API Docs (ReDoc): http://localhost:8000/redoc"
        echo ""
        echo "To check container logs:"
        echo "  docker logs financial-risk-api"
        echo ""
        echo "To stop the container:"
        echo "  docker stop financial-risk-api"
        echo ""
        echo "To remove the container:"
        echo "  docker rm financial-risk-api"
    else
        echo "✗ Failed to start container"
        exit 1
    fi
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

