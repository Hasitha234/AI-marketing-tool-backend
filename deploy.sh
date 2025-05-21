#!/bin/bash

echo "Deploying AI Marketing Tool Authentication API"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Build and deploy the containers
echo "Building and deploying containers..."
docker-compose down
docker-compose build
docker-compose up -d

# Run database migrations
echo "Running database migrations..."
docker-compose exec api alembic upgrade head

# Check if the containers are running
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "API is available at http://localhost:8000"
    echo "API documentation is available at http://localhost:8000/docs"
    echo "Database migrations completed successfully"
else
    echo "Deployment failed. Please check the logs with 'docker-compose logs'."
    exit 1
fi 