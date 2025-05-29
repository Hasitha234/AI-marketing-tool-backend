# Default environment
ENVIRONMENT=${1:-development}

echo "Deploying AI Marketing Tool Backend in $ENVIRONMENT mode"

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

# Set environment-specific variables
if [ "$ENVIRONMENT" = "production" ]; then
    export BUILD_TARGET=production
    export DEBUG=false
else
    export BUILD_TARGET=development
    export DEBUG=true
fi

# Create necessary directories
mkdir -p logs models certs tests

# Build and deploy the containers
echo "Building and deploying containers..."
docker-compose down
docker-compose build
docker-compose up -d

# Wait for the database to be ready
echo "Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "Running database migrations..."
docker-compose exec -T api alembic upgrade head

# Check if the containers are running
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "API is available at http://localhost:8000"
    echo "API documentation is available at http://localhost:8000/docs"
    echo "Database migrations completed successfully"
    
    # Show container status
    echo -e "\nContainer Status:"
    docker-compose ps
    
    # Show logs
    echo -e "\nRecent Logs:"
    docker-compose logs --tail=50
else
    echo "Deployment failed. Please check the logs with 'docker-compose logs'."
    exit 1
fi 