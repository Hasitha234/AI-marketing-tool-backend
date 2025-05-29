# Default environment
param(
    [string]$Environment = "development"
)

Write-Host "Deploying AI Marketing Tool Backend in $Environment mode"

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker Desktop for Windows and try again."
    exit 1
}

# Check if Docker Compose is installed
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Compose is not installed. Please install Docker Desktop for Windows and try again."
    exit 1
}

# Set environment-specific variables
if ($Environment -eq "production") {
    $env:BUILD_TARGET = "production"
    $env:DEBUG = "false"
} else {
    $env:BUILD_TARGET = "development"
    $env:DEBUG = "true"
}

# Create necessary directories
New-Item -ItemType Directory -Force -Path logs, models, certs, tests | Out-Null

# Build and deploy the containers
Write-Host "Building and deploying containers..."
docker-compose down
docker-compose build
docker-compose up -d

# Wait for the database to be ready
Write-Host "Waiting for database to be ready..."
Start-Sleep -Seconds 10

# Run database migrations
Write-Host "Running database migrations..."
docker-compose exec -T api alembic upgrade head

# Check if the containers are running
if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful!"
    Write-Host "API is available at http://localhost:8000"
    Write-Host "API documentation is available at http://localhost:8000/docs"
    Write-Host "Database migrations completed successfully"
    
    # Show container status
    Write-Host "`nContainer Status:"
    docker-compose ps
    
    # Show logs
    Write-Host "`nRecent Logs:"
    docker-compose logs --tail=50
} else {
    Write-Host "Deployment failed. Please check the logs with 'docker-compose logs'."
    exit 1
} 