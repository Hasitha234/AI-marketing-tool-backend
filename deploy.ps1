Write-Host "Deploying AI Marketing Tool Authentication API" -ForegroundColor Green

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker and try again." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is installed
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Compose is not installed. Please install Docker Compose and try again." -ForegroundColor Red
    exit 1
}

# Build and deploy the containers
Write-Host "Building and deploying containers..." -ForegroundColor Yellow
docker-compose down
docker-compose build
docker-compose up -d

# Run database migrations
Write-Host "Running database migrations..." -ForegroundColor Yellow
docker-compose exec api alembic upgrade head

# Check if the containers are running
if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful!" -ForegroundColor Green
    Write-Host "API is available at http://localhost:8000" -ForegroundColor Cyan
    Write-Host "API documentation is available at http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "Database migrations completed successfully" -ForegroundColor Green
} else {
    Write-Host "Deployment failed. Please check the logs with 'docker-compose logs'." -ForegroundColor Red
    exit 1
} 