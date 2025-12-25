# Docker Deployment Guide for LLM Council

This guide explains how to build, run, and deploy LLM Council using Docker and Docker Compose.

## Prerequisites

- Docker 20.10 or later
- Docker Compose 2.0 or later

## Quick Start

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd llm-council
   ```

2. **Create a `.env` file from the example:**
   ```bash
   cp .env.example .env
   ```

3. **Edit the `.env` file with your configuration:**
   - Set your API keys (at minimum `ANTHROPIC_API_KEY`)
   - Set a secure `SECRET_KEY` for JWT tokens
   - Adjust other settings as needed

4. **Build and start all services:**
   ```bash
   docker compose up --build
   ```

5. **Run database migrations (first time only):**
   ```bash
   docker compose --profile migrate up migrations
   ```

6. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8001
   - API Health Check: http://localhost:8001/

## Services

The Docker Compose configuration includes the following services:

| Service | Description | Port |
|---------|-------------|------|
| postgres | PostgreSQL database | 5432 |
| redis | Redis cache | 6379 |
| backend | FastAPI backend | 8001 |
| frontend | Vite frontend (nginx) | 5173 |
| migrations | Database migrations (one-time) | - |

## Docker Compose Commands

### Start all services
```bash
docker compose up
```

### Start in detached mode (background)
```bash
docker compose up -d
```

### Build and start
```bash
docker compose up --build
```

### Stop all services
```bash
docker compose down
```

### Stop and remove volumes (deletes data!)
```bash
docker compose down -v
```

### View logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
```

### Run migrations
```bash
docker compose --profile migrate run --rm migrations
```

## Environment Variables

Key environment variables for Docker deployment:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | (auto-generated) |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `REDIS_ENABLED` | Enable Redis caching | `true` |
| `SECRET_KEY` | JWT signing key | `change_this_...` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration | `30` |
| `BACKEND_PORT` | Backend port | `8001` |
| `FRONTEND_PORT` | Frontend port | `5173` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `REDIS_PORT` | Redis port | `6379` |

## Production Deployment

### Security Considerations

1. **Change the default SECRET_KEY:**
   ```bash
   # Generate a secure random key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Use strong database passwords**
3. **Enable HTTPS** (use a reverse proxy like nginx or Traefik)
4. **Limit exposed ports** - remove port mappings for internal services
5. **Set resource limits** to prevent container abuse

### Production Docker Compose

Create a `docker-compose.prod.yml` for production overrides:

```yaml
version: '3.8'

services:
  backend:
    restart: always
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  postgres:
    restart: always
```

Run with:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Reverse Proxy Configuration

For production, use a reverse proxy for SSL termination:

**nginx.conf example:**
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://localhost:5173/;
        proxy_set_header Host $host;
    }
}
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs backend

# Check container status
docker compose ps
```

### Database connection errors
- Ensure PostgreSQL container is healthy: `docker compose ps`
- Check DATABASE_URL environment variable
- Verify database is initialized: run migrations

### Migration errors
```bash
# Check current migration status
docker compose exec backend uv run alembic current

# Reset database (WARNING: deletes data)
docker compose exec backend uv run alembic downgrade base
docker compose --profile migrate run --rm migrations
```

### High memory usage
- Reduce Redis maxmemory in docker-compose.yml
- Set container resource limits
- Clean up unused Docker resources: `docker system prune -a`

## Development with Docker

For development, you can run the backend in development mode with hot-reload:

```bash
# Override command for development
docker compose run --rm backend uv run uvicorn backend.main:app --reload --host 0.0.0.0
```

Or use the existing development setup from `start.sh` instead of Docker.

## Building Individual Images

### Backend only
```bash
docker build -t llm-council-backend .
```

### Frontend only
```bash
docker build -t llm-council-frontend ./frontend
```

## Backup and Restore

### Backup database
```bash
docker compose exec postgres pg_dump -U llmcouncil llmcouncil > backup.sql
```

### Restore database
```bash
cat backup.sql | docker compose exec -T postgres psql -U llmcouncil llmcouncil
```

### Backup volumes
```bash
docker run --rm -v llm_council_postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```
