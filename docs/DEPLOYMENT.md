# Deployment Guide - LLM Council

Complete guide to deploy LLM Council to another machine or server.

## Quick Start (Docker - Recommended)

### Prerequisites

On the target machine, install:
- **Docker** (20.10+)
- **Docker Compose** (2.0+)

```bash
# Install Docker on Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin
```

### Step 1: Get API Keys

You need API keys for at least one LLM provider:

#### **HuggingFace API Key** (Free tier available)
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" permissions (for inference)
4. Copy the token (starts with `hf_...`)

#### **Chutes AI API Key**
1. Sign up at Chutes AI platform
2. Get your API key from account settings

### Step 2: Deploy

```bash
# 1. Clone the repository
git clone <your-repo-url> llm-council
cd llm-council

# 2. Create environment file
cp .env.example .env

# 3. Edit .env with your API keys
nano .env
```

**Edit `.env` - Minimum required:**
```bash
# Required - Your LLM Provider Keys
HUGGINGFACE_API_KEY="hf_your_key_here"
CHUTES_AI_API_KEY="your_chutes_key_here"

# Required - Security (generate random key)
SECRET_KEY="$(openssl rand -base64 32)"

# Optional - Ports (defaults shown)
BACKEND_PORT="8001"
FRONTEND_PORT="5173"
```

```bash
# 4. Start all services
docker compose up -d --build

# 5. Run database migrations
docker compose --profile migrate up migrations

# 6. Check status
docker compose ps
```

### Step 3: Access the Application

- **Frontend**: http://your-server-ip:5173
- **Backend API**: http://your-server-ip:8001
- **Health Check**: http://your-server-ip:8001/

### Default Login

First user is created on first login. Register at:
http://your-server-ip:5173/register

---

## Manual Deployment (Without Docker)

### Prerequisites

```bash
# Python 3.10+
python3 --version

# PostgreSQL 15+
psql --version

# Redis (optional, for caching)
redis-server --version
```

### Step 1: Install Dependencies

```bash
cd llm-council

# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Step 2: Set Up Database

```bash
# Create PostgreSQL database
sudo -u postgres psql
CREATE DATABASE llmcouncil;
CREATE USER llmcouncil WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE llmcouncil TO llmcouncil;
\q

# Set DATABASE_URL in .env
echo "DATABASE_URL=postgresql+asyncpg://llmcouncil:your_password@localhost:5432/llmcouncil" >> .env

# Run migrations
uv run alembic upgrade head
```

### Step 3: Configure Environment

```bash
cp .env.example .env
nano .env
```

**Required settings in `.env`:**
```bash
# LLM Provider Keys (at least one)
HUGGINGFACE_API_KEY="hf_..."
CHUTES_AI_API_KEY="..."

# Security
SECRET_KEY="your-random-secret-key"

# Database
DATABASE_URL="postgresql+asyncpg://llmcouncil:password@localhost:5432/llmcouncil"
```

### Step 4: Start Services

```bash
# Start Redis (if using)
redis-server --port 6379 &

# Start Backend
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8001

# In another terminal, Start Frontend
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## Production Deployment

### Using systemd (Auto-restart on boot)

Create `/etc/systemd/system/llm-council.service`:

```ini
[Unit]
Description=LLM Council Backend
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/llm-council
Environment="PATH=/var/www/llm-council/.venv/bin"
ExecStart=/var/www/llm-council/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable llm-council
sudo systemctl start llm-council
sudo systemctl status llm-council
```

### Using nginx (Reverse Proxy)

Create `/etc/nginx/sites-available/llm-council`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/llm-council /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### HTTPS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Provider Configuration

### Available Models

**HuggingFace:**
- `meta-llama/Llama-3-70b-chat-hf`
- `meta-llama/Llama-3-8b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `google/gemma-2-9b-it`
- `Qwen/Qwen2-72B-Instruct`

**Chutes AI:**
- `gpt-4o`
- `gpt-4o-mini`
- `claude-3-opus`
- `claude-3-sonnet`
- `llama-3-70b`

### Using Only HuggingFace and Chutes AI

The project is configured to use these providers by default. Just set the API keys in `.env`:

```bash
HUGGINGFACE_API_KEY="hf_your_key_here"
CHUTES_AI_API_KEY="your_chutes_key_here"
```

The providers will auto-detect when their API keys are present and enable themselves.

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs backend

# Check container status
docker compose ps
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check DATABASE_URL in .env
grep DATABASE_URL .env
```

### API Key errors

```bash
# Verify keys are set in .env
grep HUGGINGFACE_API_KEY .env
grep CHUTES_AI_API_KEY .env

# Test HuggingFace key
curl -H "Authorization: Bearer hf_your_key" https://huggingface.co/api/models
```

### Migration errors

```bash
# Reset and re-run migrations
docker compose exec backend uv run alembic downgrade base
docker compose --profile migrate up migrations
```

---

## Backup and Restore

### Backup database

```bash
docker compose exec postgres pg_dump -U llmcouncil llmcouncil > backup.sql
```

### Restore database

```bash
cat backup.sql | docker compose exec -T postgres psql -U llmcouncil llmcouncil
```

---

## Monitoring

### Check logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
```

### Health checks

```bash
# Backend health
curl http://localhost:8001/

# Database health
docker compose exec postgres pg_isready -U llmcouncil
```

### Resource usage

```bash
docker stats
```
