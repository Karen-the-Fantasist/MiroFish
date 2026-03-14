#!/bin/bash

echo "Stopping all services..."

# 停止前端 (vite)
pkill -f "vite" 2>/dev/null && echo "  ✓ Frontend stopped" || echo "  - Frontend not running"

# 停止后端 (Flask/Python)
pkill -f "python run.py" 2>/dev/null && echo "  ✓ Backend stopped" || echo "  - Backend not running"
pkill -f "uv run python" 2>/dev/null && echo "  ✓ Backend (uv) stopped" || true

# 停止 Docker 服务 (Neo4j + Qdrant)
docker compose down 2>/dev/null && echo "  ✓ Docker services stopped (Neo4j, Qdrant)" || echo "  - Docker services not running"

echo "All services stopped."
