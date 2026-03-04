set dotenv-load

run-backend: run-docker-compose
  uv run uvicorn app.main:app --host ${BACKEND_HOST:-localhost} --port ${BACKEND_PORT:-8000}

run-docker-compose:
  docker compose up -d
