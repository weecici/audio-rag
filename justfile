set dotenv-load

run-backend:
  uv run uvicorn app.main:app --host ${BACKEND_HOST:-localhost} --port ${BACKEND_PORT:-8000}
