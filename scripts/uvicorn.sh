#!/usr/bin/env bash

set -a
source "$(dirname "$0")/../.env"
set +a

uv run uvicorn src.main:app --host ${BACKEND_HOST:-localhost} --port ${BACKEND_PORT:-8000}