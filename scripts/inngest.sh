#!/usr/bin/env bash

set -a  
source "$(dirname "$0")/../.env"
set +a

pnpx inngest-cli@latest dev -u http://${BACKEND_HOST:-localhost}:${BACKEND_PORT:-8000}/api/inngest --no-discovery --no-poll