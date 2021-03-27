#!/bin/bash

# Exit on first error
set -euo pipefail

docker-compose down
docker image rm profanity-check || true
docker builder prune -af
docker-compose up -d

sleep 5
docker ps
