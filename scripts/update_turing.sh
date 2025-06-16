#!/bin/bash

DESTINATION="azbelikoff@turing.wpi.edu:/home/azbelikoff/projects/2025_Summer"

echo "Syncing files to $DESTINATION"

rsync -Phavz --stats \
    -e "ssh -i ~/.ssh/id_ed25519" \
    -e "ssh -i ~/.ssh/id_rsa" \
    --exclude *__pycache__* \
    llama_server_discovery.py requirements.txt scripts discovery \
    "$DESTINATION"