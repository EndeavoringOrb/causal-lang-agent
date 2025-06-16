#!/bin/bash

DESTINATION="azbelikoff@turing.wpi.edu:/home/azbelikoff/Regi"

echo "Syncing files to $DESTINATION"

rsync -Phavz --stats \
    -e "ssh -i ~/.ssh/id_ed25519" \
    -e "ssh -i ~/.ssh/id_rsa" \
    llama_server_discovery.py requirements.txt scripts discovery \
    "$DESTINATION"