#!/bin/bash

rsync -Phavz --stats --exclude='utils/__pycache__' prompts utils scripts discovery run_llama_server.py llama_server_discovery.py llama_server_no_discovery.py llama_server_no_discovery_original.py llama_server_no_discovery_qrdata_react.py prompts.py tools azbelikoff@turing.wpi.edu:/home/azbelikoff/projects/2025_Summer