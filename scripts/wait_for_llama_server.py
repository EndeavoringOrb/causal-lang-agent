import requests
import time
import signal
import sys
import os
import argparse

# Configuration defaults
HOST = "localhost"
DEFAULT_PORT = 55552
CHECK_INTERVAL = 2  # seconds
TIMEOUT = 300  # seconds

# Global variable to track the start time
start_time = time.time()


def check_server_health(health_endpoint):
    try:
        response = requests.get(health_endpoint, timeout=5)
        if response.status_code == 200:
            status = response.json().get("status", "").lower()
            if status == "ok":
                print("llama-server is ready.")
                return True
            else:
                print(f"Server status: {status}")
        else:
            print(f"Unexpected status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Health check failed: {e}")
    return False


def handle_termination(signum, frame):
    print(f"Received termination signal ({signum}). Shutting down gracefully.")
    sys.exit(0)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Check llama-server health.")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number to check (default: 55552)",
    )
    args = parser.parse_args()

    port = args.port
    health_endpoint = f"http://{HOST}:{port}/health"

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)

    print(f"Checking llama-server health at {health_endpoint}...")
    while True:
        if check_server_health(health_endpoint):
            break
        if time.time() - start_time > TIMEOUT:
            print("Timeout reached. Exiting with error.")
            sys.exit(1)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
