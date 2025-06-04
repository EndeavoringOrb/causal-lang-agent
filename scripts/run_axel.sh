#!/bin/bash

# Load required modules
module load openssl/3.4.0
module load zlib-ng/2.1.5/uctcqfl

# Get OpenSSL and Zlib-ng paths
OPENSSL_ROOT=$(dirname $(dirname $(which openssl)))
ZLIB_ROOT=$(dirname $(dirname $(module show zlib-ng/2.1.5/uctcqfl 2>&1 | \
    grep 'PKG_CONFIG_PATH' | \
    grep -o '/cm/shared/spack/opt/[^"]*')))

if [ ! -d "$ZLIB_ROOT" ]; then
    echo "Error: ZLIB_ROOT not found or invalid."
    exit 1
fi

# Define library paths
OPENSSL_LIB="$OPENSSL_ROOT/lib64"
ZLIB_LIB="$ZLIB_ROOT/lib"

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$OPENSSL_LIB:$ZLIB_LIB:$LD_LIBRARY_PATH"

# Run the actual axel executable
exec "$HOME/.local/bin/axel.real" "$@"