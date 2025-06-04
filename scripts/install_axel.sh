#!/bin/bash

set -e

log_info()  { echo -e "[INFO]  $*"; }
log_warn()  { echo -e "[WARN]  $*"; }
log_error() { echo -e "[ERROR] $*" >&2; }

log_info "Checking for existing Axel installation..."
if command -v axel &>/dev/null; then
    log_warn "Axel is already installed. Skipping installation."
    exit 0
fi

log_info "Cleaning up any previous Axel downloads..."
rm -f axel-2.17.14.tar.gz
rm -rf axel-2.17.14

log_info "Downloading Axel source..."
wget https://github.com/axel-download-accelerator/axel/releases/download/v2.17.14/axel-2.17.14.tar.gz

log_info "Extracting archive..."
tar -zxvf axel-2.17.14.tar.gz
cd axel-2.17.14

log_info "Loading required modules..."
module load openssl/3.4.0
module load gcc-runtime/13.2.0/s3f7i6x
module load zlib-ng/2.1.5/uctcqfl

log_info "Detecting OpenSSL and Zlib-ng paths..."
OPENSSL_ROOT=$(dirname $(dirname $(which openssl)))
ZLIB_ROOT=$(dirname $(dirname $(module show zlib-ng/2.1.5/uctcqfl 2>&1 | \
    grep 'PKG_CONFIG_PATH' | \
    grep -o '/cm/shared/spack/opt/[^"]*')))

if [ ! -d "$ZLIB_ROOT" ]; then
    log_error "ZLIB_ROOT not found or invalid: $ZLIB_ROOT"
    exit 1
fi

log_info "Using OpenSSL root: $OPENSSL_ROOT"
log_info "Using Zlib-ng root: $ZLIB_ROOT"

log_info "Configuring build..."
export CPPFLAGS="-I$OPENSSL_ROOT/include -I$ZLIB_ROOT/include"
export LDFLAGS="-L$OPENSSL_ROOT/lib64 -L$ZLIB_ROOT/lib"
export LIBS="-ldl -lz"
export PKG_CONFIG_PATH="$OPENSSL_ROOT/lib64/pkgconfig:$ZLIB_ROOT/lib/pkgconfig"
export ac_cv_func_ASN1_STRING_get0_data=yes

./configure --prefix=$HOME/.local --disable-nls
log_info "Cleaning previous builds (if any)..."
make clean || true

log_info "Compiling Axel..."
make

log_info "Installing Axel to $HOME/.local/bin"
make install

cd ..
log_info "Cleaning up downloaded files..."
rm -f axel-2.17.14.tar.gz
rm -rf axel-2.17.14

log_info "Renaming original axel binary to axel.real..."
mv ~/.local/bin/axel ~/.local/bin/axel.real

log_info "Creating run wrapper script at ~/.local/bin/axel.sh..."
cat << 'EOF' > ~/.local/bin/axel.sh
#!/bin/bash

module load openssl/3.4.0
module load zlib-ng/2.1.5/uctcqfl

OPENSSL_ROOT=$(dirname $(dirname $(which openssl)))
ZLIB_ROOT=$(dirname $(dirname $(module show zlib-ng/2.1.5/uctcqfl 2>&1 | \
    grep 'PKG_CONFIG_PATH' | \
    grep -o '/cm/shared/spack/opt/[^"]*')))

if [ ! -d "$ZLIB_ROOT" ]; then
    echo "[ERROR] ZLIB_ROOT not found or invalid." >&2
    exit 1
fi

OPENSSL_LIB="$OPENSSL_ROOT/lib64"
ZLIB_LIB="$ZLIB_ROOT/lib"
export LD_LIBRARY_PATH="$OPENSSL_LIB:$ZLIB_LIB:$LD_LIBRARY_PATH"

exec "$HOME/.local/bin/axel.real" "$@"
EOF

chmod +x ~/.local/bin/axel.sh

log_info "Creating symlink ~/.local/bin/axel -> axel.sh"
ln -sfn ~/.local/bin/axel.sh ~/.local/bin/axel

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    log_info "Added ~/.local/bin to PATH in ~/.bashrc"
fi

log_info "Axel installation and wrapper setup complete."