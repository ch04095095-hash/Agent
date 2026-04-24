#!/bin/bash

if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ]; then
  if [ -n "${LINARO_GCC_ROOT:-}" ]; then
    if [ -x "${LINARO_GCC_ROOT}/bin/aarch64-linux-gnu-gcc" ] && [ -x "${LINARO_GCC_ROOT}/bin/aarch64-linux-gnu-g++" ]; then
      export CC="${LINARO_GCC_ROOT}/bin/aarch64-linux-gnu-gcc"
      export CXX="${LINARO_GCC_ROOT}/bin/aarch64-linux-gnu-g++"
    elif [ -x "${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-gcc" ] && [ -x "${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-g++" ]; then
      export CC="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-gcc"
      export CXX="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-g++"
    fi
  fi
fi

if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ]; then
  if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 && command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    export CC="$(command -v aarch64-linux-gnu-gcc)"
    export CXX="$(command -v aarch64-linux-gnu-g++)"
  elif command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1 && command -v aarch64-none-linux-gnu-g++ >/dev/null 2>&1; then
    export CC="$(command -v aarch64-none-linux-gnu-gcc)"
    export CXX="$(command -v aarch64-none-linux-gnu-g++)"
  else
    echo "Please set CC/CXX or LINARO_GCC_ROOT correctly"
    exit 1
  fi
fi

if [ -d "build" ]; then
  rm -rf build
fi

mkdir build
cd build
cmake ..
make
