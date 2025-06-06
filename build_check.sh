#!/bin/bash
cd "$(dirname "$0")"
cargo check --message-format short --quiet
echo "Exit code: $?"