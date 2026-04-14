#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File-based command relay client.
# Run this from the host / Claude Code side. It sends commands to the server
# running inside Docker by writing files to the shared relay directory.
#
# Usage:
#   bash client.sh handshake              - Connect to server
#   bash client.sh run <command...>        - Run a command and print output
#   bash client.sh cancel                 - Cancel the running command
#   bash client.sh status                  - Check server status
#
# Options:
#   --relay-dir <path>   Path to relay directory (default: <script_dir>/.relay)
#   --timeout <secs>     Timeout waiting for result (default: 600)

set -euo pipefail

RELAY_DIR=""
TIMEOUT=600
POLL_INTERVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse global options before subcommand
while [[ $# -gt 0 ]]; do
    case "$1" in
        --relay-dir) RELAY_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ -z "$RELAY_DIR" ]]; then
    RELAY_DIR="$SCRIPT_DIR/.relay"
fi

CMD_DIR="$RELAY_DIR/cmd"
RESULT_DIR="$RELAY_DIR/result"

SUBCOMMAND="${1:-}"
shift || true

case "$SUBCOMMAND" in
    handshake)
        # Check server is ready
        if [[ ! -f "$RELAY_DIR/server.ready" ]]; then
            echo "ERROR: Server not ready. Start server.sh in Docker first."
            exit 1
        fi
        SERVER_INFO=$(cat "$RELAY_DIR/server.ready")
        echo "Server found: $SERVER_INFO"

        # Send client handshake
        echo "$(hostname):$$:$(date -Iseconds)" > "$RELAY_DIR/client.ready"

        # Wait for server acknowledgment
        elapsed=0
        while [[ ! -f "$RELAY_DIR/handshake.done" ]]; do
            sleep "$POLL_INTERVAL"
            elapsed=$((elapsed + POLL_INTERVAL))
            if [[ $elapsed -ge 120 ]]; then
                echo "ERROR: Handshake timed out after 120s."
                exit 1
            fi
        done

        echo "Handshake complete."
        ;;

    run)
        # Verify handshake was done
        if [[ ! -f "$RELAY_DIR/handshake.done" ]]; then
            echo "ERROR: Not connected. Run 'client.sh handshake' first."
            exit 1
        fi

        # Generate a unique command ID (timestamp + PID to avoid collisions)
        cmd_id="$(date +%s%N)_$$"

        echo "[client] Running: $*"

        # Write the command file atomically (tmp + mv)
        echo "$*" > "$CMD_DIR/$cmd_id.sh.tmp"
        mv "$CMD_DIR/$cmd_id.sh.tmp" "$CMD_DIR/$cmd_id.sh"

        # Wait for result
        elapsed=0
        while [[ ! -f "$RESULT_DIR/$cmd_id.exit" ]]; do
            # Check if server is still alive
            if [[ ! -f "$RELAY_DIR/server.ready" ]]; then
                echo "ERROR: Server appears to have stopped."
                rm -f "$CMD_DIR/$cmd_id.sh"
                exit 1
            fi
            sleep "$POLL_INTERVAL"
            elapsed=$((elapsed + POLL_INTERVAL))
            if [[ $elapsed -ge $TIMEOUT ]]; then
                echo "ERROR: Command timed out after ${TIMEOUT}s."
                # Cancel the running command only if it is OUR command
                if [[ -f "$RELAY_DIR/running" ]]; then
                    running_info=$(cat "$RELAY_DIR/running" 2>/dev/null) || true
                    running_id="${running_info%%:*}"
                    if [[ "$running_id" == "$cmd_id" ]]; then
                        echo "Sending cancel signal..."
                        echo "$cmd_id" > "$RELAY_DIR/cancel"
                        for _ in $(seq 1 10); do
                            [[ -f "$RELAY_DIR/running" ]] || break
                            sleep 1
                        done
                    fi
                fi
                # Clean up command and any orphaned result files
                rm -f "$CMD_DIR/$cmd_id.sh"
                rm -f "$RESULT_DIR/$cmd_id.exit" "$RESULT_DIR/$cmd_id.log"
                exit 1
            fi
        done

        # Read and display results
        exit_code=$(cat "$RESULT_DIR/$cmd_id.exit")
        if ! [[ "$exit_code" =~ ^[0-9]+$ ]]; then
            echo "WARNING: Invalid exit code '$exit_code', defaulting to 1."
            exit_code=1
        fi
        if [[ -f "$RESULT_DIR/$cmd_id.log" ]]; then
            cat "$RESULT_DIR/$cmd_id.log"
        fi

        # Clean up result files
        rm -f "$RESULT_DIR/$cmd_id.exit" "$RESULT_DIR/$cmd_id.log"

        exit "$exit_code"
        ;;

    status)
        if [[ -f "$RELAY_DIR/server.ready" ]]; then
            echo "Server: $(cat "$RELAY_DIR/server.ready")"
        else
            echo "Server: not running"
        fi
        if [[ -f "$RELAY_DIR/handshake.done" ]]; then
            echo "Handshake: complete"
        elif [[ -f "$RELAY_DIR/client.ready" ]]; then
            echo "Handshake: pending"
        else
            echo "Handshake: not started"
        fi
        if [[ -f "$RELAY_DIR/running" ]]; then
            echo "Running: $(cat "$RELAY_DIR/running")"
        else
            echo "Running: (idle)"
        fi
        if [[ -d "$CMD_DIR" ]]; then
            pending=$(find "$CMD_DIR" -maxdepth 1 -type f -name '*.sh' 2>/dev/null | wc -l)
        else
            pending=0
        fi
        echo "Pending commands: $pending"
        ;;

    flush)
        if [[ -f "$RELAY_DIR/running" ]]; then
            echo "ERROR: A command is currently running. Cancel it first or wait for it to finish."
            exit 1
        fi
        if [[ -d "$RELAY_DIR" ]]; then
            # Clear handshake and command/result files, but keep server.ready
            rm -f "$RELAY_DIR/client.ready" "$RELAY_DIR/handshake.done"
            rm -rf "$CMD_DIR" "$RESULT_DIR"
            mkdir -p "$CMD_DIR" "$RESULT_DIR"
            echo "Relay state cleared (server.ready preserved): $RELAY_DIR"
        else
            echo "Relay directory does not exist: $RELAY_DIR"
        fi
        ;;

    cancel)
        # Check if there's a running command
        if [[ -f "$RELAY_DIR/running" ]]; then
            running_info=$(cat "$RELAY_DIR/running" 2>/dev/null) || true
            running_id="${running_info%%:*}"
            echo "Cancelling running command: $running_id"

            # Write cancel signal with cmd_id so server can verify the target
            echo "$running_id" > "$RELAY_DIR/cancel"

            # Wait for the server to process the cancellation
            elapsed=0
            while [[ -f "$RELAY_DIR/running" ]]; do
                sleep "$POLL_INTERVAL"
                elapsed=$((elapsed + POLL_INTERVAL))
                if [[ $elapsed -ge 30 ]]; then
                    echo "WARNING: Cancel signal sent but command still running after 30s."
                    exit 1
                fi
            done
            echo "Command cancelled."
        else
            echo "No command is currently running."
        fi

        # Report pending commands
        if [[ -d "$CMD_DIR" ]]; then
            pending=$(find "$CMD_DIR" -maxdepth 1 -type f -name '*.sh' 2>/dev/null | wc -l)
            if [[ "$pending" -gt 0 ]]; then
                echo "$pending pending command(s) in queue. Use 'flush' to clear them."
            fi
        fi
        ;;

    *)
        echo "Usage: $0 [--relay-dir <path>] [--timeout <secs>] <subcommand>"
        echo ""
        echo "Subcommands:"
        echo "  handshake   Connect to the server"
        echo "  run <cmd>   Execute a command on the server"
        echo "  cancel      Cancel the currently running command"
        echo "  status      Check connection status"
        echo "  flush       Clear the relay directory"
        exit 1
        ;;
esac
