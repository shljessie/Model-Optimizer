#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt Slack Bot — connects Claude Code CLI with custom skills to Slack.

Uses Slack Socket Mode (no public URL needed) to receive messages and
Claude Code CLI (subprocess) to process them with ModelOpt skills (ptq,
deployment, evaluation, modelopt orchestrator).
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]  # xoxb-...
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]  # xapp-...

# Path to the modelopt_agent repo root (where .claude/skills/ lives)
SKILLS_CWD = os.environ.get(
    "SKILLS_CWD",
    str(Path(__file__).resolve().parent.parent),  # default: parent of slack-bot/
)

# Maximum response length for Slack (Slack truncates at 40k chars)
MAX_SLACK_LENGTH = 39000

# Timeout for Claude CLI calls (seconds)
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "600"))  # 10 min default

# ─── Slack App ───────────────────────────────────────────────────────

app = AsyncApp(token=SLACK_BOT_TOKEN)


def strip_bot_mention(text: str) -> str:
    """Remove the @bot mention from the message text."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def truncate_response(text: str) -> str:
    """Truncate response to fit Slack's message limit."""
    if len(text) <= MAX_SLACK_LENGTH:
        return text
    return text[:MAX_SLACK_LENGTH] + "\n\n... (truncated)"


async def run_claude(prompt: str, thread_context: str = "") -> str:
    """Run a prompt through Claude Code CLI as a subprocess."""
    full_prompt = prompt
    if thread_context:
        full_prompt = f"Previous context:\n{thread_context}\n\nUser: {prompt}"

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return "Error: `claude` CLI not found in PATH. Install Claude Code first."

    cmd = [
        claude_bin,
        "--print",  # output text only (no interactive UI)
        "--dangerously-skip-permissions",  # non-interactive mode
        "--verbose",  # log tool calls for debugging
        "-p",
        full_prompt,
    ]

    logger.info("Running: claude --print -p '%s...'", full_prompt[:80])

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=SKILLS_CWD,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT,
        )

        if result.returncode != 0:
            logger.error("Claude CLI stderr: %s", result.stderr[:500])
            if result.stdout:
                return result.stdout  # partial output may still be useful
            return f"Claude CLI error (exit {result.returncode}): {result.stderr[:500]}"

        return result.stdout or "No response from Claude."

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after %ds", CLAUDE_TIMEOUT)
        return f"Request timed out after {CLAUDE_TIMEOUT}s. Try a simpler request."
    except Exception as e:
        logger.error("Claude CLI error: %s", e)
        return f"Error running Claude: {e}"


# ─── Event Handlers ──────────────────────────────────────────────────

# Store thread contexts: {(channel, thread_ts): [messages]}
thread_history: dict[tuple[str, str], list[str]] = {}
MAX_HISTORY = 20


@app.event("app_mention")
async def handle_mention(event, say):
    """Handle @bot mentions in channels."""
    text = strip_bot_mention(event.get("text", ""))
    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])

    if not text:
        await say("How can I help? Try: `quantize Qwen3-0.6B with fp8`", thread_ts=thread_ts)
        return

    # Build thread context
    key = (channel, thread_ts)
    history = thread_history.get(key, [])
    context = "\n".join(history[-MAX_HISTORY:])

    # Acknowledge receipt
    await say("Working on it...", thread_ts=thread_ts)

    # Run Claude
    response = await run_claude(text, thread_context=context)
    response = truncate_response(response)

    # Save to thread history
    history.append(f"User: {text}")
    history.append(f"Assistant: {response[:500]}")  # truncate for context
    thread_history[key] = history[-MAX_HISTORY:]

    await say(text=response, thread_ts=thread_ts)


@app.event("message")
async def handle_dm(event, say):
    """Handle direct messages."""
    # Skip bot's own messages and threaded replies handled by app_mention
    if event.get("bot_id") or event.get("subtype"):
        return

    # Only handle DMs (channel type "im")
    if event.get("channel_type") != "im":
        return

    text = event.get("text", "").strip()
    if not text:
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])

    key = (channel, thread_ts)
    history = thread_history.get(key, [])
    context = "\n".join(history[-MAX_HISTORY:])

    response = await run_claude(text, thread_context=context)
    response = truncate_response(response)

    history.append(f"User: {text}")
    history.append(f"Assistant: {response[:500]}")
    thread_history[key] = history[-MAX_HISTORY:]

    await say(text=response, thread_ts=thread_ts)


# ─── Main ────────────────────────────────────────────────────────────


async def main():
    """Start the ModelOpt Slack Bot."""
    logger.info("Starting ModelOpt Slack Bot...")
    logger.info("Skills directory: %s", SKILLS_CWD)
    logger.info("Looking for skills in: %s/.claude/skills/", SKILLS_CWD)

    # Verify claude CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        logger.info("Claude CLI: %s", claude_bin)
    else:
        logger.error("Claude CLI not found in PATH — bot will not work")

    # Verify skills directory exists
    skills_path = Path(SKILLS_CWD) / ".claude" / "skills"
    if skills_path.exists():
        skills = [d.name for d in skills_path.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
        logger.info("Found skills: %s", ", ".join(skills))
    else:
        logger.warning("Skills directory not found: %s", skills_path)

    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
