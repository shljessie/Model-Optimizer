#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code subprocess runner for jobs.

Runs Claude CLI in a job's working directory with the user's auth credentials.
Supports streaming output for real-time Slack relay.
"""

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "600"))  # 10 min default


@dataclass
class StreamChunk:
    """A chunk of Claude's streamed output."""

    text: str
    is_final: bool = False
    is_error: bool = False


async def run_claude(
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    session_id: str | None = None,
    timeout: int = CLAUDE_TIMEOUT,
    system_prompt_extra: str | None = None,
) -> str:
    """Run Claude CLI and return the full output.

    Args:
        prompt: The user's prompt
        cwd: Working directory (job dir with skills)
        env: Environment variables (includes user's API key)
        session_id: Optional session ID for multi-turn within a job
        timeout: Max seconds to wait
        system_prompt_extra: Extra context appended to system prompt
    """
    cmd = _build_cmd(prompt, session_id, system_prompt_extra=system_prompt_extra)

    logger.info("Running claude in %s: %.80s...", cwd, prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")

        if proc.returncode != 0:
            logger.error("Claude CLI error (rc=%d): %s", proc.returncode, stderr_text[:500])
            return stdout_text or f"Claude CLI error (exit {proc.returncode}): {stderr_text[:500]}"

        return stdout_text or "No response from Claude."

    except asyncio.TimeoutError:
        logger.error("Claude CLI timed out after %ds", timeout)
        try:
            proc.kill()
        except Exception:
            pass
        return f"Request timed out after {timeout}s. The job may still be running on the cluster — check with `/modelopt status`."

    except Exception as e:
        logger.error("Claude CLI error: %s", e)
        return f"Error running Claude: {e}"


async def run_claude_streaming(
    prompt: str,
    cwd: Path,
    env: dict[str, str],
    session_id: str | None = None,
    timeout: int = CLAUDE_TIMEOUT,
    system_prompt_extra: str | None = None,
) -> AsyncIterator[StreamChunk]:
    """Run Claude CLI with streaming output (stream-json format).

    Yields StreamChunk objects as Claude produces output. Useful for
    real-time relay to Slack (update message progressively).
    """
    cmd = _build_cmd(prompt, session_id, streaming=True, system_prompt_extra=system_prompt_extra)

    logger.info("Running claude (streaming) in %s: %.80s...", cwd, prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as e:
        yield StreamChunk(text=f"Error starting Claude: {e}", is_final=True, is_error=True)
        return

    buffer = ""
    try:
        async def _read_stream():
            nonlocal buffer
            while True:
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=timeout)
                if not chunk:
                    break
                buffer += chunk.decode(errors="replace")

                # Parse complete JSON lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        text = _extract_text_from_event(event)
                        if text:
                            yield StreamChunk(text=text)
                    except json.JSONDecodeError:
                        # Not JSON — might be plain text output
                        yield StreamChunk(text=line)

        async for chunk in _read_stream():
            yield chunk

    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        yield StreamChunk(
            text=f"\n\nTimed out after {timeout}s.",
            is_final=True,
            is_error=True,
        )
        return

    await proc.wait()

    if proc.returncode != 0:
        stderr = await proc.stderr.read()
        stderr_text = stderr.decode(errors="replace")[:500]
        yield StreamChunk(
            text=f"\nClaude CLI error (exit {proc.returncode}): {stderr_text}",
            is_final=True,
            is_error=True,
        )
    else:
        yield StreamChunk(text="", is_final=True)


def _build_cmd(
    prompt: str,
    session_id: str | None = None,
    streaming: bool = False,
    system_prompt_extra: str | None = None,
) -> list[str]:
    """Build the claude CLI command."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise FileNotFoundError("`claude` CLI not found in PATH")

    cmd = [
        claude_bin,
        "--print",
        "--dangerously-skip-permissions",
    ]

    if system_prompt_extra:
        cmd.extend(["--append-system-prompt", system_prompt_extra])

    if streaming:
        cmd.extend(["--output-format", "stream-json"])

    if session_id:
        cmd.extend(["--session-id", session_id])

    cmd.extend(["-p", prompt])
    return cmd


def _extract_text_from_event(event: dict) -> str:
    """Extract displayable text from a stream-json event."""
    # Assistant text messages
    if event.get("type") == "assistant" and "message" in event:
        content = event["message"].get("content", [])
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block["text"])
        return "".join(parts)

    # Result message (final)
    if event.get("type") == "result":
        return ""  # Already captured via assistant messages

    return ""
