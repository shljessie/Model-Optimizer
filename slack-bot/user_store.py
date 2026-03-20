#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-user data management: auth, cluster config, onboarding state.

Directory layout per user:
    <data_dir>/users/<slack_uid>/
        auth.json          — auth method + encrypted credentials
        clusters.yaml      — SSH/cluster configs
        jobs/              — per-job working directories
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path

from key_store import KeyStore

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    SHARED_KEY = "shared_key"  # Use the server's default ANTHROPIC_API_KEY
    OWN_KEY = "own_key"  # User provided their own sk-ant-... key
    LOGIN = "login"  # User authenticated via `claude auth login` (headless browser flow)


class UserStore:
    """Manages per-user data: auth credentials, cluster configs, onboarding state."""

    def __init__(self, data_dir: str | Path, key_store: KeyStore):
        self._data_dir = Path(data_dir)
        self._users_dir = self._data_dir / "users"
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._key_store = key_store

    # ── User Directory ───────────────────────────────────────────────

    def user_dir(self, user_id: str) -> Path:
        return self._users_dir / user_id

    def jobs_dir(self, user_id: str) -> Path:
        d = self.user_dir(user_id) / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_registered(self, user_id: str) -> bool:
        return (self.user_dir(user_id) / "auth.json").exists()

    # ── Auth ─────────────────────────────────────────────────────────

    def setup_shared_key(self, user_id: str) -> None:
        """Register user with the shared/default API key."""
        self._ensure_user_dir(user_id)
        self._write_auth(user_id, {"method": AuthMethod.SHARED_KEY})
        logger.info("User %s registered with shared key", user_id)

    def setup_own_key(self, user_id: str, api_key: str) -> None:
        """Register user with their own Anthropic API key."""
        self._ensure_user_dir(user_id)
        self._key_store.store_key(user_id, api_key)
        self._write_auth(user_id, {"method": AuthMethod.OWN_KEY})
        logger.info("User %s registered with own API key", user_id)

    def setup_login_auth(self, user_id: str, config_dir: str) -> None:
        """Register user who authenticated via `claude auth login`.

        The config_dir contains the .credentials.json from the login flow.
        We store this path so we can set CLAUDE_CONFIG_DIR when running Claude.
        """
        self._ensure_user_dir(user_id)
        # Copy the credentials into the user's persistent dir
        import shutil
        user_auth_dir = self.user_dir(user_id) / "claude-config"
        if user_auth_dir.exists():
            shutil.rmtree(user_auth_dir)
        shutil.copytree(config_dir, str(user_auth_dir))
        self._write_auth(user_id, {"method": AuthMethod.LOGIN, "config_dir": str(user_auth_dir)})
        logger.info("User %s registered with claude login auth", user_id)

    def get_auth_method(self, user_id: str) -> AuthMethod | None:
        auth = self._read_auth(user_id)
        if auth is None:
            return None
        return AuthMethod(auth["method"])

    def get_api_key(self, user_id: str) -> str | None:
        """Get the API key to use for this user's Claude session."""
        auth = self._read_auth(user_id)
        if auth is None:
            return None

        method = AuthMethod(auth["method"])
        if method == AuthMethod.SHARED_KEY:
            # Use server's default key
            return os.environ.get("ANTHROPIC_API_KEY")
        elif method == AuthMethod.OWN_KEY:
            return self._key_store.get_key(user_id)
        return None

    def get_claude_env(self, user_id: str) -> dict[str, str]:
        """Build environment variables for this user's Claude subprocess."""
        env = os.environ.copy()
        auth = self._read_auth(user_id)
        if auth is None:
            return env

        method = AuthMethod(auth["method"])
        if method == AuthMethod.SHARED_KEY:
            # ANTHROPIC_API_KEY already in env (server default)
            pass
        elif method == AuthMethod.OWN_KEY:
            key = self._key_store.get_key(user_id)
            if key:
                env["ANTHROPIC_API_KEY"] = key
        elif method == AuthMethod.LOGIN:
            # Point Claude CLI at the user's stored credentials
            config_dir = auth.get("config_dir", "")
            if config_dir:
                env["CLAUDE_CONFIG_DIR"] = config_dir

        return env

    def remove_auth(self, user_id: str) -> bool:
        """Remove user's auth credentials."""
        self._key_store.remove_key(user_id)
        auth_file = self.user_dir(user_id) / "auth.json"
        if auth_file.exists():
            auth_file.unlink()
            return True
        return False

    # ── Cluster Config ───────────────────────────────────────────────

    def get_clusters_yaml_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "clusters.yaml"

    def has_clusters(self, user_id: str) -> bool:
        return self.get_clusters_yaml_path(user_id).exists()

    def save_clusters_yaml(self, user_id: str, content: str) -> None:
        """Write cluster config for a user."""
        self._ensure_user_dir(user_id)
        path = self.get_clusters_yaml_path(user_id)
        path.write_text(content, encoding="utf-8")
        logger.info("Saved cluster config for user %s", user_id)

    def read_clusters_yaml(self, user_id: str) -> str | None:
        path = self.get_clusters_yaml_path(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    # ── User Info ────────────────────────────────────────────────────

    def user_info(self, user_id: str) -> dict | None:
        """Get summary info about a user."""
        if not self.is_registered(user_id):
            return None
        auth = self._read_auth(user_id)
        jobs_path = self.user_dir(user_id) / "jobs"
        job_count = len(list(jobs_path.iterdir())) if jobs_path.exists() else 0
        return {
            "user_id": user_id,
            "auth_method": auth.get("method", "unknown") if auth else "unknown",
            "has_clusters": self.has_clusters(user_id),
            "job_count": job_count,
        }

    def list_users(self) -> list[str]:
        """List all registered user IDs."""
        if not self._users_dir.exists():
            return []
        return [
            d.name
            for d in self._users_dir.iterdir()
            if d.is_dir() and (d / "auth.json").exists()
        ]

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure_user_dir(self, user_id: str):
        d = self.user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "jobs").mkdir(exist_ok=True)

    def _write_auth(self, user_id: str, auth: dict):
        path = self.user_dir(user_id) / "auth.json"
        path.write_text(json.dumps(auth, indent=2), encoding="utf-8")
        path.chmod(0o600)

    def _read_auth(self, user_id: str) -> dict | None:
        path = self.user_dir(user_id) / "auth.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
