"""AWS Bedrock AgentCore Memory plugin — MemoryProvider interface.

Provides automated long-term memory with semantic search, user preference
extraction, conversation summarization, and episodic memory via AWS Bedrock
AgentCore Memory service.

Architecture (hybrid with builtin memory):
  - Layer 1 (Builtin): Fast, local, bounded — agent-managed curated facts
  - Layer 2 (AgentCore): Automatic, cloud, unbounded — strategy-driven extraction

SELF-CONTAINED PLUGIN ARCHITECTURE:
This plugin operates in "primary mode" where AgentCore is the source of truth
for builtin memory, synchronized via MemoryProvider hooks without requiring
modifications to run_agent.py or tools/memory_tool.py.

Hooks used:
- initialize(): Startup sync (cloud → local cache)
- on_memory_write(): Post-write sync (local → cloud primary storage)

Config via environment variables:
  AWS_REGION                     — AWS region (default: us-east-1)
  AGENTCORE_MEMORY_ID            — Memory Store ID (required)
  AGENTCORE_MEMORY_NAMESPACE     — Namespace prefix (default: /)
  AGENTCORE_PRIMARY_MODE         — Enable primary mode (default: true)

Or via $HERMES_HOME/agentcore_memory.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after consecutive failures, pause API calls
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Primary mode constants
MAX_QUEUE_SIZE = 100
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/agentcore_memory.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "memory_id": os.environ.get("AGENTCORE_MEMORY_ID", ""),
        "region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
        "namespace_prefix": os.environ.get("AGENTCORE_MEMORY_NAMESPACE", "/"),
        "primary_mode": os.environ.get("AGENTCORE_PRIMARY_MODE", "true").lower() == "true",
    }

    config_path = get_hermes_home() / "agentcore_memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "agentcore_search",
    "description": (
        "Search long-term memories by meaning using AgentCore semantic search. "
        "Returns relevant facts, preferences, and insights extracted from past "
        "conversations. Use when you need to recall context from previous sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in long-term memory.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 5, max: 20).",
            },
        },
        "required": ["query"],
    },
}

LIST_MEMORIES_SCHEMA = {
    "name": "agentcore_list",
    "description": (
        "List all long-term memory records from AgentCore. Returns extracted "
        "facts, preferences, and summaries. Use for a broad overview of what "
        "the system has learned about the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Max records to return (default: 20, max: 100).",
            },
        },
        "required": [],
    },
}

RECENT_TURNS_SCHEMA = {
    "name": "agentcore_recent",
    "description": (
        "Retrieve recent conversation turns from AgentCore short-term memory. "
        "Use to recall what was discussed earlier in this or recent sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "k": {
                "type": "integer",
                "description": "Number of recent turns to retrieve (default: 5, max: 20).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class AgentCoreMemoryProvider(MemoryProvider):
    """AWS Bedrock AgentCore Memory with automated extraction and semantic search."""

    def __init__(self):
        self._config: dict = {}
        self._memory_id: str = ""
        self._region: str = "us-east-1"
        self._namespace_prefix: str = "/"
        self._primary_mode: bool = True
        self._actor_id: str = "default-user"
        self._session_id: str = ""
        self._session_manager = None
        self._session = None
        self._manager_lock = threading.Lock()

        # Background thread state
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

        # Circuit breaker
        self._consecutive_failures: int = 0
        self._breaker_open_until: float = 0.0
        self._breaker_lock = threading.Lock()

        # HA backup state
        self._backup_thread: Optional[threading.Thread] = None
        self._backup_namespace_memory = "/builtin-backup/memory/"
        self._backup_namespace_user = "/builtin-backup/user/"

        # Primary builtin memory storage namespaces
        self._builtin_namespace_memory = "/builtin-primary/memory/"
        self._builtin_namespace_user = "/builtin-primary/user/"

        # Primary mode state
        self._sync_queue: List[Dict[str, Any]] = []
        self._sync_queue_lock = threading.Lock()
        self._startup_sync_done = False

    @property
    def name(self) -> str:
        return "agentcore"

    def is_available(self) -> bool:
        """Check if AgentCore Memory is configured (no network calls)."""
        try:
            import bedrock_agentcore  # noqa: F401
        except ImportError:
            return False
        cfg = _load_config()
        return bool(cfg.get("memory_id"))

    # -- Config schema for `hermes memory setup` ------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "memory_id",
                "description": "AgentCore Memory Store ID (from AWS Console or `agentcore add memory`)",
                "secret": False,
                "required": True,
                "env_var": "AGENTCORE_MEMORY_ID",
                "url": "https://console.aws.amazon.com/bedrock/home#/agentcore/memories",
            },
            {
                "key": "region",
                "description": "AWS Region for AgentCore",
                "default": "us-east-1",
            },
            {
                "key": "namespace_prefix",
                "description": "Namespace prefix for memory organization",
                "default": "/",
            },
            {
                "key": "primary_mode",
                "description": "Enable primary mode (AgentCore as source of truth)",
                "default": True,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/agentcore_memory.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "agentcore_memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    # -- Core lifecycle -------------------------------------------------------

    def _get_session_manager(self):
        """Lazy-init the MemorySessionManager (thread-safe)."""
        with self._manager_lock:
            if self._session_manager is not None:
                return self._session_manager
            try:
                from bedrock_agentcore.memory import MemorySessionManager
                self._session_manager = MemorySessionManager(
                    memory_id=self._memory_id,
                    region_name=self._region,
                )
                return self._session_manager
            except ImportError:
                raise RuntimeError(
                    "bedrock-agentcore package not installed. "
                    "Run: pip install bedrock-agentcore"
                )

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session and handle startup sync."""
        self._config = _load_config()
        self._memory_id = self._config.get("memory_id", "")
        self._region = self._config.get("region", "us-east-1")
        self._namespace_prefix = self._config.get("namespace_prefix", "/")
        self._primary_mode = self._config.get("primary_mode", True)

        # Map platform user_id → AgentCore actorId for per-user memory isolation
        self._actor_id = kwargs.get("user_id") or "default-user"
        self._session_id = session_id

        # Skip initialization for non-primary contexts
        agent_context = kwargs.get("agent_context", "primary")
        if agent_context not in ("primary", "flush"):
            logger.debug("AgentCore: skipping init for context=%s", agent_context)
            return

        if not self._memory_id:
            logger.warning("AgentCore Memory: memory_id not configured")
            return

        # Create a MemorySession for this conversation
        try:
            mgr = self._get_session_manager()
            self._session = mgr.create_memory_session(
                actor_id=self._actor_id,
                session_id=session_id,
            )
            logger.info(
                "AgentCore Memory initialized: memory_id=%s actor=%s session=%s primary=%s",
                self._memory_id, self._actor_id, session_id, self._primary_mode,
            )
        except Exception as e:
            logger.warning("AgentCore Memory init failed: %s", e)
            self._session = None
            return

        # STARTUP SYNC: Handle migration/sync if primary mode is enabled
        if self._primary_mode and not self._startup_sync_done:
            self._startup_sync_done = True
            self._perform_startup_sync()

    def _perform_startup_sync(self):
        """Handle startup sync/migration logic."""
        if not self._session:
            return

        try:
            # Use migration lock to prevent concurrent migration
            from hermes_constants import get_hermes_home
            migration_lock_path = get_hermes_home() / "memories" / ".migration_lock"
            migration_lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Try to acquire migration lock
            import fcntl
            try:
                with open(migration_lock_path, "w") as migration_fd:
                    try:
                        # Non-blocking lock - if another process is migrating, skip
                        fcntl.flock(migration_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                        # Check if this is first-time setup - migrate local to cloud
                        local_entries_exist = self._check_local_entries_exist()
                        cloud_entries_exist = self._check_cloud_entries_exist()

                        if local_entries_exist and not cloud_entries_exist:
                            # First-time migration: upload local to cloud
                            self._migrate_local_to_cloud()
                            # Rebuild local cache from cloud after migration
                            self._sync_all_targets_from_cloud()
                        else:
                            # Subsequent runs: sync local cache from cloud
                            self._sync_all_targets_from_cloud()

                    except BlockingIOError:
                        logger.info("Migration already in progress by another process, skipping")

            except (OSError, IOError) as lock_error:
                logger.warning("Could not acquire migration lock: %s", lock_error)
                # Continue without migration lock as fallback
                self._sync_all_targets_from_cloud()

        except Exception as e:
            logger.warning("AgentCore startup sync failed: %s", e)

    def _check_local_entries_exist(self) -> bool:
        """Check if local memory files have entries."""
        from hermes_constants import get_hermes_home
        for target in ("memory", "user"):
            filename = "USER.md" if target == "user" else "MEMORY.md"
            file_path = get_hermes_home() / "memories" / filename
            if file_path.exists():
                entries = self._read_file_entries(file_path)
                if entries:
                    return True
        return False

    def _check_cloud_entries_exist(self) -> bool:
        """Check if cloud storage has entries."""
        try:
            for target in ("memory", "user"):
                cloud_entries = self._read_from_agentcore_primary(target)
                if cloud_entries:
                    return True
        except Exception:
            pass
        return False

    def _migrate_local_to_cloud(self):
        """Migrate local files to AgentCore primary storage."""
        from hermes_constants import get_hermes_home

        migrated = {"memory": False, "user": False}

        for target in ("memory", "user"):
            filename = "USER.md" if target == "user" else "MEMORY.md"
            file_path = get_hermes_home() / "memories" / filename

            if not file_path.exists():
                continue

            entries = self._read_file_entries(file_path)
            if not entries:
                continue

            # Backup local file
            backup_path = file_path.with_suffix(file_path.suffix + ".pre-migration")
            try:
                file_path.rename(backup_path)
                logger.info("Backed up %s to %s", filename, backup_path.name)
            except Exception as e:
                logger.warning("Could not backup %s: %s", filename, e)
                continue

            # Upload to AgentCore primary
            try:
                self._write_entries_to_agentcore_primary(target, entries)
                migrated[target] = True
                logger.info("Migrated %d entries from %s to AgentCore primary", len(entries), filename)
            except Exception as e:
                # Restore backup on failure
                try:
                    backup_path.rename(file_path)
                    logger.warning("Migration failed for %s, restored backup: %s", filename, e)
                except Exception:
                    logger.error("Migration failed for %s and could not restore backup: %s", filename, e)

        logger.info("AgentCore migration completed: %s", migrated)

    def _sync_all_targets_from_cloud(self):
        """Sync all targets from cloud to local cache."""
        sync_results = {}
        for target in ("memory", "user"):
            try:
                self._sync_target_from_cloud(target)
                sync_results[target] = True
            except Exception as e:
                sync_results[target] = False
                logger.error("Startup sync failed for %s: %s", target, e)

        if not all(sync_results.values()):
            failed_targets = [k for k, v in sync_results.items() if not v]
            logger.warning("Partial startup sync failure for targets: %s", failed_targets)
        else:
            logger.info("All targets synced successfully from AgentCore: %s", list(sync_results.keys()))

    def system_prompt_block(self) -> str:
        """Return status info for the system prompt."""
        if not self._memory_id:
            return ""
        status = "connected" if self._session else "unavailable"
        mode = "primary" if self._primary_mode else "sync"
        return (
            "# AgentCore Memory\n"
            f"Status: {status} ({mode} mode). Actor: {self._actor_id}.\n"
            "AgentCore automatically extracts and stores long-term memories "
            "(facts, preferences, summaries) from conversations.\n"
            "Use agentcore_search to find past memories, agentcore_list for overview, "
            "agentcore_recent for recent conversation turns."
        )

    # -- Circuit breaker ------------------------------------------------------

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
                logger.warning(
                    "AgentCore circuit breaker tripped after %d failures. "
                    "Pausing for %ds.",
                    self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
                )

    # -- Turn sync (write events) ---------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Write conversation turn as AgentCore Event (non-blocking)."""
        if not self._session or self._is_breaker_open():
            return

        def _sync():
            try:
                from bedrock_agentcore.memory.constants import (
                    ConversationalMessage, MessageRole,
                )
                # Truncate very long messages to avoid API limits
                max_len = 10000
                user_text = user_content[:max_len] if user_content else ""
                asst_text = assistant_content[:max_len] if assistant_content else ""

                if not user_text and not asst_text:
                    return

                messages = []
                if user_text:
                    messages.append(ConversationalMessage(user_text, MessageRole.USER))
                if asst_text:
                    messages.append(ConversationalMessage(asst_text, MessageRole.ASSISTANT))

                mgr = self._get_session_manager()
                mgr.add_turns(
                    actor_id=self._actor_id,
                    session_id=self._session_id,
                    messages=messages,
                )
                self._record_success()
                logger.debug("AgentCore: synced turn (user=%d, asst=%d chars)",
                             len(user_text), len(asst_text))
            except Exception as e:
                self._record_failure()
                logger.warning("AgentCore sync_turn failed: %s", e)

        # Wait for previous sync to finish
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
            if self._sync_thread.is_alive():
                logger.warning("AgentCore: previous sync still running, skipping this sync")
                return

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="agentcore-sync"
        )
        self._sync_thread.start()

    # -- Prefetch (semantic recall) -------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched memories for injection into context."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## AgentCore Memory (recalled)\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background semantic search for the next turn."""
        if not self._session or self._is_breaker_open() or not query:
            return

        def _run():
            try:
                mgr = self._get_session_manager()
                records = mgr.search_long_term_memories(
                    query=query,
                    namespace_prefix=self._namespace_prefix,
                    top_k=5,
                    max_results=10,
                )
                if records:
                    lines = []
                    for r in records:
                        content = self._extract_record_content_text(r)
                        if content:
                            lines.append(f"- {content}")
                    if lines:
                        with self._prefetch_lock:
                            self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("AgentCore prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="agentcore-prefetch"
        )
        self._prefetch_thread.start()

    # -- Tool schemas & handlers ----------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Always return schemas — tools are registered before initialize() is called.
        # handle_tool_call() handles the "not initialized" case gracefully.
        if not self._memory_id:
            # Only skip if memory_id is definitively not configured
            cfg = _load_config()
            if not cfg.get("memory_id"):
                return []
        return [SEARCH_SCHEMA, LIST_MEMORIES_SCHEMA, RECENT_TURNS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "AgentCore API temporarily unavailable. Will retry automatically."
            })

        if not self._session:
            return tool_error("AgentCore Memory not initialized")

        try:
            mgr = self._get_session_manager()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "agentcore_search":
            return self._handle_search(mgr, args)
        elif tool_name == "agentcore_list":
            return self._handle_list(mgr, args)
        elif tool_name == "agentcore_recent":
            return self._handle_recent(args)
        else:
            return tool_error(f"Unknown tool: {tool_name}")

    def _handle_search(self, mgr, args: dict) -> str:
        """Semantic search across long-term memories."""
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        top_k = min(int(args.get("top_k", 5)), 20)

        try:
            records = mgr.search_long_term_memories(
                query=query,
                namespace_prefix=self._namespace_prefix,
                top_k=top_k,
            )
            self._record_success()

            if not records:
                return json.dumps({"result": "No relevant memories found."})

            items = []
            for r in records:
                item = self._extract_record_content(r)
                if item:
                    items.append(item)

            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Search failed: {e}")

    def _handle_list(self, mgr, args: dict) -> str:
        """List all long-term memory records."""
        max_results = min(int(args.get("max_results", 20)), 100)

        try:
            records = mgr.list_long_term_memory_records(
                namespace_prefix=self._namespace_prefix,
                max_results=max_results,
            )
            self._record_success()

            if not records:
                return json.dumps({"result": "No long-term memories stored yet."})

            items = []
            for r in records:
                item = self._extract_record_content(r)
                if item:
                    items.append(item)

            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            self._record_failure()
            return tool_error(f"List failed: {e}")

    def _handle_recent(self, args: dict) -> str:
        """Retrieve recent conversation turns."""
        k = min(int(args.get("k", 5)), 20)

        try:
            mgr = self._get_session_manager()
            turns = mgr.get_last_k_turns(
                actor_id=self._actor_id,
                session_id=self._session_id,
                k=k,
            )
            self._record_success()

            if not turns:
                return json.dumps({"result": "No recent turns found."})

            formatted = []
            for turn in turns:
                for msg in turn:
                    role = ""
                    content = ""
                    if hasattr(msg, "get"):
                        role = msg.get("role", "")
                        raw_content = msg.get("content", "")
                        if isinstance(raw_content, dict):
                            content = raw_content.get("text", "")
                        else:
                            content = str(raw_content) if raw_content else ""
                    elif hasattr(msg, "role"):
                        role = str(msg.role) if hasattr(msg.role, "value") else str(msg.role)
                        raw_content = msg.content if hasattr(msg, "content") else ""
                        if isinstance(raw_content, dict):
                            content = raw_content.get("text", "")
                        else:
                            content = str(raw_content) if raw_content else ""
                    if content:
                        formatted.append({"role": role, "content": content[:500]})

            return json.dumps({"turns": formatted, "count": len(formatted)})
        except Exception as e:
            self._record_failure()
            return tool_error(f"Recent turns failed: {e}")

    @staticmethod
    def _extract_record_content(record) -> Optional[dict]:
        """Extract content from a MemoryRecord (dict-like or object)."""
        raw_content: Any = ""
        record_type = ""

        if hasattr(record, "get"):
            raw_content = record.get("content", "") or record.get("memory", "") or ""
            record_type = record.get("strategyId", "") or record.get("type", "")
        elif hasattr(record, "content"):
            raw_content = record.content or ""
            record_type = getattr(record, "strategyId", "") or ""

        # Unwrap nested {'text': '...'} format
        if isinstance(raw_content, dict):
            content = raw_content.get("text", "") or str(raw_content)
        else:
            content = str(raw_content) if raw_content else ""

        if not content:
            return None

        result = {"content": content}
        if record_type:
            result["type"] = record_type
        return result

    def _extract_record_content_text(self, record) -> str:
        """Extract just the text content from a record."""
        extracted = self._extract_record_content(record)
        return extracted["content"] if extracted else ""

    # -- Primary mode memory operations --------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Handle builtin memory writes: sync to AgentCore and backup."""
        if not self._session or self._is_breaker_open():
            return

        def _handle_write():
            try:
                # 1. Event mirroring (existing behavior)
                from bedrock_agentcore.memory.constants import (
                    ConversationalMessage, MessageRole,
                )
                note = f"[Builtin memory {action} on {target}]: {content}"
                mgr = self._get_session_manager()
                mgr.add_turns(
                    actor_id=self._actor_id,
                    session_id=self._session_id,
                    messages=[
                        ConversationalMessage(note, MessageRole.OTHER),
                    ],
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("AgentCore on_memory_write event failed: %s", e)

            # 2. Primary mode sync (NEW)
            if self._primary_mode:
                try:
                    self._sync_builtin_memory_to_primary(target)
                except Exception as e:
                    logger.warning("Primary mode sync failed for %s: %s", target, e)

            # 3. HA backup (existing behavior)
            try:
                self._backup_builtin_file(target)
            except Exception as e:
                logger.debug("AgentCore HA backup failed for %s: %s", target, e)

        # Use existing threading pattern
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5.0)
            if self._backup_thread.is_alive():
                logger.warning("AgentCore: previous write handling still running, skipping")
                return

        self._backup_thread = threading.Thread(
            target=_handle_write, daemon=True, name="agentcore-memory-write"
        )
        self._backup_thread.start()

    def _sync_builtin_memory_to_primary(self, target: str):
        """Sync local memory file to AgentCore primary storage."""
        from hermes_constants import get_hermes_home

        filename = "USER.md" if target == "user" else "MEMORY.md"
        file_path = get_hermes_home() / "memories" / filename

        if not file_path.exists():
            # File doesn't exist, clear primary storage
            self._clear_agentcore_primary_namespace(target)
            return

        entries = self._read_file_entries(file_path)
        self._write_entries_to_agentcore_primary(target, entries)
        logger.debug("Synced %s to AgentCore primary: %d entries", target, len(entries))

    def _sync_target_from_cloud(self, target: str):
        """Sync a target from AgentCore primary to local file."""
        try:
            cloud_entries = self._read_from_agentcore_primary(target)
            self._write_entries_to_local_file(target, cloud_entries)
            logger.debug("Synced %s from AgentCore: %d entries", target, len(cloud_entries))
        except Exception as e:
            logger.error("Failed to sync %s from cloud: %s", target, e)
            raise

    def _read_from_agentcore_primary(self, target: str) -> List[str]:
        """Read entries from AgentCore primary storage."""
        ns = self._builtin_namespace_memory if target == "memory" else self._builtin_namespace_user

        try:
            mgr = self._get_session_manager()
            records = mgr.list_long_term_memory_records(
                namespace_prefix=ns,
                max_results=100,
            )

            entries = []
            for r in records:
                extracted = self._extract_record_content(r)
                if extracted and extracted.get("content"):
                    entries.append(extracted["content"])

            return entries
        except Exception as e:
            logger.warning("Failed to read from AgentCore primary %s: %s", target, e)
            raise

    def _write_entries_to_agentcore_primary(self, target: str, entries: List[str]):
        """Write entries to AgentCore primary storage."""
        ns = self._builtin_namespace_memory if target == "memory" else self._builtin_namespace_user

        try:
            # Clear namespace first
            self._clear_agentcore_primary_namespace(target)

            if not entries:
                return

            # Write all entries
            client = self._get_boto3_client()
            from datetime import datetime, timezone
            import uuid

            records = []
            for entry in entries:
                records.append({
                    "requestIdentifier": str(uuid.uuid4()),
                    "namespaces": [ns],
                    "content": {"text": entry},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            client.batch_create_memory_records(memoryId=self._memory_id, records=records)
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.warning("Failed to write entries to AgentCore primary %s: %s", target, e)
            raise

    def _clear_agentcore_primary_namespace(self, target: str):
        """Clear a primary namespace in AgentCore."""
        ns = self._builtin_namespace_memory if target == "memory" else self._builtin_namespace_user
        try:
            mgr = self._get_session_manager()
            mgr.delete_all_long_term_memories_in_namespace(ns)
        except Exception as e:
            logger.warning("Failed to clear AgentCore primary namespace %s: %s", ns, e)
            raise

    def _write_entries_to_local_file(self, target: str, entries: List[str]):
        """Write entries to local memory file."""
        from hermes_constants import get_hermes_home

        filename = "USER.md" if target == "user" else "MEMORY.md"
        file_path = get_hermes_home() / "memories" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = "\n§\n".join(entries) if entries else ""
        self._atomic_write(file_path, content)

    def _read_file_entries(self, file_path) -> List[str]:
        """Read entries from a memory file."""
        try:
            if not file_path.exists():
                return []

            raw = file_path.read_text(encoding="utf-8")
            if not raw.strip():
                return []

            entries = [e.strip() for e in raw.split("\n§\n")]
            return [e for e in entries if e]
        except Exception:
            return []

    def _get_boto3_client(self):
        """Get a boto3 bedrock-agentcore client for direct API calls."""
        import boto3
        return boto3.client(
            "bedrock-agentcore",
            region_name=self._region,
        )

    def _backup_builtin_file(self, target: str) -> None:
        """Snapshot a full builtin memory file to AgentCore backup namespace."""
        from pathlib import Path
        from hermes_constants import get_hermes_home
        import uuid
        from datetime import datetime, timezone

        ns = (self._backup_namespace_memory if target == "memory"
              else self._backup_namespace_user)
        filename = "USER.md" if target == "user" else "MEMORY.md"
        file_path = get_hermes_home() / "memories" / filename

        if not file_path.exists():
            return

        file_content = file_path.read_text(encoding="utf-8").strip()
        if not file_content:
            return

        client = self._get_boto3_client()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Create new snapshot
        try:
            client.batch_create_memory_records(
                memoryId=self._memory_id,
                records=[{
                    "requestIdentifier": str(uuid.uuid4()),
                    "namespaces": [ns],
                    "content": {"text": file_content},
                    "timestamp": now_iso,
                }],
            )
            logger.debug("AgentCore HA: backed up %s (%d chars) to %s",
                        filename, len(file_content), ns)
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.warning("AgentCore HA: backup %s failed: %s", filename, e)

    @staticmethod
    def _atomic_write(path, content: str) -> None:
        """Write content to file atomically."""
        import fcntl
        import tempfile

        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "a+") as fd:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
                # Atomic write via temp file + rename
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=".tmp", prefix=".restore_"
                )
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        f.write(content)
                    os.replace(tmp_path, str(path))
                except BaseException:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)

    # -- Optional hooks -------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Ensure all pending writes complete before session ends."""
        for t in (self._sync_thread, self._prefetch_thread, self._backup_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Write messages about to be compressed to AgentCore for preservation."""
        if not self._session or self._is_breaker_open():
            return ""

        # Extract key content from messages about to be lost
        content_parts = []
        for msg in messages[-10:]:  # Last 10 messages being compressed
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                content_parts.append(f"{role}: {content[:500]}")

        if content_parts:
            try:
                from bedrock_agentcore.memory.constants import (
                    ConversationalMessage, MessageRole,
                )
                summary = "\n".join(content_parts)[:5000]
                mgr = self._get_session_manager()
                mgr.add_turns(
                    actor_id=self._actor_id,
                    session_id=self._session_id,
                    messages=[
                        ConversationalMessage(
                            f"[Context being compressed]: {summary}",
                            MessageRole.OTHER,
                        ),
                    ],
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("AgentCore on_pre_compress failed: %s", e)

        return ""  # No contribution to compression summary

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Record subagent task/result as an event."""
        if not self._session or self._is_breaker_open():
            return

        def _record():
            try:
                from bedrock_agentcore.memory.constants import (
                    ConversationalMessage, MessageRole,
                )
                note = (
                    f"[Delegated task]: {task[:2000]}\n"
                    f"[Result]: {result[:3000]}"
                )
                mgr = self._get_session_manager()
                mgr.add_turns(
                    actor_id=self._actor_id,
                    session_id=self._session_id,
                    messages=[
                        ConversationalMessage(note, MessageRole.OTHER),
                    ],
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("AgentCore on_delegation failed: %s", e)

        t = threading.Thread(target=_record, daemon=True, name="agentcore-delegation")
        t.start()

    # -- Shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown — wait for pending writes."""
        for t in (self._prefetch_thread, self._sync_thread, self._backup_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)
        with self._manager_lock:
            self._session_manager = None
            self._session = None


def register(ctx) -> None:
    """Register AgentCore as a memory provider plugin."""
    ctx.register_memory_provider(AgentCoreMemoryProvider())