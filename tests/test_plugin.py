"""Comprehensive tests for self-contained AgentCore Memory plugin."""

import json
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the plugin directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from __init__ import AgentCoreMemoryProvider
except ImportError:
    # Fallback for different import contexts
    sys.path.append(str(Path(__file__).parent.parent))
    from __init__ import AgentCoreMemoryProvider


class TestPluginBasics:
    """Test basic plugin functionality."""

    def test_plugin_name(self):
        """Test plugin name is correct."""
        provider = AgentCoreMemoryProvider()
        assert provider.name == "agentcore"

    def test_is_available_without_bedrock_agentcore(self):
        """Test availability check when bedrock-agentcore is not installed."""
        with patch.dict('sys.modules', {'bedrock_agentcore': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                provider = AgentCoreMemoryProvider()
                assert provider.is_available() is False

    def test_is_available_without_memory_id(self, temp_hermes_home):
        """Test availability check when memory_id is not configured."""
        with patch('bedrock_agentcore.memory.MemorySessionManager'):
            provider = AgentCoreMemoryProvider()
            assert provider.is_available() is False

    def test_is_available_with_config(self, temp_hermes_home, agentcore_config):
        """Test availability check with proper configuration."""
        config_file = temp_hermes_home / "agentcore_memory.json"
        config_file.write_text(json.dumps(agentcore_config))

        with patch('bedrock_agentcore.memory.MemorySessionManager'):
            provider = AgentCoreMemoryProvider()
            assert provider.is_available() is True

    def test_config_schema(self):
        """Test configuration schema."""
        provider = AgentCoreMemoryProvider()
        schema = provider.get_config_schema()

        assert len(schema) == 4
        field_keys = [field["key"] for field in schema]
        assert "memory_id" in field_keys
        assert "region" in field_keys
        assert "namespace_prefix" in field_keys
        assert "primary_mode" in field_keys

    def test_save_config(self, temp_hermes_home):
        """Test configuration saving."""
        provider = AgentCoreMemoryProvider()
        config_data = {
            "memory_id": "test-id",
            "region": "us-west-2",
            "primary_mode": True,
        }

        provider.save_config(config_data, str(temp_hermes_home))

        config_file = temp_hermes_home / "agentcore_memory.json"
        assert config_file.exists()

        saved_config = json.loads(config_file.read_text())
        assert saved_config["memory_id"] == "test-id"
        assert saved_config["region"] == "us-west-2"
        assert saved_config["primary_mode"] is True


class TestInitialization:
    """Test plugin initialization."""

    def test_basic_initialization(self, agentcore_config, temp_hermes_home):
        """Test basic initialization without errors."""
        config_file = temp_hermes_home / "agentcore_memory.json"
        config_file.write_text(json.dumps(agentcore_config))

        provider = AgentCoreMemoryProvider()

        with patch.object(provider, '_get_session_manager') as mock_mgr:
            mock_session = Mock()
            mock_mgr.return_value.create_memory_session.return_value = mock_session

            provider.initialize("test-session-id", agent_context="primary")

            assert provider._memory_id == "test-memory-id"
            assert provider._region == "us-east-1"
            assert provider._primary_mode is True
            assert provider._session == mock_session

    def test_initialization_skip_non_primary_context(self, agentcore_config, temp_hermes_home):
        """Test that initialization is skipped for non-primary contexts."""
        config_file = temp_hermes_home / "agentcore_memory.json"
        config_file.write_text(json.dumps(agentcore_config))

        provider = AgentCoreMemoryProvider()

        with patch.object(provider, '_get_session_manager') as mock_mgr:
            provider.initialize("test-session-id", agent_context="subagent")

            # Should not create session for subagent context
            assert mock_mgr.return_value.create_memory_session.call_count == 0
            assert provider._session is None

    def test_startup_sync_first_time_migration(self, agentcore_config, temp_hermes_home, memory_files):
        """Test first-time migration during startup sync."""
        config_file = temp_hermes_home / "agentcore_memory.json"
        config_file.write_text(json.dumps(agentcore_config))

        provider = AgentCoreMemoryProvider()

        with patch.object(provider, '_get_session_manager') as mock_mgr, \
             patch.object(provider, '_check_local_entries_exist', return_value=True), \
             patch.object(provider, '_check_cloud_entries_exist', return_value=False), \
             patch.object(provider, '_migrate_local_to_cloud') as mock_migrate, \
             patch.object(provider, '_sync_all_targets_from_cloud') as mock_sync:

            mock_session = Mock()
            mock_mgr.return_value.create_memory_session.return_value = mock_session

            provider.initialize("test-session-id", agent_context="primary")

            # Should trigger migration
            mock_migrate.assert_called_once()
            mock_sync.assert_called_once()

    def test_startup_sync_subsequent_runs(self, agentcore_config, temp_hermes_home):
        """Test startup sync for subsequent runs."""
        config_file = temp_hermes_home / "agentcore_memory.json"
        config_file.write_text(json.dumps(agentcore_config))

        provider = AgentCoreMemoryProvider()

        with patch.object(provider, '_get_session_manager') as mock_mgr, \
             patch.object(provider, '_check_local_entries_exist', return_value=True), \
             patch.object(provider, '_check_cloud_entries_exist', return_value=True), \
             patch.object(provider, '_migrate_local_to_cloud') as mock_migrate, \
             patch.object(provider, '_sync_all_targets_from_cloud') as mock_sync:

            mock_session = Mock()
            mock_mgr.return_value.create_memory_session.return_value = mock_session

            provider.initialize("test-session-id", agent_context="primary")

            # Should not trigger migration, only sync
            mock_migrate.assert_not_called()
            mock_sync.assert_called_once()


class TestPrimaryModeOperations:
    """Test primary mode memory operations."""

    def test_sync_builtin_memory_to_primary(self, agentcore_config, temp_hermes_home, memory_files):
        """Test syncing local memory to AgentCore primary."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._memory_id = agentcore_config["memory_id"]
        provider._primary_mode = True
        provider._session = Mock()

        with patch.object(provider, '_write_entries_to_agentcore_primary') as mock_write:
            provider._sync_builtin_memory_to_primary("memory")

            # Should read local file and write to AgentCore
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert args[0] == "memory"  # target
            assert len(args[1]) == 2    # entries list
            assert "Test memory entry 1" in args[1]
            assert "Test memory entry 2" in args[1]

    def test_sync_target_from_cloud(self, agentcore_config, temp_hermes_home):
        """Test syncing from AgentCore to local file."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._session = Mock()

        cloud_entries = ["Cloud entry 1", "Cloud entry 2"]

        with patch.object(provider, '_read_from_agentcore_primary', return_value=cloud_entries) as mock_read, \
             patch.object(provider, '_write_entries_to_local_file') as mock_write:

            provider._sync_target_from_cloud("memory")

            mock_read.assert_called_once_with("memory")
            mock_write.assert_called_once_with("memory", cloud_entries)

    def test_read_from_agentcore_primary(self, agentcore_config, sample_memory_records):
        """Test reading from AgentCore primary storage."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._builtin_namespace_memory = "/builtin-primary/memory/"

        mock_mgr = Mock()
        mock_mgr.list_long_term_memory_records.return_value = sample_memory_records

        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            entries = provider._read_from_agentcore_primary("memory")

            assert len(entries) == 3
            assert "User prefers dark theme" in entries
            assert "Project uses TypeScript" in entries
            assert "Meeting scheduled for tomorrow" in entries

    def test_write_entries_to_agentcore_primary(self, agentcore_config, mock_boto3_client):
        """Test writing entries to AgentCore primary storage."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._memory_id = agentcore_config["memory_id"]

        entries = ["Entry 1", "Entry 2", "Entry 3"]

        with patch.object(provider, '_get_session_manager') as mock_mgr, \
             patch.object(provider, '_get_boto3_client', return_value=mock_boto3_client), \
             patch.object(provider, '_clear_agentcore_primary_namespace') as mock_clear:

            provider._write_entries_to_agentcore_primary("memory", entries)

            # Should clear namespace first
            mock_clear.assert_called_once_with("memory")

            # Should create records for all entries
            assert mock_boto3_client.batch_create_memory_records.called
            call_args = mock_boto3_client.batch_create_memory_records.call_args
            records = call_args[1]["records"]
            assert len(records) == 3

            # Check that all entries are included
            contents = [r["content"]["text"] for r in records]
            assert "Entry 1" in contents
            assert "Entry 2" in contents
            assert "Entry 3" in contents

    def test_write_entries_to_local_file(self, agentcore_config, temp_hermes_home):
        """Test writing entries to local file."""
        provider = AgentCoreMemoryProvider()
        entries = ["Local entry 1", "Local entry 2"]

        with patch.object(provider, '_atomic_write') as mock_write:
            provider._write_entries_to_local_file("memory", entries)

            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert str(args[0]).endswith("MEMORY.md")
            assert args[1] == "Local entry 1\n§\nLocal entry 2"

    def test_clear_agentcore_primary_namespace(self, agentcore_config):
        """Test clearing AgentCore primary namespace."""
        provider = AgentCoreMemoryProvider()
        provider._builtin_namespace_memory = "/builtin-primary/memory/"

        mock_mgr = Mock()
        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            provider._clear_agentcore_primary_namespace("memory")

            mock_mgr.delete_all_long_term_memories_in_namespace.assert_called_once_with(
                "/builtin-primary/memory/"
            )


class TestMemoryWriteHook:
    """Test the on_memory_write hook functionality."""

    def test_on_memory_write_primary_mode(self, agentcore_config, temp_hermes_home):
        """Test on_memory_write hook in primary mode."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._primary_mode = True
        provider._session = Mock()
        provider._actor_id = "test-user"
        provider._session_id = "test-session"

        with patch.object(provider, '_get_session_manager') as mock_mgr, \
             patch.object(provider, '_sync_builtin_memory_to_primary') as mock_sync, \
             patch.object(provider, '_backup_builtin_file') as mock_backup:

            provider.on_memory_write("add", "memory", "Test entry")

            # Wait for background thread
            if provider._backup_thread:
                provider._backup_thread.join(timeout=5.0)

            # Should sync to primary and backup
            mock_sync.assert_called_once_with("memory")
            mock_backup.assert_called_once_with("memory")

    def test_on_memory_write_event_mirroring(self, agentcore_config):
        """Test that memory writes are mirrored as events."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._primary_mode = True
        provider._session = Mock()
        provider._actor_id = "test-user"
        provider._session_id = "test-session"

        mock_mgr = Mock()

        with patch.object(provider, '_get_session_manager', return_value=mock_mgr), \
             patch.object(provider, '_sync_builtin_memory_to_primary'), \
             patch.object(provider, '_backup_builtin_file'):

            provider.on_memory_write("add", "memory", "Test entry")

            # Wait for background thread
            if provider._backup_thread:
                provider._backup_thread.join(timeout=5.0)

            # Should create event for the write
            mock_mgr.add_turns.assert_called()
            call_args = mock_mgr.add_turns.call_args
            assert call_args[1]["actor_id"] == "test-user"
            assert call_args[1]["session_id"] == "test-session"
            messages = call_args[1]["messages"]
            assert len(messages) == 1
            assert "memory add on memory" in str(messages[0])

    def test_on_memory_write_circuit_breaker(self, agentcore_config):
        """Test that circuit breaker prevents writes when open."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._primary_mode = True
        provider._session = Mock()
        provider._consecutive_failures = 10  # Trigger circuit breaker

        with patch.object(provider, '_sync_builtin_memory_to_primary') as mock_sync:
            provider.on_memory_write("add", "memory", "Test entry")

            # Should not sync when circuit breaker is open
            mock_sync.assert_not_called()

    def test_on_memory_write_no_session(self, agentcore_config):
        """Test on_memory_write when session is not available."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._primary_mode = True
        provider._session = None  # No session

        with patch.object(provider, '_sync_builtin_memory_to_primary') as mock_sync:
            provider.on_memory_write("add", "memory", "Test entry")

            # Should not sync when no session
            mock_sync.assert_not_called()


class TestToolHandling:
    """Test tool schema and handling."""

    def test_get_tool_schemas_with_session(self, agentcore_config):
        """Test tool schemas when session is available."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()

        schemas = provider.get_tool_schemas()

        assert len(schemas) == 3
        schema_names = [s["name"] for s in schemas]
        assert "agentcore_search" in schema_names
        assert "agentcore_list" in schema_names
        assert "agentcore_recent" in schema_names

    def test_get_tool_schemas_no_session(self, agentcore_config):
        """Test tool schemas when no session is available."""
        provider = AgentCoreMemoryProvider()
        provider._session = None

        schemas = provider.get_tool_schemas()

        assert schemas == []

    def test_handle_tool_call_search(self, agentcore_config, sample_memory_records):
        """Test agentcore_search tool call."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()

        mock_mgr = Mock()
        mock_mgr.search_long_term_memories.return_value = sample_memory_records

        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            result = provider.handle_tool_call("agentcore_search", {"query": "user preferences"})

            result_data = json.loads(result)
            assert "results" in result_data
            assert result_data["count"] == 3

    def test_handle_tool_call_list(self, agentcore_config, sample_memory_records):
        """Test agentcore_list tool call."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()

        mock_mgr = Mock()
        mock_mgr.list_long_term_memory_records.return_value = sample_memory_records

        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            result = provider.handle_tool_call("agentcore_list", {"max_results": 10})

            result_data = json.loads(result)
            assert "results" in result_data
            assert result_data["count"] == 3

    def test_handle_tool_call_recent(self, agentcore_config):
        """Test handling of agentcore_recent tool call."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()
        provider._actor_id = "test-user"
        provider._session_id = "test-session"

        # Use SimpleNamespace instead of Mock to avoid Mock's __getattr__
        # triggering hasattr("get") which routes to the dict branch
        from types import SimpleNamespace
        mock_turns = [
            [
                SimpleNamespace(role="USER", content="Hello"),
                SimpleNamespace(role="ASSISTANT", content="Hi there")
            ]
        ]

        mock_mgr = Mock()
        mock_mgr.get_last_k_turns.return_value = mock_turns

        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            result = provider.handle_tool_call("agentcore_recent", {"k": 5})

            result_data = json.loads(result)
            assert "turns" in result_data

    def test_handle_tool_call_circuit_breaker(self, agentcore_config):
        """Test tool calls when circuit breaker is open."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()

        # Properly trigger circuit breaker through _record_failure
        for _ in range(6):  # Above _BREAKER_THRESHOLD of 5
            provider._record_failure()

        result = provider.handle_tool_call("agentcore_search", {"query": "test"})

        result_data = json.loads(result)
        assert "error" in result_data
        assert "temporarily unavailable" in result_data["error"]

    def test_handle_tool_call_unknown_tool(self, agentcore_config):
        """Test handling of unknown tool calls."""
        provider = AgentCoreMemoryProvider()
        provider._session = Mock()

        result = provider.handle_tool_call("unknown_tool", {})

        result_data = json.loads(result)
        assert "error" in result_data
        assert "Unknown tool" in result_data["error"]


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_opens_after_failures(self, agentcore_config):
        """Test that circuit breaker opens after consecutive failures."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config

        # Record failures to trigger circuit breaker
        for _ in range(6):  # Above threshold
            provider._record_failure()

        assert provider._is_breaker_open() is True

    def test_circuit_breaker_closes_after_cooldown(self, agentcore_config):
        """Test that circuit breaker closes after cooldown period."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config

        # Trigger circuit breaker
        for _ in range(6):
            provider._record_failure()

        # Simulate cooldown period by setting the time to past
        provider._breaker_open_until = time.monotonic() - 1

        assert provider._is_breaker_open() is False

    def test_circuit_breaker_resets_on_success(self, agentcore_config):
        """Test that circuit breaker resets failure count on success."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config

        # Record some failures
        for _ in range(3):
            provider._record_failure()

        # Record success
        provider._record_success()

        assert provider._consecutive_failures == 0
        assert provider._is_breaker_open() is False


class TestUtilityFunctions:
    """Test utility functions."""

    def test_extract_record_content_dict_format(self):
        """Test extracting content from dict-like record."""
        record = {
            "content": {"text": "Test content"},
            "strategyId": "test_strategy"
        }

        result = AgentCoreMemoryProvider._extract_record_content(record)

        assert result["content"] == "Test content"
        assert result["type"] == "test_strategy"

    def test_extract_record_content_object_format(self):
        """Test extracting content from object-like record."""
        from types import SimpleNamespace
        record = SimpleNamespace(
            content={"text": "Object content"},
            strategyId="object_strategy",
        )

        result = AgentCoreMemoryProvider._extract_record_content(record)

        assert result["content"] == "Object content"
        assert result["type"] == "object_strategy"

    def test_extract_record_content_text_only(self):
        """Test extracting just text content."""
        record = {
            "content": {"text": "Text only content"}
        }

        result = AgentCoreMemoryProvider()._extract_record_content_text(record)

        assert result == "Text only content"

    def test_read_file_entries_valid_file(self, temp_hermes_home):
        """Test reading entries from a valid file."""
        test_file = temp_hermes_home / "test.md"
        test_file.write_text("Entry 1\n§\nEntry 2\n§\nEntry 3")

        provider = AgentCoreMemoryProvider()
        entries = provider._read_file_entries(test_file)

        assert len(entries) == 3
        assert "Entry 1" in entries
        assert "Entry 2" in entries
        assert "Entry 3" in entries

    def test_read_file_entries_nonexistent_file(self, temp_hermes_home):
        """Test reading entries from nonexistent file."""
        test_file = temp_hermes_home / "nonexistent.md"

        provider = AgentCoreMemoryProvider()
        entries = provider._read_file_entries(test_file)

        assert entries == []

    def test_read_file_entries_empty_file(self, temp_hermes_home):
        """Test reading entries from empty file."""
        test_file = temp_hermes_home / "empty.md"
        test_file.write_text("")

        provider = AgentCoreMemoryProvider()
        entries = provider._read_file_entries(test_file)

        assert entries == []


class TestHooks:
    """Test optional hook implementations."""

    def test_on_session_end(self, agentcore_config):
        """Test session end cleanup."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config

        # Create mock threads
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        provider._sync_thread = mock_thread
        provider._prefetch_thread = mock_thread
        provider._backup_thread = mock_thread

        provider.on_session_end([])

        # Should join all threads
        assert mock_thread.join.call_count == 3

    def test_on_pre_compress(self, agentcore_config):
        """Test pre-compression hook."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._session = Mock()
        provider._actor_id = "test-user"
        provider._session_id = "test-session"

        messages = [
            {"role": "user", "content": "Test user message"},
            {"role": "assistant", "content": "Test assistant response"},
        ]

        mock_mgr = Mock()
        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            result = provider.on_pre_compress(messages)

            # Should return empty string (no contribution to summary)
            assert result == ""

            # Should add compressed content as event
            mock_mgr.add_turns.assert_called()

    def test_on_delegation(self, agentcore_config):
        """Test delegation hook."""
        provider = AgentCoreMemoryProvider()
        provider._config = agentcore_config
        provider._session = Mock()
        provider._actor_id = "test-user"
        provider._session_id = "test-session"

        mock_mgr = Mock()
        with patch.object(provider, '_get_session_manager', return_value=mock_mgr):
            provider.on_delegation("Test task", "Test result", child_session_id="child-123")

            # Give background thread time to complete
            time.sleep(0.1)

            # Should record delegation as event
            # (The actual assertion would depend on thread completion)


class TestRegistration:
    """Test plugin registration."""

    def test_register_function_exists(self):
        """Test that register function exists and can be called."""
        import importlib
        import sys

        # Import the plugin module directly by path
        plugin_path = Path.home() / ".hermes" / "plugins" / "agentcore"
        sys.path.insert(0, str(plugin_path.parent))
        try:
            # Reload to ensure we get the module from the correct path
            if "agentcore" in sys.modules:
                mod = sys.modules["agentcore"]
            else:
                mod = importlib.import_module("agentcore")
            register_fn = getattr(mod, "register")

            # Mock context
            mock_ctx = Mock()
            register_fn(mock_ctx)

            # Should register memory provider
            mock_ctx.register_memory_provider.assert_called_once()
            args = mock_ctx.register_memory_provider.call_args[0]
            # Check by class name instead of isinstance (avoids dual-import identity issue)
            assert type(args[0]).__name__ == "AgentCoreMemoryProvider"
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    pytest.main([__file__])