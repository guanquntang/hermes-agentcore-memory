"""Test fixtures for AgentCore Memory plugin tests."""

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class MockAgentCoreProvider:
    """Mock AgentCore provider for testing."""

    def __init__(self):
        self.session = True
        self.memory_data = {}  # namespace -> list of entries
        self.fail_writes = False
        self.fail_reads = False
        self.write_calls = []
        self.read_calls = []
        self.delete_calls = []

    def write_builtin_memory(self, target, action, content, old_text=""):
        """Mock write method."""
        self.write_calls.append((target, action, content, old_text))

        if self.fail_writes:
            raise RuntimeError("AgentCore write failed")

        namespace = f"/builtin-primary/{target}/"
        if namespace not in self.memory_data:
            self.memory_data[namespace] = []

        if action == "add":
            if content not in self.memory_data[namespace]:
                self.memory_data[namespace].append(content)
        elif action == "replace":
            for i, entry in enumerate(self.memory_data[namespace]):
                if old_text in entry:
                    self.memory_data[namespace][i] = content
                    break
        elif action == "remove":
            self.memory_data[namespace] = [
                entry for entry in self.memory_data[namespace]
                if old_text not in entry
            ]

        return {"success": True, "message": f"Mock {action} completed"}

    def read_builtin_memory(self, target):
        """Mock read method."""
        self.read_calls.append(target)

        if self.fail_reads:
            raise RuntimeError("AgentCore read failed")

        namespace = f"/builtin-primary/{target}/"
        return self.memory_data.get(namespace, [])

    def sync_local_cache(self, target):
        """Mock sync method."""
        self.read_builtin_memory(target)
        return True

    def delete_all_long_term_memories_in_namespace(self, namespace):
        """Mock delete method."""
        self.delete_calls.append(namespace)
        self.memory_data[namespace] = []


class MockSessionManager:
    """Mock AgentCore session manager."""

    def __init__(self):
        self.memory_records = []
        self.turns = []
        self.search_results = []

    def create_memory_session(self, actor_id, session_id):
        return Mock()

    def add_turns(self, actor_id, session_id, messages):
        self.turns.extend(messages)

    def search_long_term_memories(self, query, namespace_prefix=None, top_k=5, max_results=None):
        return self.search_results

    def list_long_term_memory_records(self, namespace_prefix=None, max_results=20):
        return self.memory_records

    def get_last_k_turns(self, actor_id, session_id, k=5):
        return self.turns[-k:]

    def delete_all_long_term_memories_in_namespace(self, namespace):
        # Filter out records from this namespace
        self.memory_records = [r for r in self.memory_records if r.get("namespace") != namespace]


@pytest.fixture
def mock_agentcore_provider():
    """Provide a mock AgentCore provider."""
    return MockAgentCoreProvider()


@pytest.fixture
def mock_session_manager():
    """Provide a mock session manager."""
    return MockSessionManager()


@pytest.fixture
def temp_hermes_home(tmp_path):
    """Provide a temporary HERMES_HOME directory."""
    with patch('hermes_constants.get_hermes_home', return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def agentcore_config():
    """Provide test AgentCore configuration."""
    return {
        "memory_id": "test-memory-id",
        "region": "us-east-1",
        "namespace_prefix": "/",
        "primary_mode": True,
    }


@pytest.fixture
def memory_files(temp_hermes_home):
    """Create test memory files."""
    memories_dir = temp_hermes_home / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)

    memory_file = memories_dir / "MEMORY.md"
    memory_file.write_text("Test memory entry 1\n§\nTest memory entry 2")

    user_file = memories_dir / "USER.md"
    user_file.write_text("User is a software engineer\n§\nPrefers Python over JavaScript")

    return {
        "memory_file": memory_file,
        "user_file": user_file,
        "memories_dir": memories_dir,
    }


@pytest.fixture
def mock_boto3_client():
    """Provide a mock boto3 client."""
    client = Mock()
    client.batch_create_memory_records = Mock()
    return client


@pytest.fixture
def sample_memory_records():
    """Provide sample memory records for testing."""
    return [
        {
            "content": {"text": "User prefers dark theme"},
            "strategyId": "user_preference",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        {
            "content": {"text": "Project uses TypeScript"},
            "strategyId": "fact",
            "timestamp": "2024-01-01T01:00:00Z"
        },
        {
            "content": {"text": "Meeting scheduled for tomorrow"},
            "strategyId": "episodic",
            "timestamp": "2024-01-01T02:00:00Z"
        },
    ]