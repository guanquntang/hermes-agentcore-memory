"""Test the conftest.py fixtures."""

import pytest


def test_mock_agentcore_provider(mock_agentcore_provider):
    """Test that mock AgentCore provider works."""
    provider = mock_agentcore_provider

    # Test write operation
    result = provider.write_builtin_memory("memory", "add", "Test entry")
    assert result["success"] is True

    # Test read operation
    entries = provider.read_builtin_memory("memory")
    assert "Test entry" in entries

    # Test that calls are recorded
    assert len(provider.write_calls) == 1
    assert len(provider.read_calls) == 1


def test_mock_session_manager(mock_session_manager):
    """Test that mock session manager works."""
    mgr = mock_session_manager

    # Test session creation
    session = mgr.create_memory_session("test-actor", "test-session")
    assert session is not None

    # Test adding turns
    from unittest.mock import Mock
    message = Mock()
    mgr.add_turns("test-actor", "test-session", [message])
    assert len(mgr.turns) == 1


def test_temp_hermes_home(temp_hermes_home):
    """Test that temporary HERMES_HOME works."""
    assert temp_hermes_home.exists()
    assert temp_hermes_home.is_dir()

    # Test that we can create files
    test_file = temp_hermes_home / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_agentcore_config(agentcore_config):
    """Test that AgentCore config fixture works."""
    assert agentcore_config["memory_id"] == "test-memory-id"
    assert agentcore_config["region"] == "us-east-1"
    assert agentcore_config["primary_mode"] is True


def test_memory_files(memory_files):
    """Test that memory files fixture works."""
    files = memory_files

    # Check that files exist
    assert files["memory_file"].exists()
    assert files["user_file"].exists()
    assert files["memories_dir"].exists()

    # Check content
    memory_content = files["memory_file"].read_text()
    assert "Test memory entry 1" in memory_content
    assert "Test memory entry 2" in memory_content

    user_content = files["user_file"].read_text()
    assert "software engineer" in user_content
    assert "Python" in user_content


def test_sample_memory_records(sample_memory_records):
    """Test that sample memory records fixture works."""
    records = sample_memory_records

    assert len(records) == 3
    assert any("dark theme" in str(record) for record in records)
    assert any("TypeScript" in str(record) for record in records)
    assert any("Meeting" in str(record) for record in records)


def test_mock_boto3_client(mock_boto3_client):
    """Test that mock boto3 client works."""
    client = mock_boto3_client

    # Should have the required method
    assert hasattr(client, 'batch_create_memory_records')

    # Should be callable
    client.batch_create_memory_records(
        memoryId="test",
        records=[{"test": "data"}]
    )
    assert client.batch_create_memory_records.called