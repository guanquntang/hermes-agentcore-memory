# AgentCore Memory Plugin (Self-Contained)

This is a **self-contained** AgentCore Memory plugin that implements primary storage mode through MemoryProvider hooks without requiring modifications to `run_agent.py` or `tools/memory_tool.py`.

## Architecture

The plugin operates in "primary mode" where AgentCore is the source of truth for builtin memory, synchronized via hooks:

- **Writes**: Local first (fast) → `on_memory_write()` hook → sync to AgentCore primary storage
- **Reads**: Always from local cache (zero latency)
- **Startup**: `initialize()` hook → pull from AgentCore → update local cache
- **Offline**: Local works fine, writes queue up, sync on reconnect

## Key Benefits

1. **No Core Modifications**: Works entirely through MemoryProvider hooks
2. **Git-Pull Safe**: No scattered modifications that get overwritten by updates
3. **Self-Contained**: All functionality in `~/.hermes/plugins/agentcore/`
4. **Backward Compatible**: Falls back gracefully when AgentCore unavailable

## Installation

The plugin is automatically installed at `~/.hermes/plugins/agentcore/` and takes precedence over the bundled version.

### Configuration

Set up via environment variables:
```bash
export AGENTCORE_MEMORY_ID="your-memory-store-id"
export AGENTCORE_PRIMARY_MODE="true"  # Enable primary mode (default)
export AWS_REGION="us-east-1"
```

Or via `~/.hermes/agentcore_memory.json`:
```json
{
  "memory_id": "your-memory-store-id",
  "primary_mode": true,
  "region": "us-east-1",
  "namespace_prefix": "/"
}
```

## Hook Implementation

### `initialize(session_id, **kwargs)`
- Handles startup sync/migration logic (previously in `run_agent.py`)
- First-time: migrates local files to AgentCore primary storage
- Subsequent runs: syncs from AgentCore to local cache
- Uses migration locks to prevent race conditions

### `on_memory_write(action, target, content)`
- Called after builtin memory writes (add/replace/remove)
- Syncs local file state to AgentCore primary storage
- Maintains HA backup functionality
- Handles event mirroring for strategy extraction

## Files Structure

```
~/.hermes/plugins/agentcore/
├── __init__.py              # Main plugin implementation
├── plugin.yaml              # Plugin metadata
├── README.md                # This documentation
└── tests/
    ├── conftest.py           # Test fixtures
    ├── test_plugin.py        # Comprehensive tests
    ├── test_conftest.py      # Fixture tests
    └── run_tests.sh          # Test runner script
```

## Testing

Run the test suite:
```bash
~/.hermes/plugins/agentcore/tests/run_tests.sh
```

Or run basic functionality test:
```bash
cd ~/.hermes/plugins/agentcore
python3 -c "from __init__ import AgentCoreMemoryProvider; p = AgentCoreMemoryProvider(); print('✅ Plugin works!')"
```

## Migration from Bundled Version

The self-contained plugin automatically replaces the bundled version. The core modifications in `run_agent.py` and `tools/memory_tool.py` have been reverted, making the system cleaner and more maintainable.

### What Was Removed

1. **run_agent.py**: AgentCore-primary startup sync logic (lines 1331-1414)
2. **run_agent.py**: Memory tool dispatch with primary mode checking (lines 7422-7439)
3. **tools/memory_tool.py**: `agentcore_provider` parameter
4. **plugins/memory/agentcore/**: Bundled plugin directory

### What Was Added

- Complete self-contained plugin at `~/.hermes/plugins/agentcore/`
- Primary mode logic implemented through hooks
- Comprehensive test suite
- Plugin metadata and documentation

## Features

- ✅ Semantic search (`agentcore_search`)
- ✅ Memory listing (`agentcore_list`) 
- ✅ Recent turns (`agentcore_recent`)
- ✅ Automatic extraction and summarization
- ✅ User preference detection
- ✅ Primary storage mode (AgentCore as source of truth)
- ✅ HA backup/restore
- ✅ Circuit breaker for API failures
- ✅ Content security scanning
- ✅ Migration from local files
- ✅ Startup sync from cloud
- ✅ Offline resilience with write queue

## Debug Information

Check plugin discovery:
```bash
python3 -c "
from plugins.memory import discover_memory_providers
for name, desc, avail in discover_memory_providers():
    if name == 'agentcore':
        print(f'AgentCore: {desc} (available: {avail})') 
"
```

Check plugin loading:
```bash
python3 -c "
from plugins.memory import load_memory_provider
provider = load_memory_provider('agentcore')
print(f'Loaded: {provider.name if provider else None}')
"
```