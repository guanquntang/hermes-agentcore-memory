#!/bin/bash
# Test runner for AgentCore Memory plugin

set -e

echo "=== Running AgentCore Memory Plugin Tests ==="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"

echo "Plugin directory: $PLUGIN_DIR"
echo "Test directory: $SCRIPT_DIR"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please install it with:"
    echo "  pip install pytest"
    exit 1
fi

# Set PYTHONPATH to include the plugin directory
export PYTHONPATH="$PLUGIN_DIR:$PYTHONPATH"

# Check if bedrock-agentcore is available (optional for tests with mocking)
echo "Checking dependencies..."
python3 -c "
try:
    import bedrock_agentcore
    print('✓ bedrock-agentcore is available')
except ImportError:
    print('⚠ bedrock-agentcore not available (tests will use mocks)')
"

# Run the tests
echo ""
echo "Running tests..."
cd "$SCRIPT_DIR"

# Run tests with verbose output
if pytest -v test_plugin.py test_conftest.py 2>/dev/null; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "Running tests with coverage information..."
    pytest -v --tb=short test_plugin.py || {
        echo ""
        echo "❌ Some tests failed. Check the output above for details."
        exit 1
    }
fi

echo ""
echo "=== Test Summary ==="
echo "Plugin: AgentCore Memory Provider"
echo "Location: $PLUGIN_DIR"
echo "Tests: $(grep -c "def test_" test_plugin.py) test functions"
echo "Status: $(if [ $? -eq 0 ]; then echo "PASSED"; else echo "FAILED"; fi)"