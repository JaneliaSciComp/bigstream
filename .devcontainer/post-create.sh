#!/bin/bash
set -e

# Initialize network firewall (restricts outbound to allowed domains)
echo "Initializing network firewall..."
sudo /usr/local/bin/init-firewall.sh

mkdir -p ~/.npm-global
npm config set prefix ~/.npm-global

# Install Claude Code globally via npm (provided by pixi)
if ! command -v claude &> /dev/null; then
    echo "Installing Claude Code..."
    npm install -g @anthropic-ai/claude-code
fi

WS="$(pwd)"
echo "source ${WS}/.devcontainer/settings.sh" >> ~/.bashrc
echo "source ${WS}/.devcontainer/settings.sh" >> ~/.zshrc

echo ""
echo "=========================================="
echo "Dev container setup complete!"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  claude --dangerously-skip-permissions  - Start Claude Code"
echo ""
