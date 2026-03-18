# ModelOpt Slack Bot

Connect ModelOpt agent skills (PTQ, deployment, evaluation) to Slack via Claude Agent SDK.

## Setup

### 1. Create Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Enable **Socket Mode** (Settings > Socket Mode > Enable)
3. Generate an **App-Level Token** with `connections:write` scope — this is your `SLACK_APP_TOKEN` (xapp-...)
4. Add **Bot Token Scopes** (OAuth & Permissions):
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:read`
5. **Subscribe to Events** (Event Subscriptions):
   - `app_mention`
   - `message.im`
6. Install the app to your workspace
7. Copy the **Bot User OAuth Token** — this is your `SLACK_BOT_TOKEN` (xoxb-...)

### 2. Install Dependencies

```bash
cd slack-bot
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your tokens
```

### 4. Run

```bash
source .env
python bot.py
```

## Usage

In Slack, mention the bot or DM it:

```text
@modelopt quantize Qwen3-0.6B with fp8
@modelopt deploy ./qwen3-0.6b-fp8 with vLLM
@modelopt evaluate my quantized model on mmlu
@modelopt quantize and evaluate Llama-3.1-8B with nvfp4
```

The bot auto-discovers skills from `.claude/skills/` in the parent directory and routes requests to the appropriate skill (ptq, deployment, evaluation, or the modelopt orchestrator).

## Architecture

```text
Slack (Socket Mode)
  ↓
bot.py (slack-bolt async)
  ↓
Claude Agent SDK
  ↓
.claude/skills/
  ├── ptq/          → quantization
  ├── deployment/   → model serving
  ├── evaluation/   → accuracy benchmarks
  └── modelopt/     → orchestrator
```
