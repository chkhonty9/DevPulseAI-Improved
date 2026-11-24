# OpenRouter Foundation

## What changed

- `config/openrouter_settings.py` resolves keys and model configuration.
- `services/llm.py` defines the provider-neutral client contract.
- `services/openrouter_client.py` implements that contract with OpenRouter's
  OpenAI-compatible endpoint.
- Relevance and risk analysis now send their AI requests through the shared
  client.
- Synthesis receives the same client but intentionally remains deterministic
  until the evidence-cited AI synthesis stage.
- The Streamlit sidebar supports local configuration or a masked user key.

## Key modes

### Local configuration

Use either an environment variable:

```powershell
$env:OPENROUTER_API_KEY="your-rotated-key"
```

or copy `.streamlit/secrets.example.toml` to
`.streamlit/secrets.toml` and replace the placeholder. The real secrets file
is ignored by Git.

### User session

Select `Enter my own key` in the Streamlit sidebar. The value remains in the
current Streamlit session and is not copied into an environment variable,
source file, report, or log.

## Model

The default is `openrouter/free`. A user can replace it with any compatible
OpenRouter `provider/model` identifier in the sidebar.

## Current boundary

This stage changes the provider foundation only. The next stage removes the
remaining silent heuristic fallbacks and replaces them with visible failures
and controlled retries.
