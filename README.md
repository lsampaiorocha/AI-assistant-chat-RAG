# Minimal AI Chat UI

This project provides a minimal, clean web UI for chatting with an AI agent, inspired by modern AI chat apps. It's framework-agnostic (HTML/CSS/JS) and includes simple hooks to integrate your own API (non-streaming or streaming).

## Features
- Modern chat layout (left/right bubbles, avatars, auto-scroll)
- Input textarea with send via Enter and Shift+Enter for newline
- New chat, regenerate last reply, and stop-generation controls
- Model/status indicator and typing indicator
- Light/dark theme with system preference support
- Simple API integration hooks (non-streaming and streaming)
- Optional local mock mode for quick testing

## Getting started

1) Open locally
- You can open `public/index.html` directly in your browser, but some features (like streaming) work best over HTTP.

2) Serve locally (recommended)
- With Python (3.x):
  ```bash
  cd public && python -m http.server 5173
  ```
  Then visit `http://localhost:5173`.

- With Node.js (if installed):
  ```bash
  npx serve public -l 5173
  ```

3) Run the backend (optional, for real LLM)
- In another terminal:
  ```bash
  cd agent
  pip install -r requirements.txt
  # set envs: OPENAI_API_KEY, CORS_ALLOW_ORIGINS=http://localhost:5173
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
  Then set `USE_MOCK = false` in `public/script.js` and `API_CONFIG.endpoint = 'http://localhost:8000/api/chat'`.

## Integrating your API

Edit `public/script.js` and update the `callAgent` function and/or the `API_CONFIG` object.

### Non‑streaming example
```js
// Replace with your endpoint
const response = await fetch(API_CONFIG.endpoint, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', ...API_CONFIG.headers },
  body: JSON.stringify({
    message, // latest user message
    history, // optional prior messages
    // ...any agent options
  })
});
const data = await response.json();
return { text: data.reply || data.text };
```

Expected JSON response shape (customize as needed):
```json
{ "reply": "Hello! How can I help you today?" }
```

### Streaming example (SSE or fetch stream)
If your API streams text chunks (Server-Sent Events or chunked fetch), implement the `onToken` callback:
```js
await callAgent(message, history, { onToken: (token) => appendToken(token) });
```
Inside `callAgent`, read from the response body reader and call `onToken` for each chunk decoded to text.

### Configuration
Update these values in `public/script.js`:
- `API_CONFIG.endpoint`: Your API URL
- `API_CONFIG.headers`: Any required auth headers
- Toggle `USE_MOCK` to `false` to use your real API

## Commands / Shortcuts
- Enter: Send
- Shift+Enter: New line

## File structure
- `public/index.html` — markup
- `public/styles.css` — styles (responsive, dark mode)
- `public/script.js` — chat logic and API hooks

## Notes
- This UI is intentionally minimal; extend as you need (attachments, message actions, markdown rendering, code highlighting, etc.).
- If you enable streaming, ensure your server sets appropriate CORS headers and uses chunked transfer without buffering.

## License
MIT

