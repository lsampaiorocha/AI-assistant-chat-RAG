// ===============================
// Configuration
// ===============================

/**
 * API backend configuration.
 * - endpoint: URL of the FastAPI server (/api/chat).
 * - headers: optional custom headers (e.g. Authorization).
 */
const API_CONFIG = {
  endpoint: 'http://localhost:8000/api/chat',
  headers: {
    // 'Authorization': 'Bearer YOUR_TOKEN',
  },
};

/**
 * Toggle to use mock responses instead of the real backend.
 * - true: runs locally with hardcoded responses (no backend needed).
 * - false: calls the FastAPI agent backend.
 */
const USE_MOCK = false;


// ===============================
// DOM element references
// ===============================

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const regenerateBtn = document.getElementById('regenerateBtn');
const stopBtn = document.getElementById('stopBtn');
const typingEl = document.getElementById('typing');
const statusTag = document.getElementById('statusTag');

let messages = [];       // stores conversation history
let currentAbort = null; // active AbortController for streaming

// A stable thread id (one per chat). LangGraph checkpointer uses this.
// Persist it so a page refresh continues the same conversation.
const makeThreadId = () => `web-${Math.random().toString(36).slice(2, 10)}`;
let threadId = localStorage.getItem("thread_id") || makeThreadId();
localStorage.setItem("thread_id", threadId);


// ===============================
// UI helpers
// ===============================

/** Show/hide typing indicator */
function setTyping(isTyping) {
  typingEl.classList.toggle('hidden', !isTyping);
  statusTag.textContent = isTyping ? 'Thinking…' : 'Ready';
}

/** Auto-scroll chat to bottom */
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Build a DOM node for a chat message */
function createMessageNode(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'assistant' ? '🤖' : '🧑';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}

/** Render full chat history into the chat window */
function renderMessages() {
  messagesEl.innerHTML = '';
  for (const m of messages) {
    messagesEl.appendChild(createMessageNode(m.role, m.content));
  }
  scrollToBottom();
}

/** Append a new message to history and DOM */
function appendMessage(role, text) {
  messages.push({ role, content: text });
  messagesEl.appendChild(createMessageNode(role, text));
  scrollToBottom();
}

/** Update the bubble text of the last assistant message */
function updateLastAssistantMessage(text) {
  for (let i = messagesEl.children.length - 1; i >= 0; i -= 1) {
    const node = messagesEl.children[i];
    if (node.classList.contains('assistant')) {
      const bubble = node.querySelector('.bubble');
      bubble.textContent = text;
      break;
    }
  }
}

/** Enable/disable chat controls while processing */
function setControlsBusy(busy) {
  sendBtn.disabled = busy;
  regenerateBtn.disabled = busy;
  newChatBtn.disabled = busy;
  stopBtn.disabled = !busy;
}


// ===============================
// Core chat logic
// ===============================

/**
 * Send a user message and handle assistant response.
 * - Updates UI state (busy, typing).
 * - Calls backend (real or mock).
 * - Handles streaming or JSON responses.
 */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || currentAbort) return;

  inputEl.value = '';
  autoResizeTextarea(inputEl);
  appendMessage('user', text);

  setControlsBusy(true);
  setTyping(true);

  const abortController = new AbortController();
  currentAbort = abortController;

  let assistantText = '';
  appendMessage('assistant', ''); // placeholder for streaming

  try {
    const history = messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(0, -1); // exclude placeholder

    const onToken = (token) => {
      assistantText += token;
      updateLastAssistantMessage(assistantText);
    };

    let res = null;
    if (USE_MOCK) {
      res = await callAgentMock(text, history, { onToken, signal: abortController.signal });
    } else {
      res = await callAgent(text, history, { onToken, signal: abortController.signal });
      if (res && res.text && assistantText.length === 0) {
        assistantText = res.text;
        updateLastAssistantMessage(assistantText);
      }
    }

    // finalize state
    messages[messages.length - 1].content = assistantText || (res && res.text) || '';

    // optionally keep citations
    if (res && res.citations) {
      messages[messages.length - 1].citations = res.citations;
      console.log("Citations:", res.citations);
    }

    // show phase if backend returned one (safe no-op if not present)
    if (res && res.phase) {
      statusTag.textContent = `Phase: ${res.phase}`;
    }

  } catch (err) {
    if (err.name === 'AbortError') {
      updateLastAssistantMessage(assistantText + '\n[stopped]');
    } else {
      console.error(err);
      const msg = (err && err.message) ? err.message : 'Sorry, something went wrong.';
      updateLastAssistantMessage(`[error] ${msg}`);
      statusTag.textContent = 'Error';
    }
  } finally {
    currentAbort = null;
    setTyping(false);
    setControlsBusy(false);
  }
}

/** Regenerate assistant’s last answer */
async function regenerateLast() {
  if (currentAbort) return;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') {
      if (i === messages.length - 2 && messages[messages.length - 1].role === 'assistant') {
        messages.pop();
        messagesEl.removeChild(messagesEl.lastChild);
      }
      inputEl.value = messages[i].content;
      autoResizeTextarea(inputEl);
      return sendMessage();
    }
  }
}

/** Start a new empty chat */
function newChat() {
  if (currentAbort) return;
  messages = [];
  messagesEl.innerHTML = '';
  statusTag.textContent = 'Ready';
  // Start a brand-new conversation thread (and persist it)
  threadId = makeThreadId();
  localStorage.setItem("thread_id", threadId);
  appendMessage('assistant', "Hi! I'm your AI mentor. Let's get started!");
  inputEl.focus();
}

/** Stop ongoing streaming generation */
function stopGeneration() {
  if (currentAbort) {
    currentAbort.abort();
  }
}


// ===============================
// Backend integration
// ===============================

/**
 * Call the FastAPI agent backend.
 * - Supports both JSON and streaming responses.
 * - Returns { text, citations, phase? }.
 */
async function callAgent(message, history, { onToken, signal } = {}) {
  const response = await fetch(API_CONFIG.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...API_CONFIG.headers },
    body: JSON.stringify({
      message,
      history,
      stream: false, // set true for streaming tokens if the server streams
      thread_id: threadId, // 👈 IMPORTANT: used by LangGraph checkpointer
      // system_prompt: "You are a helpful AI assistant.",
    }),
    signal,
  });

  // Handle non-2xx with readable error
  if (!response.ok) {
    let detail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) detail = errJson.detail;
    } catch (_) { /* ignore parse error */ }
    throw new Error(detail);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const data = await response.json();
    return {
      text: data.reply || '',
      citations: data.citations || [],
      phase: data.phase || null,
    };
  }

  // Streaming fallback (if server returns text/plain chunked)
  if (response.body && onToken) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      onToken(chunk);
    }

    return { text: buffer.trim(), citations: [], phase: null };
  }

  return { text: '', citations: [], phase: null };
}


// ===============================
// Mock backend (for local testing)
// ===============================

async function callAgentMock(message, history, { onToken, signal } = {}) {
  const reply = mockReply(message, history);
  for (const token of tokenize(reply)) {
    if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
    await sleep(20 + Math.random() * 40);
    onToken(token);
  }
  return { text: reply, citations: [], phase: 'mock' };
}

function mockReply(message, history) {
  if (/hello|hi|hey/i.test(message)) return 'Hello! How can I help you today?';
  if (/help|feature/i.test(message)) return 'Sure! Ask me anything about your project or ideas.';
  return `You said: "${message}"`;
}

function* tokenize(text) {
  for (const ch of text) yield ch;
}
const sleep = (ms) => new Promise(r => setTimeout(r, ms));


// ===============================
// UI wiring
// ===============================

/** Auto-resize textarea height */
function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 220) + 'px';
}
inputEl.addEventListener('input', () => autoResizeTextarea(inputEl));

/** Keyboard: Enter = send, Shift+Enter = newline */
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', newChat);
regenerateBtn.addEventListener('click', regenerateLast);
stopBtn.addEventListener('click', stopGeneration);

// Initial greeting
appendMessage('assistant', "Hi! I'm your AI mentor. Let's get started!");
