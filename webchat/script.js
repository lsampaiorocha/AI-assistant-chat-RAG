// ===============================
// Configuration
// ===============================

const isLocalhost =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1' ||
  window.location.hostname.endsWith('.local');

const API_CONFIG = {
  endpoint: isLocalhost
    ? 'http://127.0.0.1:8000/api/chat' // Local dev (Uvicorn)
    : 'https://ai-mentor-api-clean-195914501585.southamerica-east1.run.app/api/chat', 
  headers: {
    // 'Authorization': 'Bearer YOUR_TOKEN',
  },
};

console.log('Environment:', isLocalhost ? 'LOCAL' : 'PRODUCTION');
console.log('Using API endpoint:', API_CONFIG.endpoint);

const USE_MOCK = false;

// ===============================
// Thread management (persistent per user)
// ===============================

// Try to restore a previous thread_id from localStorage.
// If none exists, create a new one (e.g., user-<random-hex>)
let THREAD_ID = localStorage.getItem('ai_mentor_thread_id');

if (!THREAD_ID) {
  THREAD_ID = 'user-' + Math.random().toString(36).substring(2, 10);
  localStorage.setItem('ai_mentor_thread_id', THREAD_ID);
  console.log('Created new thread_id:', THREAD_ID);
} else {
  console.log('Resuming existing thread_id:', THREAD_ID);
}


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

let messages = [];
let currentAbort = null;

// ===============================
// UI helpers
// ===============================

function setTyping(isTyping) {
  typingEl.classList.toggle('hidden', !isTyping);
  statusTag.textContent = isTyping ? 'Thinking…' : 'Ready';
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Renderiza mensagens com markdown */
function createMessageNode(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'assistant' ? '🤖' : '🧑';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = marked.parse(text || ""); 
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}

function renderMessages() {
  messagesEl.innerHTML = '';
  for (const m of messages) {
    messagesEl.appendChild(createMessageNode(m.role, m.content));
  }
  scrollToBottom();
}

function appendMessage(role, text) {
  messages.push({ role, content: text });
  messagesEl.appendChild(createMessageNode(role, text));
  scrollToBottom();
}

function updateLastAssistantMessage(text) {
  for (let i = messagesEl.children.length - 1; i >= 0; i -= 1) {
    const node = messagesEl.children[i];
    if (node.classList.contains('assistant')) {
      const bubble = node.querySelector('.bubble');
      bubble.innerHTML = marked.parse(text || ""); 
      break;
    }
  }
}

function setControlsBusy(busy) {
  sendBtn.disabled = busy;
  regenerateBtn.disabled = busy;
  newChatBtn.disabled = busy;
  stopBtn.disabled = !busy;
}

// ===============================
// Core chat logic
// ===============================

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
  appendMessage('assistant', '');

  try {
    const onToken = (token) => {
      assistantText += token;
      updateLastAssistantMessage(assistantText);
    };

    let res = null;
    if (USE_MOCK) {
      res = await callAgentMock(text, { onToken, signal: abortController.signal });
    } else {
      res = await callAgent(text, { onToken, signal: abortController.signal });
      if (res && res.text && assistantText.length === 0) {
        assistantText = res.text;
        updateLastAssistantMessage(assistantText);
      }
    }

    messages[messages.length - 1].content = assistantText || (res && res.text) || '';

    if (res && res.citations) {
      messages[messages.length - 1].citations = res.citations;
      console.log("Citations:", res.citations);
    }

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

function newChat() {
  if (currentAbort) return;
  messages = [];
  messagesEl.innerHTML = '';
  statusTag.textContent = 'Ready';

  // First friendly greeting
  appendMessage(
    'assistant',
    "Hi! I'm your AI Startup mentor — here to challenge your ideas, sharpen your vision, and help you build something remarkable. "
  );

  // Show typing indicator briefly before the next message
  setTyping(true);

  setTimeout(() => {
    setTyping(false);
    appendMessage(
      'assistant',
      "You can ask me about anything — and if you’d like other views, just tag the **CTO**, **PM**, or **VC**. You can also call the **committee** to hear from everyone at once."
    );
  }, 1200);

  inputEl.focus();
}


function stopGeneration() {
  if (currentAbort) {
    currentAbort.abort();
  }
}

// ===============================
// Backend integration
// ===============================

async function callAgent(message, { onToken, signal } = {}) {
  console.log('Sending to backend with thread_id =', THREAD_ID);
  const response = await fetch(API_CONFIG.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...API_CONFIG.headers },
    body: JSON.stringify({
      message,
      stream: false,
      thread_id: THREAD_ID, 
    }),
    signal,
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) detail = errJson.detail;
    } catch (_) {}
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
// Mock backend
// ===============================

async function callAgentMock(message, { onToken, signal } = {}) {
  const reply = mockReply(message);
  for (const token of tokenize(reply)) {
    if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
    await sleep(20 + Math.random() * 40);
    onToken(token);
  }
  return { text: reply, citations: [], phase: 'mock' };
}

function mockReply(message) {
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

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 220) + 'px';
}
inputEl.addEventListener('input', () => autoResizeTextarea(inputEl));

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

// Mensagem inicial + NÃO altera thread
newChat();
