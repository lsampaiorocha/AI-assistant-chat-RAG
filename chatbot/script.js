// Configuration for API integration
const API_CONFIG = {
  endpoint: 'http://localhost:8000/api/chat',
  headers: {
    // 'Authorization': 'Bearer YOUR_TOKEN',
  },
};

// Toggle to use mock responses without a backend
const USE_MOCK = false;

// DOM elements
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

function setTyping(isTyping) {
  typingEl.classList.toggle('hidden', !isTyping);
  statusTag.textContent = isTyping ? 'Thinking…' : 'Ready';
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

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
  // Find the last assistant node and update its bubble text
  for (let i = messagesEl.children.length - 1; i >= 0; i -= 1) {
    const node = messagesEl.children[i];
    if (node.classList.contains('assistant')) {
      const bubble = node.querySelector('.bubble');
      bubble.textContent = text;
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
      .slice(0, -1); // exclude the placeholder we just added

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

    // finalize assistant message in state
    messages[messages.length - 1].content = assistantText || (res && res.text) || '';

    // opcional: guardar citations (se backend retornou)
    if (res && res.citations) {
      messages[messages.length - 1].citations = res.citations;
      console.log("Citations:", res.citations); // 👈 debug, pode usar na UI depois
    }

  } catch (err) {
    if (err.name === 'AbortError') {
      updateLastAssistantMessage(assistantText + '\n[stopped]');
    } else {
      console.error(err);
      updateLastAssistantMessage('Sorry, something went wrong.');
    }
  } finally {
    currentAbort = null;
    setTyping(false);
    setControlsBusy(false);
  }
}


async function regenerateLast() {
  if (currentAbort) return;
  // Find last user message
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'user') {
      // Remove trailing assistant if it directly follows that user message
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
  inputEl.focus();
}

function stopGeneration() {
  if (currentAbort) {
    currentAbort.abort();
  }
}

// Integration with FastAPI agent backend
async function callAgent(message, history, { onToken, signal } = {}) {
  const response = await fetch(API_CONFIG.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...API_CONFIG.headers },
    body: JSON.stringify({
      message,
      history,
      stream: false,  // 👈 muda para false se quiser JSON completo (com citations)
      // system_prompt: "You are a helpful AI assistant.", // opcional
    }),
    signal,
  });

  // Non-streaming JSON response
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const data = await response.json();
    return {
      text: data.reply || '',
      citations: data.citations || []   // 👈 garante que citations venha junto
    };
  }

  // Streaming response (chunked text)
  if (response.body && onToken) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      onToken(chunk);  // 👈 repassa os tokens ao UI
    }

    return { text: buffer.trim(), citations: [] }; // streaming não retorna citations
  }

  return { text: '', citations: [] };
}


// Mock streaming assistant for local testing
async function callAgentMock(message, history, { onToken, signal } = {}) {
  const reply = mockReply(message, history);
  // stream token by token
  for (const token of tokenize(reply)) {
    if (signal && signal.aborted) throw new DOMException('Aborted', 'AbortError');
    await sleep(20 + Math.random() * 40);
    onToken(token);
  }
  return { text: reply };
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

// Auto-resize textarea
function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 220) + 'px';
}
inputEl.addEventListener('input', () => autoResizeTextarea(inputEl));

// Keyboard handling: Enter to send, Shift+Enter for newline
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

// Initial greeting (optional)
appendMessage('assistant', 'Hi! I\'m your AI assistant. How can I help?');


