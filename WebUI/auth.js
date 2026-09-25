/**
 * API keys for Handbook Studio. Keys are kept in this browser's localStorage
 * and attached to every /api/ request by the fetch wrapper below:
 *   - OpenAI or GIBD key  -> X-API-Key       (also identifies the user)
 *   - Anthropic key       -> X-Anthropic-Key (Claude Agent SDK provider only)
 * The server hashes the key into a user id; it never stores the key.
 */
const OPENAI_KEY_LS = 'agm_openai_api_key';
const ANTHROPIC_KEY_LS = 'agm_anthropic_api_key';

function _storedKey(name) {
    const raw = localStorage.getItem(name);
    return raw ? raw.replace(/\s/g, '') : null;
}

// FormData requests get the keys as form fields -- a custom header on a
// multipart upload can corrupt its Content-Type boundary.
const _originalFetch = window.fetch.bind(window);
window.fetch = function (url, options = {}) {
    try {
        const apiKey = _storedKey(OPENAI_KEY_LS);
        const anthropicKey = _storedKey(ANTHROPIC_KEY_LS);
        if ((apiKey || anthropicKey) && typeof url === 'string' && url.includes('/api/')) {
            if (options.body instanceof FormData) {
                if (apiKey) options.body.append('api_key', apiKey);
                if (anthropicKey) options.body.append('anthropic_api_key', anthropicKey);
            } else {
                options = { ...options };
                const headers = new Headers(options.headers || {});
                if (apiKey) headers.set('X-API-Key', apiKey);
                if (anthropicKey) headers.set('X-Anthropic-Key', anthropicKey);
                options.headers = headers;
            }
        }
    } catch (e) {
        console.warn('[fetch override] Could not inject API key:', e);
    }
    return _originalFetch(url, options);
};

function _flashStatus(id, text, color, ms = 3000) {
    const status = document.getElementById(id);
    if (!status) return;
    status.textContent = text;
    status.style.color = color;
    setTimeout(() => { status.textContent = ''; }, ms);
}

function toggleConfigDropdown() {
    document.getElementById('config-dropdown').classList.toggle('show');
}

function toggleKeyVisibility(inputId) {
    const input = document.getElementById(inputId);
    input.type = input.type === 'text' ? 'password' : 'text';
}

function saveApiKey() {
    const key = document.getElementById('GIBD-API-key').value.trim();
    if (!key) return _flashStatus('api-key-status', 'Key is empty.', 'red');
    // GIBD key (gibd-...), OpenAI key (sk-...), or "no-api" for a self-hosted
    // model server. The backend routes by prefix.
    if (key !== 'no-api' && !/^(gibd[-_]|sk-)/i.test(key)) {
        return _flashStatus('api-key-status',
            'Enter a GIBD key (starts with "gibd-"), an OpenAI key (starts with "sk-"), or "no-api" if you only use a self-hosted model.',
            'red', 4000);
    }
    localStorage.setItem(OPENAI_KEY_LS, key.replace(/\s/g, ''));
    refreshUserIdBadge();
    _flashStatus('api-key-status', 'Key saved!', 'green');
}

function clearApiKey() {
    document.getElementById('GIBD-API-key').value = '';
    localStorage.removeItem(OPENAI_KEY_LS);
    // Tell the backend to clear the key from os.environ
    _originalFetch('/api/clear-key', { method: 'POST' }).catch(() => {});
    refreshUserIdBadge();
    _flashStatus('api-key-status', 'Key cleared!', 'orange');
}

function saveAnthropicApiKey() {
    const key = document.getElementById('anthropic-API-key').value.trim();
    if (!key) return _flashStatus('anthropic-api-key-status', 'Key is empty.', 'red');
    if (!/^sk-ant-/i.test(key)) {
        return _flashStatus('anthropic-api-key-status',
            'Enter an Anthropic API key (starts with "sk-ant-").', 'red', 4000);
    }
    localStorage.setItem(ANTHROPIC_KEY_LS, key.replace(/\s/g, ''));
    _flashStatus('anthropic-api-key-status', 'Key saved!', 'green');
}

function clearAnthropicApiKey() {
    document.getElementById('anthropic-API-key').value = '';
    localStorage.removeItem(ANTHROPIC_KEY_LS);
    _flashStatus('anthropic-api-key-status', 'Key cleared!', 'orange');
}

// Show the user_id the server derives from the saved key (sha256(key)[:16]).
async function refreshUserIdBadge() {
    const row = document.getElementById('api-key-user-id');
    const value = document.getElementById('api-key-user-id-value');
    if (!row || !value) return;
    if (_storedKey(OPENAI_KEY_LS)) {
        try {
            const resp = await fetch('/api/whoami');
            const data = await resp.json();
            if (resp.ok && data.success && data.user_id) {
                value.textContent = data.user_id;
                row.style.display = '';
                return;
            }
        } catch (e) { /* server down or key rejected: just don't show an id */ }
    }
    row.style.display = 'none';
    value.textContent = '';
}

function copyUserId() {
    const value = document.getElementById('api-key-user-id-value');
    if (!value || !value.textContent) return;
    navigator.clipboard.writeText(value.textContent)
        .then(() => _flashStatus('api-key-status', 'ID copied', 'green', 2000))
        .catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
    const openaiInput = document.getElementById('GIBD-API-key');
    const anthropicInput = document.getElementById('anthropic-API-key');
    if (openaiInput) openaiInput.value = localStorage.getItem(OPENAI_KEY_LS) || '';
    if (anthropicInput) anthropicInput.value = localStorage.getItem(ANTHROPIC_KEY_LS) || '';
    refreshUserIdBadge();
});
