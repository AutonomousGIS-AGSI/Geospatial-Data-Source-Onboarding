/**
 * "Generate & Test" — a simpler, linear, OpenAI-only counterpart to
 * Experiment 2. One source, one handbook, one retrieval test, with a human
 * review pause in between. Deliberately independent of Experiment2Mode's
 * multi-mechanism research state; reuses its exp2-* CSS classes and the same
 * SSE-streaming/stage-card conventions.
 */
(function () {
    'use strict';

    // Mirrors WebUI/handbook_studio_store.py's MECHANISMS and
    // agents/data_agent/handbook_generator.py's _ACCESS_MECHANISMS -- keep
    // all three in sync. "other" is a UI-only sentinel: picking it reveals a
    // free-text box, and the value actually saved as access_mechanism is
    // whatever custom text the user types (see wireSetupInputs), not the
    // literal string "other".
    const MECHANISMS = {
        rest: { label: 'REST API', hint: 'HTTP endpoints and query parameters' },
        stac: { label: 'STAC', hint: 'Catalog, collection, item and asset search' },
        ogc: { label: 'OGC Service', hint: 'WMS, WFS, WCS and OGC API services' },
        sql: { label: 'SQL Interface', hint: 'Database and query-based access' },
        bulk: { label: 'Bulk Download', hint: 'Files, archives and repositories' },
        http_file: { label: 'HTTP File Download', hint: 'A single, direct static file link' },
        arcgis_featureserver: { label: 'ArcGIS FeatureServer', hint: 'Esri FeatureServer/MapServer REST endpoint' },
    };
    const OTHER_MECHANISM_VALUE = 'other';
    // True once a service type is set that isn't one of the fixed options --
    // covers both "Other was just picked" (value is literally the sentinel)
    // and "a custom type was already typed and saved" (value is that text).
    const isCustomMechanism = (s) => Boolean(s.access_mechanism) && !MECHANISMS[s.access_mechanism];

    // Mirrors index.html's #model-select and
    // WebUI/handbook_studio_store.py's MODEL_OPTIONS -- keep both in sync.
    const MODELS = [
        ['gpt-5.6-sol', 'GPT-5.6 Sol'], ['gpt-5.2', 'GPT-5.2'], ['gpt-5.3', 'GPT-5.3'], ['gpt-5.4', 'GPT-5.4'],
        ['gpt-4.1', 'GPT-4.1'], ['gpt-4.1-mini', 'GPT-4.1 Mini'], ['gpt-4.1-nano', 'GPT-4.1 Nano'],
        ['gpt-4o', 'GPT-4o'], ['gpt-4o-mini', 'GPT-4o Mini'], ['gpt-4-turbo', 'GPT-4 Turbo'],
    ];
    const DEFAULT_HANDBOOK_GEN_MODEL = 'gpt-5.2';
    const DEFAULT_DATA_RETRIEVAL_MODEL = 'gpt-4o';

    // The Claude Agent SDK provider option -- mirrors
    // WebUI/handbook_studio_store.py's CLAUDE_MODEL_OPTIONS.
    const CLAUDE_MODELS = [
        ['claude-opus-5', 'Claude Opus 5'],
        ['claude-sonnet-5', 'Claude Sonnet 5'],
        ['claude-haiku-4-5', 'Claude Haiku 4.5'],
    ];
    const DEFAULT_CLAUDE_HANDBOOK_MODEL = 'claude-sonnet-5';
    const DEFAULT_CLAUDE_DATA_RETRIEVAL_MODEL = 'claude-sonnet-5';
    const DEFAULT_PROVIDER = 'openai';

    function modelsForProvider(provider) {
        return provider === 'claude'
            ? { list: CLAUDE_MODELS, genDefault: DEFAULT_CLAUDE_HANDBOOK_MODEL, retrievalDefault: DEFAULT_CLAUDE_DATA_RETRIEVAL_MODEL }
            : { list: MODELS, genDefault: DEFAULT_HANDBOOK_GEN_MODEL, retrievalDefault: DEFAULT_DATA_RETRIEVAL_MODEL };
    }

    // Which models support variable reasoning effort, and what levels each
    // accepts (mirrors WebUI/script.js's REASONING_EFFORT_MAP and
    // WebUI/handbook_studio_store.py's REASONING_EFFORT_MODELS -- keep all
    // three in sync). Models not listed here (gpt-4.x family) don't
    // support it -- the control is hidden entirely for them.
    //
    // gpt-5.6-sol's "max" is real (verified live against the API) but only
    // on the Responses API surface -- the backend downgrades it to "xhigh"
    // for the Chat-Completions-only call sites (refine/evaluate/analyze,
    // and all of DataRetrieverAgent) rather than erroring, so it's still
    // safe to offer here.
    const REASONING_EFFORT_MODELS = {
        'gpt-5.6-sol': ['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        'gpt-5.2': ['none', 'low', 'medium', 'high', 'xhigh'],
        'gpt-5.3': ['low', 'medium', 'high', 'xhigh'],
        'gpt-5.4': ['none', 'low', 'medium', 'high', 'xhigh'],
    };
    const DEFAULT_REASONING_EFFORT = 'medium';

    // Per-attempt wall-clock budget for the generated retrieval code
    // (mirrors WebUI/handbook_studio_store.py's
    // MIN/MAX/UNLIMITED/DEFAULT_EXECUTION_TIMEOUT_SECONDS -- keep both in
    // sync). Configurable per session since sources vary widely, and
    // unlimited by default so a slow-but-working bulk/STAC download isn't
    // cut off before the user has a reason to set a limit. 0 is a distinct
    // sentinel meaning "no limit", not just a very large number.
    const MIN_EXECUTION_TIMEOUT_SECONDS = 60;
    const MAX_EXECUTION_TIMEOUT_SECONDS = 21600; // 6 hours
    const UNLIMITED_EXECUTION_TIMEOUT = 0;
    const DEFAULT_EXECUTION_TIMEOUT_SECONDS = UNLIMITED_EXECUTION_TIMEOUT;

    const state = {
        open: false,
        returnToGenerator: false,
        tab: 'setup',
        session: null,
        // Shared (public-link) view: everything renders, nothing can be
        // changed or run. Set by openShared(), cleared by open().
        readOnly: false,
        dirty: false,
        saving: false,
        running: false,
        runKeys: {},
        requiredKeys: [],
        lastScrolledStage: null,
        openDetails: new Set(),
        collapsedStages: new Set(),
        outputPreviews: new Map(),
        // outputPreviewKey of the preview shown full-window, or null. Held
        // here rather than on the element because render() rebuilds cards.
        fullPreview: null,
        // Slim section bar (numbers and titles only), remembered per browser.
        tabBarCollapsed: (() => {
            try { return localStorage.getItem('hbs.tabBarCollapsed') === '1'; } catch (_) { return false; }
        })(),
        // Retrieval code streamed from a run that is still going, keyed by
        // stage key -> Map(attempt number -> record). Held apart from the
        // session record because these are provisional: the moment the run is
        // saved, the stage's real attempts render from the run itself and
        // this is dropped.
        liveAttempts: new Map(),
        // Timer id for the rejoin poll (see watchActiveRun). Null when the
        // session has no run going, or when this tab is the one running it.
        activeRunTimer: null,
        // Open feedback boxes on code cards, their drafts, and which are
        // in flight -- all keyed by the same rerunKey the edit machinery uses.
        feedbackOpenFor: new Set(),
        feedbackDrafts: new Map(),
        feedbackSending: new Set(),
        // Model output for an in-flight revision, keyed the same way. Appended
        // straight into the DOM as it arrives; kept here so a render for any
        // other reason redraws what has streamed so far.
        feedbackStream: new Map(),
        // Which code cards have their version list open, keyed by rerunKey.
        versionsOpenFor: new Set(),
        versionSwitching: new Set(),
        // Which version rows have their change trace expanded, keyed
        // "<card key>::<version id>" so two cards never share a row's state.
        versionTraceOpen: new Set(),
        // Re-run results per test run, keyed by run id. Kept out of the session
        // record on purpose: a re-run is fresh evidence about the code, not a
        // revision of what the original run produced.
        rerunResults: new Map(),
        // Hand edits to retrieval code, keyed exactly like rerunResults.
        // `codeEditing` is which attempt cards are open in the editor and
        // `codeDrafts` the text typed into them -- both live outside the
        // session record so a live run's heartbeat re-render can't discard
        // what is being typed, and an unsaved draft is never mistaken for
        // the code the next re-run would actually execute.
        codeEditing: new Set(),
        codeDrafts: new Map(),
        codeSaving: new Set(),
        // Caret/scroll of the editor being typed in, restored after a
        // re-render replaces the textarea underneath the cursor.
        codeCaret: null,
        // Detail panels already rendered once, so a default-open panel
        // opens on first appearance only.
        seenDetails: new Set(),
        manualFormOpenFor: new Set(),
        // Manual review is blinded by default: while the rubric is open, the
        // other methods' verdicts for that run stay hidden, because a
        // reviewer who has already read "LLM judge: passed" is anchored.
        // Revealing is allowed but recorded on the verdict (blinded: false).
        manualRevealed: new Set(),
        manualStartedAt: {},
        lastRater: '',
        // The session picker's list and the text filtering it. Held in state
        // so the grid can be re-filtered without re-fetching or rebuilding
        // the search box under the user's cursor.
        sessionList: [],
        sessionQuery: '',
        providerAvailability: null, // filled in lazily by loadProviderAvailability()
    };

    const byId = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[ch]);
    // Python source as syntax-coloured HTML (highlight.js, loaded by
    // index.html); plain escaped text when the library is unavailable.
    // render() rebuilds cards wholesale, so results are cached by source.
    const pyHighlightCache = new Map();
    const highlightPython = (code) => {
        const text = String(code ?? '');
        if (!text || !window.hljs) return esc(text);
        let html = pyHighlightCache.get(text);
        if (html === undefined) {
            try {
                html = window.hljs.highlight(text, { language: 'python', ignoreIllegals: true }).value;
            } catch (_) {
                html = esc(text);
            }
            if (pyHighlightCache.size >= 200) pyHighlightCache.clear();
            pyHighlightCache.set(text, html);
        }
        return html;
    };
    const safeUrl = (value) => /^https?:\/\//i.test(String(value || '').trim())
        ? esc(String(value).trim()) : '';

    function clearRunViewState() {
        state.outputPreviews.forEach((record) => {
            const url = record?.preview?.object_url;
            if (url) URL.revokeObjectURL(url);
        });
        stopWatchingActiveRun();
        state.outputPreviews.clear();
        state.liveAttempts.clear();
        state.collapsedStages.clear();
        state.openDetails.clear();
        state.lastScrolledStage = null;
        // An open adjudication belongs to the run it was opened on; carrying
        // its blinding/timer state into another session would misreport both.
        state.manualFormOpenFor.clear();
        state.manualRevealed.clear();
        state.manualStartedAt = {};
        // Run ids are unique per session, but an unsaved draft belongs to the
        // card it was typed on; carrying it over would offer to save one
        // session's edit onto another's code.
        state.codeEditing.clear();
        state.codeDrafts.clear();
        state.codeSaving.clear();
        state.codeCaret = null;
        // Credentials persist across retries within a session (see startTest);
        // only clear them when actually switching to a different session.
        state.runKeys = {};
    }

    function blankSession() {
        return {
            source_name: '', documentation_url: '', retrieval_task: '', validation_spec: '',
            access_mechanism: '', access_mechanism_source: '', access_mechanism_why: '',
            provider: DEFAULT_PROVIDER,
            handbook_gen_model: DEFAULT_HANDBOOK_GEN_MODEL,
            data_retrieval_model: DEFAULT_DATA_RETRIEVAL_MODEL,
            handbook_gen_reasoning_effort: DEFAULT_REASONING_EFFORT,
            data_retrieval_reasoning_effort: DEFAULT_REASONING_EFFORT,
            execution_timeout_seconds: DEFAULT_EXECUTION_TIMEOUT_SECONDS,
            status: 'draft', generated_source: {}, test_result: {}, test_runs: [], pipeline_state: {},
        };
    }

    function markDirty() {
        state.dirty = true;
        const label = byId('handbook-studio-save-state');
        if (label) {
            label.textContent = 'Unsaved changes';
            label.className = 'exp2-save-state dirty';
        }
    }

    function savedState(text = 'Saved') {
        state.dirty = false;
        const label = byId('handbook-studio-save-state');
        if (label) {
            label.textContent = text;
            label.className = 'exp2-save-state saved';
        }
    }

    function toast(message, kind = 'success') {
        const el = byId('handbook-studio-toast');
        if (!el) return;
        el.textContent = message;
        el.className = `exp2-toast ${kind}`;
        el.hidden = false;
        clearTimeout(toast.timer);
        toast.timer = setTimeout(() => { el.hidden = true; }, 3500);
    }

    async function api(path, options = {}) {
        const response = await fetch(path, options);
        let data = {};
        try { data = await response.json(); } catch (_) { /* non-JSON */ }
        if (!response.ok || data.success === false) {
            throw new Error(data.error || `Request failed (${response.status})`);
        }
        return data;
    }

    // Fetched once per page load (not per session) -- whether the server
    // can actually run the Claude Agent SDK provider (package + CLI both
    // present). Lets the setup UI gray out the option with a real reason
    // instead of only failing on first use.
    async function loadProviderAvailability() {
        if (state.providerAvailability) return state.providerAvailability;
        try {
            const data = await api('/api/handbook-studio-providers');
            state.providerAvailability = data.providers || {};
        } catch (_) {
            // Availability check failing shouldn't block the whole UI --
            // just assume Claude might work and let the real request surface
            // the actual error if it doesn't.
            state.providerAvailability = { openai: { available: true }, claude: { available: true } };
        }
        return state.providerAvailability;
    }

    function open() {
        // Opened from the header (Generator not showing) => closing returns to
        // the main app; opened from inside the Generator => closing goes back
        // there, as before.
        const generatorWorkspace = byId('handbook-workspace');
        state.returnToGenerator = !!(generatorWorkspace && !generatorWorkspace.hidden);
        if (window.HandbookGenerator) HandbookGenerator.close();
        const exp2Workspace = byId('exp2-workspace');
        if (window.Experiment2Mode && exp2Workspace && !exp2Workspace.hidden) {
            Experiment2Mode.close();
        }
        const workspace = byId('handbook-studio-workspace');
        if (!workspace) return;
        if (!state.session) state.session = blankSession();
        state.open = true;
        state.readOnly = false;
        workspace.hidden = false;
        document.body.classList.add('exp2-open');
        byId('handbook-studio-nav-btn')?.classList.add('active');
        render();
        loadProviderAvailability().then(() => { if (state.open && state.tab === 'setup') render(); });
    }

    function close() {
        if (state.running) {
            toast('Stop the running step before leaving Handbook Studio.', 'info');
            return;
        }
        if (state.dirty && !confirm('Close Handbook Studio with unsaved changes?')) return;
        const workspace = byId('handbook-studio-workspace');
        if (workspace) workspace.hidden = true;
        document.body.classList.remove('exp2-open');
        byId('handbook-studio-nav-btn')?.classList.remove('active');
        state.open = false;
        if (state.returnToGenerator && window.HandbookGenerator) HandbookGenerator.open();
    }

    function switchTab(tab) {
        // 'validation' is intentionally not listed: the tab was removed from the
        // UI (renderValidation is kept for now but unreachable).
        if (!['setup', 'review', 'results'].includes(tab)) return;
        state.tab = tab;
        document.querySelectorAll('[data-hbs-tab]').forEach((button) => {
            button.classList.toggle('active', button.dataset.hbsTab === tab);
        });
        render();
    }

    function render() {
        const main = byId('handbook-studio-main');
        if (!main || !state.session) return;
        // Exposes the active session id on the workspace element (rather than
        // module-private `state`) so view_router.js can read it without an
        // API surface just for that; it drives the /handbook-studio/<id>
        // deep link.
        const workspace = byId('handbook-studio-workspace');
        if (workspace) {
            const sid = state.session.id || '';
            if (workspace.dataset.sessionId !== sid) workspace.dataset.sessionId = sid;
            const shared = state.readOnly ? '1' : '';
            if ((workspace.dataset.shared || '') !== shared) workspace.dataset.shared = shared;
            workspace.classList.toggle('hbs-readonly', state.readOnly);
        }
        syncTabBar();
        // A preview that was hidden or cleared (a re-run resets them) cannot
        // stay full-window.
        if (state.fullPreview && !state.outputPreviews.get(state.fullPreview)?.open) {
            state.fullPreview = null;
        }
        renderShareControls();
        const subtitle = byId('handbook-studio-subtitle');
        if (subtitle) {
            subtitle.textContent = state.session.source_name
                || 'Generate one handbook, review it, then test it';
        }
        // Captured before the markup is replaced: a code editor that had the
        // cursor gets it back below (see restoreCodeEditor).
        const focusedCodeKey = document.activeElement
            && document.activeElement.dataset
            && document.activeElement.dataset.codeKey || null;
        if (state.tab === 'review') renderReview(main);
        else if (state.tab === 'validation') renderValidation(main);
        else if (state.tab === 'results') renderResults(main);
        else renderSetup(main);
        restoreOpenDetails(main);
        if (focusedCodeKey) restoreCodeEditor(main, focusedCodeKey);
        if (state.readOnly) applyReadOnly(main);
    }

    // Controls that only READ the session and are safe for a visitor: tab
    // switches, expanding stages, opening/downloading outputs. Every other
    // button (generate, test, refine, save, validate, re-run, revise, delete,
    // pickers, key entry, ...) mutates or spends money and is removed.
    const READ_ONLY_SAFE = /toggleOutputPreview|toggleOutputFull|toggleTabBar|downloadOutput|downloadRun|toggleStageCard|scrollToStage|switchTab|openMapFull|closeMapFull|toggleAttributes|toggleEvidence/;

    function applyReadOnly(main) {
        main.querySelectorAll('input, textarea, select').forEach((el) => {
            el.disabled = true;
            el.readOnly = true;
        });
        main.querySelectorAll('button').forEach((el) => {
            const handler = el.getAttribute('onclick') || '';
            const safe = READ_ONLY_SAFE.test(handler) || el.hasAttribute('data-hbs-tab')
                || el.closest('.exp2-output-preview') || el.classList.contains('exp2-output-file');
            if (!safe) el.remove();
        });
        main.querySelectorAll('[contenteditable="true"]').forEach((el) => {
            el.setAttribute('contenteditable', 'false');
        });
    }

    // Header pieces that depend on ownership / sharing state. The header is
    // static markup, so it is refreshed here rather than re-rendered.
    function renderShareControls() {
        const share = byId('handbook-studio-share');
        if (share) {
            const shareable = !!state.session?.id && !state.readOnly && state.session.status !== 'draft';
            share.hidden = state.readOnly;
            share.disabled = !shareable || state.saving;
            share.title = shareable ? 'Anyone with the link can view this session (read-only)'
                : 'Save and run the session first';
            const isPublic = !!state.session?.public;
            share.textContent = isPublic ? 'Public link: on' : 'Share';
            share.classList.toggle('is-public', isPublic);
        }
        const copy = byId('handbook-studio-copy-link');
        if (copy) copy.hidden = state.readOnly || !state.session?.public;
        // Same rule as the picker: a draft has no handbook worth reusing yet.
        const duplicate = byId('handbook-studio-duplicate');
        if (duplicate) {
            duplicate.hidden = state.readOnly || !state.session?.id || state.session.status === 'draft';
            duplicate.disabled = state.running || state.saving;
        }
    }

    function shareUrl() {
        return `${location.origin}/handbook-studio/shared/${encodeURIComponent(state.session.id)}`;
    }

    async function toggleShare() {
        if (!state.session?.id || state.readOnly) return;
        const makePublic = !state.session.public;
        if (!makePublic && !confirm('Turn the public link off? Anyone who has it will no longer be able to open this session.')) return;
        try {
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/share`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ public: makePublic }) });
            state.session.public = data.public;
            state.session.public_since = data.public_since;
            renderShareControls();
            if (data.public) await copyShareLink();
            else toast('Public link turned off.', 'info');
        } catch (error) { toast(error.message, 'error'); }
    }

    async function copyShareLink() {
        const url = shareUrl();
        try {
            await navigator.clipboard.writeText(url);
            toast(`Public link copied: ${url}`, 'success');
        } catch (_) {
            toast(`Public link: ${url}`, 'info');
        }
    }

    // Read-only entry for /handbook-studio/shared/<id>: the session is served
    // to anyone once its owner flagged it public, so no key is needed.
    async function openShared(id) {
        open();
        state.readOnly = true;
        try {
            const data = await api(`/api/handbook-studio-sessions/${encodeURIComponent(id)}`);
            clearRunViewState();
            state.session = data.session;
            state.requiredKeys = [];
            savedState('Read-only');
            switchTab(data.session.status === 'draft' ? 'setup' : 'review');
            watchActiveRun();
        } catch (error) {
            toast(error.message === 'No API key provided'
                ? 'This session is not shared publicly.' : error.message, 'error');
        }
    }

    // Live runs re-render on every heartbeat, which would otherwise collapse
    // any panel the user has expanded to read.
    function restoreOpenDetails(main) {
        main.querySelectorAll('details[data-detail-key]').forEach((element) => {
            const key = element.dataset.detailKey;
            // A panel marked default-open starts open the FIRST time it
            // appears — used for the output of a re-run that produced nothing,
            // where the reason is the whole point. After that first render the
            // reviewer's own toggle wins, so it never springs back open.
            if (element.dataset.detailDefaultOpen === '1' && !state.seenDetails.has(key)) {
                state.seenDetails.add(key);
                state.openDetails.add(key);
            }
            element.open = state.openDetails.has(key);
        });
        if (main.dataset.detailsWired) return;
        main.dataset.detailsWired = '1';
        main.addEventListener('toggle', (event) => {
            const key = event.target?.dataset?.detailKey;
            if (!key) return;
            if (event.target.open) state.openDetails.add(key);
            else state.openDetails.delete(key);
        }, true);
    }

    function readiness() {
        const s = state.session;
        const issues = [];
        if (!String(s.source_name || '').trim()
                && !String(s.documentation_url || '').trim()) {
            issues.push('Enter a data source name or documentation URL.');
        }
        // Retrieval task is optional here — it can be added or edited on
        // Review & test before the handbook is actually tested (which does
        // require one).
        return { ready: issues.length === 0, issues };
    }

    // Renders the reasoning-effort <select> for a model that supports it,
    // or an empty (but still-labeled) placeholder for one that doesn't --
    // keeping the grid slot occupied rather than collapsing so the layout
    // doesn't jump when the model selection changes.
    function reasoningEffortField(id, modelId, currentEffort) {
        const levels = REASONING_EFFORT_MODELS[modelId];
        if (!levels) {
            return `<label class="exp2-field-disabled" id="${id}-wrap"><span>Reasoning effort</span><small class="exp2-field-note">Not supported by this model</small></label>`;
        }
        const effort = levels.includes(currentEffort) ? currentEffort : DEFAULT_REASONING_EFFORT;
        return `<label id="${id}-wrap"><span>Reasoning effort</span>
            <select id="${id}">
                ${levels.map(level => `<option value="${level}" ${effort === level ? 'selected' : ''}>${esc(level.charAt(0).toUpperCase() + level.slice(1))}</option>`).join('')}
            </select>
        </label>`;
    }

    function renderProviderPicker(s) {
        const avail = state.providerAvailability || {};
        const claudeInfo = avail.claude || {};
        const claudeDisabled = claudeInfo.available === false;
        const claudeReason = claudeDisabled
            ? (claudeInfo.reason || 'Claude Agent SDK is not available on this server.')
            : '';
        const savedAnthropicKey = (localStorage.getItem('agm_anthropic_api_key') || '').trim();
        return `
            <div class="exp2-form-grid two">
                <label><span>Provider</span>
                    <select id="hbs-provider">
                        <option value="openai" ${s.provider !== 'claude' ? 'selected' : ''}>OpenAI</option>
                        <option value="claude" ${s.provider === 'claude' ? 'selected' : ''} ${claudeDisabled ? 'disabled' : ''}>Claude Agent SDK${claudeDisabled ? ' (unavailable)' : ''}</option>
                    </select>
                    ${claudeDisabled ? `<small class="exp2-input-help">${esc(claudeReason)}</small>` : ''}
                </label>
                ${s.provider === 'claude' ? `
                <label><span>Anthropic API key</span>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <input type="password" id="hbs-anthropic-key" placeholder="sk-ant-..." value="${esc(savedAnthropicKey)}" autocomplete="off" style="flex:1;">
                        <button type="button" id="hbs-anthropic-key-toggle" title="Show/hide key" style="background:none;border:1px solid #ccc;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:14px;">👁</button>
                    </div>
                    <small class="exp2-input-help" id="hbs-anthropic-key-status">${savedAnthropicKey ? 'Saved on this device — used for every Claude request.' : 'Not saved yet — required to generate with Claude.'}</small>
                </label>` : ''}
            </div>
            ${s.provider === 'claude' ? `
            <div class="exp2-input-help" style="margin:4px 0 12px;">
                Every step (source research, drafting, self-verification, retrieval code generation and debugging) runs as a Claude Agent SDK session instead of a plain chat completion. One trade-off: the per-request HTTP-call log that OpenAI-generated retrieval trials record isn't available for Claude-driven trials — only the files actually produced.
            </div>` : ''}`;
    }

    function renderSetup(main) {
        const s = state.session;
        const ready = readiness();
        main.innerHTML = `
            <div class="exp2-page hbs-setup-page">
                <section class="exp2-page-heading hbs-heading-bare">
                    <span class="exp2-readiness ${ready.ready ? 'ready' : ''}" id="hbs-readiness-badge">${ready.ready ? 'Setup complete' : `${ready.issues.length} setup item${ready.issues.length === 1 ? '' : 's'} remaining`}</span>
                </section>

                <section class="exp2-card">
                    <div class="exp2-card-head"><div><h4>Data source</h4><p>Provide the source name, documentation URL, or both.</p></div></div>
                    <div class="exp2-form-grid two">
                        <label><span>Data source name</span><input id="hbs-source-name" value="${esc(s.source_name)}" placeholder="e.g. OpenTopography"></label>
                        <label><span>Documentation URL</span><input id="hbs-documentation-url" type="url" value="${esc(s.documentation_url)}" placeholder="https://official-documentation.example"></label>
                        <label class="wide"><span>Retrieval task <small class="exp2-field-note">(optional here — can be added on Generate &amp; retrieve before testing)</small></span><textarea id="hbs-retrieval-task" rows="4" placeholder="Describe exactly what data should be retrieved. Can be left blank and added later, before testing.">${esc(s.retrieval_task)}</textarea></label>
                    </div>
                </section>


                <section class="exp2-card hbs-condition${s.use_handbook === false ? ' control' : ''}">
                    <div class="exp2-card-head"><div><h4>Experimental condition</h4>
                    <p>Whether this session uses a generated handbook. Set it once, here — every run in the session then belongs to the same arm.</p></div></div>
                    <div class="exp2-form-grid">
                        <label class="hbs-check wide">
                            <input type="checkbox" id="hbs-no-handbook" ${s.use_handbook === false ? 'checked' : ''}
                                   onchange="HandbookStudioMode.setNoHandbook(this.checked)">
                            <span><strong>Do not use a handbook (control condition)</strong>
                           <!-- <small>The model writes the retrieval code from its own knowledge of the source. Handbook generation is skipped, and no refinement is attempted. Everything else — task, source, model, execution — is identical to a normal run, so the two arms differ by the handbook alone.</small>--></span>
                        </label>
                        ${s.use_handbook === false ? `<p class="exp2-field-note hbs-cred-note">Credentials are collected when they are needed: if the generated code reads an API key, the run pauses and asks for it by name. Nothing needs declaring up front, and credential values are never saved to the session.</p>` : ''}
                    </div>
                </section>

                <section class="exp2-card">
                    <div class="exp2-card-head"><div><h4>Service type</h4><p>Optional-leave blank to let the model choose the best fit.</p></div></div>
                    <div class="exp2-form-grid two">
                        <label><span>Service type</span>
                            <select id="hbs-mechanism">
                                <option value="" ${!s.access_mechanism ? 'selected' : ''}>Auto-select</option>
                                ${Object.entries(MECHANISMS).map(([id, m]) => `<option value="${id}" ${s.access_mechanism === id ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}
                                <option value="${OTHER_MECHANISM_VALUE}" ${isCustomMechanism(s) ? 'selected' : ''}>Other</option>
                            </select>
                            <small class="exp2-input-help">${s.access_mechanism_source === 'auto' && s.access_mechanism ? `Last auto-selected: ${esc(MECHANISMS[s.access_mechanism]?.label || s.access_mechanism)}` : ''}</small>
                        </label>
                        ${isCustomMechanism(s) ? `<label><span>Custom service type</span><input id="hbs-mechanism-other" value="${esc(s.access_mechanism === OTHER_MECHANISM_VALUE ? '' : s.access_mechanism)}" placeholder="e.g. Custom SOAP API"></label>` : ''}
                    </div>
                </section>

                <section class="exp2-card">
                    <div class="exp2-card-head"><div><h4>Provider &amp; models</h4><p>Handbook generation covers drafting, self-verification, refinement, and result evaluation; data retrieval is the model that writes and debugs the code actually run against the source.</p></div></div>
                    ${renderProviderPicker(s)}
                    <div class="exp2-form-grid two">
                        <label><span>Handbook generation model</span>
                            <select id="hbs-handbook-gen-model">
                                ${modelsForProvider(s.provider).list.map(([id, label]) => `<option value="${id}" ${(s.handbook_gen_model || modelsForProvider(s.provider).genDefault) === id ? 'selected' : ''}>${esc(label)}</option>`).join('')}
                            </select>
                        </label>
                        ${s.provider === 'claude'
                            ? '<label class="exp2-field-disabled"><span>Reasoning effort</span><small class="exp2-field-note">Not applicable to the Claude Agent SDK</small></label>'
                            : reasoningEffortField('hbs-handbook-gen-effort', s.handbook_gen_model || DEFAULT_HANDBOOK_GEN_MODEL, s.handbook_gen_reasoning_effort)}
                        <label><span>Data retrieval model</span>
                            <select id="hbs-data-retrieval-model">
                                ${modelsForProvider(s.provider).list.map(([id, label]) => `<option value="${id}" ${(s.data_retrieval_model || modelsForProvider(s.provider).retrievalDefault) === id ? 'selected' : ''}>${esc(label)}</option>`).join('')}
                            </select>
                        </label>
                        ${s.provider === 'claude'
                            ? '<label class="exp2-field-disabled"><span>Reasoning effort</span><small class="exp2-field-note">Not applicable to the Claude Agent SDK</small></label>'
                            : reasoningEffortField('hbs-data-retrieval-effort', s.data_retrieval_model || DEFAULT_DATA_RETRIEVAL_MODEL, s.data_retrieval_reasoning_effort)}
                        <label><span>Execution timeout (seconds)</span>
                            <input id="hbs-execution-timeout" type="number"
                                min="${UNLIMITED_EXECUTION_TIMEOUT}" max="${MAX_EXECUTION_TIMEOUT_SECONDS}" step="30"
                                value="${esc(s.execution_timeout_seconds ?? DEFAULT_EXECUTION_TIMEOUT_SECONDS)}">
                            <small class="exp2-input-help">How long one retrieval attempt is allowed to run before it's treated as hung. Raise this for sources with large bulk downloads [${MIN_EXECUTION_TIMEOUT_SECONDS}-${MAX_EXECUTION_TIMEOUT_SECONDS}s, or 0 for no limit.]</small>
                        </label>
                    </div>
                </section>

                <section class="exp2-readiness-card" id="hbs-readiness-card" ${ready.ready ? 'hidden' : ''}>
                    <h4>Before continuing</h4>
                    <div id="hbs-readiness-issues">${ready.issues.map(issue => `<div><span>○</span>${esc(issue)}</div>`).join('')}</div>
                </section>
                <div class="exp2-page-footer"><button class="exp2-btn primary large" onclick="HandbookStudioMode.saveAndContinue()">Save and continue →</button></div>
            </div>`;
        wireSetupInputs();
    }

    function updateReadinessBanner() {
        const ready = readiness();
        const badge = byId('hbs-readiness-badge');
        if (badge) {
            badge.className = `exp2-readiness ${ready.ready ? 'ready' : ''}`;
            badge.textContent = ready.ready ? 'Setup complete'
                : `${ready.issues.length} setup item${ready.issues.length === 1 ? '' : 's'} remaining`;
        }
        const card = byId('hbs-readiness-card');
        if (card) card.hidden = ready.ready;
        const issuesEl = byId('hbs-readiness-issues');
        if (issuesEl) {
            issuesEl.innerHTML = ready.issues.map(issue => `<div><span>○</span>${esc(issue)}</div>`).join('');
        }
    }

    function wireSetupInputs() {
        const map = {
            'hbs-source-name': 'source_name',
            'hbs-documentation-url': 'documentation_url',
            'hbs-retrieval-task': 'retrieval_task',
        };
        Object.entries(map).forEach(([id, field]) => {
            byId(id)?.addEventListener('input', (event) => {
                state.session[field] = event.target.value; markDirty();
                updateReadinessBanner();
            });
        });
        byId('hbs-mechanism')?.addEventListener('change', (event) => {
            state.session.access_mechanism = event.target.value;
            state.session.access_mechanism_source = event.target.value ? 'user' : '';
            markDirty();
            render(); // show/hide the custom service-type box
        });
        byId('hbs-mechanism-other')?.addEventListener('input', (event) => {
            state.session.access_mechanism = event.target.value;
            state.session.access_mechanism_source = 'user';
            markDirty();
        });
        byId('hbs-provider')?.addEventListener('change', (event) => {
            const provider = event.target.value === 'claude' ? 'claude' : 'openai';
            state.session.provider = provider;
            // An OpenAI model id is meaningless once provider="claude" and
            // vice versa -- reset both model fields to that provider's
            // defaults rather than carrying over a stale, mismatched id.
            const models = modelsForProvider(provider);
            state.session.handbook_gen_model = models.genDefault;
            state.session.data_retrieval_model = models.retrievalDefault;
            state.session.handbook_gen_reasoning_effort = '';
            state.session.data_retrieval_reasoning_effort = '';
            markDirty();
            render();
        });
        // Device-local credential, same as the main Settings key -- saved to
        // localStorage (never to state.session / the server-persisted
        // session), and picked up automatically by script.js's fetch
        // wrapper as the X-Anthropic-Key header on every /api/ request.
        // Deliberately does NOT call render() on input so typing doesn't
        // trigger a full innerHTML rebuild (and cursor jump) on every
        // keystroke -- only the status line next to it is updated directly.
        byId('hbs-anthropic-key')?.addEventListener('input', (event) => {
            const val = event.target.value.trim().replace(/\s/g, '');
            if (val) localStorage.setItem('agm_anthropic_api_key', val);
            else localStorage.removeItem('agm_anthropic_api_key');
            const status = byId('hbs-anthropic-key-status');
            if (status) {
                status.textContent = val
                    ? 'Saved on this device — used for every Claude request.'
                    : 'Not saved yet — required to generate with Claude.';
            }
        });
        byId('hbs-anthropic-key-toggle')?.addEventListener('click', () => {
            const input = byId('hbs-anthropic-key');
            if (input) input.type = input.type === 'text' ? 'password' : 'text';
        });
        byId('hbs-handbook-gen-model')?.addEventListener('change', (event) => {
            state.session.handbook_gen_model = event.target.value; markDirty();
            render(); // re-render so the reasoning-effort field matches the new model
        });
        byId('hbs-data-retrieval-model')?.addEventListener('change', (event) => {
            state.session.data_retrieval_model = event.target.value; markDirty();
            render();
        });
        byId('hbs-handbook-gen-effort')?.addEventListener('change', (event) => {
            state.session.handbook_gen_reasoning_effort = event.target.value; markDirty();
        });
        byId('hbs-data-retrieval-effort')?.addEventListener('change', (event) => {
            state.session.data_retrieval_reasoning_effort = event.target.value; markDirty();
        });
        byId('hbs-execution-timeout')?.addEventListener('change', (event) => {
            const parsed = parseInt(event.target.value, 10);
            const value = Number.isNaN(parsed) ? DEFAULT_EXECUTION_TIMEOUT_SECONDS
                : parsed <= UNLIMITED_EXECUTION_TIMEOUT ? UNLIMITED_EXECUTION_TIMEOUT
                : Math.max(MIN_EXECUTION_TIMEOUT_SECONDS, Math.min(MAX_EXECUTION_TIMEOUT_SECONDS, parsed));
            event.target.value = value;
            state.session.execution_timeout_seconds = value; markDirty();
        });
    }

    function pipelineEmptyState() {
        return `<div class="exp2-pipeline-empty">
            <span>01</span>
            <div><strong>Generation will appear here stage by stage</strong><p>Each stage keeps its messages, artifacts, and status inside its own card.</p></div>
        </div>`;
    }

    function cleanActivityLine(raw) {
        let text = String((raw && typeof raw === 'object') ? raw.text : raw || '').trim()
            .replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '')
            .replace(/^\[(?:chat|stage:\w+:\w+|artifact:\w+)\]\s*/, '');
        if (!text || text.includes('```') || text.includes('[REDACTED]')) return '';
        if (/^[{\["']/.test(text)) return '';
        text = text.replace(/[*`#]/g, '').replace(/\s+/g, ' ').trim();
        return text.length > 260 ? `${text.slice(0, 257)}…` : text;
    }

    function stageActivity(stage) {
        const seen = new Set();
        return (stage.details || []).map(cleanActivityLine).filter((text) => {
            if (!text || seen.has(text)) return false;
            seen.add(text);
            return true;
        });
    }

    function elapsedLabel(stage) {
        const seconds = Number.isFinite(stage.elapsed_seconds)
            ? stage.elapsed_seconds
            : (stage.started_at
                ? Math.round((Date.now() - new Date(stage.started_at).getTime()) / 1000)
                : null);
        if (seconds == null || seconds < 0) return '';
        return seconds >= 60
            ? `${Math.floor(seconds / 60)}m ${seconds % 60}s`
            : `${seconds}s`;
    }

    // Shown when the session record says a run is going but this tab is not
    // the one streaming it -- after a reload, or with the session open in a
    // second tab. Explains why the page is waiting instead of leaving a stage
    // apparently frozen.
    function rejoinBanner() {
        if (state.running) return '';
        const info = activeRunInfo(state.session);
        if (!info) return '';
        if (info.stale) {
            return `<section class="hbs-rejoin stale">
                <strong>A run was interrupted</strong>
                <span>${esc(info.label || info.kind)} stopped reporting — the
                server process was probably restarted. Anything it finished
                before then is saved; run it again to continue.</span>
            </section>`;
        }
        const since = Date.parse(info.started_at || '');
        const mins = Number.isFinite(since)
            ? Math.max(0, Math.round((Date.now() - since) / 60000)) : null;
        return `<section class="hbs-rejoin">
            <strong><i></i>${esc(info.label || info.kind)} is still running</strong>
            <span>This page is not streaming it — the live log is only in the
            tab that started it${mins !== null ? `, going ${mins} min` : ''}. The
            run keeps going on the server and this page updates by itself the
            moment it finishes.</span>
        </section>`;
    }

    function pipelineProgressStrip(stages) {
        if (!stages.length) return rejoinBanner();
        const finished = stages.filter(s => ['complete', 'warning', 'stopped', 'error'].includes(s.status)).length;
        const current = stages.find(s => s.status === 'running');
        return `${rejoinBanner()}<section class="exp2-pipeline-progress">
            <div class="exp2-progress-summary">
                <strong>${current ? `Now: ${esc(current.label)}` : `${finished} of ${stages.length} stages finished`}</strong>
                <small>${current ? `Stage ${stages.indexOf(current) + 1} of ${stages.length} · updates stream onto its card below` : 'Select a stage to jump to its card'}</small>
            </div>
            <div class="exp2-progress-chips">${stages.map((s, i) => `<button type="button" class="${esc(s.status || 'waiting')}" onclick="HandbookStudioMode.scrollToStage('${esc(s.key)}')" title="${esc(s.label || s.key)}"><b>${i + 1}</b><span>${esc(s.label || s.key)}</span></button>`).join('')}</div>
        </section>`;
    }

    // Code from a run still in flight. Deliberately a separate, simpler card
    // than attemptCard: there is no run id yet, so nothing here can be edited,
    // re-run, or reverted -- offering those buttons before the run is saved
    // would point them at a record that does not exist. Once the run lands,
    // the full cards replace these.
    function liveAttemptsSection(stageKey) {
        const live = state.liveAttempts.get(stageKey);
        if (!live || !live.size) return '';
        const attempts = [...live.values()].sort((a, b) => a.attempt - b.attempt);
        const labels = {
            running: 'Running now…', succeeded: 'Succeeded',
            failed: 'Failed', timed_out: 'Timed out',
        };
        return `<details class="hbs-attempts live" data-detail-key="${esc(stageKey)}:live-code" data-detail-default-open="1">
            <summary class="hbs-attempts-head">
                <h5>Retrieval code</h5>
                <span>${attempts.length > 1
                    ? `Attempt ${attempts.length} — the debugger revised the code after each failure`
                    : 'Streaming as the agent runs it'}</span>
            </summary>
            ${attempts.map((item) => {
                const good = item.status === 'succeeded';
                const running = item.status === 'running';
                const code = item.code || '';
                return `<article class="hbs-attempt ${good ? 'ok' : running ? 'live' : 'bad'}">
                    <header>
                        <span class="hbs-attempt-badge">Attempt ${esc(item.attempt)}</span>
                        <strong class="${good ? 'good' : running ? '' : 'bad'}">${esc(labels[item.status] || item.status)}</strong>
                        <span class="hbs-spacer"></span>
                        <span class="hbs-attempt-lines">${code ? code.split('\n').length : 0} lines</span>
                    </header>
                    ${item.note ? `<p class="hbs-attempt-note">${esc(item.note)}</p>` : ''}
                    <pre class="hbs-attempt-code"><code class="language-python">${highlightPython(code)}</code></pre>
                    ${item.error ? `<details class="hbs-attempt-error" data-detail-key="${esc(stageKey)}:live-err-${esc(item.attempt)}"><summary>Error</summary><pre>${esc(item.error)}</pre></details>` : ''}
                </article>`;
            }).join('')}
        </details>`;
    }

    function pipelineStageCard(stage, index, extraBody = '') {
        const status = stage.status || 'waiting';
        const collapsed = state.collapsedStages.has(stage.key);
        const activity = stageActivity(stage);
        let activityBlock = '';
        if (status === 'running') {
            const hint = cleanActivityLine(stage.hint);
            const lines = activity.length ? activity : (hint ? [hint] : []);
            const elapsed = elapsedLabel(stage);
            activityBlock = `<div class="exp2-stage-current-activity" role="status" aria-live="polite">
                <span><i></i><b data-live-label>Working now${elapsed ? ` · ${esc(elapsed)} elapsed` : ''}</b></span>
                <div class="exp2-live-log" data-live-log>${lines.length
                    ? lines.map(text => `<p data-live-line>${esc(text)}</p>`).join('')
                    : '<p data-live-line class="exp2-field-note">This stage is in progress. Live updates will appear here as they arrive.</p>'}</div>
                ${hint && hint !== lines[lines.length - 1] ? `<em class="exp2-stage-hint" data-live-hint>${esc(hint)}</em>` : ''}
            </div>${liveAttemptsSection(stage.key)}`;
        } else if (activity.length) {
            // stage.showActivity (set by mergeDraftAndVerification) opens the
            // log on first render so the reviewer sees what the verification
            // did without hunting for the toggle; their own toggle wins after.
            activityBlock = `<details class="exp2-stage-log" data-detail-key="${esc(stage.key)}:log"${stage.showActivity ? ' data-detail-default-open="1"' : ''}><summary>Stage activity · ${activity.length} update${activity.length === 1 ? '' : 's'}</summary><ul>${activity.map(text => `<li>${esc(text)}</li>`).join('')}</ul></details>`;
        }
        return `<article class="exp2-pipeline-stage ${esc(status)} ${collapsed ? 'collapsed' : ''}" data-stage-key="${esc(stage.key)}" data-stage-keys="${esc((stage.keys || [stage.key]).join(' '))}">
            <div class="exp2-pipeline-step"><span>${String(index + 1).padStart(2, '0')}</span><i></i></div>
            <div class="exp2-pipeline-card">
                <header>
                    <div><strong>${esc(stage.label || stage.key)}</strong><p>${esc(stage.description || '')}</p></div>
                    <div class="exp2-stage-header-actions">
                        <span class="exp2-stage-status">${esc(status.replaceAll('_', ' '))}</span>
                        <button type="button" class="exp2-stage-collapse" aria-expanded="${collapsed ? 'false' : 'true'}" onclick="HandbookStudioMode.toggleStageCard('${esc(stage.key)}')">
                            <span>${collapsed ? 'Expand' : 'Collapse'}</span><i aria-hidden="true">⌃</i>
                        </button>
                    </div>
                </header>
                <div class="exp2-stage-card-body" ${collapsed ? 'hidden' : ''}>
                    ${stage.message ? `<div class="exp2-stage-message">${esc(stage.message)}</div>` : ''}
                    ${activityBlock}
                    ${pipelineArtifact(stage.artifact, stage.key)}
                    ${extraBody}
                    ${stage.started_at ? `<footer><span>Started ${esc(new Date(stage.started_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}))}</span>${stage.finished_at ? `<span>Finished ${esc(new Date(stage.finished_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}))}</span>` : '<span>Working now…</span>'}</footer>` : ''}
                </div>
            </div>
        </article>`;
    }

    function toggleStageCard(key) {
        const collapsed = !state.collapsedStages.has(key);
        if (collapsed) state.collapsedStages.add(key);
        else state.collapsedStages.delete(key);
        const article = document.querySelector(
            `.exp2-pipeline-stage[data-stage-key="${CSS.escape(key)}"]`);
        if (!article) return;
        article.classList.toggle('collapsed', collapsed);
        const body = article.querySelector('.exp2-stage-card-body');
        if (body) body.hidden = collapsed;
        const button = article.querySelector('.exp2-stage-collapse');
        if (button) {
            button.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            const label = button.querySelector('span');
            if (label) label.textContent = collapsed ? 'Expand' : 'Collapse';
        }
    }

    function stageArticle(key) {
        const k = CSS.escape(key);
        return document.querySelector(
            `.exp2-pipeline-stage[data-stage-key="${k}"], .exp2-pipeline-stage[data-stage-keys~="${k}"]`);
    }

    function updateLiveStageDom(stage) {
        if (!stage || stage.status !== 'running') return;
        const article = stageArticle(stage.key);
        if (!article) return;
        const activity = stageActivity(stage);
        const hint = cleanActivityLine(stage.hint);
        const elapsed = elapsedLabel(stage);
        const label = article.querySelector('[data-live-label]');
        if (label) label.textContent = `Working now${elapsed ? ` · ${elapsed} elapsed` : ''}`;

        // Append only the NEW lines to the live log rather than replacing a
        // single message — the log should stream line-by-line, not
        // overwrite itself on every heartbeat/progress event.
        const log = article.querySelector('[data-live-log]');
        if (log) {
            const existing = log.querySelectorAll('[data-live-line]:not(.exp2-field-note)').length;
            if (activity.length > existing) {
                const placeholder = log.querySelector('.exp2-field-note');
                if (placeholder) placeholder.remove();
                const wasAtBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
                for (const text of activity.slice(existing)) {
                    const p = document.createElement('p');
                    p.dataset.liveLine = '';
                    p.textContent = text;
                    log.appendChild(p);
                }
                if (wasAtBottom) log.scrollTop = log.scrollHeight;
            }
        }

        const hintEl = article.querySelector('[data-live-hint]');
        const latestLine = activity[activity.length - 1] || '';
        if (hint && hint !== latestLine) {
            if (hintEl) hintEl.textContent = hint;
        } else if (hintEl) {
            hintEl.remove();
        }
    }

    function scrollToStage(key) {
        stageArticle(key)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function outputPreviewKey(scope, index) {
        return `${scope}:output:${index}`;
    }

    function outputRequestUrl(file, mode = '') {
        const params = new URLSearchParams();
        if (file?.artifact_ref) params.set('ref', file.artifact_ref);
        else {
            params.set('name', file?.name || '');
            params.set('size', String(file?.size_bytes ?? -1));
        }
        if (mode) params.set(mode, '1');
        return `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/output-preview?${params}`;
    }

    // One preview at a time fills the window; the map frame reloads at the
    // new size through the usual previewMapReady path.
    function toggleOutputFull(scope, index) {
        const key = outputPreviewKey(scope, index);
        state.fullPreview = state.fullPreview === key ? null : key;
        render();
    }

    function syncTabBar() {
        byId('handbook-studio-workspace')?.classList.toggle('hbs-bar-collapsed', state.tabBarCollapsed);
        const button = byId('hbs-tabbar-toggle');
        if (button) {
            button.setAttribute('aria-expanded', String(!state.tabBarCollapsed));
            button.title = state.tabBarCollapsed ? 'Expand the section bar' : 'Collapse the section bar';
        }
    }

    function toggleTabBar() {
        state.tabBarCollapsed = !state.tabBarCollapsed;
        try { localStorage.setItem('hbs.tabBarCollapsed', state.tabBarCollapsed ? '1' : '0'); } catch (_) { /* per-browser nicety only */ }
        syncTabBar();
    }

    // Drag handle under a data table. The height lives in custom properties on
    // the workspace (which render() never rebuilds), so it survives re-renders;
    // full-window and in-card tables keep separate heights.
    const RESIZE_GRIP = '<div class="hbs-resize-grip" role="separator" aria-orientation="horizontal" tabindex="0" title="Drag to resize · double-click to reset"></div>';
    const TABLE_MIN_HEIGHT = 120;

    function setTableHeight(grip, height) {
        const workspace = byId('handbook-studio-workspace');
        if (!workspace) return;
        const prop = grip.closest('.hbs-preview-full') ? '--hbs-tbl-h-full' : '--hbs-tbl-h-inline';
        if (height == null) workspace.style.removeProperty(prop);
        else workspace.style.setProperty(prop, `${Math.max(TABLE_MIN_HEIGHT, Math.round(height))}px`);
    }

    document.addEventListener('pointerdown', (event) => {
        const grip = event.target.closest?.('.hbs-resize-grip');
        const table = grip?.previousElementSibling;
        if (!table) return;
        event.preventDefault();
        const startY = event.clientY;
        const startHeight = table.getBoundingClientRect().height;
        const move = (e) => setTableHeight(grip, startHeight + e.clientY - startY);
        const stop = () => {
            document.removeEventListener('pointermove', move);
            document.removeEventListener('pointerup', stop);
            document.body.classList.remove('hbs-resizing');
        };
        document.addEventListener('pointermove', move);
        document.addEventListener('pointerup', stop);
        // Map iframes would otherwise swallow the pointer mid-drag.
        document.body.classList.add('hbs-resizing');
    });
    document.addEventListener('dblclick', (event) => {
        const grip = event.target.closest?.('.hbs-resize-grip');
        if (grip) setTableHeight(grip, null);
    });

    document.addEventListener('keydown', (event) => {
        const grip = event.target.closest?.('.hbs-resize-grip');
        if (grip && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
            event.preventDefault();
            const table = grip.previousElementSibling;
            if (table) setTableHeight(grip, table.getBoundingClientRect().height
                + (event.key === 'ArrowDown' ? 40 : -40));
            return;
        }
        if (event.key === 'Escape' && state.open && state.fullPreview) {
            state.fullPreview = null;
            render();
        }
    });

    function outputPreviewPanel(scope, index) {
        const record = state.outputPreviews.get(outputPreviewKey(scope, index));
        if (!record?.open) return '';
        if (record.loading) {
            return `<div class="exp2-output-preview loading"><span class="exp2-output-preview-spinner"></span><p>Preparing a safe in-card preview…</p></div>`;
        }
        if (record.error) {
            return `<div class="exp2-output-preview error"><strong>Preview unavailable</strong><p>${esc(record.error)}</p></div>`;
        }
        const preview = record.preview || {};
        let content = '';
        if (preview.kind === 'geojson') {
            // Geometry belongs on a map, not in a text box. The frame is the
            // same Mapbox viewer the analysis workspace uses, so basemaps,
            // zoom and the layer legend all behave the way they do elsewhere.
            const n = preview.feature_count;
            content = `<div class="hbs-inline-map" id="map-${esc(scope)}-${index}">
                    <iframe src="map.html" title="Map of ${esc(preview.name || 'retrieved data')}"
                        onload="HandbookStudioMode.previewMapReady('${esc(scope)}', ${index})"></iframe>
                </div>
                <p class="exp2-field-note">${n != null ? `${n} feature${n === 1 ? '' : 's'}` : 'Geometry'}${preview.truncated ? ` — first ${(preview.data?.features || []).length} drawn` : ''}.
                Drag to pan, scroll to zoom.</p>
                ${outputAttributesSection(scope, index)}`;
        } else if (preview.kind === 'raster') {
            // A raster with no bbox cannot be put anywhere on a map. Rendering
            // the frame anyway drew an empty basemap, which reads as "the file
            // is empty" when the pixels are usually fine and only the location
            // is missing -- so the reason takes the map's place.
            const placeable = Boolean(preview.bbox);
            content = `${placeable ? `<div class="hbs-inline-map" id="map-${esc(scope)}-${index}">
                    <iframe src="map.html" title="Footprint of ${esc(preview.name || 'raster')}"
                        onload="HandbookStudioMode.previewMapReady('${esc(scope)}', ${index})"></iframe>
                </div>` : `<div class="exp2-notice neutral"><strong>Not placeable on a map</strong><p>${esc(preview.reason || 'This raster carries no CRS, so where it belongs is unknown.')}</p></div>`}
                <div class="hbs-stats">
                    <div><span>Size</span><b>${esc(preview.width)}&times;${esc(preview.height)}</b></div>
                    <div><span>Bands</span><b>${esc(preview.bands)}</b></div>
                    <div><span>Pixel type</span><b>${esc(preview.dtype)}</b></div>
                    <div><span>CRS</span><b>${esc(preview.crs || 'none')}</b></div>
                </div>
                <p class="exp2-field-note">${!placeable
                    ? 'The pixels themselves are readable \u2014 "Show raster attributes" below reports their statistics.'
                    : preview.has_pixels
                    ? 'Imagery drawn above, contrast-stretched (2\u201398th percentile) for display only. Drag to pan, scroll to zoom.'
                    : 'Footprint drawn above \u2014 this raster could not be rendered, so only its extent is shown.'}</p>
                ${outputAttributesSection(scope, index)}`;
        } else if (preview.kind === 'csv') {
            content = hbsCsvTable(preview.content, preview.delimiter || ',')
                + (preview.truncated ? '<small>Preview truncated to the first 300 KB.</small>' : '');
        } else if (preview.kind === 'text') {
            content = `<pre><code>${esc(preview.content || '')}</code></pre>${preview.truncated ? '<small>Preview truncated to the first 300 KB.</small>' : ''}`;
        } else if (preview.kind === 'image' && preview.object_url) {
            content = `<img src="${esc(preview.object_url)}" alt="Preview of ${esc(preview.name || 'output file')}">`;
        } else if (preview.kind === 'pdf' && preview.object_url) {
            content = `<iframe src="${esc(preview.object_url)}" title="Preview of ${esc(preview.name || 'PDF output')}"></iframe>`;
        } else {
            content = `<div class="exp2-output-binary"><strong>Structured preview is not available for this file type.</strong><p>You can still download and inspect the complete output.</p></div>`;
        }
        const full = state.fullPreview === outputPreviewKey(scope, index);
        return `<div class="exp2-output-preview${full ? ' hbs-preview-full' : ''}" ${full ? 'role="dialog" aria-modal="true"' : ''}>
            <header><div><strong>${esc(preview.name || 'Output')}</strong><small>${esc(preview.mime_type || 'Unknown file type')} · ${esc(preview.size_bytes ?? 0)} bytes</small></div><span class="hbs-preview-actions"><button type="button" class="hbs-preview-full-btn" onclick="HandbookStudioMode.toggleOutputFull('${esc(scope)}', ${index})" title="${full ? 'Exit full screen (Esc)' : 'View full screen'}">${full ? '✕ Exit full screen' : '⛶ Full screen'}</button><button type="button" data-ref="${esc(preview.artifact_ref || '')}" data-name="${esc(preview.name || 'output')}" onclick="HandbookStudioMode.downloadOutput(this)">Download</button></span></header>
            ${content}
        </div>`;
    }

    // Pipeline stages are keyed "retrieval_test_<n>", which is the same <n> as
    // the test run's test_number -- that is how a stage card in Review and test
    // finds the run whose code it can re-run. Any other stage (generation,
    // trace analysis, refinement) has no run behind it and gets no button.
    function runIdForStage(scope) {
        const match = /^retrieval_test_(\d+)$/.exec(String(scope || ''));
        if (!match) return null;
        const number = Number(match[1]);
        const run = (state.session.test_runs || []).find(
            (candidate) => Number(candidate.test_number) === number);
        return run ? run.id : null;
    }

    // What a re-run produced, shown directly under the code that produced it:
    // the outcome line, whatever the process wrote, and the files themselves,
    // clickable straight onto the map or into a table like any other output.
    function rerunOutcome(record, scope) {
        if (record.running) {
            return `<div class="hbs-rerun-out running">
                <span class="hbs-rerun-spinner" aria-hidden="true"></span>
                <p>Running this code again…</p></div>`;
        }
        if (record.error) {
            return `<div class="hbs-rerun-out failed">
                <p class="hbs-rerun-line bad"><strong>Could not re-run</strong> ${esc(record.error)}</p></div>`;
        }
        const files = record.files || [];
        const exited = record.returncode === 0 && !record.timed_out;
        // Exiting 0 having written nothing is the common shape of a real
        // failure here: retrieval code usually catches a dead endpoint, prints
        // why, and returns normally. Reporting that as "completed" hides the
        // one thing the reviewer needs, so an empty run is called out and its
        // console output is opened rather than tucked behind a summary.
        const empty = exited && !files.length;
        const state = record.timed_out || !exited ? 'failed' : empty ? 'warned' : 'passed';
        // A stored outcome is being read back from the session, not produced a
        // moment ago -- saying "freshly fetched just now" about last week's
        // files would misrepresent the evidence.
        const heading = record.from_original_run ? 'This round\u2019s output'
            : record.stored ? 'Stored re-run' : 'Re-run';
        const when = record.stored && record.at
            ? ` \u00b7 ${esc(new Date(record.at).toLocaleString())}` : '';
        const label = record.timed_out ? 'timed out'
            : !exited ? `exited with code ${record.returncode}`
            : empty ? 'ran, but produced no files'
            : 'completed';
        return `<div class="hbs-rerun-out ${state}">
            <p class="hbs-rerun-line ${state === 'passed' ? '' : 'bad'}">
                <strong>${heading} ${esc(record.from_original_run ? '' : label)}</strong>${when}
                <span>${record.elapsed_seconds != null ? esc(record.elapsed_seconds) + 's' : ''}</span>
                <span>${files.length} file${files.length === 1 ? '' : 's'}</span>
                <span>${(record.http_requests || []).length} request${(record.http_requests || []).length === 1 ? '' : 's'}</span>
            </p>
            ${empty ? `<p class="hbs-rerun-why">The code ran to completion without writing anything. Retrieval code
                usually catches a failed download and exits normally, so the reason is in the output below —
                commonly a source URL that has moved or now needs a key.</p>` : ''}
            ${files.length ? `<p class="hbs-rerun-caveat">${record.from_original_run
                ? 'The files this round recorded.'
                : record.stored
                ? 'Kept from when this version was last run; the files are on disk, not re-fetched now.'
                : 'Freshly fetched just now — evidence about the code, not a replacement for what the original run returned.'}</p>
            <div class="exp2-output-files">${
                files.map((file, index) => outputFileControl(file, `${scope}`, index)).join('')
            }</div>${files.map((file, index) => outputPreviewPanel(`${scope}`, index)).join('')}` : ''}
            ${record.stderr ? `<details class="hbs-attempt-error" ${state !== 'passed' ? 'data-detail-default-open="1"' : ''} data-detail-key="${esc(scope)}:stderr"><summary>Error output</summary><pre>${esc(record.stderr)}</pre></details>` : ''}
            ${record.stdout ? `<details ${empty ? 'data-detail-default-open="1"' : ''} data-detail-key="${esc(scope)}:stdout"><summary>Console output</summary><pre>${esc(record.stdout)}</pre></details>` : ''}
            ${!record.stdout && !record.stderr && empty ? '<p class="hbs-rerun-caveat">The process printed nothing at all.</p>' : ''}
        </div>`;
    }

    // The files the ORIGINAL run recorded. Re-run output is shown under its own
    // attempt instead of replacing this, so the two are never confused.
    // What a quota-limited host promises about this run's files. Nothing is
    // shown on an unlimited (desktop) install, where retention is off.
    function retentionNote(run) {
        const info = run?.retention;
        if (!info || !info.policy_enabled) return '';
        const until = info.bulk_kept_until ? new Date(info.bulk_kept_until) : null;
        const untilText = until && !Number.isNaN(until.getTime())
            ? until.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '';
        const smallMb = Math.round((info.keep_small_bytes || 0) / 1048576) || 1;
        if (info.over_cap) {
            return `<p class="hbs-retention-note warn"><strong>Large output (${hbsBytes(info.run_size_bytes)}).</strong>
                Files over ${smallMb} MB are kept on this server only until <b>${esc(untilText)}</b> — download them before then.
                Small files (handbook, manifests, samples) are kept permanently.</p>`;
        }
        return `<p class="hbs-retention-note">Files over ${smallMb} MB are kept on this server until ${esc(untilText)};
            small files are kept permanently. Re-running the session regenerates anything removed.</p>`;
    }

    function outputFileSection(result, scope, runId, supersededBy, run) {
        const files = result.downloaded_files || [];
        if (!files.length) return '';
        const downloadAll = runId
            ? `<button type="button" class="hbs-download-all" data-run="${esc(runId)}" onclick="HandbookStudioMode.downloadRun(this)" title="Everything this run produced, as one zip">⤓ Download all (zip)</button>`
            : '';
        const body = `${retentionNote(run)}<div class="exp2-output-files">${
            files.map((file, index) => outputFileControl(file, scope, index)).join('')
        }</div>${files.map((file, index) => outputPreviewPanel(scope, index)).join('')}`;
        // Once a newer version of the code has been run, these files are the
        // OLD result. Leaving them open underneath the new one read as if the
        // round had produced both, so they fold away into a labelled summary
        // that still says exactly what they are and opens on a click.
        if (supersededBy) {
            return `<details class="hbs-original-outputs" data-detail-key="${esc(scope)}:original-outputs">
                <summary>Data outputs from the original run \u00b7 ${files.length} file${files.length === 1 ? '' : 's'}
                    <em>\u2014 superseded by ${esc(supersededBy)}</em></summary>
                ${body}</details>`;
        }
        return `<div class="hbs-outputs-head"><h5>Data outputs</h5><span>${files.length} file${files.length === 1 ? '' : 's'} from this round</span>${downloadAll}</div>
        ${body}`;
    }

    async function downloadRun(button) {
        const runId = button?.dataset?.run || '';
        if (!runId || !state.session?.id) return;
        button.disabled = true;
        try {
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/test-runs/${encodeURIComponent(runId)}/download-all`);
            if (!response.ok) {
                let message = `Download failed (${response.status}).`;
                try { message = (await response.json()).error || message; } catch (_) { /* not JSON */ }
                throw new Error(message);
            }
            const disposition = response.headers.get('Content-Disposition') || '';
            const match = /filename="([^"]+)"/.exec(disposition);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = url;
            link.download = match ? match[1] : 'handbook-studio-run.zip';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (error) {
            toast(error.message, 'error');
        } finally {
            button.disabled = false;
        }
    }

    // The newest version across this round's code cards that has actually been
    // re-run -- what the round's original files have been superseded BY.
    function newerRunLabel(result) {
        const slots = [result].concat(((result.execution || {}).attempts) || []);
        let newest = null;
        slots.forEach((slot) => {
            (slot.code_versions || []).forEach((version) => {
                if (!version.run || version.run.from_original_run) return;
                if (!newest || String(version.run.at || '') > String(newest.run.at || '')) {
                    newest = version;
                }
            });
        });
        return newest ? `${newest.label}${newest.run.at
            ? ` (re-run ${new Date(newest.run.at).toLocaleString()})` : ''}` : null;
    }

    // attemptIndex null re-runs the round's final code; a number re-runs that
    // specific attempt, into its own folder so results never mix.
    async function rerunCode(runId, attemptIndex) {
        if (state.running) return;
        const key = rerunKey(runId, attemptIndex ?? null);
        const prior = state.rerunResults.get(key);
        if (prior && prior.running) return;
        state.rerunResults.set(key, { running: true });
        // Previews are keyed per scope and point at paths this may rewrite.
        state.outputPreviews.clear();
        render();
        try {
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                + `/test-runs/${encodeURIComponent(runId)}/rerun-code`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(
                      attemptIndex === null || attemptIndex === undefined
                          ? {} : { attempt: attemptIndex }) });
            state.rerunResults.set(key, {
                running: false, returncode: data.returncode,
                timed_out: data.timed_out, elapsed_seconds: data.elapsed_seconds,
                files: data.files || [], stderr: (data.stderr || '').slice(-4000),
                stdout: (data.stdout || '').slice(-4000),
                http_requests: data.http_requests || [],
            });
            // The outcome is stored against the version that ran, so the
            // session is re-read: what the card shows is then what survives a
            // refresh, not a copy that only exists in this tab.
            try {
                const latest = await api(
                    `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
                if (latest && latest.session) state.session = latest.session;
            } catch (_) { /* keep what is on screen */ }
            const n = (data.files || []).length;
            const clean = data.returncode === 0 && !data.timed_out;
            toast(!clean ? `Re-run exited with code ${data.returncode}.`
                    : n ? `Re-run finished — ${n} file${n === 1 ? '' : 's'}.`
                    : 'Re-run finished but produced no files — see the output.',
                  clean && n ? 'success' : 'info');
        } catch (error) {
            state.rerunResults.set(key, { running: false, error: error.message, files: [] });
            toast(error.message, 'error');
        }
        render();
    }

    // Extensions whose preview opens a map rather than a text box. Shown on the
    // chip so a reviewer knows what clicking will give them.
    const HBS_MAP_EXT = /\.(geojson|gpkg|shp|tif|tiff)$/i;
    const HBS_TABLE_EXT = /\.(csv|tsv)$/i;

    function outputFileControl(file, scope, index) {
        const record = state.outputPreviews.get(outputPreviewKey(scope, index));
        const name = file.name || 'Output file';
        const opens = HBS_MAP_EXT.test(name) ? 'map'
            : HBS_TABLE_EXT.test(name) ? 'table' : 'preview';
        const icon = opens === 'map' ? '◎' : opens === 'table' ? '▦' : '▤';
        const hint = record?.open ? 'Hide'
            : opens === 'map' ? 'Click to open on a map'
            : opens === 'table' ? 'Click to open as a table'
            : 'Click to preview';
        return `<button type="button" class="exp2-output-file ${record?.open ? 'active' : ''} opens-${opens}" data-ref="${esc(file.artifact_ref || '')}" data-name="${esc(name)}" data-size="${esc(file.size_bytes ?? 0)}" onclick="HandbookStudioMode.toggleOutputPreview('${esc(scope)}', ${index}, this)">
            <span aria-hidden="true">${icon}</span><span><b>${esc(name)}</b><small>${hbsBytes(file.size_bytes)} · ${hint}</small></span>
        </button>`;
    }

    // The rendered PNG for a raster, as a data URL the map document can use.
    //
    // Fetched HERE, in the parent, rather than by handing the map an /api/
    // URL: the API key is attached by script.js's window.fetch wrapper, which
    // only exists in this document. map.html is a separate document with an
    // untouched fetch, so an image source pointed at the endpoint is requested
    // with no X-API-Key header, comes back 401, and draws nothing -- a bare
    // basemap with no way to tell it apart from a raster that simply failed.
    // A data URL also matches what the GeoTIFF path already feeds Mapbox, so
    // the basemap-switch restore path handles it unchanged.
    async function rasterPixelsDataUrl(record) {
        if (record.pixels_data_url) return record.pixels_data_url;
        const response = await fetch(outputRequestUrl(record.file, 'pixels'));
        if (!response.ok) throw new Error(`Raster render failed (${response.status}).`);
        const blob = await response.blob();
        record.pixels_data_url = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error('Could not read the rendered raster.'));
            reader.readAsDataURL(blob);
        });
        return record.pixels_data_url;
    }

    // Rendered after the iframe loads: the map is a separate document, so the
    // layer has to be posted in rather than written into the markup.
    async function previewMapReady(scope, index) {
        const record = state.outputPreviews.get(outputPreviewKey(scope, index));
        const preview = record && record.preview;
        if (!preview) return;
        // Re-read on every post: an await below can outlive the iframe this
        // was called for, because render() rebuilds the card wholesale.
        const post = (layer) => {
            const frame = byId(`map-${scope}-${index}`)?.querySelector('iframe');
            if (frame && frame.contentWindow) {
                frame.contentWindow.postMessage({ type: 'ADD_LAYER', layer }, '*');
            }
        };
        if (preview.kind === 'geojson' && preview.data) {
            post({ type: 'geojson', id: `out-${index}`,
                   name: preview.name || 'Retrieved data', geoJSON: preview.data });
            return;
        }
        if (preview.kind !== 'raster' || !preview.bbox) return;
        const b = preview.bbox;
        // Draw the actual pixels when the server can render them. The
        // footprint rectangle below is the fallback for a raster with no
        // CRS or an unsupported layout -- it says where the data lies but
        // shows nothing of what the data IS, which is useless for judging
        // whether the right scene came back.
        if (preview.has_pixels && record.file) {
            try {
                const url = await rasterPixelsDataUrl(record);
                post({
                    type: 'image', id: `out-${index}`,
                    name: preview.name || 'Raster',
                    url,
                    // Mapbox image-source corner order: TL, TR, BR, BL. The
                    // server warps to EPSG:4326 first, so these corners and the
                    // pixels agree; sending UTM pixels here would misplace them.
                    coordinates: [
                        [b.west, b.north], [b.east, b.north],
                        [b.east, b.south], [b.west, b.south],
                    ],
                });
                return;
            } catch (error) {
                // Rendering can fail server-side (422) for a raster rasterio
                // cannot warp. Fall through to the footprint rather than
                // leaving an empty basemap, and say why in the console.
                console.warn('Raster pixels unavailable, drawing footprint:', error);
                preview.has_pixels = false;
                render();
            }
        }
        post({
            type: 'geojson', id: `out-${index}`,
            name: `${preview.name || 'Raster'} — footprint`,
            geoJSON: { type: 'FeatureCollection', features: [{
                type: 'Feature', properties: { name: preview.name },
                geometry: { type: 'Polygon', coordinates: [[
                    [b.west, b.south], [b.east, b.south],
                    [b.east, b.north], [b.west, b.north], [b.west, b.south],
                ]] } }] },
        });
    }

    // Seconds, because that is the unit the pipeline measures in and the one
    // that compares across runs. A long run also gets a m/s reading beside it,
    // since "1,284.3s" is precise and unreadable at a glance.
    function secs(value) {
        const n = Number(value) || 0;
        return `${n >= 100 ? Math.round(n) : n.toFixed(1)}s`;
    }

    function clock(value) {
        const n = Math.round(Number(value) || 0);
        const h = Math.floor(n / 3600);
        const m = Math.floor((n % 3600) / 60);
        const s = n % 60;
        return h ? `${h}h ${m}m ${s}s` : `${m}m ${s}s`;
    }

    function hbsBytes(n) {
        if (n == null) return 'unknown size';
        if (n < 1024) return n + ' bytes';
        if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
        return (n / 1048576).toFixed(1) + ' MB';
    }

    function hbsCsvTable(text, delimiter) {
        const sep = delimiter || ',';
        const lines = String(text || '').replace(/\s+$/, '').split(/\r?\n/);
        if (!lines.length || !lines[0]) return '<p class="exp2-field-note">Empty file.</p>';
        const rows = lines.slice(0, 201).map((l) => l.split(sep));
        const dataRows = rows.length - 1;
        return `<div class="hbs-tblwrap hbs-tblwrap-resizable"><table class="hbs-table">
            <thead><tr>${rows[0].map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>
            <tbody>${dataRows > 0
                ? rows.slice(1).map((cells) => `<tr>${cells.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')
                : `<tr><td colspan="${rows[0].length}" class="hbs-norows">Header only — this file contains no data rows.</td></tr>`}
            </tbody></table></div>${RESIZE_GRIP}
            ${lines.length > 201 ? `<p class="exp2-field-note">Showing the first 200 of ${lines.length - 1} rows.</p>` : ''}`;
    }

    // "Show attribute table" under the preview's map: the data the map is
    // drawing, read as a table. Loaded on demand and kept on the record, so a
    // large output pays for it only when a reviewer actually asks.
    function outputAttributesSection(scope, index) {
        const record = state.outputPreviews.get(outputPreviewKey(scope, index));
        if (!record) return '';
        const isRaster = (record.preview || {}).kind === 'raster';
        const label = record.attributesLoading ? 'Reading\u2026'
            : isRaster
                ? (record.attributesOpen ? 'Hide raster attributes' : 'Show raster attributes')
                : (record.attributesOpen ? 'Hide attribute table' : 'Show attribute table');
        const button = `<button type="button" class="exp2-btn tiny secondary hbs-attr-toggle"
            ${record.attributesLoading ? 'disabled' : ''}
            onclick="HandbookStudioMode.toggleOutputAttributes('${esc(scope)}', ${index})">${label}</button>`;
        if (!record.attributesOpen) return button;
        if (record.attributesLoading) {
            return `${button}<p class="exp2-field-note">Reading the attributes\u2026</p>`;
        }
        if (record.attributesError) {
            return `${button}<p class="exp2-field-note">Attributes unavailable — ${esc(record.attributesError)}</p>`;
        }
        const data = record.attributes;
        if (!data) return button;
        return button + (data.kind === 'raster'
            ? hbsRasterAttributes(data) : hbsAttributeTable(data));
    }

    // Vector or tabular output: one row per feature, geometry left out.
    function hbsAttributeTable(data) {
        const columns = data.columns || [];
        const rows = data.rows || [];
        if (!columns.length) {
            return '<p class="exp2-field-note">This file carries no attribute columns.</p>';
        }
        const total = data.total_rows ?? rows.length;
        return `<div class="hbs-tblwrap hbs-tblwrap-resizable"><table class="hbs-table">
            <thead><tr><th>#</th>${columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead>
            <tbody>${rows.length
                ? rows.map((row, i) => `<tr><td>${i + 1}</td>${columns.map(
                    (c) => `<td>${esc(row[c] == null ? '' : row[c])}</td>`).join('')}</tr>`).join('')
                : `<tr><td colspan="${columns.length + 1}" class="hbs-norows">No rows — this file has columns but no records.</td></tr>`}
            </tbody></table></div>${RESIZE_GRIP}
            <p class="exp2-field-note">${total} row${total === 1 ? '' : 's'}${
                total > rows.length ? ` — first ${rows.length} shown` : ''} · ${
                columns.length} column${columns.length === 1 ? '' : 's'}.</p>`;
    }

    // A raster has no feature table, so this is what stands in for one:
    // properties, per-band statistics, and the value counts of a classified
    // raster (which IS its attribute table, in everything but name).
    function hbsRasterAttributes(data) {
        const p = data.properties || {};
        const bands = data.bands || [];
        const cats = data.categories;
        const num = (v, digits) => {
            if (v === null || v === undefined || v === '') return '\u2014';
            if (typeof v === 'number') {
                return Number.isInteger(v) ? v.toLocaleString()
                    : String(Number(v.toPrecision(digits || 6)));
            }
            return String(v);
        };
        const b = p.wgs84_bounds;
        // Size, bands, pixel type and CRS are deliberately absent: the raster
        // preview prints those four immediately above this, and repeating
        // them here reads as a rendering mistake rather than as detail.
        const facts = [
            ['File', p.file_name], ['Format', p.driver],
            ['NoData', p.nodata],
            ['Pixel size', p.pixel_size_x != null ? `${num(p.pixel_size_x)} × ${num(p.pixel_size_y)}` : null],
            ['Pixels', p.total_pixels],
        ].filter((f) => f[1] !== null && f[1] !== undefined && f[1] !== '');
        if (b) {
            facts.push(['Extent (WGS84)',
                `W ${num(b.west)}  S ${num(b.south)}  E ${num(b.east)}  N ${num(b.north)}`]);
        }

        const bandRows = bands.map((band) => `<tr>
            <td>${esc(band.band)}</td><td>${esc(band.description || band.color_interp || '\u2014')}</td>
            <td>${esc(band.dtype || '\u2014')}</td><td>${num(band.nodata)}</td>
            <td>${num(band.min)}</td><td>${num(band.max)}</td>
            <td>${num(band.mean)}</td><td>${num(band.std)}</td>
            <td>${num(band.valid_pixels)}</td></tr>`).join('');

        const catRows = (cats || []).map((row) => `<tr>
            <td>${num(row.value)}</td><td>${num(row.count)}</td>
            <td>${row.percent != null ? row.percent.toFixed(2) + '%' : '\u2014'}</td>
            <td>${row.color ? `<span class="hbs-swatch" style="background:${esc(row.color)}"></span>${esc(row.color)}` : '\u2014'}</td>
            </tr>`).join('');

        const notes = [];
        if (data.sampled && data.sample_shape) {
            notes.push(`Statistics read from a ${data.sample_shape[0]}×${data.sample_shape[1]} sample of the grid.`);
        }
        if (!cats && bands.length) {
            notes.push('No value table: this raster reads as continuous, not classified.');
        }

        return `<div class="hbs-stats">${facts.map(([k, v]) =>
                `<div${k.startsWith('Extent') ? ' class="wide"' : ''}><span>${esc(k)}</span><b>${esc(num(v))}</b></div>`).join('')}</div>
            ${bands.length ? `<p class="hbs-subhead">Band statistics</p>
            <div class="hbs-tblwrap"><table class="hbs-table">
                <thead><tr><th>Band</th><th>Description</th><th>Type</th><th>NoData</th><th>Min</th><th>Max</th><th>Mean</th><th>Std dev</th><th>Valid pixels</th></tr></thead>
                <tbody>${bandRows}</tbody></table></div>` : ''}
            ${cats && cats.length ? `<p class="hbs-subhead">Value counts</p>
            <div class="hbs-tblwrap"><table class="hbs-table">
                <thead><tr><th>Value</th><th>Pixels</th><th>% of valid</th><th>Colour</th></tr></thead>
                <tbody>${catRows}</tbody></table></div>` : ''}
            ${notes.length ? `<p class="exp2-field-note">${esc(notes.join(' '))}</p>` : ''}`;
    }

    async function toggleOutputAttributes(scope, index) {
        const key = outputPreviewKey(scope, index);
        const record = state.outputPreviews.get(key);
        if (!record || record.attributesLoading) return;
        // Already fetched: this is only ever a show/hide from here on.
        if (record.attributes || record.attributesError) {
            record.attributesOpen = !record.attributesOpen;
            render();
            return;
        }
        record.attributesOpen = true;
        record.attributesLoading = true;
        render();
        try {
            record.attributes = await api(outputRequestUrl({
                ...record.file,
                artifact_ref: (record.preview || {}).artifact_ref || record.file?.artifact_ref,
            }, 'attributes'));
        } catch (error) {
            record.attributesError = error.message;
        }
        record.attributesLoading = false;
        render();
    }

    async function toggleOutputPreview(scope, index, button) {
        const key = outputPreviewKey(scope, index);
        const existing = state.outputPreviews.get(key);
        // A failed preview is retried when reopened rather than replayed: the
        // failure may have been transient, and a cached error would otherwise
        // stick until the next re-run.
        if (existing?.preview || (existing?.error && existing.open)) {
            existing.open = !existing.open;
            render();
            return;
        }
        const file = {
            artifact_ref: button?.dataset?.ref || '',
            name: button?.dataset?.name || '',
            size_bytes: Number(button?.dataset?.size || 0),
        };
        state.outputPreviews.set(key, { open: true, loading: true, file });
        render();
        try {
            const data = await api(outputRequestUrl(file));
            const preview = data.preview || {};
            if (['image', 'pdf'].includes(preview.kind)) {
                const rawResponse = await fetch(outputRequestUrl(
                    { ...file, artifact_ref: preview.artifact_ref }, 'raw'));
                if (!rawResponse.ok) throw new Error(`Could not load binary preview (${rawResponse.status}).`);
                preview.object_url = URL.createObjectURL(await rawResponse.blob());
            }
            state.outputPreviews.set(key, {
                open: true, loading: false, file, preview,
            });
        } catch (error) {
            state.outputPreviews.set(key, {
                open: true, loading: false, file, error: error.message,
            });
        }
        render();
    }

    async function downloadOutput(button) {
        const artifactRef = button?.dataset?.ref || '';
        const filename = button?.dataset?.name || 'handbook-studio-output';
        try {
            const response = await fetch(outputRequestUrl({
                artifact_ref: artifactRef, name: filename,
            }, 'download'));
            if (!response.ok) throw new Error(`Download failed (${response.status}).`);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = url;
            link.download = filename || 'handbook-studio-output';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (error) {
            toast(error.message, 'error');
        }
    }

    function outputEvidenceCard(item) {
        const facts = [];
        if (item.feature_count != null) facts.push(`${item.feature_count} features`);
        if (item.row_count != null) facts.push(`${item.row_count} rows`);
        if (Array.isArray(item.layers)) facts.push(`${item.layers.length} layers`);
        if (item.osm_counts) facts.push(`${Object.values(item.osm_counts).reduce((sum, value) => sum + Number(value || 0), 0)} OSM elements`);
        if (Array.isArray(item.columns)) facts.push(`${item.columns.length} columns`);
        return `<article><strong>${esc(item.name || 'Output')}</strong><span>${esc(item.extension || 'unknown format')} · ${esc(item.size_bytes ?? 0)} bytes</span><p>${esc(facts.join(' · ') || item.inspection_note || 'Basic file evidence recorded')}</p></article>`;
    }

    // The clean, mechanical "did retrieval work" facts — Retrieval status,
    // Execution success, Output files, plus the raw evidence behind them
    // (files, HTTP requests, output structure, error trace, generated
    // code). Deliberately excludes anything that's a correctness JUDGMENT
    // (file validation, task completion, mechanism/parameter/spatial/
    // temporal correctness) — those live in validationFactsCard /
    // the Validation tab instead, so this card stays a quick "did it run
    // and produce something" read without correctness noise mixed in.
    function runById(runId) {
        return (state.session?.test_runs || []).find((run) => run.id === runId) || null;
    }

    function retrievalFactsCard(result, scope, runId) {
        if (!result) return '';
        // "Retrieval" is deliberately based on whether real, validly-formatted
        // data came back (validation.output_present && format_match) — NOT
        // result.status, which gets downgraded to "failed_validation" when
        // task-correctness (semantic validation) fails even though data was
        // retrieved fine. That correctness judgment belongs in the Validation
        // tab's "Task completion", not here; conflating the two is exactly
        // what made a successful download read as a failed retrieval.
        const validation = result.validation || {};
        const retrieved = Boolean(validation.output_present && validation.format_match);
        return `<div class="exp2-evaluation-artifact">
            <div class="exp2-artifact-facts">
                <div><span>Retrieval</span><strong class="${retrieved ? 'good' : 'bad'}">${retrieved ? 'Data retrieved' : 'No valid data retrieved'}</strong></div>
                <div><span>Execution success</span><strong class="${result.execution?.success ? 'good' : 'bad'}">${result.execution?.success ? 'Yes' : 'No'}</strong></div>
                <div><span>Output files</span><strong>${esc((result.downloaded_files || []).length)}</strong></div>
            </div>
            ${attemptsSection(result, scope, runId)}
            ${outputFileSection(result, scope, runId, newerRunLabel(result), runById(runId))}
            ${(result.http_requests || []).length ? `<details class="exp2-request-evidence" data-detail-key="${esc(scope)}:requests"><summary>View HTTP request evidence · ${result.http_requests.length}</summary><div>${result.http_requests.map(item => `<article><span>${esc(item.method || 'GET')}</span><code>${esc(item.url || '')}</code><small>${item.attempt ? `Attempt ${esc(item.attempt)}` : 'Recorded request'}</small></article>`).join('')}</div></details>` : ''}
            ${(result.output_evidence || []).length ? `<details class="exp2-output-structure" data-detail-key="${esc(scope)}:structure"><summary>Inspect output content evidence · ${result.output_evidence.length}</summary><div class="exp2-output-evidence">${result.output_evidence.map(outputEvidenceCard).join('')}</div></details>` : ''}
        </div>`;
    }

    // The code the agent actually ran, in the open rather than buried in a
    // collapsed panel, one card per attempt. A round can hold several: the
    // debugger rewrites the code after an error and tries again, so attempt 1
    // is usually the interesting failure and the last attempt is what produced
    // the output. Each carries its own re-run, because "does attempt 1 still
    // fail the same way?" is a different question from "can I get the data
    // back?", and they write to separate folders so one never clobbers the
    // other.
    function attemptsSection(result, scope, runId) {
        const attempts = (result.execution && result.execution.attempts) || [];
        const cards = attempts.length
            ? attempts.map((attempt, index) => attemptCard({
                  index,
                  label: `Attempt ${attempt.attempt ?? index + 1}`,
                  outcome: attempt.timed_out ? 'Timed out'
                      : attempt.success ? 'Succeeded' : 'Failed',
                  good: Boolean(attempt.success),
                  code: attempt.code || '',
                  error: attempt.error || '',
                  note: attempt.fix_explanation || '',
                  edited: Boolean(attempt.code_edited),
                  revised: Boolean(attempt.code_revised_from_feedback),
                  multiAttempt: attempts.length > 1,
                  feedbackLog: attempt.code_feedback || [],
                  hasOriginal: Boolean(attempt.original_code),
                  originalCode: attempt.original_code || '',
                  versions: attempt.code_versions || [],
                  currentVersion: attempt.current_version || null,
                  isLast: index === attempts.length - 1,
              }, scope, runId))
            : (result.generated_code ? [attemptCard({
                  index: null, label: 'Generated code',
                  outcome: result.execution && result.execution.success ? 'Succeeded' : 'Failed',
                  good: Boolean(result.execution && result.execution.success),
                  code: result.generated_code, error: result.error || '',
                  note: '',
                  edited: Boolean(result.code_edited),
                  revised: Boolean(result.code_revised_from_feedback),
                  feedbackLog: result.code_feedback || [],
                  hasOriginal: Boolean(result.original_code),
                  originalCode: result.original_code || '',
                  versions: result.code_versions || [],
                  currentVersion: result.current_version || null,
                  isLast: true,
              }, scope, runId)] : []);
        if (!cards.length) return '';
        return `<details class="hbs-attempts" data-detail-key="${esc(scope)}:code">
            <summary class="hbs-attempts-head">
                <h5>Retrieval code</h5>
                <span>${attempts.length > 1
                    ? `${attempts.length} attempts — the debugger revised the code after each failure`
                    : 'The code this round ran'}</span>
            </summary>
            ${cards.join('')}
        </details>`;
    }

    function attemptCard(attempt, scope, runId) {
        const key = rerunKey(runId, attempt.index);
        const record = state.rerunResults.get(key);
        const busy = record && record.running;
        const canRerun = Boolean(runId && (attempt.code || '').trim());
        // Editing needs a run to save against; a card rendered without one
        // (a stage with no test run behind it) stays read-only.
        const canEdit = Boolean(runId);
        const editing = state.codeEditing.has(key);
        const saving = state.codeSaving.has(key);
        const versions = codeVersionsFor(attempt);
        const currentVersionId = attempt.currentVersion || versions[versions.length - 1].id;
        const currentVersion = versions.find(v => v.id === currentVersionId) || versions[versions.length - 1];
        const versionsOpen = state.versionsOpenFor.has(key);
        const switching = state.versionSwitching.has(key);
        const feedbackOpen = state.feedbackOpenFor.has(key);
        const sendingFeedback = state.feedbackSending.has(key);
        const feedbackDraft = state.feedbackDrafts.get(key) || '';
        const draft = editing ? codeDraft(key, attempt.code || '') : (attempt.code || '');
        const lines = draft.split('\n').length;
        const dirty = editing && draft !== (attempt.code || '');
        const idx = attempt.index === null ? 'null' : attempt.index;
        const call = (fn, extra = '') =>
            `HandbookStudioMode.${fn}('${esc(runId)}', ${idx}${extra})`;
        return `<article class="hbs-attempt ${attempt.good ? 'ok' : 'bad'} ${editing ? 'editing' : ''}">
            <header>
                <span class="hbs-attempt-badge">${esc(attempt.label)}</span>
                <strong class="${attempt.good ? 'good' : 'bad'}">${esc(attempt.outcome)}</strong>
                ${attempt.multiAttempt && attempt.isLast && attempt.index !== null
                    ? '<span class="hbs-attempt-tag">produced the output</span>' : ''}
                <span class="hbs-spacer"></span>
                <span class="hbs-attempt-lines">${lines} lines${
                    versions.length > 1
                        ? ` \u00b7 v${versions.indexOf(currentVersion) + 1} of ${versions.length} \u00b7 ${esc(currentVersion.label || '')}`
                        : ''}${dirty ? ' \u00b7 unsaved' : ''}</span>
                ${editing ? `
                    <button class="exp2-btn tiny secondary" ${saving ? 'disabled' : ''}
                        onclick="${call('cancelCodeEdit')}">Cancel</button>
                    <button class="exp2-btn tiny" ${saving ? 'disabled' : ''}
                        onclick="${call('saveCode')}">${saving ? 'Saving…' : 'Save'}</button>
                    <button class="exp2-btn tiny primary" ${saving || state.running ? 'disabled' : ''}
                        onclick="${call('saveCode', ', true')}">Save &amp; re-run</button>
                ` : `

                    <button class="exp2-btn tiny secondary ${versionsOpen ? 'active' : ''}" ${switching ? 'disabled' : ''}
                        onclick="${call('toggleVersions')}"
                        title="Every stored version of this code, with the output each one produced">Versions (${versions.length})</button>
                    ${canEdit && attempt.hasOriginal ? `<button class="exp2-btn tiny secondary" ${saving || state.running ? 'disabled' : ''}
                        onclick="${call('revertCode')}" title="Restore the code the agent generated">Revert</button>` : ''}
                    ${canEdit ? `<button class="exp2-btn tiny secondary" ${busy || state.running ? 'disabled' : ''}
                        onclick="${call('editCode')}">Edit</button>` : ''}
                    ${canRerun ? `<button class="exp2-btn tiny secondary ${feedbackOpen ? 'active' : ''}" ${busy || sendingFeedback || state.running ? 'disabled' : ''}
                        onclick="${call('toggleCodeFeedback')}"
                        title="Say what is wrong and have the model revise this code">Provide feedback</button>` : ''}
                    ${canRerun ? `<button class="exp2-btn tiny ${busy ? 'secondary' : ''}" ${busy || state.running ? 'disabled' : ''}
                        onclick="${call('rerunCode')}">
                        ${busy ? 'Running…' : 'Re-run'}</button>` : ''}

                `}
            </header>
            ${attempt.note ? `<p class="hbs-attempt-note">${esc(attempt.note)}</p>` : ''}
            ${versionsOpen ? `<div class="hbs-versions">
                <p class="exp2-field-note">${versions.length > 1
                    ? 'Every stored version of this code. Loading one puts it back on the card and brings its own recorded output with it.'
                    : 'One version so far: the code the agent wrote for this round. Editing it, or sending feedback, files the result as a new version here.'}</p>
                ${versions.map((version, versionIndex) => {
                    const traceOpen = state.versionTraceOpen.has(key + '::' + version.id);
                    return `<div class="hbs-version ${version.id === currentVersionId ? 'current' : ''} ${traceOpen ? 'expanded' : ''}">
                    <button type="button" class="hbs-version-toggle" aria-expanded="${traceOpen}"
                        title="${traceOpen ? 'Hide what changed' : 'Show what changed in this version'}"
                        onclick="${call('toggleVersionTrace', `, '${esc(version.id)}'`)}">
                        <span class="hbs-version-caret">${traceOpen ? '\u25be' : '\u25b8'}</span>
                        <span class="hbs-version-tag">${esc(version.label || version.id)}</span>
                        <span class="hbs-version-meta">${version.at ? esc(new Date(version.at).toLocaleString()) : 'time not recorded'}
                            \u00b7 ${esc((version.code || '').split('\n').length)} lines${
                            version.lines_added != null ? ` \u00b7 +${esc(version.lines_added)} \u2212${esc(version.lines_removed)}` : ''}</span>
                        ${version.feedback ? `<span class="hbs-version-feedback">\u201c${esc(String(version.feedback).slice(0, 70))}\u201d</span>` : ''}
                    </button>
                    <span class="hbs-version-run">${version.run
                        ? `${esc((version.run.files || []).length)} file${(version.run.files || []).length === 1 ? '' : 's'}`
                        : version.original_output_files
                        ? `${esc(version.original_output_files)} file${version.original_output_files === 1 ? '' : 's'} (original run)`
                        : 'not run'}</span>
                    ${version.id === currentVersionId
                        ? '<span class="hbs-version-current">loaded</span>'
                        : `<button class="exp2-btn tiny secondary" ${switching ? 'disabled' : ''}
                            onclick="${call('selectCodeVersion', `, '${esc(version.id)}'`)}">${switching ? 'Loading\u2026' : 'Load'}</button>`}
                </div>
                ${traceOpen ? `<div class="hbs-version-trace">${
                    versionTrace(version, versions[versionIndex - 1] || null, key)}</div>` : ''}`;
                }).join('')}
            </div>` : ''}
            ${feedbackOpen ? `<div class="exp2-inline-form hbs-feedback-form">
                <p class="exp2-field-note">Say what is wrong with this code. The model rewrites it from your feedback and saves it here — then re-run to execute it. The agent's original stays recoverable with Revert.</p>
                <textarea class="hbs-feedback-input" rows="3" ${sendingFeedback ? 'disabled' : ''}
                    data-feedback-key="${esc(key)}"
                    placeholder="e.g. It downloads the whole state — filter to the county FIPS in the retrieval task before writing the file."
                    oninput="HandbookStudioMode.updateFeedbackDraft(this)">${esc(feedbackDraft)}</textarea>
                ${sendingFeedback ? `<p class="exp2-field-note">The model is rewriting the code — a few hundred lines takes a minute or two. Its output appears below as it arrives.</p>
                <pre class="hbs-feedback-stream" data-feedback-stream="${esc(key)}">${esc(state.feedbackStream.get(key) || '')}</pre>` : ''}
                <div class="hbs-feedback-actions">
                    <button class="exp2-btn tiny secondary" ${sendingFeedback ? 'disabled' : ''}
                        onclick="${call('toggleCodeFeedback')}">Cancel</button>
                    <button class="exp2-btn tiny primary" data-feedback-send="${esc(key)}"
                        ${sendingFeedback || !feedbackDraft.trim() ? 'disabled' : ''}
                        onclick="${call('sendCodeFeedback')}">${sendingFeedback ? 'Revising…' : 'Send — revise this code'}</button>
                </div>
            </div>` : ''}
            ${currentVersion && currentVersion.feedback ? `<div class="hbs-feedback-log">
                <p><strong>Your feedback</strong> · the version loaded here${
                    currentVersion.at ? ` · ${esc(new Date(currentVersion.at).toLocaleString())}` : ''}${
                    currentVersion.lines_added != null
                        ? ` · +${esc(currentVersion.lines_added)} −${esc(currentVersion.lines_removed)}`
                        : ''}</p>
                <blockquote>${esc(currentVersion.feedback)}</blockquote>
                ${currentVersion.explanation ? `<p><strong>What the model changed</strong></p><blockquote>${esc(currentVersion.explanation)}</blockquote>` : ''}
            </div>` : ''}
            ${editing ? `<textarea class="hbs-code-editor" spellcheck="false" wrap="off"
                    data-code-key="${esc(key)}" aria-label="Edit ${esc(attempt.label)} code"
                    oninput="HandbookStudioMode.updateCodeDraft(this)"
                    onkeydown="HandbookStudioMode.codeEditorKey(event, this)"
                    onscroll="HandbookStudioMode.trackCodeCaret(this)"
                    onclick="HandbookStudioMode.trackCodeCaret(this)"
                    onkeyup="HandbookStudioMode.trackCodeCaret(this)">
${esc(draft)}</textarea>
                <p class="hbs-code-hint">Saving replaces the stored code for this ${attempt.index === null ? 'run' : 'attempt'} — what a re-run executes. The agent's original is kept and can be restored. Tab inserts four spaces, Escape leaves the box, ⌘/Ctrl+Enter saves and re-runs.</p>`
            : `<pre class="hbs-attempt-code"><code class="language-python">${highlightPython(attempt.code)}</code></pre>`}
            ${attempt.error ? `<details class="hbs-attempt-error" data-detail-key="${esc(scope)}:err-${attempt.index}"><summary>Original error</summary><pre>${esc(attempt.error)}</pre></details>` : ''}
            ${record ? rerunOutcome(record, `${scope}:rerun-${attempt.index}`)
                : (currentVersion && currentVersion.run
                    ? rerunOutcome({ ...currentVersion.run, stored: true },
                                   `${scope}:stored-${attempt.index}-${currentVersion.id}`)
                    : '')}
        </article>`;
    }

    const rerunKey = (runId, index) => `${runId}#${index === null ? 'final' : index}`;

    const codeDraft = (key, fallback) =>
        state.codeDrafts.has(key) ? state.codeDrafts.get(key) : fallback;

    // Locating the stored code again on save: the card renders from a copy of
    // the record, so every edit action re-resolves the live slot rather than
    // trusting what was on screen when the button was drawn.
    function attemptSlot(runId, attemptIndex) {
        const run = (state.session?.test_runs || []).find((item) => item.id === runId);
        const result = run && run.result;
        if (!result) return null;
        if (attemptIndex === null || attemptIndex === undefined) {
            return { holder: result, field: 'generated_code' };
        }
        const attempt = ((result.execution || {}).attempts || [])[attemptIndex];
        return attempt ? { holder: attempt, field: 'code' } : null;
    }

    function editCode(runId, attemptIndex) {
        const slot = attemptSlot(runId, attemptIndex);
        if (!slot) return;
        const key = rerunKey(runId, attemptIndex ?? null);
        state.codeEditing.add(key);
        if (!state.codeDrafts.has(key)) {
            state.codeDrafts.set(key, slot.holder[slot.field] || '');
        }
        render();
        const field = document.querySelector(`textarea[data-code-key="${cssEscape(key)}"]`);
        if (field) {
            field.focus();
            field.setSelectionRange(0, 0);
            field.scrollIntoView({ block: 'nearest' });
        }
    }

    function cancelCodeEdit(runId, attemptIndex) {
        const key = rerunKey(runId, attemptIndex ?? null);
        const slot = attemptSlot(runId, attemptIndex);
        const stored = slot ? (slot.holder[slot.field] || '') : '';
        if (codeDraft(key, stored) !== stored
            && !confirm('Discard your unsaved edits to this code?')) return;
        state.codeEditing.delete(key);
        state.codeDrafts.delete(key);
        if (state.codeCaret && state.codeCaret.key === key) state.codeCaret = null;
        render();
    }

    function updateCodeDraft(field) {
        const key = field.dataset.codeKey;
        if (!key) return;
        // The draft is recorded FIRST and unconditionally: everything below is
        // cosmetic, and an earlier version of this let a failure in the
        // cosmetic half leave the Save button stuck disabled while the draft
        // itself was perfectly fine. Save is never gated on this running.
        state.codeDrafts.set(key, field.value);
        trackCodeCaret(field);
        // The line count is the only thing that changes per keystroke, so it
        // is patched in place -- a full render() here would replace the
        // textarea mid-word.
        try {
            const card = field.closest('.hbs-attempt');
            const counter = card && card.querySelector('.hbs-attempt-lines');
            if (!counter) return;
            const slot = parseCodeKey(key);
            const stored = slot ? (slot.holder[slot.field] || '') : '';
            const lines = field.value.split('\n').length;
            counter.textContent = `${lines} lines${field.value !== stored ? ' · unsaved' : ''}`;
        } catch (error) {
            console.warn('Code editor: line counter update failed', error);
        }
    }

    // A key is "<runId>#<index|final>"; the run id is a uuid without '#'.
    function parseCodeKey(key) {
        const cut = String(key).lastIndexOf('#');
        if (cut < 0) return null;
        const tail = key.slice(cut + 1);
        return attemptSlot(key.slice(0, cut), tail === 'final' ? null : Number(tail));
    }

    function trackCodeCaret(field) {
        if (!field || !field.dataset.codeKey) return;
        state.codeCaret = {
            key: field.dataset.codeKey,
            start: field.selectionStart, end: field.selectionEnd,
            scrollTop: field.scrollTop, scrollLeft: field.scrollLeft,
        };
    }

    // Tab has to insert rather than leave the field: this is a code editor,
    // and Python is indentation-sensitive. That would trap keyboard focus, so
    // Escape is the way back out; ⌘/Ctrl+Enter is save-and-re-run.
    function codeEditorKey(event, field) {
        const key = field.dataset.codeKey;
        if (!key) return;
        if (event.key === 'Escape') {
            event.preventDefault();
            const exit = field.closest('.hbs-attempt')?.querySelector('header button');
            if (exit) exit.focus(); else field.blur();
            return;
        }
        if (event.key === 'Tab' && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            const start = field.selectionStart;
            const end = field.selectionEnd;
            field.value = `${field.value.slice(0, start)}    ${field.value.slice(end)}`;
            field.setSelectionRange(start + 4, start + 4);
            updateCodeDraft(field);
            return;
        }
        if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            const slot = parseCodeKey(key);
            if (!slot) return;
            const cut = String(key).lastIndexOf('#');
            const tail = key.slice(cut + 1);
            saveCode(key.slice(0, cut), tail === 'final' ? null : Number(tail), true);
        }
    }

    // Saving writes the edit into the session record on the server, so the
    // next re-run (here or after a reload) executes exactly what is on screen.
    // Records written before versioning kept only `original_code` and the
    // current code. Deriving the same two entries the server would means the
    // picker works on old sessions, and the ids it offers match the ones the
    // server creates when one is actually loaded.
    function codeVersionsFor(attempt) {
        if (Array.isArray(attempt.versions) && attempt.versions.length) return attempt.versions;
        const current = attempt.code || '';
        const original = attempt.originalCode || '';
        const last = (attempt.feedbackLog || [])[(attempt.feedbackLog || []).length - 1] || {};
        if (original && original !== current) {
            return [
                { id: 'v1', origin: 'agent', label: 'Agent original',
                  code: original, at: null, run: null },
                { id: 'v2', origin: attempt.revised ? 'feedback' : 'edit',
                  label: attempt.revised ? 'Revised from feedback' : 'Hand edit',
                  code: current, at: null, run: null,
                  feedback: last.feedback || null,
                  lines_added: last.lines_added, lines_removed: last.lines_removed },
            ];
        }
        return [{ id: 'v1', origin: 'agent', label: 'Agent original',
                  code: current, at: null, run: null }];
    }

    // A line diff between two versions of the code. Common prefix and suffix
    // are trimmed first, so an edit of a few lines in a 355-line script costs
    // almost nothing to compare; only the disputed middle goes through the
    // LCS table, and a pathologically large middle is refused rather than
    // freezing the tab.
    const DIFF_CELL_BUDGET = 2000000;

    function diffCodeLines(beforeText, afterText) {
        const before = String(beforeText || '').split('\n');
        const after = String(afterText || '').split('\n');
        let start = 0;
        while (start < before.length && start < after.length
               && before[start] === after[start]) start++;
        let endBefore = before.length;
        let endAfter = after.length;
        while (endBefore > start && endAfter > start
               && before[endBefore - 1] === after[endAfter - 1]) {
            endBefore--; endAfter--;
        }
        const midBefore = before.slice(start, endBefore);
        const midAfter = after.slice(start, endAfter);
        if (!midBefore.length && !midAfter.length) {
            return { rows: [], added: 0, removed: 0, identical: true };
        }
        if (midBefore.length * midAfter.length > DIFF_CELL_BUDGET) {
            return { rows: null, added: midAfter.length, removed: midBefore.length,
                     tooBig: true };
        }

        const m = midBefore.length;
        const n = midAfter.length;
        const table = new Int32Array((m + 1) * (n + 1));
        for (let i = m - 1; i >= 0; i--) {
            for (let j = n - 1; j >= 0; j--) {
                table[i * (n + 1) + j] = midBefore[i] === midAfter[j]
                    ? table[(i + 1) * (n + 1) + (j + 1)] + 1
                    : Math.max(table[(i + 1) * (n + 1) + j], table[i * (n + 1) + (j + 1)]);
            }
        }

        const rows = [];
        let added = 0;
        let removed = 0;
        let i = 0;
        let j = 0;
        while (i < m && j < n) {
            if (midBefore[i] === midAfter[j]) {
                rows.push({ type: 'context', text: midBefore[i] });
                i++; j++;
            } else if (table[(i + 1) * (n + 1) + j] >= table[i * (n + 1) + (j + 1)]) {
                rows.push({ type: 'del', text: midBefore[i] });
                removed++; i++;
            } else {
                rows.push({ type: 'add', text: midAfter[j] });
                added++; j++;
            }
        }
        while (i < m) { rows.push({ type: 'del', text: midBefore[i++] }); removed++; }
        while (j < n) { rows.push({ type: 'add', text: midAfter[j++] }); added++; }

        // A couple of unchanged lines either side, so a change reads in context
        // rather than as a floating fragment.
        const CONTEXT = 2;
        const lead = before.slice(Math.max(0, start - CONTEXT), start)
            .map((text) => ({ type: 'context', text }));
        const tail = before.slice(endBefore, Math.min(before.length, endBefore + CONTEXT))
            .map((text) => ({ type: 'context', text }));
        return { rows: lead.concat(rows, tail), added, removed, startLine: Math.max(1, start - CONTEXT + 1) };
    }

    function versionTrace(version, previous, key) {
        const parts = [];
        if (version.feedback) {
            parts.push(`<p class="hbs-trace-head">What was asked for</p>
                <blockquote>${esc(version.feedback)}</blockquote>`);
        }
        if (version.explanation) {
            parts.push(`<p class="hbs-trace-head">What the model said it changed</p>
                <blockquote>${esc(version.explanation)}</blockquote>`);
        }
        if (!previous) {
            parts.push(`<p class="exp2-field-note">This is the first stored version \u2014 the code as the agent generated it, with nothing before it to compare against.</p>`);
            return parts.join('');
        }

        const diff = diffCodeLines(previous.code, version.code);
        if (diff.identical) {
            parts.push('<p class="exp2-field-note">The code is byte-for-byte identical to the version before it.</p>');
            return parts.join('');
        }
        if (diff.tooBig) {
            parts.push(`<p class="exp2-field-note">Too large to diff in the browser \u2014 roughly ${esc(diff.removed)} lines replaced by ${esc(diff.added)}.</p>`);
            return parts.join('');
        }
        const MAX_ROWS = 300;
        const shown = diff.rows.slice(0, MAX_ROWS);
        parts.push(`<p class="hbs-trace-head">Change against ${esc(previous.label || 'the previous version')}
            <span>+${esc(diff.added)} \u2212${esc(diff.removed)}</span></p>
            <pre class="hbs-diff">${shown.map((row) =>
                `<span class="hbs-diff-${row.type}">${esc(
                    (row.type === 'add' ? '+ ' : row.type === 'del' ? '- ' : '  ') + row.text)}</span>`
            ).join('\n')}</pre>
            ${diff.rows.length > MAX_ROWS
                ? `<p class="exp2-field-note">Showing the first ${MAX_ROWS} lines of the change.</p>` : ''}`);
        return parts.join('');
    }

    function toggleVersionTrace(runId, attemptIndex, versionId) {
        const key = rerunKey(runId, attemptIndex ?? null) + '::' + versionId;
        if (state.versionTraceOpen.has(key)) state.versionTraceOpen.delete(key);
        else state.versionTraceOpen.add(key);
        render();
    }

    function toggleVersions(runId, attemptIndex) {
        const key = rerunKey(runId, attemptIndex ?? null);
        if (state.versionsOpenFor.has(key)) state.versionsOpenFor.delete(key);
        else state.versionsOpenFor.add(key);
        render();
    }

    async function selectCodeVersion(runId, attemptIndex, versionId) {
        const key = rerunKey(runId, attemptIndex ?? null);
        if (state.versionSwitching.has(key)) return;
        state.versionSwitching.add(key);
        render();
        try {
            await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                + `/test-runs/${encodeURIComponent(runId)}/select-version`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                      attempt: attemptIndex === undefined ? null : attemptIndex,
                      version: versionId }) });
            // A re-run held in memory belongs to the version that was loaded a
            // moment ago, not to this one; dropping it lets the version's own
            // stored output show instead of the previous one's.
            state.rerunResults.delete(key);
            const latest = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
            if (latest && latest.session) state.session = latest.session;
            // Picking from the list is the end of that errand: the list folds
            // away so the code and output it just loaded are what you are
            // looking at, rather than being pushed down the page by the picker.
            state.versionsOpenFor.delete(key);
            toast('Version loaded — re-run it to regenerate its output.', 'success');
        } catch (error) {
            toast(error.message, 'error');
        }
        state.versionSwitching.delete(key);
        render();
    }

    function toggleCodeFeedback(runId, attemptIndex) {
        const key = rerunKey(runId, attemptIndex ?? null);
        if (state.feedbackSending.has(key)) return;
        if (state.feedbackOpenFor.has(key)) {
            state.feedbackOpenFor.delete(key);
            state.feedbackDrafts.delete(key);
        } else {
            state.feedbackOpenFor.add(key);
        }
        render();
    }

    // Typed straight into state rather than read at send time: the card is
    // re-rendered wholesale on every state change, so an unheld draft would
    // be wiped by any unrelated render while the reviewer is still typing.
    function updateFeedbackDraft(textarea) {
        const key = textarea.dataset.feedbackKey;
        if (!key) return;
        state.feedbackDrafts.set(key, textarea.value);
        // NEVER render from here. render() rebuilds the card wholesale, which
        // replaces this textarea with a new node -- the caret goes with it and
        // the reviewer is thrown out of the box mid-sentence. The only thing
        // typing changes on screen is whether Send is available, so that one
        // property is set directly on the existing button instead. (The key
        // contains '#', so it is matched by dataset rather than a selector.)
        const send = [...document.querySelectorAll('[data-feedback-send]')]
            .find((button) => button.dataset.feedbackSend === key);
        if (send && !state.feedbackSending.has(key)) {
            send.disabled = !textarea.value.trim();
        }
    }

    // Appended straight to the live pane rather than through render(): a
    // re-render per chunk would rebuild the whole card dozens of times a
    // second and throw away the scroll position with it.
    function appendFeedbackStream(key, text) {
        if (!text) return;
        state.feedbackStream.set(key, (state.feedbackStream.get(key) || '') + text);
        const pane = [...document.querySelectorAll('[data-feedback-stream]')]
            .find((node) => node.dataset.feedbackStream === key);
        if (pane) {
            pane.textContent = state.feedbackStream.get(key);
            pane.scrollTop = pane.scrollHeight;
        }
    }

    async function sendCodeFeedback(runId, attemptIndex) {
        const key = rerunKey(runId, attemptIndex ?? null);
        if (state.feedbackSending.has(key)) return;
        const feedback = (state.feedbackDrafts.get(key) || '').trim();
        if (!feedback) { toast('Say what should change first.', 'error'); return; }
        state.feedbackSending.add(key);
        state.feedbackStream.set(key, '');
        render();

        let finished = null;
        try {
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                + `/test-runs/${encodeURIComponent(runId)}/revise-code-stream`,
                { method: 'POST',
                  headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                  body: JSON.stringify({
                      attempt: attemptIndex === undefined ? null : attemptIndex,
                      feedback }) });
            // Pre-flight failures come back as ordinary JSON, since nothing has
            // been streamed yet.
            if (!response.ok || !response.body) {
                let message = `Request failed (${response.status}).`;
                try { message = (await response.json()).error || message; } catch (_) { /* ignore */ }
                throw new Error(message);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
                const frames = buffer.split(/\r?\n\r?\n/);
                buffer = frames.pop() || '';
                for (const frame of frames) {
                    const line = frame.split(/\r?\n/).find(item => item.startsWith('data:'));
                    if (!line) continue;
                    let event;
                    try { event = JSON.parse(line.slice(5).trim()); } catch (_) { continue; }
                    if (event.type === 'revise_delta') appendFeedbackStream(key, event.text || '');
                    else if (event.type === 'revise_done') finished = event;
                    else if (event.type === 'error') throw new Error(event.error || 'The revision failed.');
                }
                if (done) break;
            }
            if (!finished) throw new Error('The revision ended without returning any code.');
            state.feedbackOpenFor.delete(key);
            state.feedbackDrafts.delete(key);
            toast('Code revised from your feedback — re-run to execute it.', 'success');
        } catch (error) {
            // The draft is deliberately kept on failure: it is the reviewer's
            // words, and a failed model call is the moment they least want to
            // retype them.
            toast(error.message, 'error');
        } finally {
            state.feedbackSending.delete(key);
            state.feedbackStream.delete(key);
            // The server is the record. Re-reading the session here is what
            // makes the result land no matter what happened to this request --
            // a revision that completed server-side but lost its response
            // still shows up on the card instead of vanishing.
            try {
                const latest = await api(
                    `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
                if (latest && latest.session) state.session = latest.session;
            } catch (_) { /* keep what is already on screen */ }
            render();
        }
    }

    async function saveCode(runId, attemptIndex, thenRerun = false) {
        const key = rerunKey(runId, attemptIndex ?? null);
        if (state.codeSaving.has(key)) return;
        const slot = attemptSlot(runId, attemptIndex);
        // Every way out of this function says something. Returning quietly is
        // indistinguishable from a dead button, which is exactly how a broken
        // Save presented itself once already.
        if (!slot) {
            toast('Could not find this code in the session record — reload and try again.', 'error');
            return;
        }
        const stored = slot.holder[slot.field] || '';
        const code = codeDraft(key, stored);
        if (!code.trim()) { toast('Code cannot be empty.', 'error'); return; }
        if (code === stored) {
            // Nothing to write. Posting anyway would flag the attempt as
            // hand-edited and file an "original" identical to it.
            state.codeEditing.delete(key);
            state.codeDrafts.delete(key);
            if (state.codeCaret && state.codeCaret.key === key) state.codeCaret = null;
            render();
            if (thenRerun) { await rerunCode(runId, attemptIndex); return; }
            toast('No changes to save.', 'info');
            return;
        }
        state.codeSaving.add(key);
        render();
        try {
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                + `/test-runs/${encodeURIComponent(runId)}/attempt-code`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                      attempt: attemptIndex === undefined ? null : attemptIndex,
                      code }) });
            slot.holder[slot.field] = data.code;
            slot.holder.code_edited = true;
            // First edit is when the server files the agent's own version away;
            // holding it here too is what lights up Revert without a refetch.
            if (data.original_code) slot.holder.original_code = data.original_code;
            // A hand edit IS a version. Taking the list back from the response
            // is what makes it appear in the picker straight away rather than
            // after a reload.
            if (data.code_versions) slot.holder.code_versions = data.code_versions;
            if (data.current_version) slot.holder.current_version = data.current_version;
            state.codeEditing.delete(key);
            state.codeDrafts.delete(key);
            state.codeSaving.delete(key);
            if (state.codeCaret && state.codeCaret.key === key) state.codeCaret = null;
            if (thenRerun) {
                toast('Code saved — re-running.', 'success');
                await rerunCode(runId, attemptIndex);
                return;
            }
            toast('Code saved. Re-run to execute it.', 'success');
        } catch (error) {
            state.codeSaving.delete(key);
            toast(error.message, 'error');
        }
        render();
    }

    async function revertCode(runId, attemptIndex) {
        const slot = attemptSlot(runId, attemptIndex);
        if (!slot) return;
        if (!confirm('Restore the code the agent generated? Your edits are discarded.')) return;
        const key = rerunKey(runId, attemptIndex ?? null);
        try {
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                + `/test-runs/${encodeURIComponent(runId)}/attempt-code`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                      attempt: attemptIndex === undefined ? null : attemptIndex,
                      revert: true }) });
            slot.holder[slot.field] = data.code;
            slot.holder.code_edited = false;
            delete slot.holder.original_code;
            if (data.code_versions) slot.holder.code_versions = data.code_versions;
            if (data.current_version) slot.holder.current_version = data.current_version;
            state.codeEditing.delete(key);
            state.codeDrafts.delete(key);
            toast('Original code restored.', 'success');
        } catch (error) {
            toast(error.message, 'error');
        }
        render();
    }

    // CSS.escape isn't in every browser this UI is opened in.
    const cssEscape = (value) => (window.CSS && CSS.escape)
        ? CSS.escape(value) : String(value).replace(/["\\]/g, '\\$&');

    // A heartbeat re-render during a live run replaces the textarea the user
    // is typing in; without this the caret jumps to the end and the view
    // scrolls back to the top on every tick. Only the editor that HELD focus
    // when the render started gets it back, so a render while the user is
    // somewhere else never pulls the cursor into a code box.
    function restoreCodeEditor(main, key) {
        const caret = state.codeCaret;
        if (!caret || caret.key !== key) return;
        const field = main.querySelector(`textarea[data-code-key="${cssEscape(key)}"]`);
        if (!field || field === document.activeElement) return;
        field.focus({ preventScroll: true });
        try { field.setSelectionRange(caret.start, caret.end); } catch (_) { /* detached */ }
        field.scrollTop = caret.scrollTop;
        field.scrollLeft = caret.scrollLeft;
    }

    // Everything that's a correctness JUDGMENT rather than a raw execution
    // fact: File validation, Task completion, and (via semanticValidationArtifact
    // / traceAnalysisArtifact) Mechanism / Parameters / Spatial query /
    // Temporal query correctness — shown in the Validation tab, separate
    // from the plain "did it run" facts in retrievalFactsCard.
    const VALIDATION_METHOD_LABELS = {
        mechanical: 'Mechanical check (deterministic)',
        metadata: 'Mechanical check (deterministic)',
        llm: 'LLM judge (comparison arm)',
        manual: 'Human adjudication',
    };

    // The experiment's correctness metrics, in the order the rubric asks
    // about them. "Dataset / product selection" and "Provenance" have no
    // equivalent in the old five-field schema even though the experimental
    // design calls for the former.
    const VALIDATION_DIMENSIONS = [
        ['dataset_selection', 'Dataset / product selection'],
        ['mechanism_correctness', 'Access mechanism'],
        ['parameter_correctness', 'Query parameters'],
        ['spatial_query_correctness', 'Spatial query'],
        ['temporal_query_correctness', 'Temporal query'],
        ['output_correctness', 'Output validity'],
        ['provenance', 'Provenance'],
    ];
    const VERDICT_OPTIONS = ['not_applicable', 'pass', 'fail', 'undecidable'];

    // "undecidable" is styled as its own state, never as a pass — the whole
    // point of the three-valued algebra is that "we could not check this"
    // stays visible instead of being absorbed into a green tick.
    const verdictClass = v => v === 'pass' ? 'good' : v === 'fail' ? 'bad' : v === 'undecidable' ? 'warn' : '';
    const verdictLabel = v => String(v || 'unknown').replaceAll('_', ' ');

    function recordVerdict(record) {
        if (!record) return 'unknown';
        return record.verdict || (record.task_completed ? 'pass' : 'fail');
    }

    function validationFactsCard(result, scope, runId = '') {
        if (!result) return '';
        const validation = result.validation || {};
        const semantic = result.semantic_validation;
        const trace = result.trace_analysis;
        const methodLabel = VALIDATION_METHOD_LABELS[result.validation_method];
        const verdict = recordVerdict(semantic);
        // Blinding: while a reviewer has the rubric open for this run, every
        // previously recorded verdict is withheld so their judgment is their
        // own. They can reveal, and that choice is stored on the verdict.
        const blinding = runId && state.manualFormOpenFor.has(runId) && !state.manualRevealed.has(runId);
        if (blinding) {
            return `<div class="exp2-evaluation-artifact">
                <section class="exp2-notice neutral"><strong>Hidden for blind review</strong><p>Verdicts already recorded for this run are hidden while you adjudicate it, so they can't anchor your judgment. Submit your verdict to reveal them.</p><button class="exp2-btn tiny secondary" onclick="HandbookStudioMode.revealForReview('${esc(runId)}')">Reveal anyway (recorded as unblinded)</button></section>
            </div>`;
        }
        const others = (result.validations || []).filter(v => v !== semantic);
        return `<div class="exp2-evaluation-artifact">
            <div class="exp2-artifact-facts">
                <div><span>File validation</span><strong class="${validation.passed ? 'good' : 'bad'}">${validation.passed ? 'Passed' : 'Not passed'}</strong></div>
                ${semantic ? `<div><span>Task completion</span><strong class="${verdictClass(verdict)}">${esc(verdictLabel(verdict))}</strong></div>` : ''}
                ${methodLabel ? `<div><span>Reported via</span><strong>${esc(methodLabel)}</strong></div>` : ''}
            </div>
            ${semantic ? semanticValidationArtifact(semantic) : (trace ? traceAnalysisArtifact(trace) : '<p class="exp2-field-note">No correctness evaluation was recorded for this run yet.</p>')}
            ${others.length ? `<details class="exp2-validation-evidence" data-detail-key="${esc(scope)}:other-methods"><summary>${others.length} other validation record${others.length === 1 ? '' : 's'} for this run (append-only log)</summary><div>${others.map(v => semanticValidationArtifact(v)).join('')}</div></details>` : ''}
            <details class="exp2-validation-evidence" data-detail-key="${esc(scope)}:validation"><summary>View validation evidence</summary><div><p><strong>${validation.passed ? 'Passed' : 'Not passed'}</strong> ${esc(validation.message || 'No validation message was recorded.')}</p><dl><div><dt>Output present</dt><dd>${validation.output_present ? 'Yes' : 'No'}</dd></div><div><dt>Format match</dt><dd>${validation.format_match === false ? 'No' : 'Yes'}</dd></div><div><dt>Expected format</dt><dd>${esc(validation.expected_format || 'Not specified')}</dd></div></dl></div></details>
        </div>`;
    }

    // What the self-verification run actually produced: a head of each
    // output file plus the run's stdout, captured by the verifier before its
    // sandbox was deleted. Older sessions have only file names.
    function verificationSamples(verification, scope) {
        const samples = verification.samples;
        const names = verification.files || [];
        if (!samples && !names.length) return '';
        const files = samples?.files?.length ? samples.files : names.map(name => ({ name }));
        const sizeLabel = (n) => !Number.isFinite(n) ? ''
            : n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB`
            : n >= 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`;
        const items = files.map((file, i) => {
            const size = sizeLabel(file.size);
            const body = file.preview
                ? `<pre class="hbs-sample-preview">${esc(file.preview)}${file.truncated ? '\n…' : ''}</pre>`
                : `<p class="exp2-field-note">${size ? 'Binary file — no text preview.' : 'Preview not captured for this run.'}</p>`;
            return `<details class="hbs-sample-file" data-detail-key="${esc(scope)}:sample-${i}">
                <summary><code>${esc(file.name)}</code>${size ? `<small>${esc(size)}</small>` : ''}</summary>${body}</details>`;
        }).join('');
        const more = samples?.more ? `<p class="exp2-field-note">${esc(samples.more)} more file${samples.more === 1 ? '' : 's'} not shown.</p>` : '';
        const stdout = samples?.stdout
            ? `<details class="hbs-sample-file" data-detail-key="${esc(scope)}:sample-stdout"><summary><code>stdout</code><small>what the sample run printed</small></summary><pre class="hbs-sample-preview">${esc(samples.stdout)}</pre></details>`
            : '';
        return `<details class="exp2-handbook-artifact hbs-samples" data-detail-key="${esc(scope)}:samples">
            <summary><div><span>Sample outputs</span><strong>${files.length} file${files.length === 1 ? '' : 's'} produced by the verification run</strong></div></summary>
            ${items}${more}${stdout}
        </details>`;
    }

    function pipelineArtifact(artifact, scope = '') {
        if (!artifact || typeof artifact !== 'object') return '';
        const parts = [];
        const verification = artifact.verification;
        if (verification) {
            parts.push(`<div class="exp2-artifact-facts"><div><span>Verification</span><strong>${verification.verified ? 'Passed' : esc(verification.status || 'Not passed')}</strong></div><div><span>Attempts</span><strong>${esc(verification.attempts ?? 0)}</strong></div><div><span>Sample outputs</span><strong>${esc((verification.files || []).length)}</strong></div></div>`);
            if (verification.error) {
                parts.push(`<details class="exp2-validation-evidence" data-detail-key="${esc(scope)}:verify-error"><summary>View the sample-download error</summary><pre>${esc(verification.error)}</pre></details>`);
            }
            parts.push(verificationSamples(verification, scope));
        }
        const source = artifact.source;
        if (source) {
            parts.push(`<div class="exp2-handbook-artifact"><div><span>${esc(artifact.version || 'Draft')}</span><strong>${esc(source.data_source_name || 'Generated handbook')}</strong></div><details data-detail-key="${esc(scope)}:handbook"><summary>View handbook instructions</summary><pre>${esc(source.handbook || '')}</pre></details><details data-detail-key="${esc(scope)}:code"><summary>View sample retrieval code</summary><pre><code class="language-python">${highlightPython(source.code_example)}</code></pre></details></div>`);
        }
        const mechanism = artifact.mechanism;
        if (mechanism) {
            parts.push(`<div class="exp2-artifact-facts"><div><span>Selected service type</span><strong>${esc(MECHANISMS[mechanism]?.label || mechanism)}</strong></div></div>${artifact.why ? `<p class="exp2-field-note">${esc(artifact.why)}</p>` : ''}`);
        }
        const result = artifact.execution_result;
        if (result) {
            // A stage keeps its own copy of the result from the moment the
            // pipeline ran. Every edit, revision, version and re-run writes to
            // the live test_run record instead, so rendering that copy showed
            // the code as first generated -- no versions, no hand edits -- while
            // the card's own buttons were acting on something else entirely.
            // The live record wins whenever this stage has one behind it.
            const stageRunId = runIdForStage(scope);
            const liveRun = (state.session.test_runs || []).find(
                (candidate) => candidate.id === stageRunId);
            parts.push(retrievalFactsCard(
                (liveRun && liveRun.result) || result, scope, stageRunId));
        }
        const analysis = artifact.analysis;
        if (analysis && Object.hasOwn(analysis, 'task_completed')) {
            parts.push(semanticValidationArtifact(analysis));
        } else if (analysis && Object.hasOwn(analysis, 'deficiencies')) {
            parts.push(traceAnalysisArtifact(analysis));
        }
        return parts.length ? `<div class="exp2-stage-artifact">${parts.join('')}</div>` : '';
    }

    // The control arm is asked a different question (external vs. a gap in the
    // model's own knowledge of the source), so the same record must not be
    // labelled "Handbook gap" there -- that was the mislabel that made every
    // control failure read as a handbook problem. `condition` is written by
    // analyze_execution_trace; records predating it are treatment records.
    const analysisIsControl = (analysis) => analysis?.condition === 'control';

    const FAILURE_CATEGORY_LABELS = {
        handbook_deficiency: 'Handbook deficiency',
        knowledge_gap: 'Model knowledge gap',
        external: 'External failure',
        request_infeasible: 'Request outside source coverage',
        Unknown: 'Not categorised',
    };

    function traceAnalysisArtifact(analysis) {
        const deficiencies = analysis.deficiencies || [];
        return `<div class="exp2-analysis-artifact exp2-trace-artifact">
            <div class="exp2-analysis-summary ${deficiencies.length ? 'failed' : ''}"><span>${esc(FAILURE_CATEGORY_LABELS[analysis.failure_category] || analysis.failure_category || 'Failure analysis')}</span><p>${esc(analysis.summary || 'No trace analysis summary was recorded.')}</p></div>
            ${deficiencies.length ? `<div class="exp2-semantic-checks">${deficiencies.map(item => `<article class="fail"><span>${analysisIsControl(analysis) ? 'Knowledge gap' : 'Handbook gap'}</span><strong>${esc(item.handbook_gap || 'Unspecified deficiency')}</strong><p>${esc(item.evidence || 'No supporting evidence recorded.')}</p>${item.recommended_change ? `<small>${analysisIsControl(analysis) ? 'Correct fact that would have avoided it' : 'Recommended fix'}: ${esc(item.recommended_change)}</small>` : ''}</article>`).join('')}</div>` : `<p class="exp2-field-note">${analysisIsControl(analysis)
                ? 'No knowledge gap was identified — this looks like an external failure (network, credentials, or platform issue) rather than the model misunderstanding the source.'
                : 'No refinable handbook deficiency was identified — this looks like an external failure (network, credentials, or platform issue) rather than a handbook gap.'}</p>`}
        </div>`;
    }

    function semanticValidationArtifact(analysis) {
        const verdict = recordVerdict(analysis);
        const fact = (label, value) => `<div><span>${esc(label)}</span><strong class="${verdictClass(value)}">${esc(verdictLabel(value))}</strong></div>`;
        const heading = verdict === 'pass' ? 'Task completed'
            : verdict === 'undecidable' ? 'Undecidable — needs human adjudication'
            : esc(analysis.failure_category || 'Validation did not pass');
        const method = VALIDATION_METHOD_LABELS[analysis.method] || 'Validation';
        const meta = [
            method,
            analysis.rater ? `rater ${analysis.rater}` : '',
            analysis.blinded === false ? 'NOT blinded' : (analysis.blinded === true ? 'blinded' : ''),
            Number.isFinite(analysis.elapsed_seconds) ? `${Math.round(analysis.elapsed_seconds)}s` : '',
            analysis.spec_declared === false && analysis.method === 'mechanical' ? 'no spec declared' : '',
            analysis.created_at ? new Date(analysis.created_at).toLocaleString() : '',
        ].filter(Boolean).join(' · ');
        return `<div class="exp2-analysis-artifact exp2-semantic-artifact">
            <div class="exp2-analysis-summary ${verdict === 'pass' ? 'passed' : verdict === 'undecidable' ? '' : 'failed'}"><span>${heading}</span><p>${esc(analysis.summary || 'No summary was recorded.')}</p><small>${esc(meta)}${analysis.method === 'llm' ? ` · confidence ${Math.round(Number(analysis.confidence || 0) * 100)}%` : ''}</small></div>
            <div class="exp2-artifact-facts">
                ${VALIDATION_DIMENSIONS.map(([key, label]) => fact(label, analysis[key])).join('')}
            </div>
            ${(analysis.checks || []).length ? `<div class="exp2-semantic-checks">${analysis.checks.map(check => `<article class="${esc(check.status || 'unknown')}"><span>${esc(verdictLabel(check.status))}</span><strong>${esc(check.name || 'Validation check')}</strong><p>${esc(check.evidence || 'No supporting evidence recorded.')}</p></article>`).join('')}</div>` : ''}
            ${analysis.handbook_gap ? `<div class="exp2-handbook-gap"><span>Possible handbook deficiency</span><strong>${esc(analysis.handbook_gap)}</strong><small>${analysis.should_refine_handbook ? 'Eligible for refinement' : 'Recorded for review; refinement was not recommended'}</small></div>` : ''}
        </div>`;
    }

    // Rendered inline inside the "Review the handbook" pipeline stage card
    // (see pipelineStageCard's extraBody) so the editable handbook lives
    // under that stage's own collapse/expand control rather than as a
    // separate floating section. wrap=true keeps a standalone `.exp2-card`
    // usable as a fallback for the rare session with no "review" stage
    // recorded (e.g. hand-edited or pre-migration data).
    function reviewHandbookCard(source, session, canEdit, wrap = false) {
        // A control session has no handbook, so it gets no handbook editor.
        // Rendering the usual fields empty would invite editing them, and a
        // handbook typed in here would silently end the ablation.
        if (session.use_handbook === false) {
            const body = `
            <div class="exp2-card-head"><div><h4>Control condition</h4>
            <p>This session runs without a handbook.</p></div></div>
            <div class="exp2-form-grid two">
                <label class="wide"><span>Data source</span><input id="hbs-name" value="${esc(source.data_source_name || '')}" ${canEdit ? '' : 'disabled'}></label>
                <div class="exp2-field-note wide">The model writes the retrieval code from its own knowledge of this source. If the code needs a credential, the run pauses and asks for it.</div>
            </div>`;
            return wrap ? `<section class="exp2-card">${body}</section>` : body;
        }
        const mechLabel = MECHANISMS[session.access_mechanism]?.label || session.access_mechanism || 'Not selected';
        const requiresKey = String(source.requires_key || '').toLowerCase() === 'true';
        const dis = canEdit ? '' : 'disabled';
        const panel = state.reviewPanel || 'instructions';
        const editingInstructions = state.editingInstructions && canEdit;
        const lines = (text) => String(text || '').split('\n').length;
        const words = (text) => (String(text || '').trim().match(/\S+/g) || []).length;
        const tab = (key, label, count) => `<button type="button" class="${panel === key ? 'active' : ''}" data-review-tab="${key}" onclick="HandbookStudioMode.showReviewPanel('${key}')">${label}<small>${esc(count)}</small></button>`;
        const body = `
            <div class="hbs-review-head">
                <div class="hbs-review-title">
                    <span class="hbs-review-eyebrow">Generated handbook</span>
                    <input id="hbs-name" class="hbs-review-name" value="${esc(source.data_source_name || '')}" placeholder="Handbook name" aria-label="Handbook name" title="${esc(source.data_source_name || '')}" ${dis}>
                </div>
                <div class="hbs-review-meta">
                    <span title="${session.access_mechanism_source === 'auto' ? esc(session.access_mechanism_why || 'Auto-selected by the model') : 'Selected on Setup'}"><small>Service type</small><strong>${esc(mechLabel)}</strong></span>
                    <span><small>Credentials</small><strong>${requiresKey ? esc(source.key_name || 'Required') : 'None required'}</strong></span>
                    <span><small>Editing</small><strong>${canEdit ? 'Enabled' : 'Locked while running'}</strong></span>
                </div>
            </div>
            <nav class="hbs-review-tabs" aria-label="Handbook sections">
                ${tab('description', 'Description', `${words(source.brief_description)} words`)}
                ${tab('instructions', 'Instructions', `${words(source.handbook)} words`)}
                ${tab('code', 'Sample retrieval code', `${lines(source.code_example)} lines`)}
                ${tab('caveats', 'Caveats', source.caveats ? `${words(source.caveats)} words` : 'none')}
            </nav>
            <div class="hbs-review-panels">
                <section data-review-panel="description" ${panel === 'description' ? '' : 'hidden'}>
                    <textarea id="hbs-desc" class="exp2-prose-field" rows="5" placeholder="A short description of the source and what the handbook retrieves" aria-label="Description" ${dis}>${esc(source.brief_description || '')}</textarea>
                </section>
                <section data-review-panel="instructions" ${panel === 'instructions' ? '' : 'hidden'}>
                    <div class="hbs-review-panel-bar">
                        <span data-instruction-count>${instructionCountLabel(source.handbook)}</span>
                        ${canEdit ? `<button type="button" class="exp2-btn tiny secondary" data-instruction-edit onclick="HandbookStudioMode.toggleInstructionEdit()">${editingInstructions ? 'Done editing' : 'Edit'}</button>` : ''}
                    </div>
                    <ol class="hbs-instruction-list" data-instruction-list ${editingInstructions ? 'hidden' : ''}>${instructionItems(source.handbook)}</ol>
                    <textarea id="hbs-handbook" class="exp2-prose-field" rows="18" aria-label="Handbook instructions" ${dis} ${editingInstructions ? '' : 'hidden'}>${esc(source.handbook || '')}</textarea>
                    ${editingInstructions ? '<p class="hbs-review-foot">One instruction per line — each line becomes its own numbered step.</p>' : ''}
                </section>
                <section data-review-panel="code" ${panel === 'code' ? '' : 'hidden'}>
                    <textarea id="hbs-code" class="exp2-code-field" rows="18" spellcheck="false" wrap="off" aria-label="Sample retrieval code" ${dis}>${esc(source.code_example || '')}</textarea>
                </section>
                <section data-review-panel="caveats" ${panel === 'caveats' ? '' : 'hidden'}>
                    <textarea id="hbs-caveats" rows="8" placeholder="Known limitations, rate limits, quirks of this source…" aria-label="Caveats" ${dis}>${esc(source.caveats || '')}</textarea>
                </section>
            </div>
            <p class="hbs-review-foot">Edit any section before testing — changes are kept with the session and used by every test run that follows.</p>`;
        return wrap ? `<section class="exp2-card hbs-review">${body}</section>`
            : `<div class="exp2-stage-handbook-review hbs-review">${body}</div>`;
    }

    // The handbook's instructions are one per line; shown as a numbered
    // list so each reads as its own step instead of wrapped prose.
    function instructionLines(text) {
        return String(text || '').split('\n').map(l => l.trim()).filter(Boolean);
    }
    function instructionItems(text) {
        const items = instructionLines(text);
        return items.length
            ? items.map(line => `<li>${esc(line)}</li>`).join('')
            : '<li class="empty">No instructions yet — click Edit to write them.</li>';
    }
    function instructionCountLabel(text) {
        const n = instructionLines(text).length;
        return `${n} instruction${n === 1 ? '' : 's'}`;
    }

    // Swaps the Instructions panel between the numbered read view and the
    // textarea, in place; the list is rebuilt from the textarea on the way
    // back so edits show up immediately.
    function toggleInstructionEdit() {
        const root = document.querySelector('.hbs-review [data-review-panel="instructions"]');
        if (!root) return;
        const editing = !state.editingInstructions;
        state.editingInstructions = editing;
        const textarea = root.querySelector('#hbs-handbook');
        const list = root.querySelector('[data-instruction-list]');
        const button = root.querySelector('[data-instruction-edit]');
        const count = root.querySelector('[data-instruction-count]');
        if (!editing && textarea && list) {
            list.innerHTML = instructionItems(textarea.value);
            if (count) count.textContent = instructionCountLabel(textarea.value);
        }
        if (textarea) textarea.hidden = !editing;
        if (list) list.hidden = editing;
        if (button) button.textContent = editing ? 'Done editing' : 'Edit';
        root.querySelectorAll('.hbs-review-foot').forEach(el => el.remove());
        if (editing) {
            root.insertAdjacentHTML('beforeend', '<p class="hbs-review-foot">One instruction per line — each line becomes its own numbered step.</p>');
            textarea?.focus();
        }
    }

    // Switches the handbook editor between Instructions / Sample code /
    // Caveats without re-rendering (a full render would drop the cursor).
    function showReviewPanel(key) {
        state.reviewPanel = key;
        document.querySelectorAll('.hbs-review [data-review-panel]').forEach((el) => {
            el.hidden = el.dataset.reviewPanel !== key;
        });
        document.querySelectorAll('.hbs-review [data-review-tab]').forEach((el) => {
            el.classList.toggle('active', el.dataset.reviewTab === key);
        });
    }

    // Rendered inline inside the synthetic "Retrieval question" stage card
    // (see withRetrievalStage / pipelineStageCard's extraBody) — kept as its
    // own stage, separate from "Review the handbook", so the task and the
    // "Test handbook" action can be collapsed/reviewed independently of the
    // (often much longer) handbook text above it.
    function retrievalQuestionCard(session, canEdit, wrap = false) {
        const pipeline = session.pipeline_state || {};
        const requiredKeys = [...new Set([...(pipeline.required_keys || []), ...state.requiredKeys])];
        const needsCredentials = requiredKeys.length > 0;
        const body = `
            <div class="exp2-form-grid two">
                <label class="wide"><span>Retrieval task</span><textarea id="hbs-task" rows="4" ${canEdit ? '' : 'disabled'}>${esc(session.retrieval_task || '')}</textarea></label>
            </div>
            ${needsCredentials ? `<section class="exp2-credentials exp2-pipeline-credentials"><div><strong>Credentials needed to continue</strong><small>Used for this run only; never saved.</small></div><div class="exp2-credential-fields">${requiredKeys.map(name => `<label><span>${esc(name)}</span><input type="password" autocomplete="off" data-key="${esc(name)}" value="${esc(state.runKeys[name] || '')}" oninput="HandbookStudioMode.setRunKey(this.dataset.key, this.value)" placeholder="Enter credential"></label>`).join('')}</div></section>` : ''}
            <div class="exp2-page-footer">
                <button class="exp2-btn primary large" ${canEdit ? '' : 'disabled'} onclick="HandbookStudioMode.startTest()">${needsCredentials ? 'Continue' : (session.use_handbook === false
                    ? `Run retrieval${(session.test_runs || []).length ? ' again' : ''} (no handbook)`
                    : `Test handbook${(session.test_runs || []).length ? ' again' : ''}`)}</button>
                ${needsCredentials ? '' : (session.use_handbook === false
                    ? '<small class="exp2-field-note">Control condition: the model writes the retrieval code without a handbook. No refinement is attempted. If the code needs a credential, the run pauses and asks for it.</small>'
                    : '<small class="exp2-field-note">If this fails for a fixable handbook reason, it will automatically revise the handbook (H1, H2, H3) and retry — up to 3 revisions — before stopping.</small>')}
            </div>`;
        return wrap ? `<section class="exp2-card"><div class="exp2-card-head"><div><h4>Retrieval question</h4><p>What should be downloaded when this handbook is tested.</p></div></div>${body}</section>`
            : `<div class="exp2-stage-handbook-review">${body}</div>`;
    }

    // The runner records "Initial handbook draft" and "Self-verification" as
    // two stages; they read better as one card, so they are folded together
    // here for display only. The saved pipeline state is untouched. The
    // merged card carries both keys in data-stage-keys so live updates and
    // "jump to stage" for either underlying stage still land on it.
    function mergeDraftAndVerification(stages) {
        const draftIndex = stages.findIndex(s => s.key === 'draft');
        if (draftIndex === -1) return stages;
        const draft = stages[draftIndex];
        const verifyIndex = stages.findIndex(s => s.key === 'verification');
        const verify = verifyIndex === -1 ? null : stages[verifyIndex];
        const active = verify && verify.status === 'running' ? verify
            : draft.status === 'running' ? draft
            : (verify || draft);
        const merged = {
            ...active,
            key: 'draft',
            keys: ['draft', 'verification'],
            label: 'Handbook draft & self-verification',
            description: 'Create a handbook grounded in the research, then run the sample-download verification.',
            showActivity: true,
            details: [...(draft.details || []), ...(verify?.details || [])],
            artifact: { ...(draft.artifact || {}), ...(verify?.artifact || {}) },
            started_at: draft.started_at || active.started_at,
            finished_at: verify ? verify.finished_at : draft.finished_at,
        };
        const copy = stages.slice();
        copy[draftIndex] = merged;
        if (verifyIndex !== -1) copy.splice(verifyIndex, 1);
        return copy;
    }

    // "Retrieval question" isn't a real pipeline stage — the server only
    // records what the generation/test pipeline itself does — but it reads
    // as one to the person reviewing: its own numbered card, collapsible
    // independently of "Review the handbook". Spliced in right after the
    // real "review" stage (not appended at the end) so it stays there even
    // once test-run stages get appended to the real array as tests execute.
    function withRetrievalStage(stages) {
        const reviewIndex = stages.findIndex(s => s.key === 'review');
        if (reviewIndex === -1) return stages;
        const synthetic = {
            key: '__retrieval_question',
            label: 'Retrieval question',
            description: 'What should be downloaded when this handbook is tested.',
            status: 'warning',
        };
        const copy = stages.slice();
        copy.splice(reviewIndex + 1, 0, synthetic);
        return copy;
    }

    function testHistorySection(session) {
        const runs = session.test_runs || [];
        if (!runs.length) return '';
        const ordered = [...runs].reverse();
        return `<section class="exp2-card">
            <div class="exp2-card-head"><div><h4>Test history</h4><p>Every retrieval test run against this handbook, most recent first.</p></div><span class="exp2-count">${runs.length} test${runs.length === 1 ? '' : 's'}</span></div>
            <div class="exp2-refinement-list">${ordered.map((run) => `
                <details class="exp2-stage-log" data-detail-key="test-history:${esc(run.id)}">
                    <summary>Test #${esc(run.test_number)} · <strong class="${run.outcome === 'passed' ? 'good' : 'bad'}">${esc(run.outcome === 'passed' ? 'Passed' : 'Not passed')}</strong> · ${esc(new Date(run.finished_at || run.started_at).toLocaleString())}</summary>
                    <div class="exp2-field-note">${esc(run.retrieval_task || '')}</div>
                    ${retrievalFactsCard(run.result, `test-history:${run.id}`, run.id)}
                </details>`).join('')}</div>
        </section>`;
    }

    // Mirrors testHistorySection but for correctness judgments (File
    // validation, Task completion, Mechanism, Parameters, Spatial query,
    // Temporal query) instead of raw retrieval facts — the Validation tab's
    // per-run history.
    function validationHistorySection(session) {
        const runs = session.test_runs || [];
        if (!runs.length) return '';
        const ordered = [...runs].reverse();
        return `<section class="exp2-card">
            <div class="exp2-card-head"><div><h4>Validation history</h4><p>Correctness evaluation for every retrieval test run, most recent first.</p></div><span class="exp2-count">${runs.length} test${runs.length === 1 ? '' : 's'}</span></div>
            <div class="exp2-refinement-list">${ordered.map((run) => {
                const sem = run.result?.semantic_validation;
                const summaryLabel = sem
                    ? `<strong class="${verdictClass(recordVerdict(sem))}">${esc(verdictLabel(recordVerdict(sem)))}</strong>`
                    : '<strong class="exp2-tag neutral" style="display:inline">Not yet validated</strong>';
                return `
                <details class="exp2-stage-log" data-detail-key="validation-history:${esc(run.id)}">
                    <summary>Test #${esc(run.test_number)} · ${summaryLabel} · ${esc(new Date(run.finished_at || run.started_at).toLocaleString())}</summary>
                    <div class="exp2-field-note">${esc(run.retrieval_task || '')}</div>
                    ${validationFactsCard(run.result, `validation-history:${run.id}`, run.id)}
                    ${validationControls(run)}
                </details>`;
            }).join('')}</div>
        </section>`;
    }

    // Rate over test runs where `field` reached a determinate pass/fail
    // verdict (excludes 'unknown'/'not_applicable', which aren't correctness
    // data).
    function correctnessRate(runs, field) {
        const determinate = runs.filter(r => {
            const v = (r.result?.semantic_validation || {})[field];
            return v === 'pass' || v === 'fail';
        });
        if (!determinate.length) return null;
        return Math.round(determinate.filter(r => r.result.semantic_validation[field] === 'pass').length / determinate.length * 100);
    }

    // Returns {rate, count, total} rather than a bare percentage — with at
    // most 4 runs per session (H0-H3), a lone "75%" reads as more
    // statistically substantial than "3 of 4" really is, so the raw count
    // travels with the rate everywhere it's displayed.
    // Only runs with a DETERMINATE verdict form the denominator: an
    // "undecidable" run has not been measured, and folding it in either
    // direction would misstate the completion rate. It's reported separately
    // as the adjudication backlog.
    function taskCompletionRate(runs) {
        const evaluated = runs.filter(r => ['pass', 'fail'].includes(recordVerdict(r.result?.semantic_validation)));
        const total = evaluated.length;
        const count = evaluated.filter(r => recordVerdict(r.result.semantic_validation) === 'pass').length;
        const undecided = runs.filter(r => recordVerdict(r.result?.semantic_validation) === 'undecidable').length;
        return { rate: total ? Math.round(count / total * 100) : null, count, total, undecided };
    }

    // Three-stage funnel mirroring the manuscript's L3 (execution) / L4
    // (retrieval: valid output, then does it answer the request) framework.
    // Each stage requires the previous one, so drop-off is directly readable:
    // (1) did the code run at all, (2) of that, was non-empty/correctly-
    // formatted output produced, (3) of that, did it actually answer the task.
    function evaluationStage(result) {
        const execution = result?.execution || {};
        const validation = result?.validation || {};
        const semantic = result?.semantic_validation;
        if (!execution.success) {
            return { key: 'execution_failed', label: 'Execution failed' };
        }
        if (!(validation.output_present && validation.format_match)) {
            return { key: 'no_valid_output', label: 'No valid output' };
        }
        if (semantic && semantic.task_completed === false) {
            return { key: 'output_incomplete', label: 'Ran, wrong output' };
        }
        if (semantic && semantic.task_completed === true) {
            return { key: 'passed', label: 'Passed' };
        }
        // Structurally valid output but semantic validation didn't run/
        // complete — shouldn't normally happen once a test finishes, but
        // handle gracefully rather than mislabeling it as passed or failed.
        return { key: 'unvalidated', label: 'Output produced (unvalidated)' };
    }

    function executionSuccessRate(runs) {
        const total = runs.length;
        const count = runs.filter(r => (r.result?.execution || {}).success).length;
        return { rate: total ? Math.round(count / total * 100) : null, count, total };
    }

    function validOutputRate(runs) {
        const ranOk = runs.filter(r => (r.result?.execution || {}).success);
        const total = ranOk.length;
        const count = ranOk.filter(r => {
            const v = r.result?.validation || {};
            return v.output_present && v.format_match;
        }).length;
        return { rate: total ? Math.round(count / total * 100) : null, count, total };
    }

    function formatRate(value) {
        return value === null ? '—' : `${value}%`;
    }

    // Renders a funnel stat card with its "count of total" alongside the
    // rate, so the small sample size (at most 4 runs per session) is always
    // visible next to the percentage rather than requiring a cross-reference
    // to "Tests run".
    function funnelStatCard(label, stat, hint) {
        if (!stat || !stat.total) return statCard(label, '—', hint);
        return statCard(label, `${stat.rate}%`, `${stat.count} of ${stat.total} — ${hint}`);
    }

    // What actually stopped a non-converged session, described in terms of
    // the funnel stage its LAST attempt reached — so "0 revisions helped"
    // reads very differently from "execution never even succeeded."
    const _STAGE_STOP_HINTS = {
        no_valid_output: (h) => `Stopped at H${h} — execution succeeded, but never produced valid output.`,
        output_incomplete: (h) => `Stopped at H${h} — execution and output succeeded every attempt; only task correctness failed.`,
        unvalidated: (h) => `Stopped at H${h} before validation could complete.`,
    };

    // A coarse signature for "is this the same underlying failure as
    // another attempt" — mirrors WebUI/handbook_studio_runner.py's
    // _failure_signature (strip the volatile URL, keep
    // ExceptionType:HTTPStatus), kept as a parallel copy so the Results tab
    // can recompute this from persisted test_runs after a reload, not just
    // from the live stream event.
    function _failureSignature(result) {
        const err = String((result && result.error) || '').trim();
        if (!err) return null;
        const lines = err.split(/\r?\n/);
        let lastLine = lines[lines.length - 1] || '';
        lastLine = lastLine.split(/\s+for url:/)[0];
        const match = lastLine.match(/^([\w.]+(?:Error|Exception)):\s*(.*)$/);
        if (!match) return lastLine;
        const [, excType, detail] = match;
        const statusMatch = detail.match(/^(\d{3})\b/);
        return statusMatch ? `${excType}:${statusMatch[1]}` : excType;
    }

    // Refinement is a sequential, self-terminating chain (stops at the first
    // pass, or at the revision cap) — not independent parallel trials. So
    // "passed / total" is a poor summary: it always has numerator 0 or 1,
    // and penalizes a case that converged slowly as if it were "worse" than
    // one that never needed refinement at all. This reports the two things
    // that actually matter instead: did it ever converge, and how many
    // revisions that took (or were spent before giving up).
    function convergenceSummary(runs) {
        if (!runs.length) {
            return { converged: null, hint: 'No tests run yet.', revisions: null,
                     revisionsHint: 'No tests run yet.' };
        }
        const convergedRun = runs.find(r => r.outcome === 'passed') || null;
        const lastRun = runs[runs.length - 1];
        const finalRun = convergedRun || lastRun;
        const n = finalRun.refinement_iteration ?? 0;
        if (convergedRun) {
            return {
                converged: true,
                hint: n === 0
                    ? 'Passed on the first attempt (H0) — no revisions needed.'
                    : `Passed after ${n} revision${n === 1 ? '' : 's'} (H${n}).`,
                revisions: n,
                revisionsHint: 'Handbook revisions needed before a test passed',
            };
        }
        // An "external" failure_category means the automatic-refinement loop
        // itself stopped without spending a revision (WebUI/
        // handbook_studio_runner.py retries the same handbook a bounded
        // number of times first) — so this is a different story than
        // "still broken after N revisions": it's an environment/network
        // hiccup at test time, not an unresolved handbook defect.
        if (lastRun.result?.failure_category === 'external') {
            return {
                converged: false,
                external: true,
                hint: `Stopped at H${n} — the last failure looks external `
                    + '(network or server-side), not an unresolved handbook '
                    + 'defect. Refinement retried the same handbook instead '
                    + 'of spending a revision on it; try testing again later.',
                revisions: n,
                revisionsHint: 'Handbook revisions attempted before stopping, still not passing',
            };
        }
        const stage = evaluationStage(lastRun.result);
        // execution_failed gets its own case (rather than a plain
        // _STAGE_STOP_HINTS entry): whether the revisions were fixing a
        // distinct bug each time (real progress, just out of budget) or
        // stuck on the same recurring failure changes whether raising
        // max_refinements would actually help this case.
        if (stage.key === 'execution_failed') {
            const failedSigs = runs
                .filter(r => r.outcome !== 'passed')
                .map(r => _failureSignature(r.result))
                .filter(Boolean);
            const distinctFailures = new Set(failedSigs).size;
            let note = '';
            if (failedSigs.length >= 2) {
                note = distinctFailures >= 2
                    ? ` ${distinctFailures} distinct failures were seen across attempts, each fixed before the next appeared — a higher refinement limit might help.`
                    : ' The same underlying failure recurred across attempts — a higher refinement limit likely would not help.';
            }
            return {
                converged: false,
                distinctFailures,
                hint: `Stopped at H${n} — execution kept failing.${note}`,
                revisions: n,
                revisionsHint: 'Handbook revisions attempted before stopping, still not passing',
            };
        }
        const hintFn = _STAGE_STOP_HINTS[stage.key];
        return {
            converged: false,
            hint: hintFn ? hintFn(n)
                : `Did not converge within the refinement budget (stopped at H${n}).`,
            revisions: n,
            revisionsHint: 'Handbook revisions attempted before stopping, still not passing',
        };
    }

    function renderResults(main) {
        const session = state.session;
        const runs = session.test_runs || [];
        const convergence = convergenceSummary(runs);
        const execRate = executionSuccessRate(runs);
        const outputRate = validOutputRate(runs);
        const mechLabel = MECHANISMS[session.access_mechanism]?.label || session.access_mechanism || 'Not selected';

        main.innerHTML = `<div class="exp2-page">
            <section class="exp2-page-heading hbs-heading-bare">
                <div class="exp2-inline-actions">
                    <button class="exp2-btn secondary" onclick="HandbookStudioMode.exportCSV()">Export tests CSV</button>
                    <button class="exp2-btn secondary" onclick="HandbookStudioMode.exportValidationMatrix()" title="Long-format run × method × dimension matrix — the input for agreement statistics">Export validation matrix</button>
                    <button class="exp2-btn primary" onclick="HandbookStudioMode.exportJSON()">Export session JSON</button>
                </div>
            </section>
            <div class="exp2-stat-grid">
                ${statCard('Converged', convergence.converged === null ? '—' : (convergence.converged ? 'Yes' : 'No'), convergence.hint)}
                ${statCard('Revisions to converge', convergence.revisions === null ? '—' : convergence.revisions, convergence.revisionsHint)}
                ${statCard('Service type', esc(mechLabel), session.access_mechanism_source === 'auto' ? 'Auto-selected' : 'Selected by you')}
                ${statCard('Tests run', runs.length, 'Retrieval attempts against this handbook')}
            </div>
            <div class="exp2-stat-grid">
                ${funnelStatCard('1. Execution success', execRate, 'code ran without crashing')}
                ${funnelStatCard('2. Valid output', outputRate, 'non-empty, correctly-formatted output')}
            </div>
            ${usageCard(session)}
            <section class="exp2-card">
                <div class="exp2-card-head"><div><h4>Test-by-test comparison</h4><p>Every run against this handbook, in order.</p></div></div>
                ${resultComparisonTable(runs)}
            </section>
            ${!runs.length ? '<section class="exp2-notice neutral"><strong>No tests yet</strong><p>Run "Test handbook" from the Generate &amp; retrieve tab to populate the session report.</p></section>' : testHistorySection(session)}
        </div>`;
    }


    // What this handbook cost to produce. The framework's claim is that
    // automated onboarding is cheaper than an expert doing it by hand, so the
    // token bill is the measurement that claim rests on -- and it can only be
    // captured while the calls happen, never reconstructed afterwards.
    function usageCard(session) {
        const usage = session.usage || {};
        const total = usage.total || {};
        if (!total.calls) {
            return `<section class="exp2-card">
                <div class="exp2-card-head"><div><h4>Token cost</h4>
                <p>Recorded per LLM call while the pipeline runs.</p></div></div>
                <div class="exp2-card-body"><p class="exp2-field-note">
                No usage recorded for this session. Runs made before token accounting
                was added have none, and it cannot be recovered — providers expose no
                way to look up historical usage. Generate or test again to capture it.
                </p>
                <p class="exp2-field-note">Configured for
                    <b>${esc(session.handbook_gen_model || '—')}</b> (handbook generation) and
                    <b>${esc(session.data_retrieval_model || '—')}</b> (data retrieval) on
                    ${esc(session.provider === 'claude' ? 'Claude Agent SDK' : 'OpenAI')}.
                    That is the configuration, not a record of what any past run actually
                    called.</p></div></section>`;
        }
        const fmt = (n) => (n || 0).toLocaleString();
        const money = (v) => `$${(v || 0).toFixed(4)}`;
        const phases = Object.entries(total.by_phase || {});
        const models = Object.entries(total.by_model || {});
        // Sessions recorded before merge() carried elapsed time forward still
        // have it on each phase entry, so the total is recoverable by summing
        // them rather than showing a dash for runs that did measure it.
        const elapsed = total.elapsed_seconds
            ?? (usage.runs || []).reduce((sum, run) => sum + (run.elapsed_seconds || 0), 0);
        // A duplicated session reuses a handbook someone else's session paid
        // to generate. Those tokens are shown here so this handbook's cost is
        // visible, but they belong to ONE generation -- flagged, so nobody
        // adds a parent and its duplicates together and counts it twice.
        const carried = (usage.runs || []).filter((run) => run.inherited);
        const phaseInherited = (name) => {
            const runs = (usage.runs || []).filter((run) => (run.phase || '') === name);
            return runs.length > 0 && runs.every((run) => run.inherited);
        };
        return `<section class="exp2-card">
            <div class="exp2-card-head"><div><h4>Token cost</h4>
            <p>Every LLM call this handbook required, generation and retrieval together.</p></div></div>
            <div class="exp2-card-body">
                <div class="hbs-stats">
                    <div><span>Total tokens</span><b>${fmt(total.total_tokens)}</b></div>
                    <div><span>Input</span><b>${fmt(total.input_tokens)}</b></div>
                    <div><span>Output</span><b>${fmt(total.output_tokens)}</b></div>
                    ${total.reasoning_tokens ? `<div><span>Reasoning</span><b>${fmt(total.reasoning_tokens)}</b></div>` : ''}
                    <div><span>LLM calls</span><b>${fmt(total.calls)}</b></div>
                    <div><span>Cost</span><b>${usage.priced || total.cost_usd ? money(total.cost_usd) : '—'}</b></div>
                    <div><span>Wall-clock time</span><b>${secs(elapsed)}</b>${
                        elapsed >= 90 ? `<span class="hbs-stat-note">${clock(elapsed)}</span>` : ''}</div>
                </div>
                ${models.length ? `<div class="hbs-stats">
                    <div class="wide"><span>Model${models.length === 1 ? '' : 's'} used</span><b>${
                        models.map(([name]) => esc(name)).join(', ')}</b>
                        <!--<span class="hbs-stat-note">what the calls actually ran on, not what the session was configured with</span>--></div>
                </div>` : ''}
                <!--<p class="exp2-field-note">Wall-clock time is how long the phases took end to end —
                    LLM calls plus everything between them, including running the generated code and
                    downloading data. It is not the sum of the API calls' own latency.</p>-->
                ${carried.length ? `<p class="exp2-field-note">
                    <b>Generation was inherited.</b> This session reuses a handbook generated in
                    an earlier session, so that phase's tokens and time are carried over rather
                    than re-spent — which is what makes the handbook's real cost visible here.
                    They describe <b>one</b> generation: when aggregating across a session and its
                    duplicates, count generation once per handbook, not once per session.
                    Retrieval below was spent by this session.</p>` : ''}
                <!--${!usage.priced && !total.cost_usd ? `<p class="exp2-field-note">
                    Tokens are recorded; cost is not priced. Set <code>HANDBOOK_MODEL_PRICES</code>
                    to your per-model rates, e.g.
                    <code>{"gpt-5.2": {"input": 1.25, "output": 10.0}}</code> in USD per 1M tokens.
                    Rates are configuration rather than a constant in the code, so a stale figure
                    never turns into a confidently wrong cost.</p>` : ''}
                ${total.cost_is_partial ? `<p class="exp2-field-note bad">
                    Some models have no rate, so this cost is a lower bound.</p>` : ''}-->
                ${total.calls_missing_usage ? `<p class="exp2-field-note bad">
                    ${fmt(total.calls_missing_usage)} of ${fmt(total.calls)} calls reported no usage
                    (older backends do not return it), so the totals are a lower bound too.</p>` : ''}
                ${phases.length ? `<div class="hbs-outputs-head"><h5>By phase</h5><span>where the tokens went</span></div>
                <div class="hbs-tblwrap"><table class="hbs-table">
                    <thead><tr><th>Phase</th><th>Calls</th><th>Input</th><th>Output</th><th>Total</th><th>Time</th><th>Cost</th></tr></thead>
                    <tbody>${phases.map(([name, b]) => `<tr><td>${esc(name)}${
                        phaseInherited(name) ? ' <em>(inherited)</em>' : ''}</td><td>${fmt(b.calls)}</td>
                        <td>${fmt(b.input_tokens)}</td><td>${fmt(b.output_tokens)}</td>
                        <td>${fmt(b.total_tokens)}</td><td>${b.elapsed_seconds ? secs(b.elapsed_seconds) : '—'}</td>
                        <td>${b.cost_usd ? money(b.cost_usd) : '—'}</td></tr>`).join('')}
                    </tbody></table></div>` : ''}
                ${models.length ? `<div class="hbs-outputs-head"><h5>By model</h5><span>${
                    models.length === 1 ? 'every call ran on one model' : `${models.length} models`}</span></div>
                <div class="hbs-tblwrap"><table class="hbs-table">
                    <thead><tr><th>Model</th><th>Calls</th><th>Total tokens</th><th>Cost</th></tr></thead>
                    <tbody>${models.map(([name, b]) => `<tr><td>${esc(name)}</td><td>${fmt(b.calls)}</td>
                        <td>${fmt(b.total_tokens)}</td><td>${b.cost_usd ? money(b.cost_usd) : '—'}</td></tr>`).join('')}
                    </tbody></table></div>` : ''}
            </div></section>`;
    }

    function renderValidation(main) {
        const session = state.session;
        const runs = session.test_runs || [];
        const taskRate = taskCompletionRate(runs);
        const paramRate = correctnessRate(runs, 'parameter_correctness');
        const spatialRate = correctnessRate(runs, 'spatial_query_correctness');
        const temporalRate = correctnessRate(runs, 'temporal_query_correctness');

        main.innerHTML = `<div class="exp2-page">
            <section class="exp2-page-heading">
                <div><span class="exp2-section-number">03</span><h3>Validation</h3><p>Correctness judgments across every retrieval test run — file validation, task completion, and mechanism/parameter/spatial/temporal correctness.</p></div>
            </section>
            <div class="exp2-stat-grid">
                ${funnelStatCard('3. Task completion', taskRate, 'output actually answered the request')}
                ${statCard('Parameter correctness', formatRate(paramRate), 'Correct endpoint/parameter usage, where evaluated')}
                ${statCard('Spatial correctness', formatRate(spatialRate), 'Correct spatial filtering, where applicable')}
                ${statCard('Temporal correctness', formatRate(temporalRate), 'Correct temporal filtering, where applicable')}
            </div>
            <section class="exp2-card">
                <div class="exp2-card-head"><div><h4>Validation comparison</h4><p>Every run against this handbook, in order.</p></div></div>
                ${validationComparisonTable(runs)}
            </section>
            ${!runs.length ? '<section class="exp2-notice neutral"><strong>No tests yet</strong><p>Run "Test handbook" from the Generate &amp; retrieve tab to populate validation.</p></section>' : validationHistorySection(session)}
        </div>`;
    }

    function resultComparisonTable(runs) {
        if (!runs.length) return '<div class="exp2-empty compact"><p>No completed tests yet.</p></div>';
        const stageTagClass = {
            execution_failed: 'bad', no_valid_output: 'bad',
            output_incomplete: 'neutral', unvalidated: 'neutral', passed: '',
        };
        const rows = runs.map(r => {
            const stage = evaluationStage(r.result);
            return `<tr>
                <td><strong>#${esc(r.test_number)}</strong></td>
                <td>${r.refinement_iteration === 0 ? 'H0 (baseline)' : `H${esc(r.refinement_iteration ?? '—')}`}</td>
                <td>${r.manual_intervention ? `<span class="exp2-tag neutral">${[r.handbook_changed ? 'Handbook' : '', r.task_changed ? 'Task' : ''].filter(Boolean).join(' + ')} edited</span>` : '—'}</td>
                <td><span class="exp2-tag ${stageTagClass[stage.key] || 'neutral'}">${esc(stage.label)}</span></td>
                <td>${esc(new Date(r.finished_at || r.started_at).toLocaleString())}</td>
            </tr>`;
        });
        return `<div class="exp2-table-wrap"><table class="exp2-table"><thead><tr><th>Test</th><th>Iteration</th><th>Manual edit</th><th>Outcome (1. Execution → 2. Output)</th><th>When</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
    }

    function validationComparisonTable(runs) {
        if (!runs.length) return '<div class="exp2-empty compact"><p>No completed tests yet.</p></div>';
        const fmtStatus = v => v === 'pass' ? 'Pass' : v === 'fail' ? 'Fail'
            : v === 'undecidable' ? 'Undecid.' : v === 'not_applicable' ? 'N/A' : '—';
        const rows = runs.map(r => {
            const sem = r.result?.semantic_validation;
            const failureCategory = r.result?.failure_category;
            return `<tr>
                <td><strong>#${esc(r.test_number)}</strong></td>
                <td>${r.refinement_iteration === 0 ? 'H0 (baseline)' : `H${esc(r.refinement_iteration ?? '—')}`}</td>
                <td>${sem ? `<span class="${verdictClass(recordVerdict(sem))}">${esc(verdictLabel(recordVerdict(sem)))}</span>` : (failureCategory ? esc(failureCategory) : '—')}</td>
                <td>${sem ? esc(VALIDATION_METHOD_LABELS[sem.method] || sem.method || '—') : '—'}</td>
                ${VALIDATION_DIMENSIONS.map(([key]) => `<td>${sem ? fmtStatus(sem[key]) : '—'}</td>`).join('')}
                <td>${esc(new Date(r.finished_at || r.started_at).toLocaleString())}</td>
            </tr>`;
        });
        return `<div class="exp2-table-wrap"><table class="exp2-table"><thead><tr><th>Test</th><th>Iteration</th><th>Verdict</th><th>Method</th>${VALIDATION_DIMENSIONS.map(([, label]) => `<th>${esc(label)}</th>`).join('')}<th>When</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
    }

    // Validation is deliberately separate from — and never auto-triggered
    // by — the Review & Test tab (see run_test_stream/run_validation_stream
    // on the backend): this renders the per-run action controls that let a
    // reviewer pick which method judges task-correctness for this run.
    function validationControls(run) {
        const result = run.result || {};
        if (!result.validation?.output_present) {
            return '<p class="exp2-field-note">No retrieved data to validate yet — run the retrieval test first.</p>';
        }
        const sem = result.semantic_validation;
        const busy = state.running;
        const manualOpen = state.manualFormOpenFor.has(run.id);
        const specDeclared = Boolean((state.session.validation_spec || '').trim());
        // Refinement needs a determinate failure. An "undecidable" verdict
        // establishes nothing, and feeding it to the reviser is how a
        // requirement gets argued away instead of met — the backend refuses
        // it too.
        const canRefine = sem && recordVerdict(sem) === 'fail';
        const dimRow = ([key, label]) => `
            <label><span>${esc(label)}</span>
                <select id="hbs-dim-${esc(run.id)}-${key}">
                    ${VERDICT_OPTIONS.map(v => `<option value="${v}">${esc(verdictLabel(v))}</option>`).join('')}
                </select>
            </label>
            <label><span>Evidence for ${esc(label.toLowerCase())}</span><input id="hbs-ev-${esc(run.id)}-${key}" placeholder="Which URL, header, or artifact property did you check?"></label>`;
        return `<div class="exp2-validation-controls">
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">
                <button class="exp2-btn tiny secondary" ${busy ? 'disabled' : ''} onclick="HandbookStudioMode.runValidation('${esc(run.id)}', 'mechanical')" title="${specDeclared ? 'Deterministic checks against the pre-registered spec' : 'No spec declared — this will report undecidable'}">Run mechanical check</button>
                <button class="exp2-btn tiny secondary" ${busy ? 'disabled' : ''} onclick="HandbookStudioMode.toggleManualForm('${esc(run.id)}')">${manualOpen ? 'Cancel adjudication' : 'Adjudicate manually'}</button>
                <button class="exp2-btn tiny secondary" ${busy ? 'disabled' : ''} onclick="HandbookStudioMode.runValidation('${esc(run.id)}', 'llm')" title="Comparison arm — never the reported measurement when a mechanical or human verdict exists">Run LLM judge</button>
                ${canRefine ? `<button class="exp2-btn tiny" ${busy ? 'disabled' : ''} onclick="HandbookStudioMode.refineFromValidation('${esc(run.id)}')">Send back for refinement</button>` : ''}
            </div>
            ${!specDeclared ? '<p class="exp2-field-note">No validation spec is declared for this session, so the mechanical check can only confirm that an artifact exists — it will report <strong>undecidable</strong>. Add one on the Setup tab.</p>' : ''}
            <div class="exp2-inline-form" ${manualOpen ? '' : 'hidden'}>
                <p class="exp2-field-note">Judge each dimension separately and cite what you checked. Leave a dimension at <em>not applicable</em> only when the task genuinely does not constrain it — use <em>undecidable</em> when it does but you could not settle it.</p>
                <div class="exp2-form-grid two">
                    ${VALIDATION_DIMENSIONS.map(dimRow).join('')}
                    <label><span>Reviewer ID</span><input id="hbs-manual-rater-${esc(run.id)}" value="${esc(state.lastRater || '')}" placeholder="e.g. R1 — needed for inter-rater agreement"></label>
                    <label><span>Overall verdict</span>
                        <select id="hbs-manual-verdict-${esc(run.id)}">
                            <option value="">Roll up from the dimensions above</option>
                            ${VERDICT_OPTIONS.filter(v => v !== 'not_applicable').map(v => `<option value="${v}">${esc(verdictLabel(v))}</option>`).join('')}
                        </select>
                    </label>
                    <label class="wide"><span>Notes</span><textarea id="hbs-manual-summary-${esc(run.id)}" rows="2" placeholder="What did you check, and why does it settle the question?"></textarea></label>
                </div>
                <button class="exp2-btn tiny primary" ${busy ? 'disabled' : ''} onclick="HandbookStudioMode.submitManualVerdict('${esc(run.id)}')">Save adjudication</button>
            </div>
        </div>`;
    }

    function toggleManualForm(runId) {
        if (state.manualFormOpenFor.has(runId)) {
            state.manualFormOpenFor.delete(runId);
            delete state.manualStartedAt[runId];
        } else {
            state.manualFormOpenFor.add(runId);
            // Adjudication time is one of the experiment's reported metrics
            // ("manual intervention required"), so it's measured rather than
            // estimated afterwards.
            state.manualStartedAt[runId] = Date.now();
            state.manualRevealed.delete(runId);
        }
        render();
    }

    function revealForReview(runId) {
        state.manualRevealed.add(runId);
        render();
    }

    async function insertSpecTemplate() {
        const field = byId('hbs-validation-spec');
        if (!field) return;
        if (field.value.trim() && !confirm('Replace the current validation spec with the template?')) return;
        try {
            const data = await api('/api/handbook-studio/validation-spec-template');
            field.value = data.template;
            state.session.validation_spec = data.template;
            markDirty();
        } catch (error) { toast(error.message, 'error'); }
    }

    async function postValidation(runId, body) {
        if (state.running) return;
        state.running = true;
        render();
        try {
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/test-runs/${encodeURIComponent(runId)}/validate-stream`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                    body: JSON.stringify(body),
                }
            );
            await consumeStream(response);
            const latest = await api(`/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
            state.session = latest.session;
            savedState('Progress saved');
        } catch (error) {
            applyRunEvent({ type: 'error', error: error.message });
            toast(error.message, 'error');
        } finally {
            state.manualFormOpenFor.delete(runId);
            state.running = false;
            watchActiveRun();
            render();
        }
    }

    async function runValidation(runId, method) {
        await postValidation(runId, { method });
    }

    async function submitManualVerdict(runId) {
        const dimensions = {}, evidence = {};
        VALIDATION_DIMENSIONS.forEach(([key]) => {
            dimensions[key] = byId(`hbs-dim-${runId}-${key}`)?.value || 'not_applicable';
            evidence[key] = byId(`hbs-ev-${runId}-${key}`)?.value || '';
        });
        const rater = byId(`hbs-manual-rater-${runId}`)?.value.trim() || '';
        const cited = VALIDATION_DIMENSIONS.filter(([k]) => dimensions[k] !== 'not_applicable' && !evidence[k].trim());
        if (cited.length && !confirm(`${cited.length} dimension(s) have a verdict but no cited evidence. Save anyway?`)) return;
        state.lastRater = rater;
        const startedAt = state.manualStartedAt[runId];
        const manual_verdict = {
            dimensions, evidence, rater,
            verdict: byId(`hbs-manual-verdict-${runId}`)?.value || '',
            summary: byId(`hbs-manual-summary-${runId}`)?.value || '',
            blinded: !state.manualRevealed.has(runId),
            elapsed_seconds: startedAt ? (Date.now() - startedAt) / 1000 : null,
        };
        await postValidation(runId, { method: 'manual', manual_verdict });
        state.manualRevealed.delete(runId);
        delete state.manualStartedAt[runId];
    }

    async function refineFromValidation(runId) {
        if (state.running) return;
        state.running = true;
        render();
        try {
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/test-runs/${encodeURIComponent(runId)}/refine-from-validation-stream`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                    body: JSON.stringify({}),
                }
            );
            await consumeStream(response);
            const latest = await api(`/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
            state.session = latest.session;
            savedState('Progress saved');
            toast('Handbook revised — go to Review & Test to re-run it.', 'success');
        } catch (error) {
            applyRunEvent({ type: 'error', error: error.message });
            toast(error.message, 'error');
        } finally {
            state.running = false;
            watchActiveRun();
            render();
        }
    }

    function statCard(label, value, hint) {
        return `<article class="exp2-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(hint)}</small></article>`;
    }

    function download(filename, content, type) {
        const blob = new Blob([content], { type });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob); link.download = filename; link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 0);
    }
    const csvCell = v => `"${String(v ?? '').replaceAll('"', '""')}"`;
    const slug = v => String(v || 'handbook_studio').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

    function exportJSON() {
        download(`${slug(state.session.source_name)}_session.json`, JSON.stringify(state.session, null, 2), 'application/json');
    }

    function exportCSV() {
        const headers = [
            'test_number', 'refinement_iteration', 'handbook_changed', 'task_changed',
            'manual_intervention', 'outcome', 'evaluation_stage', 'status', 'retrieval_task',
            'started_at', 'finished_at', 'execution_success', 'output_present', 'format_match',
            'reported_method', 'reported_verdict', 'task_completed', 'confidence',
            ...VALIDATION_DIMENSIONS.map(([key]) => key),
            'should_refine_handbook', 'handbook_gap', 'failure_category',
            // Which question the failure analysis asked. In a control row
            // handbook_gap holds a gap in the model's knowledge of the
            // source, not a gap in any handbook.
            'analysis_condition',
            'trace_deficiencies', 'execution_attempts_used', 'error',
            // Cost of this round. Recorded at call time because provider APIs
            // expose no way to look usage up afterwards.
            'llm_calls', 'llm_calls_missing_usage', 'input_tokens',
            'output_tokens', 'total_tokens', 'reasoning_tokens',
            'cost_usd', 'cost_is_partial',
            // Which model did the work and how long the round took end to end
            // -- both are per-round facts the totals card only shows summed.
            'models_used', 'wall_clock_seconds',
        ];
        const rows = (state.session.test_runs || []).map(r => {
            const res = r.result || {};
            const sem = res.semantic_validation || {};
            const trace = res.trace_analysis || {};
            const exec = res.execution || {};
            const val = res.validation || {};
            const stage = evaluationStage(res);
            const use = r.usage || {};
            const deficiencies = (trace.deficiencies || [])
                .map(d => d.handbook_gap).filter(Boolean).join(' | ');
            return [
                r.test_number, r.refinement_iteration ?? '', r.handbook_changed ?? '',
                r.task_changed ?? '', r.manual_intervention ?? '',
                r.outcome, stage.key, res.status, r.retrieval_task, r.started_at, r.finished_at,
                exec.success ?? '', val.output_present ?? '', val.format_match ?? '',
                sem.method || '', recordVerdict(res.semantic_validation),
                sem.task_completed ?? '', sem.confidence ?? '',
                ...VALIDATION_DIMENSIONS.map(([key]) => sem[key] || ''),
                sem.should_refine_handbook ?? '', sem.handbook_gap || '',
                sem.failure_category || '',
                (res.trace_analysis || {}).condition
                    || (r.use_handbook === false ? 'control' : ''),
                deficiencies,
                exec.attempts_used ?? '', res.error || '',
                use.calls ?? '', use.calls_missing_usage ?? '',
                use.input_tokens ?? '', use.output_tokens ?? '',
                use.total_tokens ?? '', use.reasoning_tokens ?? '',
                use.cost_usd ?? '', use.cost_is_partial ?? '',
                Object.keys(use.by_model || {}).join(' | '), use.elapsed_seconds ?? '',
            ];
        });
        const csv = [headers, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
        download(`${slug(state.session.source_name)}_tests.csv`, csv, 'text/csv;charset=utf-8');
    }

    // One row per (run x validation method x dimension) — the long-format
    // matrix the results table and the inter-rater / LLM-agreement statistics
    // are computed from. Exports the whole append-only log, not just the
    // reported verdict, so every method's judgment of the same run can be
    // compared side by side.
    function exportValidationMatrix() {
        const headers = [
            'session_id', 'source_name', 'test_number', 'run_id', 'retrieval_task',
            'validation_id', 'method', 'rater', 'blinded', 'elapsed_seconds',
            'created_at', 'overall_verdict', 'spec_declared', 'dimension',
            'verdict', 'evidence',
        ];
        const rows = [];
        (state.session.test_runs || []).forEach(run => {
            const records = run.result?.validations || [];
            records.forEach(rec => {
                const checks = rec.checks || [];
                VALIDATION_DIMENSIONS.forEach(([key, label]) => {
                    const evidence = checks
                        .filter(c => (c.name || '').toLowerCase().includes(label.toLowerCase().split(' ')[0]))
                        .map(c => c.evidence).join(' || ');
                    rows.push([
                        state.session.id, state.session.source_name, run.test_number, run.id,
                        run.retrieval_task, rec.id || '', rec.method || '', rec.rater || '',
                        rec.blinded ?? '', rec.elapsed_seconds ?? '', rec.created_at || '',
                        rec.verdict || '', rec.spec_declared ?? '', key, rec[key] || '', evidence,
                    ]);
                });
            });
        });
        if (!rows.length) { toast('No validation records to export yet.', 'error'); return; }
        const csv = [headers, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
        download(`${slug(state.session.source_name)}_validation_matrix.csv`, csv, 'text/csv;charset=utf-8');
    }

    function wireReviewInputs() {
        const map = {
            'hbs-name': 'data_source_name', 'hbs-desc': 'brief_description',
            'hbs-handbook': 'handbook', 'hbs-code': 'code_example',
            'hbs-caveats': 'caveats',
        };
        Object.entries(map).forEach(([id, field]) => {
            byId(id)?.addEventListener('input', (event) => {
                state.session.generated_source = state.session.generated_source || {};
                state.session.generated_source[field] = event.target.value;
                markDirty();
            });
        });
        byId('hbs-task')?.addEventListener('input', (event) => {
            state.session.retrieval_task = event.target.value;
            markDirty();
        });
    }

    function renderReview(main) {
        const session = state.session;
        const pipeline = session.pipeline_state || {};
        const stages = pipeline.stages || [];
        const status = pipeline.status || 'not_started';
        const isRunning = state.running;
        const hasHandbook = !!(session.generated_source && session.generated_source.data_source_name);
        const canEdit = !isRunning && [
            'awaiting_review', 'awaiting_credentials', 'complete', 'interrupted', 'error',
        ].includes(status);

        const statusLabel = isRunning ? 'Working…'
            : status === 'awaiting_review' ? 'Ready to review'
            : status === 'awaiting_credentials' ? 'Waiting for credentials'
            : status === 'testing' ? 'Testing…'
            : status === 'complete' ? 'Test complete'
            : status === 'interrupted' ? 'Stopped — ready to resume'
            : status === 'error' ? 'Needs attention'
            : status === 'running' ? 'Generating…'
            : 'Not started';

        const mergedStages = mergeDraftAndVerification(stages);
        const displayStages = hasHandbook ? withRetrievalStage(mergedStages) : mergedStages;
        const hasReviewStage = stages.some(s => s.key === 'review');

        main.innerHTML = `<div class="exp2-page exp2-pipeline-page">
            <section class="exp2-page-heading hbs-heading-bare">
                <div class="exp2-pipeline-actions">
                    <span class="exp2-pipeline-status ${esc(status)}"><i class="${isRunning ? 'active' : ''}"></i>${esc(statusLabel)}</span>
                    ${isRunning ? '<button class="exp2-btn danger" onclick="HandbookStudioMode.cancelRun()">Stop safely</button>' : ''}
                </div>
            </section>

            ${pipelineProgressStrip(displayStages)}
            <section class="exp2-pipeline-timeline" aria-live="polite">
                ${displayStages.length ? displayStages.map((s, i) => pipelineStageCard(
                    s, i,
                    s.key === 'review' ? reviewHandbookCard(session.generated_source, session, canEdit)
                        : s.key === '__retrieval_question' ? retrievalQuestionCard(session, canEdit)
                        : ''
                )).join('') : pipelineEmptyState()}
            </section>

            ${hasHandbook && !hasReviewStage ? `<section class="exp2-card">
                ${reviewHandbookCard(session.generated_source, session, canEdit)}
                ${retrievalQuestionCard(session, canEdit)}
            </section>` : ''}
        </div>`;
        if (hasHandbook) wireReviewInputs();
    }


    // The condition is a property of the session, not of a single run: mixing
    // arms inside one session would make the case impossible to attribute.
    function setNoHandbook(checked) {
        state.session.use_handbook = !checked;
        markDirty();
        render();
    }

    function setRunKey(name, value) {
        state.runKeys[name] = value;
    }

    function applyRunEvent(event) {
        if (event.session) {
            state.session = event.session;
            savedState('Progress saved');
        }
        const pipeline = state.session.pipeline_state || (state.session.pipeline_state = {
            status: 'running', stages: [], required_keys: [],
        });
        const stages = pipeline.stages || (pipeline.stages = []);
        const targetStage = (key) => stages.find(item => item.key === key)
            || [...stages].reverse().find(item => item.status === 'running');
        let liveStage = null;

        if (event.type === 'stage' && event.stage) {
            const index = stages.findIndex(item => item.key === event.stage.key);
            if (index < 0) stages.push(event.stage);
            else {
                const previous = stages[index];
                stages[index] = {
                    ...previous,
                    ...event.stage,
                    details: (event.stage.details || []).length
                        ? event.stage.details : (previous.details || []),
                };
                if (event.stage.status !== 'running') delete stages[index].hint;
            }
            // Once the stage stops running its saved run record carries the
            // real attempts (with re-run and edit); the provisional copies
            // would otherwise show the same code twice.
            if (event.stage.status && event.stage.status !== 'running') {
                state.liveAttempts.delete(event.stage.key);
            }
        }
        if (event.type === 'stage_progress') {
            const stage = targetStage(event.stage_key);
            const message = String(event.message || '').trim();
            if (stage && message) {
                const details = stage.details || (stage.details = []);
                if (details[details.length - 1] !== message) details.push(message);
                if (details.length > 30) details.splice(0, details.length - 30);
                delete stage.hint;
            }
            liveStage = stage;
        }
        if (event.type === 'heartbeat') {
            const stage = targetStage(event.stage_key);
            if (stage) {
                stage.hint = String(event.message || '').trim();
                stage.elapsed_seconds = event.elapsed_seconds;
            }
            liveStage = stage;
        }
        // Retrieval code arriving mid-run: one event when an attempt starts
        // executing and another when it settles, so the code is on screen
        // while it runs rather than after the whole trial returns.
        if (event.type === 'attempt_code') {
            const key = event.stage_key || (targetStage() || {}).key;
            if (key) {
                let live = state.liveAttempts.get(key);
                if (!live) state.liveAttempts.set(key, live = new Map());
                const number = Number(event.attempt) || live.size + 1;
                live.set(number, {
                    attempt: number,
                    status: event.status || 'running',
                    // The settle event carries the same code; keep whatever
                    // arrived rather than blanking a card if it ever omits it.
                    code: event.code || (live.get(number) || {}).code || '',
                    note: event.note || (live.get(number) || {}).note || '',
                    error: event.error || '',
                });
            }
        }
        if (event.type === 'pipeline_started') pipeline.status = 'running';
        if (event.type === 'credential_required') {
            const names = event.required_keys || [];
            state.requiredKeys = [...new Set([...state.requiredKeys, ...names])];
            pipeline.required_keys = state.requiredKeys;
            pipeline.status = 'awaiting_credentials';
        }
        if (event.type === 'handbook_ready_for_review') {
            pipeline.status = 'awaiting_review';
        }
        if (event.type === 'test_finished') {
            pipeline.status = 'complete';
            pipeline.outcome = event.outcome || '';
        }
        if (event.type === 'refinement_started') {
            pipeline.status = 'testing';
        }
        if (event.type === 'refinement_retry') {
            pipeline.status = 'testing';
            toast(event.message || 'Retrying after an external failure.', 'info');
        }
        if (event.type === 'refinement_finished') {
            pipeline.status = 'complete';
            pipeline.outcome = event.outcome || '';
            const labels = {
                stopped: 'Reached the automatic refinement limit without a passing test.',
                not_refinable: 'The failure was not a fixable handbook deficiency, so refinement stopped.',
                no_change: 'The proposed refinement made no change to the handbook, so refinement stopped.',
                external_failure: 'Stopped after repeated external/transient failures — not a handbook defect.',
            };
            toast(labels[event.outcome] || event.message || 'Automatic refinement finished.',
                event.outcome === 'stopped' ? 'info' : 'info');
        }
        if (event.type === 'validation_finished') {
            pipeline.status = 'complete';
            toast(event.task_completed ? 'Validated: task completed.' : 'Validated: not completed.',
                event.task_completed ? 'success' : 'info');
        }
        if (event.type === 'pipeline_cancelled') pipeline.status = 'interrupted';
        if (event.type === 'error') {
            pipeline.status = 'error';
            pipeline.error = event.error || 'Pipeline error';
        }
        if (liveStage && ['stage_progress', 'heartbeat'].includes(event.type)) {
            updateLiveStageDom(liveStage);
        } else {
            render();
        }
    }

    async function consumeStream(response) {
        if (!response.ok || !response.body) {
            let message = `Request failed (${response.status}).`;
            try { message = (await response.json()).error || message; } catch (_) { /* ignore */ }
            throw new Error(message);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            const frames = buffer.split(/\r?\n\r?\n/);
            buffer = frames.pop() || '';
            frames.forEach((frame) => {
                const line = frame.split(/\r?\n/).find(item => item.startsWith('data:'));
                if (!line) return;
                try {
                    const event = JSON.parse(line.slice(5).trim());
                    applyRunEvent(event);
                    if (event.type === 'error') toast(event.error || 'Step failed.', 'error');
                } catch (_) { /* ignore malformed/partial event */ }
            });
            if (done) break;
        }
    }

    async function startGenerate() {
        if (state.running) return;
        state.running = true;
        render();
        try {
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/generate-stream`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                    body: JSON.stringify({}),
                }
            );
            await consumeStream(response);
            const latest = await api(`/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
            state.session = latest.session;
            savedState('Progress saved');
        } catch (error) {
            applyRunEvent({ type: 'error', error: error.message });
            toast(error.message, 'error');
        } finally {
            state.running = false;
            watchActiveRun();
            render();
        }
    }


    async function startTest() {
        if (state.running) return;
        state.running = true;
        render();
        try {
            // A control session runs the single-trial endpoint: refinement
            // revises a handbook, and this condition has none, so offering it
            // would quietly turn the control back into the treatment.
            const control = state.session.use_handbook === false;
            const endpoint = control ? 'test-stream' : 'test-refine-stream';
            const payload = {
                retrieval_task: state.session.retrieval_task,
                generated_source: state.session.generated_source,
                data_source_keys: state.runKeys,
                // Sent explicitly as well as being stored on the session.
                // Leaving it out is what silently ran the control arm through
                // the treatment path.
                use_handbook: !control,
            };
            if (!control) payload.max_refinements = 3;
            const response = await fetch(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/${endpoint}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                    body: JSON.stringify(payload),
                }
            );
            await consumeStream(response);
            const latest = await api(`/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`);
            state.session = latest.session;
            savedState('Progress saved');
        } catch (error) {
            applyRunEvent({ type: 'error', error: error.message });
            toast(error.message, 'error');
        } finally {
            // Credentials intentionally survive this reset — see
            // clearRunViewState, which clears them only on session switch —
            // so a retry after a non-credential failure doesn't require
            // retyping a key that already worked.
            state.running = false;
            watchActiveRun();
            render();
        }
    }

    async function cancelRun() {
        if (!state.running || !state.session.id) return;
        try {
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}/cancel`,
                { method: 'POST' }
            );
            applyRunEvent({ type: 'heartbeat', message: data.message });
            toast('Safe stop requested.', 'info');
        } catch (error) {
            toast(error.message, 'error');
        }
    }

    async function save() {
        if (state.running) {
            toast('Progress is being saved automatically.', 'info');
            return false;
        }
        if (state.saving) return false;
        state.saving = true;
        const label = byId('handbook-studio-save-state');
        if (label) label.textContent = 'Saving…';
        try {
            const path = state.session.id
                ? `/api/handbook-studio-sessions/${encodeURIComponent(state.session.id)}`
                : '/api/handbook-studio-sessions';
            const method = state.session.id ? 'PUT' : 'POST';
            const data = await api(path, {
                method, headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.session),
            });
            state.session = data.session;
            savedState(`Saved ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`);
            toast('Session saved.');
            render();
            return true;
        } catch (error) {
            markDirty(); toast(error.message, 'error'); return false;
        } finally { state.saving = false; }
    }

    async function saveAndContinue() {
        const ready = readiness();
        if (!ready.ready) {
            toast(ready.issues[0], 'error');
            return;
        }
        const ok = await save();
        if (!ok) return;
        switchTab('review');
        setTimeout(() => startGenerate(), 120);
    }

    async function openSessionPicker() {
        const main = byId('handbook-studio-main');
        try {
            const data = await api('/api/handbook-studio-sessions');
            state.sessionList = data.sessions || [];
            // A search typed on an earlier visit must not silently hide
            // sessions the next time the list is opened.
            state.sessionQuery = '';
            main.innerHTML = sessionPickerMarkup();
            if (state.sessionList.length > SESSION_SEARCH_MIN) {
                byId('hbs-session-search')?.focus();
            }
        } catch (error) { toast(error.message, 'error'); }
    }

    // Below this many sessions the whole list fits on screen and a search box
    // is just another thing to look past.
    const SESSION_SEARCH_MIN = 4;

    function sessionPickerMarkup() {
        const searchable = state.sessionList.length > SESSION_SEARCH_MIN;
        return `<div class="exp2-page">
            <section class="exp2-page-heading">
                <div><h3>Handbook Studio sessions</h3><p>Continue a saved session or start a new one.</p></div>
                <button class="exp2-btn primary" onclick="HandbookStudioMode.newSession()">+ New session</button>
            </section>
            ${searchable ? `<div class="hbs-session-search">
                <label class="hbs-session-search-box">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
                    <input type="search" id="hbs-session-search" autocomplete="off" spellcheck="false"
                        aria-label="Search sessions" aria-controls="hbs-session-grid"
                        placeholder="Search source, session name, status or mechanism…"
                        value="${esc(state.sessionQuery)}"
                        oninput="HandbookStudioMode.filterSessions(this.value)"
                        onkeydown="HandbookStudioMode.sessionSearchKey(event)">
                </label>
                <span class="hbs-session-count" id="hbs-session-count" role="status" aria-live="polite">${sessionCountLabel()}</span>
            </div>` : ''}
            <div class="exp2-study-grid" id="hbs-session-grid">${sessionCardsMarkup()}</div>
        </div>`;
    }

    // Every token must match somewhere, in any field and any order, so
    // "rest draft" narrows to draft sessions on a REST source. Matching is
    // substring rather than word-prefix: source names run together
    // ("NASA/LANCE FIRMS") and a reviewer types the fragment they remember.
    // The retrieval task is included because that -- not the source -- is
    // what distinguishes several sessions against the same API.
    function sessionMatches(session, tokens) {
        const haystack = [
            session.source_name || '',
            session.name || '',
            session.retrieval_task || '',
            (session.status || 'draft').replaceAll('_', ' '),
            MECHANISMS[session.access_mechanism]?.label || session.access_mechanism || '',
        ].join(' ').toLowerCase();
        return tokens.every((token) => haystack.includes(token));
    }

    const sessionQueryTokens = () =>
        state.sessionQuery.toLowerCase().split(/\s+/).filter(Boolean);

    const visibleSessions = () => {
        const tokens = sessionQueryTokens();
        return tokens.length
            ? state.sessionList.filter((session) => sessionMatches(session, tokens))
            : state.sessionList;
    };

    function sessionCountLabel() {
        const total = state.sessionList.length;
        const shown = visibleSessions().length;
        if (shown === total) return `${total} session${total === 1 ? '' : 's'}`;
        return `${shown} of ${total} sessions`;
    }

    function sessionCardsMarkup() {
        const total = state.sessionList.length;
        if (!total) {
            return '<div class="exp2-empty"><strong>No saved sessions</strong><p>Create your first Handbook Studio session.</p></div>';
        }
        const shown = visibleSessions();
        if (!shown.length) {
            return `<div class="exp2-empty"><strong>No sessions match &ldquo;${esc(state.sessionQuery.trim())}&rdquo;</strong>
                <p>Searches cover the source, session name, status and access mechanism.</p>
                <button class="exp2-btn secondary" onclick="HandbookStudioMode.filterSessions('')">Clear search</button></div>`;
        }
        return shown.map(sessionCard).join('');
    }

    function sessionCard(session) {
        // The session's own name is shown only when it adds something the
        // heading doesn't already say -- otherwise a name-only search match
        // would look like a card matching nothing at all. The retrieval task
        // is shown for the same reason, clamped to two lines by CSS.
        const name = (session.name || '').trim();
        const source = (session.source_name || '').trim();
        const subtitle = name && name.toLowerCase() !== source.toLowerCase() ? name : '';
        const task = (session.retrieval_task || '').trim();
        return `<article class="exp2-study-card"><span class="exp2-tag neutral">${esc((session.status || 'draft').replaceAll('_', ' '))}</span><h4>${esc(source || 'Untitled source')}</h4>${subtitle ? `<p class="hbs-session-name">${esc(subtitle)}</p>` : ''}${task ? `<p class="hbs-session-task" title="${esc(task)}">${esc(task)}</p>` : ''}<p>Updated ${esc(new Date(session.updated_at).toLocaleString())}</p><div><span>${esc(MECHANISMS[session.access_mechanism]?.label || 'Mechanism: auto-select')}</span></div><div class="exp2-inline-actions"><button class="exp2-btn primary" onclick="HandbookStudioMode.loadSession('${session.id}')">Open session</button>${session.status && session.status !== 'draft' ? `<button class="exp2-btn secondary" onclick="HandbookStudioMode.duplicateSession('${session.id}')" title="Reuse this handbook for a different retrieval task, without regenerating it">Duplicate for new task</button>` : ''}<span class="hbs-spacer"></span><button class="exp2-icon-btn danger" onclick="HandbookStudioMode.deleteSession('${session.id}')" title="Delete this session" aria-label="Delete session ${esc(source || name || 'untitled')}"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg></button></div></article>`;
    }

    // Deleting throws away the handbook, every test run and every recorded
    // verdict -- the verdicts especially cannot be reconstructed, so this
    // asks first and names what is going.
    async function deleteSession(id) {
        const session = state.sessionList.find((item) => item.id === id);
        const label = (session?.source_name || session?.name || '').trim() || 'this session';
        if (state.running && state.session && state.session.id === id) {
            toast('This session is running — cancel the run before deleting it.', 'error');
            return;
        }
        const runs = state.session && state.session.id === id
            ? (state.session.test_runs || []).length : 0;
        if (!confirm(`Delete "${label}"?\n\nIts handbook${runs ? `, ${runs} test run${runs === 1 ? '' : 's'}` : ''} and any recorded verdicts are permanently removed. This cannot be undone.`)) return;
        try {
            await api(`/api/handbook-studio-sessions/${encodeURIComponent(id)}`,
                      { method: 'DELETE' });
            state.sessionList = state.sessionList.filter((item) => item.id !== id);
            // Deleting the session currently open would leave the workspace
            // editing a record the server no longer has, and the next Save
            // would silently recreate it under the same id.
            if (state.session && state.session.id === id) {
                clearRunViewState();
                state.session = blankSession();
                state.requiredKeys = [];
                savedState('Not saved');
            }
            const main = byId('handbook-studio-main');
            if (main) main.innerHTML = sessionPickerMarkup();
            toast(`Deleted "${label}".`, 'success');
        } catch (error) { toast(error.message, 'error'); }
    }

    // Only the grid and the count are rewritten, never the input itself:
    // replacing the field the user is typing in would drop the caret.
    function filterSessions(value) {
        state.sessionQuery = String(value ?? '');
        const field = byId('hbs-session-search');
        // Set only when they disagree -- assigning .value while typing would
        // move the caret to the end mid-word.
        if (field && field.value !== state.sessionQuery) field.value = state.sessionQuery;
        const grid = byId('hbs-session-grid');
        if (grid) grid.innerHTML = sessionCardsMarkup();
        const count = byId('hbs-session-count');
        if (count) count.textContent = sessionCountLabel();
    }

    function sessionSearchKey(event) {
        if (event.key === 'Escape' && state.sessionQuery) {
            event.preventDefault();  // don't let it bubble out and close anything
            filterSessions('');
        }
    }

    function newSession() {
        if (state.dirty && !confirm('Discard unsaved changes and start a new session?')) return;
        clearRunViewState();
        state.session = blankSession();
        state.requiredKeys = [];
        savedState('Not saved');
        switchTab('setup');
    }

    // ---- Rejoining a run this tab is not streaming -------------------------
    // SSE is one-shot: reload the page and the stream is gone for good, even
    // though the pipeline itself keeps going (the server detaches it). Without
    // this the page would sit on a stage marked "running" that never moves,
    // and the finished run -- already saved -- would only appear if the user
    // happened to reload again later. So when the session says a run is
    // active, poll the record until it lands.

    const ACTIVE_RUN_STALE_MS = 180000;   // matches the server's staleness rule
    const ACTIVE_RUN_POLL_MS = 4000;

    function activeRunInfo(session) {
        const marker = session && session.active_run;
        if (!marker || typeof marker !== 'object' || !marker.kind) return null;
        const beat = Date.parse(marker.heartbeat || marker.started_at || '');
        // A marker left behind by a killed process must not trap the page in a
        // permanent "waiting" state, so anything long past its last heartbeat
        // is reported as stale and the UI stops waiting on it.
        const stale = Number.isFinite(beat)
            ? (Date.now() - beat) > ACTIVE_RUN_STALE_MS
            : false;
        return { ...marker, stale };
    }

    function stopWatchingActiveRun() {
        if (state.activeRunTimer) {
            clearTimeout(state.activeRunTimer);
            state.activeRunTimer = null;
        }
    }

    function watchActiveRun() {
        stopWatchingActiveRun();
        // The tab driving the run has the live stream; polling on top of it
        // would fight the stream for the same session object.
        if (state.running) return;
        const info = activeRunInfo(state.session);
        if (!info || info.stale) return;

        const sessionId = state.session.id;
        state.activeRunTimer = setTimeout(async () => {
            state.activeRunTimer = null;
            if (state.running || !state.session || state.session.id !== sessionId) return;
            try {
                const data = await api(
                    `/api/handbook-studio-sessions/${encodeURIComponent(sessionId)}`);
                // The user may have moved on while the request was in flight.
                if (!state.session || state.session.id !== sessionId || state.running) return;
                state.session = data.session;
                render();
            } catch (_) {
                // A failed poll is not worth a toast -- the next tick retries.
            }
            watchActiveRun();
        }, ACTIVE_RUN_POLL_MS);
    }

    async function loadSession(id) {
        try {
            const data = await api(`/api/handbook-studio-sessions/${encodeURIComponent(id)}`);
            clearRunViewState();
            state.session = data.session;
            state.requiredKeys = [];
            savedState('Saved');
            switchTab(data.session.status === 'draft' ? 'setup' : 'review');
            // A run may still be going from before the reload.
            watchActiveRun();
        } catch (error) { toast(error.message, 'error'); }
    }

    async function duplicateSession(id) {
        try {
            // If the session being duplicated is the one currently open, its
            // in-memory credentials are still good for the identical handbook
            // — carry them over instead of making the user retype them.
            // clearRunViewState() (below) would otherwise wipe them, and
            // credentials are never persisted server-side to carry over any
            // other way.
            const carriedKeys = (state.session && state.session.id === id)
                ? { ...state.runKeys } : null;
            const data = await api(
                `/api/handbook-studio-sessions/${encodeURIComponent(id)}/duplicate`,
                { method: 'POST' });
            clearRunViewState();
            state.session = data.session;
            state.requiredKeys = Array.isArray(data.session?.pipeline_state?.required_keys)
                ? data.session.pipeline_state.required_keys : [];
            if (carriedKeys && Object.keys(carriedKeys).length) {
                state.runKeys = carriedKeys;
            }
            savedState('Saved');
            switchTab('review');
            toast('Handbook reused — update the retrieval task, then test.', 'success');
        } catch (error) { toast(error.message, 'error'); }
    }

    // Header shortcut for the picker's "Duplicate for new task". The server
    // copies the STORED session, so unsaved edits are saved first -- otherwise
    // the duplicate would silently miss them.
    async function duplicateCurrent() {
        if (state.readOnly || !state.session?.id) return;
        if (state.running) {
            toast('Wait for the running step to finish before duplicating.', 'info');
            return;
        }
        if (state.dirty && !(await save())) return;
        await duplicateSession(state.session.id);
    }

    window.HandbookStudioMode = {
        open, close, switchTab, save, saveAndContinue,
        startGenerate, startTest, cancelRun, setRunKey, setNoHandbook,
        openSessionPicker, newSession, loadSession, duplicateSession, duplicateCurrent,
        filterSessions, sessionSearchKey, deleteSession,
        scrollToStage, toggleStageCard, toggleOutputPreview, toggleOutputFull, toggleTabBar,
        downloadOutput, downloadRun,
        openShared, toggleShare, copyShareLink,
        toggleOutputAttributes,
        exportCSV, exportJSON, exportValidationMatrix,
        rerunCode, previewMapReady,
        editCode, cancelCodeEdit, saveCode, revertCode,
        toggleCodeFeedback, updateFeedbackDraft, sendCodeFeedback,
        toggleVersions, selectCodeVersion, toggleVersionTrace,
        updateCodeDraft, codeEditorKey, trackCodeCaret,
        toggleManualForm, runValidation, submitManualVerdict, refineFromValidation,
        revealForReview, insertSpecTemplate, showReviewPanel, toggleInstructionEdit,
    };
})();
