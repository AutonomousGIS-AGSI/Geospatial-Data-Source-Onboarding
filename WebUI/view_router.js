/**
 * Keeps the address bar in sync with Handbook Studio so sessions are
 * shareable, bookmarkable URLs:
 *   /                             -> a blank/new session
 *   /handbook-studio/<session-id> -> one specific saved session
 *   /handbook-studio/shared/<id>  -> read-only view of a session its owner made public
 * Watches `data-session-id` / `data-shared` on the workspace (set by
 * handbook_studio_mode.js's render()) rather than patching HandbookStudioMode,
 * so every entry point - Sessions picker, Duplicate, Save - stays in sync.
 */
(function () {
    'use strict';

    const STUDIO_PATH = '/handbook-studio';
    const STUDIO_RE = /^\/handbook-studio(?:\/([^/]+))?\/?$/;
    const SHARED_RE = /^\/handbook-studio\/shared\/([^/]+)\/?$/;

    const workspace = () => document.getElementById('handbook-studio-workspace');
    const currentSessionId = () => workspace()?.dataset.sessionId || null;
    const isSharedView = () => workspace()?.dataset.shared === '1';

    function currentTargetPath() {
        const id = currentSessionId();
        if (!id) return '/';
        if (isSharedView()) return `${STUDIO_PATH}/shared/${encodeURIComponent(id)}`;
        return `${STUDIO_PATH}/${encodeURIComponent(id)}`;
    }

    // Set while a deep-link transition (open, then async loadSession) is in
    // flight, so the observer doesn't push the intermediate blank session.
    let suspendSync = false;
    let syncScheduled = false;
    function scheduleSync() {
        if (suspendSync || syncScheduled) return;
        syncScheduled = true;
        queueMicrotask(() => {
            syncScheduled = false;
            const target = currentTargetPath();
            if (location.pathname !== target) {
                history.pushState({ hbView: target }, '', target);
            }
        });
    }

    async function applyRoute(path) {
        const studio = window.HandbookStudioMode;
        if (!studio) return;
        suspendSync = true;
        try {
            const sharedMatch = path.match(SHARED_RE);
            const studioMatch = sharedMatch ? null : path.match(STUDIO_RE);
            if (sharedMatch) {
                const id = decodeURIComponent(sharedMatch[1]);
                if (currentSessionId() !== id || !isSharedView()) await studio.openShared(id);
            } else {
                const id = studioMatch && studioMatch[1] ? decodeURIComponent(studioMatch[1]) : null;
                if (workspace()?.hidden || isSharedView()) studio.open();
                if (currentSessionId() !== id) {
                    if (id) await studio.loadSession(id);
                    else studio.newSession();
                }
            }
        } finally {
            suspendSync = false;
        }
        scheduleSync();
    }

    window.addEventListener('popstate', () => {
        // If a load was refused (unsaved changes, a run in progress, ...) put
        // the URL back in sync with whatever is really on screen.
        applyRoute(location.pathname).then(scheduleSync);
    });

    const el = workspace();
    if (el) {
        new MutationObserver(scheduleSync).observe(el, {
            attributes: true, attributeFilter: ['hidden', 'data-session-id', 'data-shared'],
        });
    }

    // Defer a tick so other init scripts finish mounting first.
    setTimeout(() => applyRoute(location.pathname), 0);
})();
