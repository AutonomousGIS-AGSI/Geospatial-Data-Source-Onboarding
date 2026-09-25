"""
Flask backend for Handbook Studio: onboard a geospatial data source by
generating a handbook for it, testing retrieval, and refining from the result.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file, Response, stream_with_context, g, redirect
from flask_cors import CORS
import hashlib
import sys
import os
import json
import mimetypes
import re
import subprocess
import threading
import time

# Load the project-root .env into os.environ at startup so configuration vars
# are available at request time regardless of the worker's working directory.
# override=False keeps any vars already set by the hosting environment.
try:
    from dotenv import load_dotenv
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        override=False,
    )
except Exception:
    pass


class GeoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Shapely geometry objects."""
    def default(self, obj):
        try:
            import shapely.geometry
            if isinstance(obj, shapely.geometry.base.BaseGeometry):
                import shapely
                return shapely.geometry.mapping(obj)
        except ImportError:
            pass
        return super().default(obj)

# Add the parent directory to path to import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='/static')
CORS(app, allow_headers=['Content-Type', 'X-API-Key', 'X-Anthropic-Key', 'Authorization', 'Accept'])  # Enable CORS for frontend requests


def hash_api_key(api_key):
    """Derive a stable user_id from an API key (SHA-256, first 16 hex chars).
    The key itself is never stored."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


@app.before_request
def extract_api_key():
    """Extract user-provided API key from X-API-Key header (JSON) or api_key form field (FormData).
    IMPORTANT: Only access request.form for multipart/form-data requests.
    Accessing request.form on a JSON request consumes the input stream,
    causing 'read of closed file' errors in streaming endpoints."""
    key = request.headers.get('X-API-Key', '').strip()
    anthropic_key = request.headers.get('X-Anthropic-Key', '').strip()
    if not key or not anthropic_key:
        ct = request.content_type or ''
        if 'multipart/form-data' in ct or 'application/x-www-form-urlencoded' in ct:
            try:
                key = key or (request.form.get('api_key') or '').strip()
                anthropic_key = anthropic_key or (
                    request.form.get('anthropic_api_key') or '').strip()
            except Exception:
                pass
    g.api_key = key or None

    # Anthropic key for the "Claude Agent SDK" provider option (Generate &
    # Test) -- a separate header/field, since a caller may hold an OpenAI
    # key, an Anthropic key, or both at once.
    g.anthropic_api_key = anthropic_key or None

    # Also propagate into os.environ so that background threads (which
    # cannot access Flask's thread-local ``g``) pick it up via
    # ``load_OpenAI_key()``'s ``os.getenv`` fallback. Only propagate values
    # shaped like a real key — a sentinel like "no-api" (meaning "use the
    # server's own configured key/local model") must not clobber the real
    # OPENAI_API_KEY for the rest of the process.
    if key and (key.startswith('sk-') or key.startswith(('gibd-', 'gibd_'))):
        os.environ['OPENAI_API_KEY'] = key


def _no_key_event():
    """SSE error event for missing API key."""
    return f"data: {json.dumps({'type': 'error', 'error': 'No API key provided. Enter your OpenAI API key in Settings.'})}\n\n"


def sse_data(payload):
    """Encode one payload as an SSE data frame."""
    return f"data: {json.dumps(payload, cls=GeoJSONEncoder)}\n\n"


def _studio_identity_key():
    """OpenAI or Anthropic key, whichever is present -- Handbook Studio
    ("Generate & Test") sessions can run on either provider, so a caller
    with only an Anthropic key must still get a stable user_id to hash for
    session storage. Prefers the OpenAI key when both are set, matching
    every other part of the app that identifies callers by g.api_key."""
    return g.api_key or g.anthropic_api_key


@app.route('/api/clear-key', methods=['POST'])
def clear_api_key():
    """Clear the API key from the server environment."""
    os.environ.pop('OPENAI_API_KEY', None)
    return jsonify({'status': 'ok'})



def sse_response(generator):
    """Create a properly-configured SSE Response with headers that prevent
    caching and buffering, ensuring subsequent streams work correctly."""
    resp = Response(stream_with_context(generator), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['X-Accel-Buffering'] = 'no'
    # Note: 'Connection' is a hop-by-hop header forbidden in WSGI (PEP 3333).
    # Waitress manages connection persistence itself.
    return resp


def _drain_detached(generator, label):
    """Finish a pipeline whose client has gone away, discarding its events."""
    try:
        for _ in generator:
            pass
    except Exception:
        app.logger.exception("Detached pipeline %s failed after disconnect", label)
    else:
        app.logger.info("Detached pipeline %s finished after disconnect", label)


def detachable_stream(generator, label=""):
    """Let a pipeline run to completion even if the browser goes away.

    Reloading the page or closing the tab makes the WSGI server close this
    response, which raises GeneratorExit inside the pipeline generator at
    whichever yield it is parked on. Everything after that point -- building
    the run record, appending it to the session, persisting -- is skipped, so a
    run that had already done all its work and spent all its tokens left no
    trace. Worse, it was invisible: the retrieval thread carried on and
    finished into nothing, and the session simply had no run in it.

    On disconnect the pipeline is handed to a background thread that keeps
    pulling events and throwing them away. The pipeline never learns the client
    left, reaches its own end, and persists exactly as it normally would.
    Nothing here writes to the session -- the pipeline stays the single writer,
    so this cannot introduce a partial or duplicate record.
    """
    detached = False
    try:
        for event in generator:
            yield event
    except GeneratorExit:
        detached = True
        threading.Thread(target=_drain_detached, args=(generator, label),
                         daemon=True).start()
        raise
    finally:
        if not detached:
            generator.close()



@app.route('/')
@app.route('/handbook-studio')
@app.route('/handbook-studio/')
@app.route('/handbook-studio/<session_id>')
def serve_studio(session_id=None):
    """Every Studio URL serves the SPA shell; view_router.js reads the path
    on load and opens the matching session. session_id is unused here on
    purpose."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/map.html')
def serve_map():
    """The map viewer, with the Mapbox token filled in from MAPBOX_TOKEN.
    Kept out of the source so it isn't committed; without one the viewer
    falls back to OpenLayers + OSM tiles."""
    with open(os.path.join(BASE_DIR, 'map.html'), encoding='utf-8') as f:
        page = f.read()
    token = os.environ.get('MAPBOX_TOKEN', '').strip()
    if not re.fullmatch(r'pk\.[A-Za-z0-9._-]+', token):
        token = ''
    response = Response(page.replace('__MAPBOX_TOKEN__', token), mimetype='text/html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@app.route('/handbook-generator/generate-test')
@app.route('/handbook-generator/generate-test/<session_id>')
def redirect_legacy_studio_views(session_id=None):
    """Studio used to live under /handbook-generator/generate-test; keep those
    links (workbooks, notes, bookmarks) resolving to the new /handbook-studio
    path."""
    target = '/handbook-studio' + (f'/{session_id}' if session_id else '')
    return redirect(target, code=301)


@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files (CSS, JS, images, etc.)"""
    response = send_from_directory(BASE_DIR, filename)
    # Prevent browser caching of JS/CSS so changes take effect immediately
    if filename.endswith(('.js', '.css', '.html')):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response



# ── Admin / identity ─────────────────────────────────────────────────────────

def _is_admin(api_key):
    """A caller is an admin if their key (or its hash) is listed in the
    AGM_ADMIN_API_KEYS / AGM_ADMIN_USER_IDS environment variables
    (comma-separated)."""
    if not api_key:
        return False
    raw = {k.strip() for k in os.environ.get('AGM_ADMIN_API_KEYS', '').split(',') if k.strip()}
    ids = {k.strip() for k in os.environ.get('AGM_ADMIN_USER_IDS', '').split(',') if k.strip()}
    if api_key in raw:
        return True
    return hash_api_key(api_key) in ids


@app.route('/api/whoami', methods=['GET'])
def whoami_endpoint():
    """Return the caller's user_id (hash) and whether they are an admin — used
    by the UI to decide whether to show the review queue."""
    api_key = g.api_key
    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    return jsonify({
        'success': True,
        'user_id': hash_api_key(api_key),
        'is_admin': _is_admin(api_key),
    })




# ── "Generate & Test" handbook sessions ─────────────────────────────────────
# A separate, simpler, OpenAI-only mode: one source, one handbook, one
# retrieval test, with a human review pause in between. Deliberately
# independent of the Experiment 2 routes/store/runner above.

@app.route('/api/handbook-studio-providers', methods=['GET'])
def handbook_studio_providers():
    """Which "Generate & Test" LLM providers are actually usable right now --
    lets the setup UI gray out Claude Agent SDK with a real reason instead of
    just failing on first use if the server-side package/CLI isn't there."""
    from agents.data_agent import claude_agent_provider
    return jsonify({
        'success': True,
        'providers': {
            'openai': {'available': True},
            'claude': {
                'available': claude_agent_provider.available(),
                'reason': claude_agent_provider.unavailable_reason(),
            },
        },
    })


@app.route('/api/handbook-studio/validation-spec-template')
def handbook_studio_validation_spec_template():
    """The commented starter validation spec, served from the one place it
    is defined so the UI template can't drift from the parser."""
    from agents.data_agent.validation import SPEC_TEMPLATE
    return jsonify({'success': True, 'template': SPEC_TEMPLATE})


@app.route('/api/handbook-studio-sessions', methods=['GET', 'POST'])
def handbook_studio_sessions_endpoint():
    """List sessions or create a new server-persisted Generate & Test session."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    if request.method == 'GET':
        return jsonify({'success': True, 'sessions': store.list_sessions(user_id)})
    try:
        body = request.get_json(silent=True) or {}
        session = store.save_session(user_id, body)
        return jsonify({'success': True, 'session': session}), 201
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def _studio_session_for_read(session_id):
    """(owner_user_id, session, is_owner) for a READ of one Studio session.

    The caller's own session wins. Otherwise -- no key, or a key that isn't
    the owner's -- a session the owner has flagged public is served read-only,
    which is how a manuscript's appendix links open for any reader. A private
    session and an unknown id both come back as (None, None, False) so the
    response can't be used to probe which ids exist.
    """
    from WebUI import handbook_studio_store as store
    identity_key = _studio_identity_key()
    if identity_key:
        user_id = hash_api_key(identity_key)
        try:
            session = store.get_session(user_id, session_id)
        except ValueError:
            session = None
        if session is not None:
            return user_id, session, True
    found = store.find_public_session(session_id)
    if found:
        owner_id, session = found
        return owner_id, session, False
    return None, None, False


@app.route('/handbook-studio/shared/<session_id>')
def serve_shared_studio_session(session_id):
    """Read-only deep link to a public Studio session. The client router
    (view_router.js) opens the Studio in shared mode; no key is required."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/handbook-studio-sessions/<session_id>/share', methods=['POST'])
def handbook_studio_share_endpoint(session_id):
    """Owner-only: make a session public (read-only link) or private again."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    body = request.get_json(silent=True) or {}
    public = bool(body.get('public', not session.get('public')))
    updated = store.save_session(user_id, {**session, 'public': public},
                                 session_id=session_id)
    return jsonify({
        'success': True,
        'public': updated.get('public', False),
        'public_since': updated.get('public_since'),
        'url': f"{request.host_url.rstrip('/')}/handbook-studio/shared/{session_id}",
        'session': updated,
    })


@app.route('/api/handbook-studio-sessions/<session_id>',
           methods=['GET', 'PUT', 'DELETE'])
def handbook_studio_session_endpoint(session_id):
    """Read, update, or delete one caller-owned Generate & Test session.
    A public session can be READ by anyone (see _studio_session_for_read)."""
    if request.method == 'GET':
        _owner, session, is_owner = _studio_session_for_read(session_id)
        if session is None:
            if not _studio_identity_key():
                return jsonify({'success': False, 'error': 'No API key provided'}), 401
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        return jsonify({'success': True, 'session': session,
                        'is_owner': is_owner, 'shared_view': not is_owner})
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    try:
        session = store.get_session(user_id, session_id)
        if session is None:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        if request.method == 'DELETE':
            store.delete_session(user_id, session_id)
            return jsonify({'success': True})
        updated = store.save_session(
            user_id, request.get_json(silent=True) or {}, session_id=session_id)
        return jsonify({'success': True, 'session': updated})
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/handbook-studio-sessions/<session_id>/generate-stream',
           methods=['POST'])
def handbook_studio_generate_stream(session_id):
    """Generate one handbook (OpenAI or Claude, per the session's provider)
    and pause for human review, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_generate_stream
        try:
            for event in run_generate_stream(
                    user_id, api_key_value, session_id,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test handbook generation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'generate:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-stream',
           methods=['POST'])
def handbook_studio_test_stream(session_id):
    """Run one controlled retrieval trial against the reviewed handbook, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    edited_task = body.get('retrieval_task')
    edited_source = body.get('generated_source') or {}
    data_source_keys = body.get('data_source_keys') or {}
    if not isinstance(edited_source, dict):
        return jsonify({
            'success': False, 'error': 'generated_source must be an object.'
        }), 400
    if not isinstance(data_source_keys, dict):
        return jsonify({
            'success': False, 'error': 'data_source_keys must be an object.'
        }), 400
    user_id = hash_api_key(identity_key)
    # Which arm this run belongs to. The SESSION is the authority: it is what
    # the author set, what the UI shows, and what is persisted. This used to
    # default a missing body field to True, which silently overrode a control
    # session -- the runner only consults the session when it receives None, so
    # a caller that forgot the field turned every control run into a treatment
    # run while the UI still displayed "no handbook". Fifteen runs were lost
    # that way. None means "not specified, use the session"; the body can still
    # override deliberately.
    use_handbook = (None if body.get('use_handbook') is None
                    else body.get('use_handbook') is not False)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_test_stream
        try:
            for event in run_test_stream(
                    user_id, api_key_value, session_id,
                    edited_task=edited_task, edited_source=edited_source,
                    user_keys=data_source_keys,
                    anthropic_api_key=anthropic_key_value,
                    use_handbook=use_handbook):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test retrieval test failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'test:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-refine-stream',
           methods=['POST'])
def handbook_studio_test_refine_stream(session_id):
    """Run a controlled retrieval trial, automatically revising the handbook
    (H1, H2, H3) and retrying on evidence-grounded failures, over SSE."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    edited_task = body.get('retrieval_task')
    edited_source = body.get('generated_source') or {}
    data_source_keys = body.get('data_source_keys') or {}
    max_refinements = body.get('max_refinements', 3)
    if not isinstance(edited_source, dict):
        return jsonify({
            'success': False, 'error': 'generated_source must be an object.'
        }), 400
    if not isinstance(data_source_keys, dict):
        return jsonify({
            'success': False, 'error': 'data_source_keys must be an object.'
        }), 400
    try:
        max_refinements = max(0, min(10, int(max_refinements)))
    except (TypeError, ValueError):
        max_refinements = 3
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_test_with_refinement_stream
        try:
            for event in run_test_with_refinement_stream(
                    user_id, api_key_value, session_id,
                    edited_task=edited_task, edited_source=edited_source,
                    user_keys=data_source_keys, max_refinements=max_refinements,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Generate & Test retrieval test (with refinement) failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'test-refine:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/validate-stream',
           methods=['POST'])
def handbook_studio_validate_stream(session_id, run_id):
    """Judge one completed test run's correctness on demand, over SSE --
    separate from and not triggered by Review & Test. method is one of
    "mechanical" (deterministic checks against the session's pre-registered
    validation spec, no LLM; "metadata" is accepted as its former name),
    "manual" (a per-dimension human adjudication rubric), or "llm" (the
    evidence-grounded LLM judge, kept as a comparison arm). Each call
    APPENDS to the run's validation log; nothing is overwritten."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    body = request.get_json(silent=True) or {}
    method = body.get('method')
    manual_verdict = body.get('manual_verdict') or {}
    if not isinstance(manual_verdict, dict):
        return jsonify({
            'success': False, 'error': 'manual_verdict must be an object.'
        }), 400
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_validation_stream
        try:
            for event in run_validation_stream(
                    user_id, api_key_value, session_id, run_id, method,
                    manual_verdict=manual_verdict,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Handbook Studio validation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'validate:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/refine-from-validation-stream',
           methods=['POST'])
def handbook_studio_refine_from_validation_stream(session_id, run_id):
    """Send a Validation-judged-incomplete test run back for a single
    handbook revision, over SSE -- triggered deliberately from the
    Validation tab. Does not auto-retest afterward."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return sse_response(iter([_no_key_event()]))
    user_id = hash_api_key(identity_key)

    # Read off the request context BEFORE the generator starts: a stream that
    # outlives its client (see detachable_stream) finishes on a plain worker
    # thread, where flask.g no longer exists.
    api_key_value = g.api_key
    anthropic_key_value = g.anthropic_api_key

    def generate():
        from WebUI.handbook_studio_runner import run_validation_refine_stream
        try:
            for event in run_validation_refine_stream(
                    user_id, api_key_value, session_id, run_id,
                    anthropic_api_key=anthropic_key_value):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except (ValueError, TypeError) as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            app.logger.exception("Handbook Studio refine-from-validation failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return sse_response(detachable_stream(generate(), f'validate-refine:{session_id}'))


@app.route('/api/handbook-studio-sessions/<session_id>/cancel', methods=['POST'])
def handbook_studio_cancel(session_id):
    """Request cooperative cancellation of a running Generate & Test session."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    from WebUI.handbook_studio_runner import request_cancel
    user_id = hash_api_key(identity_key)
    try:
        if store.get_session(user_id, session_id) is None:
            return jsonify({
                'success': False, 'error': 'Session not found'
            }), 404
        request_cancel(user_id, session_id)
        return jsonify({
            'success': True,
            'message': 'Cancellation requested. The active step will stop at '
                       'the next safe checkpoint.',
        })
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


def _studio_preview_geometry(path, extension, max_features=5000):
    """Vector artifact -> GeoJSON the browser can draw without its own GDAL.

    Returns None when the file cannot be read as vector data, so the caller
    can fall back rather than reporting a broken map layer.
    """
    try:
        if extension == '.geojson':
            if os.path.getsize(path) > 40 * 1024 * 1024:
                return {'kind': 'binary'}
            with open(path, encoding='utf-8', errors='replace') as handle:
                data = json.load(handle)
            features = data.get('features') if isinstance(data, dict) else None
            truncated = False
            if isinstance(features, list) and len(features) > max_features:
                data = dict(data, features=features[:max_features])
                truncated = True
            return {'kind': 'geojson', 'data': data,
                    'feature_count': len(features) if isinstance(features, list) else None,
                    'truncated': truncated}
        import geopandas as gpd
        frame = gpd.read_file(path)
        if frame.crs is not None and str(frame.crs) != 'EPSG:4326':
            frame = frame.to_crs(epsg=4326)
        total = len(frame)
        truncated = total > max_features
        if truncated:
            frame = frame.iloc[:max_features]
        return {'kind': 'geojson', 'data': json.loads(frame.to_json()),
                'feature_count': total, 'truncated': truncated}
    except Exception:
        return None


def _sidecar_crs(path):
    """The CRS from an ESRI .prj sitting beside a raster that carries none.

    A world file plus a .prj is a normal way to georeference a plain TIFF --
    it is exactly what a PIL-written mosaic gets -- and GDAL applies the .tfw
    but not the .prj. rasterio then reports a correct transform with no CRS,
    and everything downstream treats the file as unplaceable even though the
    projection is sitting in a text file next to it.
    """
    try:
        from rasterio.crs import CRS
    except Exception:
        return None
    base = os.path.splitext(path)[0]
    for candidate in (base + '.prj', base + '.PRJ'):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding='utf-8', errors='replace') as handle:
                wkt = handle.read().strip()
            if wkt:
                return CRS.from_wkt(wkt)
        except Exception:
            return None
    return None


def _studio_render_raster_png(path, max_dim=1024):
    """Render a raster to a WGS84-aligned RGBA PNG for map display.

    Three things have to happen here or satellite imagery is unreadable, which
    is why this is done server-side rather than in the browser:

    * **Reprojection.** Sentinel-2 and Landsat ship in UTM. Mapbox image
      sources take four lon/lat corners, so the pixels must be warped to
      EPSG:4326 first; pasting UTM pixels onto a lon/lat rectangle puts them in
      the wrong place, and the browser-side loader in map.html simply refuses
      the file instead.
    * **A percentile stretch.** Reflectance rasters are uint16 with a long
      bright tail, so a min/max stretch maps almost everything to near-black.
      The 2nd-98th percentile of the valid pixels is what makes the scene
      visible; this is a display stretch only and is never fed back into any
      analysis.
    * **Downsampling before reading.** A full Sentinel tile is ~10980 px square
      per band; decoding that to send to a browser preview would cost hundreds
      of megabytes for an image displayed a few hundred pixels wide.

    Returns (png_bytes, info) or (None, None) when the file cannot be rendered.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.enums import ColorInterp, Resampling
        from rasterio.vrt import WarpedVRT
        from PIL import Image
        import io

        from affine import Affine
        from rasterio.warp import calculate_default_transform

        with rasterio.open(path) as src:
            # No CRS means nothing can be placed on a map; the footprint
            # summary still reports the raw bounds.
            source_crs = src.crs or _sidecar_crs(path)
            if not source_crs:
                return None, {'reason': 'This raster has no CRS, and no .prj '
                                        'sits beside it, so it cannot be '
                                        'placed on a map.'}
            # Overviews are what make a big raster cheap to preview: GDAL reads
            # a small pre-built pyramid level instead of the full grid. A COG
            # (Sentinel-2, most STAC assets) has them, and a 10980x10980 tile
            # renders in well under a second. A plain tiled TIFF may not, and
            # then every pixel must be decompressed to produce any view at all
            # -- a 96000x96000 LZW DEM measured at 258s, which no browser
            # request survives. Refuse those with a reason rather than hanging.
            pixels = src.width * src.height
            if pixels > 50_000_000 and not src.overviews(1):
                return None, {'reason': (
                    f'This raster is {src.width}x{src.height} with no overview '
                    'pyramid, so a preview would require decompressing every '
                    'pixel. Build overviews (gdaladdo, or save as a COG) to '
                    'preview it.')}
            # Size the warp DOWN FRONT rather than warping at native
            # resolution and shrinking afterwards. A default WarpedVRT on a
            # 96000x96000 continental tile reprojects every pixel before the
            # downsample ever runs -- measured at 258s, which no browser
            # request survives. Scaling the transform first makes the warp
            # itself produce only the pixels that will be displayed.
            transform, vw, vh = calculate_default_transform(
                source_crs, 'EPSG:4326', src.width, src.height, *src.bounds)
            scale = max(vw, vh) / float(max_dim)
            if scale > 1:
                transform = transform * Affine.scale(scale, scale)
                vw = max(1, int(vw / scale))
                vh = max(1, int(vh / scale))
            with WarpedVRT(src, src_crs=source_crs, crs='EPSG:4326', transform=transform,
                           width=vw, height=vh,
                           resampling=Resampling.bilinear) as vrt:
                out_w, out_h = vrt.width, vrt.height

                # Prefer declared colour interpretation; fall back to the first
                # three bands, which is the convention for a true-colour
                # composite. A single band is rendered as grayscale.
                interp = list(vrt.colorinterp)
                rgb = [interp.index(c) + 1 for c in
                       (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                       if c in interp]
                if len(rgb) != 3:
                    rgb = [1, 2, 3][:vrt.count]
                indexes = rgb if len(rgb) == 3 else [1]

                data = vrt.read(indexes, out_shape=(len(indexes), out_h, out_w),
                                resampling=Resampling.bilinear,
                                masked=True).astype('float32')
                # The warp fills outside the source footprint; that fill and
                # any nodata become transparent rather than black bars.
                mask = vrt.read_masks(indexes[0], out_shape=(out_h, out_w),
                                      resampling=Resampling.nearest)

                bands = []
                for i in range(data.shape[0]):
                    band = data[i]
                    valid = band.compressed() if hasattr(band, 'compressed') else band.ravel()
                    valid = valid[np.isfinite(valid)]
                    if valid.size == 0:
                        bands.append(np.zeros((out_h, out_w), dtype='uint8'))
                        continue
                    lo, hi = np.percentile(valid, [2, 98])
                    if hi <= lo:
                        lo, hi = float(valid.min()), float(valid.max())
                    if hi <= lo:
                        hi = lo + 1.0
                    stretched = np.clip((np.asarray(band, dtype='float32') - lo)
                                        / (hi - lo), 0, 1) * 255.0
                    bands.append(stretched.astype('uint8'))

                if len(bands) == 1:
                    bands = bands * 3
                alpha = np.where(mask > 0, 255, 0).astype('uint8')
                # A Sentinel/Landsat scene is a square grid holding a diagonal
                # swath, and the corners outside the swath are plain zeros with
                # no nodata value declared -- so the mask calls them valid and
                # they render as an opaque black wedge sitting over the
                # basemap. For a multi-band composite, all-bands-zero means no
                # data by near-universal convention, so drop those to
                # transparent. NOT applied to single-band rasters, where zero
                # is usually a real measurement (sea level in a DEM, zero
                # population in a density grid).
                if data.shape[0] >= 3 and src.nodata is None:
                    blank = np.all(np.asarray(data) == 0, axis=0)
                    alpha[blank] = 0
                rgba = np.dstack(bands[:3] + [alpha])
                buffer = io.BytesIO()
                Image.fromarray(rgba, mode='RGBA').save(buffer, format='PNG',
                                                        optimize=True)
                west, south, east, north = vrt.bounds
                return buffer.getvalue(), {
                    'bbox': {'west': west, 'south': south,
                             'east': east, 'north': north},
                    'width': out_w, 'height': out_h,
                    'bands_used': indexes,
                    'source_size': [src.width, src.height],
                }
    except Exception:
        app.logger.exception('Raster render failed for %s', path)
        return None, {'reason': 'This raster could not be decoded.'}


def _studio_preview_raster(path):
    """Raster artifact -> footprint and band summary (no pixels shipped)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(path) as src:
            source_crs = src.crs or _sidecar_crs(path)
            summary = {
                'kind': 'raster', 'width': src.width, 'height': src.height,
                'bands': src.count, 'dtype': str(src.dtypes[0]),
                'crs': str(source_crs) if source_crs else None,
                'bbox': None,
            }
            if not source_crs:
                # The old code returned the raw bounds here as though they were
                # lon/lat. For a projected raster that is a bbox in the
                # millions of degrees: the map silently drew nothing, which
                # read as "the file is empty" rather than "we don't know where
                # this goes".
                summary['reason'] = (
                    'This raster carries no CRS and has no .prj beside it, so '
                    'it cannot be placed on a map. Its data is fine — only its '
                    'location is unknown.')
                return summary
            west, south, east, north = transform_bounds(
                source_crs, 'EPSG:4326', *src.bounds)
            summary['bbox'] = {'west': west, 'south': south,
                               'east': east, 'north': north}
            return summary
    except Exception:
        return None


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/rerun-code',
           methods=['POST'])
def handbook_studio_rerun_code(session_id, run_id):
    """Re-execute one test run's stored retrieval code to regenerate its files.

    Artifacts are written to a temp directory that the OS eventually sweeps, so
    a run's data can vanish long before anyone reviews it. The code survives in
    the session record, and re-running it puts the files back.

    This re-runs the SAME code; it does not re-generate code and does not touch
    the handbook or the recorded verdict. What comes back is today's data from
    the source, which is the right evidence for judging where a query pointed
    and which parameters it used, but NOT for judging completeness of the
    original run -- a near-real-time source has moved on since. The response is
    labelled `rerun: True` so callers never conflate the two.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    from utils.python_exe import python_executable
    from WebUI.handbook_studio_runner import _session_output_root
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    # A round can hold several attempts (the debugger retrying after an error).
    # Re-running a specific one is how you check whether an earlier failure
    # still reproduces; omitting the index runs the round's final code.
    result = run.get('result') or {}
    body = request.get_json(silent=True) or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    label = 'final'
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            chosen = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return jsonify({'success': False,
                            'error': f'No attempt {attempt_index} on this run'}), 400
        code = chosen.get('code') or ''
        label = f"attempt {chosen.get('attempt', attempt_index + 1)}"
    else:
        code = result.get('generated_code') or ''
    if not code.strip():
        return jsonify({'success': False,
                        'error': f'This run has no stored code for {label}'}), 400

    run_dir = _inspect_run_dir(user_id, session_id, run)
    if run_dir is None:
        # Original directory is gone (or the run recorded no artifact_ref);
        # write into a fresh, clearly-named directory under the same root.
        run_dir = os.path.join(_session_output_root(user_id, session_id),
                               f'rerun_{run_id[:8]}')
    if attempt_index is not None:
        # Keep each attempt's output apart, so comparing attempt 1 against
        # attempt 3 is a matter of looking at two folders, not one clobbered.
        run_dir = os.path.join(run_dir, f'attempt_{attempt_index + 1}')
    os.makedirs(run_dir, exist_ok=True)

    rewritten, rewrites = _studio_rewrite_output_paths(code, run_dir)
    user_script = os.path.join(run_dir, '_rerun_user.py')
    with open(user_script, 'w', encoding='utf-8') as handle:
        handle.write(rewritten)
    boot = os.path.join(run_dir, '_rerun_boot.py')
    with open(boot, 'w', encoding='utf-8') as handle:
        handle.write(_STUDIO_RERUN_SHIM
                     + f"\nimport runpy\nrunpy.run_path({user_script!r}, run_name='__main__')\n")

    http_log = os.path.join(run_dir, '_rerun_http.jsonl')
    open(http_log, 'w').close()
    env = dict(os.environ)
    env['RERUN_HTTP_LOG'] = http_log
    env['PYTHONUNBUFFERED'] = '1'

    timeout = int(session.get('execution_timeout_seconds') or 600)
    started = time.time()
    try:
        proc = subprocess.run([python_executable(), boot], cwd=run_dir, env=env,
                              capture_output=True, text=True, timeout=timeout)
        returncode, stdout, stderr, timed_out = (
            proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = -1, True
        stdout = exc.stdout.decode('utf-8', 'replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode('utf-8', 'replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')

    # Collect anything written since the run started, across the WHOLE session
    # root rather than just run_dir. Generated code composes its own output
    # paths, and a hardcoded absolute directory that only partly rewrites can
    # land the files beside the run folder instead of inside it. Scanning by
    # modification time finds them wherever they went, so a successful run
    # never reports "0 files" while its output sits one directory over.
    internal = {'_rerun_user.py', '_rerun_boot.py', '_rerun_http.jsonl'}
    session_root = _session_output_root(user_id, session_id)
    cutoff = started - 2  # small allowance for filesystem timestamp skew
    files = []
    for base, dirs, names in os.walk(session_root):
        dirs[:] = [d for d in dirs if d != '_handbook']
        for name in sorted(names):
            if name in internal:
                continue
            full = os.path.join(base, name)
            try:
                if os.path.getmtime(full) < cutoff:
                    continue
                size = os.path.getsize(full)
            except OSError:
                continue
            rel = os.path.relpath(full, session_root)
            files.append({
                'name': name,
                'artifact_ref': rel.replace(os.sep, '/'),
                'size_bytes': size,
                'extension': os.path.splitext(name)[1].lower(),
            })

    requests_seen = []
    try:
        with open(http_log, encoding='utf-8') as handle:
            for line in handle:
                try:
                    requests_seen.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass

    from datetime import datetime as _dt, timezone as _tz
    run_record = {
        'at': _dt.now(_tz.utc).isoformat(),
        'returncode': returncode, 'timed_out': timed_out,
        'elapsed_seconds': round(time.time() - started, 1),
        'stdout': stdout[-8000:], 'stderr': stderr[-8000:],
        'files': files, 'http_requests': requests_seen,
    }
    # Stored against the code that produced it. Held only in the browser
    # before, so a refresh discarded the evidence while its files stayed on
    # disk with nothing pointing at them.
    target = chosen if attempt_index is not None else result
    if attempt_index is None:
        run['result'] = result
    field = 'code' if attempt_index is not None else 'generated_code'
    is_final = attempt_index is None or attempt_index == len(attempts) - 1
    _record_version_run(target, field, run_record, run=run, is_final=is_final)
    store.save_session(user_id, session, session_id)

    return jsonify({
        'success': True, 'rerun': True, 'ran': label,
        'attempt': attempt_index,
        'returncode': returncode, 'timed_out': timed_out,
        'elapsed_seconds': run_record['elapsed_seconds'],
        'stdout': run_record['stdout'], 'stderr': run_record['stderr'],
        'files': files, 'path_rewrites': rewrites,
        'http_requests': requests_seen,
        'current_version': target.get('current_version'),
    })


def _version_label(origin):
    return {
        'agent': 'Agent original',
        'edit': 'Hand edit',
        'feedback': 'Revised from feedback',
    }.get(origin, 'Version')


def _ensure_code_versions(target, field, run=None, is_final=False):
    """Give a code slot a version list, backfilling one for older records.

    Records written before versioning kept only two snapshots: `original_code`
    (what the agent wrote, filed on first change) and the current code. Those
    become v1 and v2 here, deterministically -- the browser derives the same
    two for display before anything has been saved, so the ids it offers match
    the ones this produces.

    Nothing is invented: intermediate states that were never stored cannot be
    recovered, and the version list says only what the record actually holds.
    """
    versions = target.get('code_versions')
    if isinstance(versions, list) and versions:
        return versions

    current = str(target.get(field) or '')
    original = str(target.get('original_code') or '')
    history = target.get('code_feedback') or []
    last_feedback = history[-1] if isinstance(history, list) and history else {}

    versions = []
    if original and original != current:
        versions.append({
            'id': 'v1', 'n': 1, 'origin': 'agent',
            'label': _version_label('agent'), 'code': original,
            'at': (run or {}).get('finished_at') or (run or {}).get('started_at'),
            'feedback': None, 'explanation': None,
            'lines_added': None, 'lines_removed': None, 'run': None,
        })
        origin = 'feedback' if target.get('code_revised_from_feedback') else 'edit'
        versions.append({
            'id': 'v2', 'n': 2, 'origin': origin,
            'label': _version_label(origin), 'code': current,
            'at': target.get('code_revised_at') or target.get('code_edited_at'),
            'feedback': last_feedback.get('feedback'),
            'explanation': last_feedback.get('explanation'),
            'lines_added': last_feedback.get('lines_added'),
            'lines_removed': last_feedback.get('lines_removed'),
            'run': None,
        })
    else:
        versions.append({
            'id': 'v1', 'n': 1, 'origin': 'agent',
            'label': _version_label('agent'), 'code': current,
            'at': (run or {}).get('finished_at') or (run or {}).get('started_at'),
            'feedback': None, 'explanation': None,
            'lines_added': None, 'lines_removed': None, 'run': None,
        })

    # The files the original test run recorded belong to the code that produced
    # them -- the first version -- but only when this slot IS the code that ran
    # last. Attaching them to an earlier attempt would credit one attempt with
    # another's output.
    if is_final and run:
        result = run.get('result') or {}
        recorded = result.get('downloaded_files') or []
        if recorded:
            # Noted as a count, NOT as a run record. Copying the file list here
            # made the round's own outputs render twice at once -- inside the
            # card as this version's result, and again in the round's "Data
            # outputs" section -- which read as two different sets of files.
            versions[0]['original_output_files'] = len(recorded)

    target['code_versions'] = versions
    target['current_version'] = versions[-1]['id']
    return versions


def _append_code_version(target, field, code, origin, run=None, is_final=False,
                         **meta):
    """Record a new code version and make it the current one."""
    from datetime import datetime as _dt, timezone as _tz
    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    # An unchanged body is not a new version; it would clutter the picker with
    # entries a reviewer cannot tell apart.
    if versions and (versions[-1].get('code') or '') == code:
        target['current_version'] = versions[-1]['id']
        return versions[-1]
    entry = {
        'id': f'v{len(versions) + 1}',
        'n': len(versions) + 1,
        'origin': origin,
        'label': _version_label(origin),
        'code': code,
        'at': _dt.now(_tz.utc).isoformat(),
        'feedback': meta.get('feedback'),
        'explanation': meta.get('explanation'),
        'lines_added': meta.get('lines_added'),
        'lines_removed': meta.get('lines_removed'),
        'run': None,
    }
    versions.append(entry)
    target['code_versions'] = versions
    target['current_version'] = entry['id']
    return entry


def _record_version_run(target, field, run_record, run=None, is_final=False):
    """Attach a re-run's outcome to whichever version was executed.

    Re-run output used to live only in the browser's memory, so a refresh threw
    away the evidence a reviewer had just generated while the files sat on disk
    unreferenced. Stored on the version, it survives the reload and stays tied
    to the exact code that produced it.
    """
    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    current_id = target.get('current_version') or versions[-1]['id']
    for entry in versions:
        if entry.get('id') == current_id:
            entry['run'] = run_record
            return entry
    versions[-1]['run'] = run_record
    return versions[-1]


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/attempt-code',
           methods=['POST'])
def handbook_studio_edit_code(session_id, run_id):
    """Replace one test run's stored retrieval code with a hand-edited version.

    The re-run endpoint above executes whatever code the record holds, so
    editing here and re-running there is how a reviewer fixes a dead endpoint,
    a wrong bbox or a missing key by hand instead of asking the model for
    another round.

    What the model originally wrote is never lost: the first edit copies it to
    `original_code` and flags the attempt `code_edited`, so the experiment can
    still separate agent-authored code from human-repaired code (and `revert`
    puts it back). Recorded verdicts, files and usage are left alone -- they
    describe the ORIGINAL run, and rewriting them would make an edited attempt
    look like the agent had produced it.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from datetime import datetime as _dt, timezone as _tz
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    result = run.get('result') or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    revert = bool(body.get('revert'))
    # Where the code lives depends on the shape of the record: a round with a
    # debugger loop stores one entry per attempt, an older single-shot round
    # stores only `generated_code`.
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            target = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return jsonify({'success': False,
                            'error': f'No attempt {attempt_index} on this run'}), 400
    else:
        # A record whose `result` slot was missing entirely gets one, so the
        # edit has somewhere to live rather than being written to a throwaway.
        run['result'] = result
        target = result

    if revert:
        original = target.get('original_code')
        if not original:
            return jsonify({'success': False,
                            'error': 'This code has not been edited.'}), 400
        code = original
    else:
        code = body.get('code')
        if not isinstance(code, str) or not code.strip():
            return jsonify({'success': False, 'error': 'Code cannot be empty.'}), 400
        if len(code) > 400_000:
            return jsonify({'success': False,
                            'error': 'Code is too long to store (400 KB limit).'}), 400
        code = code.replace('\r\n', '\n')

    field = 'code' if attempt_index is not None else 'generated_code'
    is_final = attempt_index is None or attempt_index == len(attempts) - 1
    # BEFORE the slot is touched. Backfilling afterwards would describe the new
    # code with the old version's label and then treat the append as a no-op,
    # losing whatever was being replaced.
    _ensure_code_versions(target, field, run=run, is_final=is_final)
    if revert:
        # Revert moves back to the agent's own version rather than filing a
        # copy of it: the picker should show the history that happened, not a
        # new entry every time someone looks at the original.
        versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
        target[field] = code
        for entry in versions:
            if (entry.get('code') or '') == code:
                target['current_version'] = entry['id']
                break
        target.pop('original_code', None)
        target['code_edited'] = False
        target['code_edited_at'] = None
    else:
        # Keyed on original_code, not code_edited: a model revision from
        # reviewer feedback also files the agent's version here, and keying on
        # the hand-edit flag would overwrite it with that revision -- leaving
        # Revert pointing at a revision rather than at what the agent wrote.
        if not target.get('original_code'):
            target['original_code'] = target.get(field) or ''
        target[field] = code
        target['code_edited'] = True
        target['code_edited_at'] = _dt.now(_tz.utc).isoformat()
        _append_code_version(target, field, code, 'edit',
                             run=run, is_final=is_final)

    store.save_session(user_id, session, session_id)
    return jsonify({
        'success': True,
        'attempt': attempt_index,
        'code': code,
        'code_edited': bool(target.get('code_edited')),
        'original_code': target.get('original_code') or '',
        'code_versions': target.get('code_versions') or [],
        'current_version': target.get('current_version'),
    })




def _studio_revision_slot(session, run, body):
    """Locate the code slot a revision targets.

    Returns ``(target, field, attempt_index, error)`` where ``error`` is
    ``(message, status)`` or None. Mirrors the hand-edit endpoint's record
    shapes: a debugger loop stores one entry per attempt, an older single-shot
    round only ``generated_code``.
    """
    result = run.get('result') or {}
    attempts = ((result.get('execution') or {}).get('attempts') or [])
    attempt_index = body.get('attempt')
    if attempt_index is not None:
        try:
            attempt_index = int(attempt_index)
            target = attempts[attempt_index]
        except (TypeError, ValueError, IndexError):
            return None, None, None, (f'No attempt {attempt_index} on this run', 400)
        return target, 'code', attempt_index, None
    run['result'] = result
    return result, 'generated_code', None, None


def _studio_store_revision(user_id, session, session_id, target, field,
                           current_code, feedback, revision, provider):
    """Persist one feedback-driven revision. Returns ``(payload, error)``.

    What the agent originally wrote is preserved in ``original_code`` (so
    Revert still restores it), but the attempt is flagged
    ``code_revised_from_feedback`` rather than ``code_edited``: for the
    experiment, code a model rewrote from an instruction is not the same
    evidence as code a human typed.
    """
    from datetime import datetime as _dt, timezone as _tz
    from WebUI import handbook_studio_store as store

    revised = (revision.get('code') or '').replace('\r\n', '\n').strip()
    if not revised:
        # A reply with no code block means the model answered in prose. Storing
        # nothing and reporting success would look like a silent no-op, so the
        # reason is handed back instead.
        return None, ('The model replied without any code. It said: '
                      + ((revision.get('raw') or '').strip()[:400] or '(nothing)'), 422)
    if len(revised) > 400_000:
        return None, ('Revised code is too long to store (400 KB limit).', 400)
    if revised == current_code.strip():
        return None, ('The model returned the same code unchanged. '
                      'Try saying more specifically what to change.', 422)

    # How big the change actually was. A feedback revision is supposed to touch
    # only what was asked for; showing the size on the card is what lets a
    # reviewer notice a model that reformatted the whole file instead.
    import difflib
    diff = list(difflib.unified_diff(current_code.splitlines(),
                                     revised.splitlines(), n=0, lineterm=''))
    lines_added = sum(1 for line in diff
                      if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in diff
                        if line.startswith('-') and not line.startswith('---'))

    now = _dt.now(_tz.utc).isoformat()
    # Snapshot the code being replaced before replacing it, for the same reason
    # the edit endpoint does: a backfill that runs afterwards sees only the new
    # body and the version it was meant to preserve is gone.
    _ensure_code_versions(target, field)
    # The agent's own version is filed once and never overwritten, whether the
    # first change came from a hand edit or from feedback -- Revert has to
    # reach the code the agent actually wrote, not an intermediate revision.
    if not target.get('original_code'):
        target['original_code'] = current_code
    target[field] = revised
    target['code_revised_from_feedback'] = True
    target['code_revised_at'] = now
    history = target.get('code_feedback')
    if not isinstance(history, list):
        history = []
    history.append({
        'feedback': feedback,
        'explanation': revision.get('explanation') or '',
        'lines_added': lines_added,
        'lines_removed': lines_removed,
        'at': now,
        'model': revision.get('model') or '',
        'provider': provider,
    })
    target['code_feedback'] = history
    _append_code_version(target, field, revised, 'feedback',
                         feedback=feedback,
                         explanation=revision.get('explanation') or '',
                         lines_added=lines_added, lines_removed=lines_removed)

    store.save_session(user_id, session, session_id)
    return {
        'code': revised,
        'code_versions': target.get('code_versions') or [],
        'current_version': target.get('current_version'),
        'explanation': revision.get('explanation') or '',
        'original_code': target.get('original_code') or '',
        'code_revised_from_feedback': True,
        'code_feedback': history,
    }, None


def _studio_revision_inputs(session, run, target, field):
    """The context a revision is given besides the feedback itself."""
    result = run.get('result') or {}
    source = session.get('generated_source') or {}
    return {
        'current_code': str(target.get(field) or ''),
        'task_text': str(run.get('retrieval_task')
                         or session.get('retrieval_task') or ''),
        'handbook_text': str(source.get('handbook') or ''),
        'last_error': str(target.get('error') or result.get('error') or ''),
    }


def _studio_revise_agent(session, openai_key, anthropic_key):
    """The retrieval agent configured the way this session's tests ran."""
    from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent
    provider = str(session.get('provider') or 'openai')
    if provider not in ('openai', 'claude'):
        provider = 'openai'
    agent = DataRetrieverAgent(
        api_key=openai_key, evaluation=True,
        model=session.get('data_retrieval_model'),
        reasoning_effort=session.get('data_retrieval_reasoning_effort') or None,
        provider=provider, anthropic_api_key=anthropic_key)
    return agent, provider


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/select-version',
           methods=['POST'])
def handbook_studio_select_version(session_id, run_id):
    """Load one stored version of a run's retrieval code.

    Selecting a version puts its code back in the slot the re-run endpoint
    executes, so "load v1 and run it again" needs no copy-paste. The version's
    own recorded output travels with it, which is what makes comparing two
    versions a matter of clicking between them.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    version_id = str(body.get('version') or '').strip()
    if not version_id:
        return jsonify({'success': False, 'error': 'No version given.'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    is_final = attempt_index is None or attempt_index == len(attempts) - 1

    versions = _ensure_code_versions(target, field, run=run, is_final=is_final)
    chosen = next((v for v in versions if v.get('id') == version_id), None)
    if chosen is None:
        return jsonify({'success': False,
                        'error': f'No version {version_id} on this code.'}), 404

    target[field] = chosen.get('code') or ''
    target['current_version'] = chosen['id']
    store.save_session(user_id, session, session_id)
    return jsonify({
        'success': True, 'attempt': attempt_index,
        'version': chosen['id'], 'code': target[field],
        'code_versions': versions, 'current_version': chosen['id'],
    })


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/revise-code',
           methods=['POST'])
def handbook_studio_revise_code(session_id, run_id):
    """Rewrite one test run's retrieval code from a reviewer's written feedback.

    The sibling of the hand-edit endpoint above: instead of typing the fix
    yourself, you say what is wrong ("it downloads the whole state -- filter to
    the county in the retrieval task") and the model returns a revised program,
    which is stored exactly where a hand edit would be stored and executed by
    the same re-run endpoint.

    The UI uses the streaming variant below; this one is the plain
    request/response form, kept for scripted callers.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    feedback = str(body.get('feedback') or '').strip()
    if not feedback:
        return jsonify({'success': False,
                        'error': 'Say what should change before sending.'}), 400
    if len(feedback) > 20_000:
        return jsonify({'success': False,
                        'error': 'Feedback is too long (20 KB limit).'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]

    inputs = _studio_revision_inputs(session, run, target, field)
    if not inputs['current_code'].strip():
        return jsonify({'success': False,
                        'error': 'This attempt has no code to revise.'}), 400
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    _ensure_code_versions(target, field, run=run,
                          is_final=(attempt_index is None
                                    or attempt_index == len(attempts) - 1))

    try:
        agent, provider = _studio_revise_agent(session, g.api_key, g.anthropic_api_key)
        revision = agent.revise_code_from_feedback(
            feedback, inputs['current_code'], inputs['task_text'],
            inputs['handbook_text'], last_error=inputs['last_error'])
        revision['model'] = agent.model
    except Exception as exc:
        import traceback
        print(f"[revise-code] Error: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__,
        }), 502

    payload, error = _studio_store_revision(
        user_id, session, session_id, target, field,
        inputs['current_code'], feedback, revision, provider)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]
    return jsonify({'success': True, 'attempt': attempt_index, **payload})


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/revise-code-stream',
           methods=['POST'])
def handbook_studio_revise_code_stream(session_id, run_id):
    """The same revision, streamed as the model writes it.

    Rewriting a few hundred lines takes a minute or more. Without a stream the
    card sits on a disabled button for that whole time, which is
    indistinguishable from a dead button -- so the model's output is forwarded
    as it arrives and the stored result is sent as a final event.

    Pre-flight problems (no key, unknown session/run, empty feedback) are
    reported as ordinary JSON errors, since nothing has been streamed yet; once
    the stream opens, everything including failure is an SSE event.
    """
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    session = store.get_session(user_id, session_id)
    if session is None:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    body = request.get_json(silent=True) or {}
    feedback = str(body.get('feedback') or '').strip()
    if not feedback:
        return jsonify({'success': False,
                        'error': 'Say what should change before sending.'}), 400
    if len(feedback) > 20_000:
        return jsonify({'success': False,
                        'error': 'Feedback is too long (20 KB limit).'}), 400

    target, field, attempt_index, error = _studio_revision_slot(session, run, body)
    if error:
        return jsonify({'success': False, 'error': error[0]}), error[1]

    inputs = _studio_revision_inputs(session, run, target, field)
    if not inputs['current_code'].strip():
        return jsonify({'success': False,
                        'error': 'This attempt has no code to revise.'}), 400
    attempts = ((run.get('result') or {}).get('execution') or {}).get('attempts') or []
    _ensure_code_versions(target, field, run=run,
                          is_final=(attempt_index is None
                                    or attempt_index == len(attempts) - 1))

    # Read out of the request context before the generator starts: by the time
    # it runs, `g` belongs to no request.
    openai_key, anthropic_key = g.api_key, g.anthropic_api_key

    def generate():
        import queue
        import threading

        yield sse_data({'type': 'revise_started', 'attempt': attempt_index,
                        'lines': len(inputs['current_code'].splitlines())})

        chunks = queue.Queue()
        box = {}

        def worker():
            # Storing happens HERE, in the worker, not after the send loop
            # below: if the browser goes away mid-stream, the generator is
            # closed at its next yield and anything after it never runs. A
            # revision the model has already been paid for would then be lost,
            # which is the one outcome a reviewer cannot recover from -- they
            # would see nothing on the card and have no way to know it ran.
            try:
                agent, provider = _studio_revise_agent(
                    session, openai_key, anthropic_key)
                box['provider'] = provider
                revision = agent.revise_code_from_feedback(
                    feedback, inputs['current_code'], inputs['task_text'],
                    inputs['handbook_text'], last_error=inputs['last_error'],
                    stream_callback=lambda text: chunks.put(text))
                revision['model'] = agent.model
                payload, store_error = _studio_store_revision(
                    user_id, session, session_id, target, field,
                    inputs['current_code'], feedback, revision, provider)
                if store_error:
                    box['error'] = store_error[0]
                else:
                    box['payload'] = payload
            except Exception as exc:          # surfaced as an SSE error below
                import traceback
                print(f"[revise-code-stream] Error: {traceback.format_exc()}")
                box['error'] = (f'{type(exc).__name__}: {exc}'
                                if str(exc) else type(exc).__name__)
            finally:
                chunks.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            chunk = chunks.get()
            if chunk is None:
                break
            if chunk:
                yield sse_data({'type': 'revise_delta', 'text': chunk})

        if box.get('error'):
            yield sse_data({'type': 'error', 'error': box['error']})
            return
        yield sse_data({'type': 'revise_done', 'attempt': attempt_index,
                        **(box.get('payload') or {})})

    return sse_response(generate())


# Absolute output directories the generated code baked in -- often from a
# different machine entirely (the Windows runs wrote to %TEMP%\GISHBStudio).
# The deeper "…/<user>/<session>/test_<hex>" tail is OPTIONAL: some generated
# code stores only the GISHBStudio root and joins subdirectories itself, and
# missing those left the path pointing at a drive letter that does not exist on
# this machine, so the re-run wrote nowhere.
# ONE pattern, ONE pass. The Windows and POSIX roots are alternatives of a
# single regex rather than two successive substitutions, because the
# replacement path itself lives under the POSIX root: running a second sub over
# the output made that pattern match the text just written and substitute
# again, yielding ".../attempt_1/<user>/<session>/rerun_x/attempt_1/file".
# re.sub never re-scans what it has already emitted, so a single pass cannot
# double up.
#
# A trailing separator IS consumed, and _studio_rewrite_output_paths puts one
# back — that pairing keeps every shape valid. Consuming without restoring
# glued a filename onto the directory; restoring without consuming left a
# stray Windows backslash mid-path.
_STUDIO_OUT_DIR_RE = re.compile(
    r'(?:'
    r'[A-Za-z]:(?:\\{1,2}|/)(?:[^"\'\n]*?)GISHBStudio'
    r'(?:(?:\\{1,2}|/)[^"\'\n]*?(?:test_[0-9a-f]+|_handbook))?'
    r'|'
    r'/tmp/gis-co-scientist-hbstudio(?:/[^"\'\n]*?test_[0-9a-f]+)?'
    r'|'
    # The durable root (handbook_studio_runner._default_output_root) on any
    # OS. Code generated on Windows and re-run on Linux must be caught here:
    # left alone, the backslashes are legal filename characters there, and the
    # whole Windows path becomes one oddly named directory the UI cannot open.
    r'(?:[A-Za-z]:(?:\\{1,2}|/)|/)(?:[^"\'\n]*?)gis-co-scientist(?:\\{1,2}|/)outputs'
    r'(?:(?:\\{1,2}|/)[^"\'\n]*?(?:test_[0-9a-f]+|rerun_[0-9a-f]+|_handbook))?'
    r')'
    r'(?:\\{1,2}|/)?'
)

# Logs each request with its FINAL url (parameters merged) and POST bodies --
# what the original capture missed, which is why query parameters could not be
# checked mechanically for most runs.
_STUDIO_RERUN_SHIM = '''
import json as _json, os as _os, time as _time
_LOG = _os.environ.get("RERUN_HTTP_LOG")
def _install():
    try:
        import requests
    except Exception:
        return
    _orig = requests.sessions.Session.request
    def _wrapped(self, method, url, *a, **kw):
        rec = {"method": method, "url": url, "t": _time.time()}
        try:
            resp = _orig(self, method, url, *a, **kw)
            rec["url"] = resp.url
            rec["status"] = resp.status_code
            return resp
        except Exception as exc:
            rec["error"] = type(exc).__name__
            raise
        finally:
            body = kw.get("data") or kw.get("json")
            if body is not None:
                rec["body"] = str(body)[:4000]
            if _LOG:
                try:
                    with open(_LOG, "a", encoding="utf-8") as fh:
                        fh.write(_json.dumps(rec) + "\\n")
                except Exception:
                    pass
    requests.sessions.Session.request = _wrapped
_install()
'''


def _studio_rewrite_output_paths(code, out_dir):
    """Point every hardcoded artifact directory at out_dir.

    Whatever followed the directory in the original string has to keep working.
    Generated code stores these three ways, and all of them must survive:

        out_dir  = r"...\\GISHBStudio\\<user>\\<session>\\test_ab12"
        out_file = r"...\\GISHBStudio\\<user>\\<session>\\test_ab12\\data.geojson"
        root     = r"...\\GISHBStudio"            # joined with more parts later

    So when the match consumed a trailing separator, the replacement keeps one.
    Dropping it is what produced paths like ".../attempt_1PADUS_NPS.geojson",
    where the filename was glued straight onto the directory.
    """
    replacement = out_dir.replace('\\', '/')
    count = 0

    def _sub(match):
        nonlocal count
        count += 1
        text = match.group(0)
        return replacement + '/' if text[-1:] in ('\\', '/') else replacement

    return _STUDIO_OUT_DIR_RE.sub(_sub, code), count


# ── Inspect tab: look at the data a test run actually returned ──────────────
# The Validation tab records a verdict; these endpoints are what let a reviewer
# SEE what they are judging -- the retrieved file itself, where it sits on a
# map, and the place the task named. Everything is scoped to the caller's own
# session output root, and paths are resolved before use so a crafted
# artifact_ref cannot escape it.

_INSPECT_MAX_PREVIEW = 5 * 1024 * 1024
_INSPECT_TEXT_EXT = {'.csv', '.tsv', '.json', '.geojson', '.txt', '.xml',
                     '.toml', '.meta', '.jsonl', '.wkt', '.prj', '.cpg'}


def _inspect_run_dir(user_id, session_id, run):
    """Absolute directory holding one test run's artifacts, or None."""
    from WebUI.handbook_studio_runner import _session_output_root
    refs = [f.get('artifact_ref') for f in
            ((run.get('result') or {}).get('downloaded_files') or [])
            if f.get('artifact_ref')]
    if not refs:
        return None
    first = refs[0].replace('\\', '/').split('/')[0]
    root = _session_output_root(user_id, session_id)
    candidate = os.path.abspath(os.path.join(root, first))
    if not candidate.startswith(os.path.abspath(root) + os.sep):
        return None
    return candidate if os.path.isdir(candidate) else None


def _inspect_find_run(session, run_id):
    for run in (session.get('test_runs') or []):
        if run.get('id') == run_id:
            return run
    return None


def _evicted_message(session_roots, artifact_ref, name):
    """Why a recorded output is missing, if the retention policy removed it.

    Distinguishes "evicted to stay within the server's storage quota" from a
    file that was never there, so the UI can tell the user to re-run rather
    than leaving them to wonder whether the run produced anything.
    """
    from WebUI import output_retention
    parts = [p for p in str(artifact_ref or '').replace('\\', '/').split('/') if p]
    run_dirs = []
    for root in session_roots:
        if parts:
            run_dirs.append(os.path.join(root, parts[0]))
        elif os.path.isdir(root):
            run_dirs.extend(os.path.join(root, d) for d in os.listdir(root)
                            if d != '_handbook')
    for run_dir in run_dirs:
        reason = output_retention.evicted_reason(run_dir, os.path.basename(name or ''))
        if reason:
            return reason
    return None


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/artifacts')
def handbook_studio_run_artifacts(session_id, run_id):
    """List the files one test run left on disk, with their recorded evidence.

    A run whose artifacts have been swept from the temp directory returns an
    empty list plus `available: False`, so the UI can say the data is gone
    rather than implying the run produced nothing.
    """
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404

    result = run.get('result') or {}
    evidence = {str(item.get('name')): item
                for item in (result.get('output_evidence') or [])
                if isinstance(item, dict)}
    run_dir = _inspect_run_dir(user_id, session_id, run)
    files = []
    if run_dir:
        for base, _dirs, names in os.walk(run_dir):
            for name in sorted(names):
                path = os.path.join(base, name)
                rel = os.path.relpath(path, run_dir).replace('\\', '/')
                ext = os.path.splitext(name)[1].lower()
                files.append({
                    'name': name,
                    'rel': rel,
                    'size_bytes': os.path.getsize(path),
                    'extension': ext,
                    'previewable': ext in _INSPECT_TEXT_EXT,
                    'evidence': evidence.get(name) or {},
                })
    from WebUI import output_retention
    return jsonify({
        'success': True,
        'available': bool(run_dir),
        'files': files,
        # Files the retention policy removed since the run, so the UI can say
        # "evicted, re-run to regenerate" instead of implying they never were.
        'evicted': output_retention.evicted_files(run_dir) if run_dir else {},
        'retention': run.get('retention') or {},
        'output_evidence': result.get('output_evidence') or [],
        'http_requests': result.get('http_requests') or [],
        'retrieval_task': run.get('retrieval_task') or session.get('retrieval_task') or '',
    })


class _ZipSink:
    """A write-only, non-seekable target for zipfile that we drain as we go.

    zipfile needs only write()/tell()/flush() on a non-seekable stream (it
    then emits data descriptors), so the whole archive never exists on disk
    or in memory -- only the chunk currently being sent.
    """

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0

    def write(self, data):
        self._buf += data
        self._pos += len(data)
        return len(data)

    def tell(self):
        return self._pos

    def flush(self):
        pass

    def drain(self):
        out = bytes(self._buf)
        self._buf.clear()
        return out


def _stream_zip(run_dir, arc_root):
    import zipfile
    sink = _ZipSink()
    with zipfile.ZipFile(sink, 'w', zipfile.ZIP_DEFLATED) as archive:
        for base, _dirs, names in os.walk(run_dir):
            for name in sorted(names):
                if name.startswith('.') and name.endswith('.json'):
                    continue  # runner bookkeeping (.retention.json, .evicted.json)
                path = os.path.join(base, name)
                arcname = os.path.join(arc_root, os.path.relpath(path, run_dir))
                try:
                    with open(path, 'rb') as src, archive.open(arcname, 'w') as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                            yield sink.drain()
                except OSError:
                    continue
                yield sink.drain()
    yield sink.drain()


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>/download-all')
def handbook_studio_run_download_all(session_id, run_id):
    """Everything one test run left on disk, as a zip streamed straight to
    the browser. Exists so a user on a quota-limited host can claim a run's
    output in one click before the retention policy evicts its bulk files."""
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    run_dir = _inspect_run_dir(user_id, session_id, run)
    if not run_dir:
        return jsonify({'success': False,
                        'error': 'Artifacts for this run are no longer on disk'}), 410
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', str(session.get('name') or 'session'))[:40]
    filename = f"{stem}_run{run.get('test_number') or ''}_{run_id[:8]}.zip"
    response = Response(stream_with_context(_stream_zip(run_dir, filename[:-4])),
                        mimetype='application/zip')
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@app.route('/api/handbook-studio-sessions/<session_id>/test-runs/<run_id>'
           '/artifacts/<path:rel>')
def handbook_studio_run_artifact_file(session_id, run_id, rel):
    """Return one artifact: parsed preview by default, raw bytes with ?raw=1."""
    user_id, session, _is_owner = _studio_session_for_read(session_id)
    if session is None:
        if not _studio_identity_key():
            return jsonify({'success': False, 'error': 'No API key provided'}), 401
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    run = _inspect_find_run(session, run_id)
    if run is None:
        return jsonify({'success': False, 'error': 'Test run not found'}), 404
    run_dir = _inspect_run_dir(user_id, session_id, run)
    if not run_dir:
        return jsonify({'success': False,
                        'error': 'Artifacts for this run are no longer on disk'}), 410

    path = os.path.abspath(os.path.join(run_dir, rel))
    if not path.startswith(run_dir + os.sep) or not os.path.isfile(path):
        from WebUI import output_retention
        evicted = output_retention.evicted_reason(run_dir, os.path.basename(rel))
        if evicted:
            return jsonify({'success': False, 'evicted': True, 'error': evicted}), 410
        return jsonify({'success': False, 'error': 'File not found'}), 404

    if request.args.get('raw'):
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path))

    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    if size > _INSPECT_MAX_PREVIEW:
        return jsonify({'success': True, 'kind': 'too_large',
                        'size_bytes': size, 'name': os.path.basename(path)})

    # GeoJSON goes back as geometry the map can draw; everything else textual
    # goes back as text for the table/code view.
    if ext in ('.geojson', '.json'):
        try:
            with open(path, encoding='utf-8', errors='replace') as handle:
                data = json.load(handle)
        except (ValueError, OSError) as exc:
            return jsonify({'success': False, 'error': f'Unreadable: {exc}'}), 400
        kind = 'geojson' if isinstance(data, dict) and data.get('type') in (
            'FeatureCollection', 'Feature', 'GeometryCollection') else 'json'
        return jsonify({'success': True, 'kind': kind, 'name': os.path.basename(path),
                        'size_bytes': size, 'data': data})

    if ext in _INSPECT_TEXT_EXT:
        with open(path, encoding='utf-8', errors='replace') as handle:
            text = handle.read(_INSPECT_MAX_PREVIEW)
        return jsonify({'success': True,
                        'kind': 'csv' if ext in ('.csv', '.tsv') else 'text',
                        'name': os.path.basename(path), 'size_bytes': size,
                        'text': text})

    # Vector formats GDAL can read (gpkg/shp) are converted to GeoJSON so the
    # map can draw them without the browser needing a GDAL of its own.
    if ext in ('.gpkg', '.shp'):
        try:
            import geopandas as gpd
            frame = gpd.read_file(path)
            if frame.crs is not None and str(frame.crs) != 'EPSG:4326':
                frame = frame.to_crs(epsg=4326)
            if len(frame) > 5000:
                frame = frame.iloc[:5000]
            return jsonify({'success': True, 'kind': 'geojson',
                            'name': os.path.basename(path), 'size_bytes': size,
                            'truncated': len(frame) >= 5000,
                            'data': json.loads(frame.to_json())})
        except Exception as exc:
            return jsonify({'success': False, 'error': f'Could not read: {exc}'}), 400

    if ext in ('.tif', '.tiff'):
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            with rasterio.open(path) as src:
                west, south, east, north = transform_bounds(
                    src.crs, 'EPSG:4326', *src.bounds) if src.crs else (
                        src.bounds.left, src.bounds.bottom,
                        src.bounds.right, src.bounds.top)
                return jsonify({
                    'success': True, 'kind': 'raster',
                    'name': os.path.basename(path), 'size_bytes': size,
                    'width': src.width, 'height': src.height,
                    'bands': src.count, 'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'bbox': {'west': west, 'south': south,
                             'east': east, 'north': north},
                })
        except Exception as exc:
            return jsonify({'success': False, 'error': f'Could not read: {exc}'}), 400

    return jsonify({'success': True, 'kind': 'binary',
                    'name': os.path.basename(path), 'size_bytes': size})


@app.route('/api/handbook-studio/reference-place')
def handbook_studio_reference_place():
    """Resolve a place name to an outline, for comparing against an extent.

    Backed by the same local-first cache the deterministic spatial check uses,
    so the map and the mechanical verdict always agree about where a place is.
    """
    place = (request.args.get('q') or '').strip()
    if not place:
        return jsonify({'success': False, 'error': 'No place given'}), 400
    from agents.data_agent.validation import reference_geometry
    allow_network = request.args.get('offline') != '1'
    geometry, provenance = reference_geometry(place, allow_network=allow_network)
    if geometry is None:
        return jsonify({'success': True, 'resolved': False,
                        'place': place, 'provenance': provenance})
    minx, miny, maxx, maxy = geometry.bounds
    rings = []
    geoms = ([geometry] if geometry.geom_type == 'Polygon'
             else list(geometry.geoms) if geometry.geom_type == 'MultiPolygon'
             else [])
    span = max(maxx - minx, maxy - miny)
    tolerance = max(span / 400.0, 1e-4)
    for poly in sorted(geoms, key=lambda p: -p.area)[:60]:
        simple = poly.simplify(tolerance, preserve_topology=True)
        ring = simple.exterior if not simple.is_empty else poly.exterior
        coords = [round(v, 5) for xy in ring.coords for v in xy]
        if len(coords) >= 8:
            rings.append(coords)
    return jsonify({
        'success': True, 'resolved': True, 'place': place,
        'provenance': provenance, 'rings': rings,
        'bbox': {'west': minx, 'south': miny, 'east': maxx, 'north': maxy},
    })


@app.route('/api/handbook-studio/basemap')
def handbook_studio_basemap():
    """Coastline rings for the Inspect map.

    No tile server is involved: the map is vector-only and self-contained, so
    it works offline and behind a proxy. Natural Earth 1:110m ships inside
    pyogrio's test fixtures, so this costs no download.
    """
    cached = getattr(app, '_hbs_basemap_cache', None)
    if cached is not None:
        return jsonify({'success': True, 'rings': cached})
    generated = os.path.join(
        BASE_DIR, '..', 'eval', 'retro', 'out', 'basemap.json')
    rings = []
    if os.path.exists(generated):
        try:
            with open(generated) as handle:
                rings = json.load(handle)
        except (ValueError, OSError):
            rings = []
    if not rings:
        try:
            import geopandas as gpd
            import pyogrio
            source = os.path.join(
                os.path.dirname(pyogrio.__file__), 'tests', 'fixtures',
                'naturalearth_lowres', 'naturalearth_lowres.shp')
            frame = gpd.read_file(source)
            frame['geometry'] = frame.geometry.simplify(0.02, preserve_topology=True)
            for geom in frame.geometry:
                if geom is None or geom.is_empty:
                    continue
                polys = ([geom] if geom.geom_type == 'Polygon'
                         else list(geom.geoms))
                for poly in polys:
                    coords = [round(v, 3) for xy in poly.exterior.coords for v in xy]
                    if len(coords) >= 8:
                        rings.append(coords)
        except Exception:
            rings = []
    app._hbs_basemap_cache = rings
    return jsonify({'success': True, 'rings': rings})


@app.route('/api/handbook-studio-sessions/<session_id>/duplicate', methods=['POST'])
def handbook_studio_duplicate(session_id):
    """Clone a session's generated handbook into a new session, skipping
    generation, so the same handbook can be tested against a different
    retrieval task (e.g. a different difficulty tier of the same source)."""
    identity_key = _studio_identity_key()
    if not identity_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 401
    from WebUI import handbook_studio_store as store
    user_id = hash_api_key(identity_key)
    try:
        new_session = store.duplicate_session(user_id, session_id)
        return jsonify({'success': True, 'session': new_session}), 201
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@app.route('/api/handbook-studio-sessions/<session_id>/output-preview',
           methods=['GET'])
def handbook_studio_output_preview(session_id):
    """Preview or download one caller-owned Generate & Test output artifact."""
    from WebUI.handbook_studio_runner import _session_output_roots

    user_id, session, _is_owner = _studio_session_for_read(session_id)
    try:
        if session is None:
            if not _studio_identity_key():
                return jsonify({'success': False, 'error': 'No API key provided'}), 401
            return jsonify({
                'success': False, 'error': 'Session not found'
            }), 404

        # Both roots: outputs written before the durable root existed still
        # live under the old /tmp path, and a run recorded months ago should
        # still open its files rather than 404 because a constant changed.
        session_roots = _session_output_roots(user_id, session_id)
        session_root = session_roots[0]
        artifact_ref = str(request.args.get('ref') or '').strip().replace('\\', '/')
        candidate = None

        def containing_root(path):
            try:
                absolute = os.path.abspath(path)
                return next((
                    root for root in session_roots
                    if os.path.commonpath([root, absolute]) == root
                ), None)
            except (OSError, ValueError):
                return None

        if artifact_ref:
            if artifact_ref.startswith('/') or '..' in artifact_ref.split('/'):
                return jsonify({
                    'success': False, 'error': 'Invalid output reference'
                }), 400
            parts = [part for part in artifact_ref.split('/') if part]
            for root in session_roots:
                possible = os.path.abspath(os.path.join(root, *parts))
                if os.path.isfile(possible):
                    candidate = possible
                    session_root = root
                    break
            if candidate is None:
                candidate = os.path.abspath(os.path.join(session_roots[0], *parts))
            if not containing_root(candidate):
                return jsonify({
                    'success': False, 'error': 'Output access denied'
                }), 403
        else:
            requested_name = os.path.basename(
                str(request.args.get('name') or '').strip())
            if not requested_name:
                return jsonify({
                    'success': False, 'error': 'Output reference is required'
                }), 400
            try:
                requested_size = int(request.args.get('size') or -1)
            except (TypeError, ValueError):
                requested_size = -1
            matches = []
            for root in session_roots:
                if not os.path.isdir(root):
                    continue
                for directory, dirnames, filenames in os.walk(root):
                    dirnames[:] = [name for name in dirnames
                                   if name != '_handbook']
                    if requested_name in filenames:
                        path = os.path.join(directory, requested_name)
                        try:
                            if (requested_size >= 0
                                    and os.path.getsize(path) != requested_size):
                                continue
                            matches.append(path)
                        except OSError:
                            continue
            if matches:
                candidate = max(matches, key=os.path.getmtime)
                session_root = containing_root(candidate) or session_roots[0]

        if (not candidate or not containing_root(candidate)
                or not os.path.isfile(candidate)):
            evicted = _evicted_message(
                session_roots, artifact_ref,
                os.path.basename(candidate or '') or request.args.get('name'))
            if evicted:
                return jsonify({'success': False, 'evicted': True,
                                'error': evicted}), 410
            app.logger.warning(
                'Studio output missing: ref=%r name=%r resolved=%r roots=%r',
                artifact_ref, request.args.get('name'), candidate, session_roots)
            return jsonify({
                'success': False, 'error': 'Output file is no longer available'
            }), 404

        session_root = containing_root(candidate) or session_root
        artifact_ref = os.path.relpath(
            candidate, session_root).replace(os.sep, '/')
        filename = os.path.basename(candidate)
        size_bytes = os.path.getsize(candidate)
        mime_type = (
            mimetypes.guess_type(filename)[0] or 'application/octet-stream')

        if request.args.get('download') == '1':
            return send_file(
                candidate, as_attachment=True, download_name=filename,
                mimetype=mime_type, conditional=True)

        if request.args.get('raw') == '1':
            if not (mime_type.startswith('image/')
                    or mime_type == 'application/pdf'):
                return jsonify({
                    'success': False,
                    'error': 'Inline binary preview is not available for this file type',
                }), 415
            response = send_file(
                candidate, as_attachment=False, download_name=filename,
                mimetype=mime_type, conditional=True, max_age=0)
            response.headers['Cache-Control'] = 'private, no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        extension = os.path.splitext(filename)[1].casefold()

        # ?attributes=1 asks for the data behind the preview rather than the
        # preview itself: the feature table for a vector or tabular output,
        # the properties/band/value description for a raster. Read here, on
        # the artifact path, so a large file is never shipped to the browser
        # just to count its rows.
        if request.args.get('attributes') == '1':
            table_exts = {'.geojson', '.json', '.gpkg', '.shp', '.zip',
                          '.csv', '.tsv', '.xlsx', '.xls', '.parquet'}
            raster_exts = {'.tif', '.tiff'}
            try:
                if extension in raster_exts:
                    return jsonify({
                        'success': True, 'kind': 'raster', 'name': filename,
                        **describe_raster(candidate),
                    })
                if extension in table_exts:
                    max_rows = request.args.get('max_rows', 500, type=int)
                    return jsonify({
                        'success': True, 'kind': 'table', 'name': filename,
                        **read_attribute_table(candidate, max_rows),
                    })
            except ImportError as exc:
                return jsonify({
                    'success': False,
                    'error': f'Missing library on the server: {exc}',
                }), 500
            except Exception as exc:
                return jsonify({
                    'success': False,
                    'error': f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__,
                }), 422
            return jsonify({
                'success': False,
                'error': 'This file type has no attribute table.',
            }), 415

        text_extensions = {
            '.txt', '.csv', '.tsv', '.json', '.geojson', '.xml', '.osm',
            '.html', '.htm', '.py', '.md', '.toml', '.yaml', '.yml', '.log',
            '.sql', '.js', '.css', '.prj', '.cpg',
        }
        image_types = {
            'image/png', 'image/jpeg', 'image/gif', 'image/webp',
        }
        preview = {
            'artifact_ref': artifact_ref,
            'name': filename,
            'size_bytes': size_bytes,
            'mime_type': mime_type,
            'kind': 'binary',
            'content': '',
            'truncated': False,
        }
        # Geospatial and tabular kinds are classified BEFORE the generic text
        # branch: a .geojson is perfectly readable as text, but a reviewer
        # judging whether the right area came back needs it on a map, and a
        # .csv needs columns, not a wall of commas.
        if mime_type in image_types:
            preview['kind'] = 'image'
        elif mime_type == 'application/pdf':
            preview['kind'] = 'pdf'
        elif extension in ('.geojson', '.gpkg', '.shp'):
            geo = _studio_preview_geometry(candidate, extension)
            if geo:
                preview.update(geo)
        elif extension in ('.tif', '.tiff'):
            # ?pixels=1 asks for the rendered image itself rather than the
            # metadata summary. Served from this route so it inherits the same
            # ownership and path-containment checks the summary already passed.
            if request.args.get('pixels'):
                # Cached on disk keyed by path+mtime+size: rendering is pure,
                # so the same file never pays for it twice, and panning back to
                # a raster is instant.
                import hashlib
                stat = os.stat(candidate)
                key = hashlib.sha256(
                    f'{candidate}|{stat.st_mtime_ns}|{stat.st_size}'.encode()
                ).hexdigest()[:32]
                cache_dir = os.path.join(session_roots[0], '_previews')
                cached = os.path.join(cache_dir, key + '.png')
                png = None
                if os.path.isfile(cached):
                    with open(cached, 'rb') as handle:
                        png = handle.read()
                if png is None:
                    png, info = _studio_render_raster_png(candidate)
                    if not png:
                        return jsonify({
                            'success': False,
                            'error': (info or {}).get(
                                'reason', 'This raster could not be rendered.')
                        }), 422
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cached, 'wb') as handle:
                            handle.write(png)
                    except OSError:
                        pass          # a cache miss is not a failure
                response = app.response_class(png, mimetype='image/png')
                response.headers['Cache-Control'] = 'private, max-age=3600'
                return response
            raster = _studio_preview_raster(candidate)
            if raster:
                preview.update(raster)
                # Tells the viewer a real image can be fetched, so it draws
                # pixels instead of falling back to the footprint rectangle.
                # Only claimed when the raster can be placed at all: promising
                # pixels for an unplaceable file sent the viewer to ?pixels=1
                # for a 422 it could do nothing with.
                preview['has_pixels'] = bool(raster.get('bbox'))
        elif extension in ('.csv', '.tsv'):
            limit = 300_000
            with open(candidate, 'rb') as handle:
                raw = handle.read(limit + 1)
            preview['truncated'] = len(raw) > limit
            preview['kind'] = 'csv'
            preview['delimiter'] = '\t' if extension == '.tsv' else ','
            preview['content'] = raw[:limit].decode('utf-8', errors='replace')
        elif (extension in text_extensions or mime_type.startswith('text/')
              or mime_type in (
                  'application/json', 'application/geo+json',
                  'application/xml', 'application/javascript')):
            limit = 300_000
            with open(candidate, 'rb') as handle:
                raw = handle.read(limit + 1)
            preview['truncated'] = len(raw) > limit
            text = raw[:limit].decode('utf-8', errors='replace')
            if extension in ('.json', '.geojson') and not preview['truncated']:
                try:
                    text = json.dumps(
                        json.loads(text), ensure_ascii=False, indent=2)
                except (ValueError, TypeError):
                    pass
            preview['kind'] = 'text'
            preview['content'] = text

        return jsonify({'success': True, 'preview': preview})
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        app.logger.exception('Could not preview Generate & Test output')
        return jsonify({'success': False, 'error': str(exc)}), 500

def _json_number(value):
    """Make a numpy / None / non-finite value JSON-safe."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number or number in (float('inf'), float('-inf')):
        return None
    return number


def read_attribute_table(path, max_rows=500):
    """Read one file's attribute table as columns + JSON-safe rows.

    Handles anything geopandas opens (.gpkg, .shp, .geojson, .zip) and the
    tabular formats pandas reads (.csv, .tsv, .xlsx, .parquet), so a caller
    can hand it whatever the user clicked without branching first.
    """
    import json
    import re

    import numpy as np
    import pandas as pd

    extension = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    if extension in ('csv', 'tsv', 'txt'):
        # sep=None lets the python engine sniff comma vs tab
        frame = pd.read_csv(path, sep=None, engine='python')
    elif extension in ('xlsx', 'xls'):
        frame = pd.read_excel(path)
    elif extension == 'parquet':
        frame = pd.read_parquet(path)
    else:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        # Drop geometry column — we only want the attribute table
        frame = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))

    total_rows = len(frame)
    if total_rows > max_rows:
        frame = frame.head(max_rows)

    # Sanitise values for JSON
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype(str)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    records_str = frame.to_json(orient='records')
    records_str = re.sub(r'\b-?Infinity\b', 'null', records_str)
    records_str = re.sub(r'\bNaN\b', 'null', records_str)
    rows = json.loads(records_str)

    return {
        'columns': [str(c) for c in frame.columns],
        'rows': rows,
        'total_rows': total_rows,
        'returned_rows': len(rows),
    }


def describe_raster(path, max_categories=60):
    """Describe a raster: properties, per-band statistics, value counts.

    A raster has no feature table, so this is what stands in for one: the file
    and CRS properties, statistics per band, and — for a classified raster
    (land cover, a model's output classes) — the count of pixels per value,
    which is the closest thing it has to an attribute table.

    Statistics come from a decimated read for large rasters; the returned
    'sampled' flag says whether that happened.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    num = _json_number

    with rasterio.open(path) as src:
        # Statistics come from a decimated read — a full read of a large
        # raster is slow and the summary numbers barely move.
        MAX_DIM = 1024
        scale = max(src.width / MAX_DIM, src.height / MAX_DIM, 1.0)
        out_h = max(1, int(round(src.height / scale)))
        out_w = max(1, int(round(src.width / scale)))
        sampled = (out_w != src.width or out_h != src.height)

        source_crs = src.crs or _sidecar_crs(path)
        try:
            wgs84 = transform_bounds(source_crs, 'EPSG:4326', *src.bounds) if source_crs else None
        except Exception:
            wgs84 = None

        res_x, res_y = src.res
        properties = {
            'file_name': os.path.basename(path),
            'driver': src.driver,
            'crs': str(source_crs) if source_crs else None,
            'width': src.width,
            'height': src.height,
            'band_count': src.count,
            'dtypes': list(src.dtypes),
            'nodata': num(src.nodata),
            'pixel_size_x': num(res_x),
            'pixel_size_y': num(res_y),
            'total_pixels': int(src.width) * int(src.height),
            'bounds': {
                'left': num(src.bounds.left),
                'bottom': num(src.bounds.bottom),
                'right': num(src.bounds.right),
                'top': num(src.bounds.top),
            },
            'wgs84_bounds': ({
                'west': num(wgs84[0]), 'south': num(wgs84[1]),
                'east': num(wgs84[2]), 'north': num(wgs84[3]),
            } if wgs84 else None),
            'transform': [num(x) for x in list(src.transform)[:6]],
        }

        try:
            colorinterp = [ci.name for ci in src.colorinterp]
        except Exception:
            colorinterp = []
        descriptions = src.descriptions or ()

        bands = []
        for i in range(1, src.count + 1):
            arr = src.read(i, out_shape=(out_h, out_w), masked=True)
            valid = arr.compressed()
            entry = {
                'band': i,
                'description': descriptions[i - 1] if i <= len(descriptions) else None,
                'dtype': src.dtypes[i - 1],
                'nodata': num(src.nodatavals[i - 1]),
                'color_interp': colorinterp[i - 1] if i <= len(colorinterp) else None,
                'valid_pixels': int(valid.size),
                'nodata_pixels': int(arr.size - valid.size),
            }
            if valid.size:
                entry.update({
                    'min': num(np.min(valid)),
                    'max': num(np.max(valid)),
                    'mean': num(np.mean(valid)),
                    'std': num(np.std(valid)),
                })
            else:
                entry.update({'min': None, 'max': None, 'mean': None, 'std': None})
            bands.append(entry)

        # A categorical raster (land cover, classified output) gets a
        # value/count table. Continuous rasters get None and the caller shows
        # statistics only.
        categories = None
        if src.count:
            first = src.read(1, out_shape=(out_h, out_w), masked=True).compressed()
            if first.size:
                looks_discrete = (np.issubdtype(first.dtype, np.integer)
                                  or bool(np.all(np.mod(first, 1) == 0)))
                if looks_discrete:
                    values, counts = np.unique(first, return_counts=True)
                    if len(values) <= max_categories:
                        try:
                            cmap = src.colormap(1)
                        except Exception:
                            cmap = None
                        total = int(counts.sum())
                        categories = []
                        for idx in np.argsort(-counts):
                            value = values[idx]
                            rgba = cmap.get(int(value)) if cmap else None
                            categories.append({
                                'value': num(value),
                                'count': int(counts[idx]),
                                'percent': (round(float(counts[idx]) * 100.0 / total, 3)
                                            if total else None),
                                'color': ('#%02x%02x%02x' % tuple(rgba[:3])) if rgba else None,
                            })

        tags = {}
        try:
            for key, value in (src.tags() or {}).items():
                tags[str(key)] = str(value)
        except Exception:
            pass

    return {
        'properties': properties,
        'bands': bands,
        'categories': categories,
        'tags': tags,
        'sampled': sampled,
        'sample_shape': [out_w, out_h],
    }



if __name__ == '__main__':
    port = int(os.environ.get('PORT', '4041'))
    print(f"Handbook Studio running at http://localhost:{port}/")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False, threaded=True)
