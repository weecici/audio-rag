import os
import uuid
import streamlit as st
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import ValidationError
from app import schema

load_dotenv()


# -------------------------------
# Config
# -------------------------------
def get_default_api_base() -> str:
    host = os.getenv("BACKEND_HOST", "localhost")
    port = os.getenv("BACKEND_PORT", "8000")
    return f"http://{host}:{port}/api/v1"


st.set_page_config(page_title="CS431 RAG Chat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content, sources?)]

# Multi-chat support
if "chats" not in st.session_state:
    # chat_id -> {"title": str, "messages": list, "created_at": float}
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "settings" not in st.session_state:
    st.session_state.settings = {
        "api_base": get_default_api_base(),
        "collection_name": "cs431",
        "top_k": 10,
        "mode": "hybrid",
        "overfetch_mul": 2.0,
        "rerank_enabled": False,
        "summarization_enabled": False,
        "model_name": "gpt-oss-120b",
    }

# Track per-chat rename/edit state (chat_id -> bool)
if "chat_editing" not in st.session_state:
    st.session_state.chat_editing = {}

# Track per-chat action menu open state (chat_id -> bool)
if "chat_menu_open" not in st.session_state:
    st.session_state.chat_menu_open = {}

# Busy state and pending input to prevent concurrent sends
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "pending_chat_id" not in st.session_state:
    st.session_state.pending_chat_id = None
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False


def _truncate_one_line(text: str, max_chars: int = 25) -> str:
    """Return a single-line truncated string with ellipsis if too long.
    - Collapses any newlines into spaces so it never wraps to a second line.
    - Cuts at max_chars and appends … if truncation occurs.
    """
    try:
        single = " ".join(text.splitlines())
    except Exception:
        single = str(text)
    if len(single) <= max_chars:
        return single
    # Reserve 1 char for ellipsis
    cutoff = max(0, max_chars - 1)
    return single[:cutoff] + "…"


def _ensure_default_chat():
    """Make sure there is at least one chat and messages points to it."""
    if not st.session_state.chats:
        chat_id = str(uuid.uuid4())[:8]
        st.session_state.chats[chat_id] = {
            "title": "New chat",
            "messages": [],
            "created_at": (
                st.time() if hasattr(st, "time") else 0.0
            ),  # fallback if time not available
        }
        st.session_state.current_chat_id = chat_id
    if st.session_state.current_chat_id not in st.session_state.chats:
        # Pick any existing
        st.session_state.current_chat_id = next(iter(st.session_state.chats.keys()))
    # Always point messages to the active chat's messages list (shared list reference)
    st.session_state.messages = st.session_state.chats[
        st.session_state.current_chat_id
    ]["messages"]


def _new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "created_at": st.time() if hasattr(st, "time") else 0.0,
    }
    # Initialize rename/edit state for this chat
    if "chat_editing" not in st.session_state:
        st.session_state.chat_editing = {}
    st.session_state.chat_editing[chat_id] = False
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]
    # Do not clear pending_input here, so background generation can continue
    return chat_id


# -------------------------------
# Sidebar (settings)
# -------------------------------
with st.sidebar:
    st.title("Config")

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button {
                text-align: left;
                justify-content: flex-start;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Settings", expanded=False):
        # Keep API base in the sidebar
        api_base = st.text_input(
            "API base URL",
            value=st.session_state.settings.get("api_base", get_default_api_base()),
            help="API URL of the backend service",
            key="api_base_sidebar",
        )
        st.session_state.settings["api_base"] = api_base.rstrip("/")

        st.session_state.settings["collection_name"] = st.text_input(
            "Collection",
            value=st.session_state.settings.get("collection_name", "documents"),
            key="sidebar_collection_name",
        )
        st.session_state.settings["top_k"] = st.slider(
            "Top-K",
            min_value=1,
            max_value=20,
            value=int(st.session_state.settings.get("top_k", 5)),
            key="sidebar_top_k",
        )
        st.session_state.settings["mode"] = st.selectbox(
            "Mode",
            options=["dense", "sparse", "hybrid"],
            index=["dense", "sparse", "hybrid"].index(
                st.session_state.settings.get("mode", "hybrid")
            ),
            key="sidebar_mode",
        )
        st.session_state.settings["overfetch_mul"] = float(
            st.number_input(
                "Overfetch multiplier",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.settings.get("overfetch_mul", 2.0)),
                step=0.5,
                key="sidebar_overfetch",
            )
        )
        st.session_state.settings["rerank_enabled"] = st.checkbox(
            "Rerank enabled",
            value=bool(st.session_state.settings.get("rerank_enabled", False)),
            key="sidebar_rerank",
        )
        st.session_state.settings["summarization_enabled"] = st.checkbox(
            "Summarization enabled",
            value=bool(st.session_state.settings.get("summarization_enabled", False)),
            key="sidebar_summarization",
        )
        model_options = [
            "gpt-oss-120b",
            "llama-3.3-70b",
            "qwen-3-235b-a22b-thinking-2507",
            "qwen-3-coder-480b",
        ]
        st.session_state.settings["model_name"] = st.selectbox(
            "LLM",
            options=model_options,
            index=model_options.index(
                st.session_state.settings.get("model_name", "gpt-oss-120b")
            ),
            key="sidebar_model",
        )

    st.markdown("---")
    st.title("Chats History")

    # Ensure chat state
    _ensure_default_chat()

    if st.button("New chat", use_container_width=True, key="new_chat_btn"):
        _new_chat()
        st.rerun()

    # Chat list with expandable option popover/expander (ellipsis menu)
    # Sort by created_at descending (newest first) if available, else insertion order reversed
    # We'll just reverse the items list for now as a proxy for "newest first"
    chats_desc = list(st.session_state.chats.items())[::-1]

    for cid, chat in chats_desc:
        if cid not in st.session_state.chat_editing:
            st.session_state.chat_editing[cid] = False

        # If we are editing this chat, show the edit UI in place of the button
        if st.session_state.chat_editing[cid]:
            with st.container():
                new_title = st.text_input(
                    "Title",
                    value=chat.get("title", "Untitled"),
                    key=f"edit_input_{cid}",
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns(2)
                if c1.button("Save", key=f"save_{cid}", use_container_width=True):
                    chat["title"] = new_title.strip() or "Untitled"
                    st.session_state.chat_editing[cid] = False
                    st.rerun()
                if c2.button("Cancel", key=f"cancel_{cid}", use_container_width=True):
                    st.session_state.chat_editing[cid] = False
                    st.rerun()
        else:
            # Normal display
            row = st.container()
            with row:
                cols = st.columns([0.9, 0.1], gap=None)
                with cols[0]:
                    full_title = chat.get("title", "Untitled")
                    label = _truncate_one_line(full_title)
                    is_active = cid == st.session_state.current_chat_id
                    if st.button(
                        label,
                        key=f"open_chat_{cid}",
                        use_container_width=True,
                        help=full_title,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.current_chat_id = cid
                        st.session_state.messages = chat["messages"]
                        st.rerun()
                with cols[1]:
                    # Streamlit popover for expandable options
                    with st.popover(
                        "",
                        type="tertiary",
                        help="Chat actions",
                        use_container_width=True,
                    ):
                        if st.button(
                            "Rename",
                            key=f"popover_rename_btn_{cid}",
                            use_container_width=True,
                        ):
                            st.session_state.chat_editing[cid] = True
                            st.rerun()
                        if st.button(
                            "Delete",
                            key=f"popover_delete_btn_{cid}",
                            use_container_width=True,
                        ):
                            try:
                                del st.session_state.chats[cid]
                                st.session_state.chat_editing.pop(cid, None)
                            except KeyError:
                                pass

                            # If we deleted the current chat, switch to another
                            if (
                                st.session_state.current_chat_id == cid
                                or st.session_state.current_chat_id
                                not in st.session_state.chats
                            ):
                                if st.session_state.chats:
                                    # Pick the first one available (which is the last one in our reversed list, i.e. the newest remaining?)
                                    # Actually just pick any.
                                    st.session_state.current_chat_id = next(
                                        iter(st.session_state.chats.keys())
                                    )
                                else:
                                    _new_chat()

                            # Update messages reference
                            st.session_state.messages = st.session_state.chats[
                                st.session_state.current_chat_id
                            ]["messages"]
                            st.rerun()


# -------------------------------
# Helpers
# -------------------------------
def post_generate(query: str) -> Optional[schema.GenerationResponse]:
    """Call the /generate endpoint with a single user query and return parsed response."""
    req = schema.GenerationRequest(
        queries=[query],
        collection_name=st.session_state.settings["collection_name"],
        top_k=st.session_state.settings["top_k"],
        mode=st.session_state.settings["mode"],
        overfetch_mul=st.session_state.settings["overfetch_mul"],
        rerank_enabled=st.session_state.settings["rerank_enabled"],
        summarization_enabled=st.session_state.settings["summarization_enabled"],
        model_name=st.session_state.settings["model_name"],
    )

    try:
        url = f"{st.session_state.settings['api_base']}/generate"
        r = requests.post(url, json=req.model_dump())
        r.raise_for_status()
        data = r.json()
        return schema.GenerationResponse.model_validate(data)
    except (requests.RequestException, ValidationError) as e:
        st.error(f"API error: {e}")
        try:
            # Best-effort show error details from server
            st.code(r.text, language="json")  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def post_ingest_documents(
    file_paths: List[str], file_dir: str, collection_name: str
) -> Optional[schema.IngestionResponse]:
    req = schema.DocumentIngestionRequest(
        file_paths=file_paths,
        file_dir=file_dir,
        collection_name=collection_name,
    )
    try:
        url = f"{st.session_state.settings['api_base']}/ingest/documents"
        r = requests.post(url, json=req.model_dump())
        r.raise_for_status()
        data = r.json()
        return schema.IngestionResponse.model_validate(data)
    except (requests.RequestException, ValidationError) as e:
        st.error(f"Ingestion error: {e}")
        try:
            st.code(r.text, language="json")  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def post_ingest_audios(
    file_paths: List[str], urls: List[str], collection_name: str
) -> Optional[schema.IngestionResponse]:
    req = schema.AudioIngestionRequest(
        file_paths=file_paths,
        urls=urls,
        collection_name=collection_name,
    )
    try:
        url = f"{st.session_state.settings['api_base']}/ingest/audios"
        r = requests.post(url, json=req.model_dump())
        r.raise_for_status()
        data = r.json()
        return schema.IngestionResponse.model_validate(data)
    except (requests.RequestException, ValidationError) as e:
        st.error(f"Audio ingestion error: {e}")
        try:
            st.code(r.text, language="json")  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def render_sources(docs: List[schema.RetrievedDocument]):
    if not docs:
        return
    for idx, d in enumerate(docs, start=1):
        meta = d.payload.metadata

        # Default values
        src_title = f"Source {idx}: {meta.file_name}"
        video_url = None
        start_time = None

        try:
            parts = meta.title.split("||")
            if len(parts) == 3:
                title, start_str, end_str = (
                    parts[0].strip(),
                    parts[1].strip(),
                    parts[2].strip(),
                )

                # Try to parse start_time
                try:
                    start_time = int(float(start_str))
                except ValueError:
                    start_time = 0

                # Try to extract video_id from file path if it follows convention
                # e.g. "some_name$VIDEO_ID.mp3"
                try:
                    stem = Path(meta.file_path).stem
                    if "$" in stem:
                        video_id = stem.split("$")[1].strip()
                        video_url = f"https://youtu.be/{video_id}"
                except Exception:
                    pass

                src_title = f"Source {idx}: {title} | {start_str}s - {end_str}s in {meta.file_name}"
            else:
                src_title = f"Source {idx}: {meta.title} (in {meta.file_name})"
        except Exception:
            # If parsing fails, keep defaults
            pass

        with st.expander(src_title):
            if video_url and start_time is not None:
                try:
                    st.video(video_url, start_time=start_time)
                    st.caption(f"▶ {meta.file_name}")
                except Exception:
                    st.warning("Could not load video.")

            st.markdown("---")
            st.write(d.payload.text)


# -------------------------------
# Pages (Top navigation with fallback to Tabs)
# -------------------------------


def render_chat():
    st.title("RAG Chat")
    st.caption("Your Reliable Teaching Assistant!")

    # Ensure chat state and bind messages to the current chat
    _ensure_default_chat()

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])  # type: ignore[arg-type]

    # Input (disabled while generating) - stays at bottom center; settings panel is fixed left
    user_input = st.chat_input(
        "Ask anything about your documents…",
        disabled=st.session_state.is_generating,
    )

    # If a new message is submitted and we're not currently generating,
    # stage it and re-run so the input becomes disabled immediately.
    if user_input and not st.session_state.is_generating:
        # Append to current chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Set chat title from first user message if still default
        chat = st.session_state.chats[st.session_state.current_chat_id]
        if chat.get("title") in (None, "", "New chat") and user_input.strip():
            chat["title"] = user_input.strip()[:30] + (
                "…" if len(user_input) > 30 else ""
            )
        st.session_state.pending_input = user_input
        st.session_state.pending_chat_id = st.session_state.current_chat_id
        st.session_state.is_generating = True
        st.rerun()

    # If we have a staged input, perform generation now
    if st.session_state.pending_input is not None:
        pending = st.session_state.pending_input
        target_chat_id = st.session_state.get("pending_chat_id")
        if target_chat_id is None:
            target_chat_id = st.session_state.current_chat_id

        is_current = target_chat_id == st.session_state.current_chat_id

        # If we are in the target chat, show the spinner in the chat flow.
        # Otherwise, show a global spinner or just run in background.
        if is_current:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    resp: Optional[schema.GenerationResponse] = None
                    try:
                        resp = post_generate(pending)
                    finally:
                        # Ensure we reset flags even on error
                        st.session_state.pending_input = None
                        st.session_state.is_generating = False

                if resp is None:
                    st.error("Failed to get a response.")
                else:
                    # Take the first response for the single query
                    answer = resp.responses[0] if resp.responses else "(empty)"
                    sources = (
                        resp.summarized_docs_list[0]
                        if resp.summarized_docs_list
                        else []
                    )

                    st.write(answer)
                    render_sources(sources)

                    # Append to the target chat (which is current)
                    if target_chat_id in st.session_state.chats:
                        st.session_state.chats[target_chat_id]["messages"].append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
        else:
            # Background generation (user switched chat)
            with st.spinner(f"Generating response for another chat..."):
                resp: Optional[schema.GenerationResponse] = None
                try:
                    resp = post_generate(pending)
                finally:
                    st.session_state.pending_input = None
                    st.session_state.is_generating = False

            if resp is not None:
                answer = resp.responses[0] if resp.responses else "(empty)"
                sources = (
                    resp.summarized_docs_list[0] if resp.summarized_docs_list else []
                )
                if target_chat_id in st.session_state.chats:
                    st.session_state.chats[target_chat_id]["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

        # Trigger a rerun to refresh the input (now enabled)
        st.rerun()


def render_docs():
    st.header("📄 Ingest Documents")
    st.caption("Provide server-side file paths or a directory to ingest.")
    col1, col2 = st.columns(2)
    with col1:
        file_dir = st.text_input("Directory (optional)", value="")
    with col2:
        collection_docs = st.text_input(
            "Collection",
            value=st.session_state.settings.get("collection_name", "documents"),
            key="docs_collection_input",
        )
    file_paths_text = st.text_area(
        "File paths (one per line)",
        placeholder="/path/to/doc1.pdf\n/path/to/doc2.txt",
        height=120,
        key="docs_paths_text",
    )
    if st.button("Ingest documents", type="primary", key="ingest_docs_btn"):
        file_paths = [p.strip() for p in file_paths_text.splitlines() if p.strip()]
        if not file_paths and not file_dir.strip():
            st.warning("Please provide at least one file path or a directory.")
        else:
            with st.spinner("Ingesting documents…"):
                resp = post_ingest_documents(
                    file_paths, file_dir.strip(), collection_docs
                )
            if resp is None:
                st.error("Document ingestion failed.")
            else:
                if resp.status == 200:
                    st.success(resp.message)
                else:
                    st.warning(f"Status {resp.status}: {resp.message}")


def render_audios():
    st.header("🎵 Ingest Audios")
    st.caption("Provide server-side audio file paths and/or YouTube URLs to ingest.")
    col1, col2 = st.columns(2)
    with col1:
        collection_audios = st.text_input(
            "Collection",
            value=st.session_state.settings.get("collection_name", "documents"),
            key="audios_collection_input",
        )
    with col2:
        st.empty()

    audio_paths_text = st.text_area(
        "Audio file paths (one per line)",
        placeholder="/path/to/audio1.mp3\n/path/to/audio2.wav",
        height=120,
        key="audio_paths_text",
    )
    youtube_urls_text = st.text_area(
        "YouTube URLs (one per line)",
        placeholder="https://youtu.be/VIDEO_ID\nhttps://www.youtube.com/watch?v=VIDEO_ID",
        height=120,
        key="youtube_urls_text",
    )
    if st.button("Ingest audios", type="primary", key="ingest_audios_btn"):
        file_paths = [p.strip() for p in audio_paths_text.splitlines() if p.strip()]
        urls = [u.strip() for u in youtube_urls_text.splitlines() if u.strip()]
        if not file_paths and not urls:
            st.warning("Please provide at least one audio file path or a YouTube URL.")
        else:
            with st.spinner("Ingesting audios…"):
                resp = post_ingest_audios(file_paths, urls, collection_audios)
            if resp is None:
                st.error("Audio ingestion failed.")
            else:
                if resp.status == 200:
                    st.success(resp.message)
                else:
                    st.warning(f"Status {resp.status}: {resp.message}")


# Prefer top navigation if available (Streamlit >= 1.31), else fallback to tabs
if hasattr(st, "navigation") and hasattr(st, "Page"):
    pages = [
        st.Page(render_chat, title="Chat", icon="💬"),
        st.Page(render_docs, title="Ingest Documents", icon="📄"),
        st.Page(render_audios, title="Ingest Audios", icon="🎵"),
    ]
    nav = st.navigation(pages)
    nav.run()
else:
    chat_tab, docs_tab, audios_tab = st.tabs(
        ["💬 Chat", "📄 Ingest Documents", "🎵 Ingest Audios"]
    )
    with chat_tab:
        render_chat()
    with docs_tab:
        render_docs()
    with audios_tab:
        render_audios()
