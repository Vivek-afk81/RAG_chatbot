import os
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from rag.loader import load_pdf
from rag.chunker import chunk_text, compare_chunk_sizes, CHUNK_PRESETS
from rag.embedder import build_index, query_index
from rag.search import semantic_search, answer_question

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

# Global state (Gradio shares one process per Space)
state = {
    "chunks": [],
    "index": None,
    "chunk_preset": "medium",
    "pdf_name": None,
    "raw_text": "",          # full extracted text — used by rebuild_index
}

# ---------------------------------------------------------------------------
# Core logic helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE_MAP = {"Small": "small", "Medium": "medium", "Large": "large"}


def process_pdf(pdf_file, chunk_size_label):
    """Load PDF, chunk it, build FAISS index. Returns stat strings."""
    if pdf_file is None:
        return "No file uploaded.", "", "", "", "", []

    preset_key = CHUNK_SIZE_MAP.get(chunk_size_label, "medium")
    preset = CHUNK_PRESETS[preset_key]

    text = load_pdf(pdf_file.name)
    chunks = chunk_text(text, **preset)
    index = build_index(chunks)

    state["chunks"] = chunks
    state["index"] = index
    state["chunk_preset"] = preset_key
    state["pdf_name"] = os.path.basename(pdf_file.name)
    state["raw_text"] = text                              # cache for rebuild

    avg_chars = int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0

    comparison = compare_chunk_sizes(text)
    comparison_rows = [
        [label, s["chunk_size"], s["total_chunks"], s["avg_chars"]]
        for label, s in comparison.items()
    ]

    return (
        str(len(chunks)),          # stat: chunks
        "384",                     # stat: dims (all-MiniLM-L6-v2 default)
        str(preset["chunk_size"]), # stat: chunk size
        str(avg_chars),            # stat: avg chars
        f"✓ Index ready — {index.ntotal} vectors ({preset_key} chunks)",
        comparison_rows,
    )


def rebuild_index(chunk_size_label):
    """Re-chunk + re-index when user changes chunk size after upload."""
    if state["index"] is None or not state["raw_text"]:
        return "", "", "", "", "Upload a PDF first.", []

    preset_key = CHUNK_SIZE_MAP.get(chunk_size_label, "medium")
    preset = CHUNK_PRESETS[preset_key]

    text = state["raw_text"]                   # use the original extracted text
    chunks = chunk_text(text, **preset)
    index = build_index(chunks)

    state["chunks"] = chunks
    state["index"] = index
    state["chunk_preset"] = preset_key

    avg_chars = int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0

    comparison = compare_chunk_sizes(text)
    comparison_rows = [
        [label, s["chunk_size"], s["total_chunks"], s["avg_chars"]]
        for label, s in comparison.items()
    ]

    return (
        str(len(chunks)),
        "384",
        str(preset["chunk_size"]),
        str(avg_chars),
        f"✓ Rebuilt — {index.ntotal} vectors ({preset_key} chunks)",
        comparison_rows,
    )


def chat_respond(user_message, history, mode):
    """
    Handle one turn of conversation.
    mode: "Chat" or "Search"
    history: list of [user, assistant] pairs (Gradio format)
    """
    if not user_message.strip():
        return history, ""

    if state["index"] is None:
        history.append([user_message, "Upload and process a PDF first."])
        return history, ""

    if mode == "Search":
        results = semantic_search(
            user_message, state["index"], state["chunks"], k=3
        )
        lines = [f"**Semantic search results for:** *{user_message}*\n"]
        for r in results:
            lines.append(
                f"**Rank {r['rank']} · score {r['score']:.3f}**\n"
                f"> {r['passage'][:400].strip()}…\n"
            )
        response = "\n".join(lines)
    else:
        try:
            raw_answer = answer_question(
                user_message, state["index"], state["chunks"], client
            )
            # query_index returns (list[str], list[float])
            retrieved_chunks, scores = query_index(
                user_message, state["index"], state["chunks"], k=3
            )
            # Find the index of each retrieved chunk in state["chunks"] for labelling
            chunk_labels = []
            for chunk_text_val in retrieved_chunks:
                try:
                    idx = state["chunks"].index(chunk_text_val)
                    chunk_labels.append(f"`chunk {idx}`")
                except ValueError:
                    chunk_labels.append("`chunk ?`")
            pill_line = "  ".join(chunk_labels)
            response = f"{raw_answer}\n\n**Sources:** {pill_line}"
        except Exception as e:
            response = f"Error: {e}"

    history.append([user_message, response])
    return history, ""


def clear_chat():
    return [], ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="RAG Document Q&A",
    theme=gr.themes.Default(
        primary_hue="blue",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
    /* top bar */
    #header { border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 4px; }
    /* stat cards */
    .stat-box textarea { font-size: 22px !important; font-weight: 500 !important;
                         text-align: center !important; border: none !important;
                         background: transparent !important; padding: 0 !important; }
    .stat-box label { text-align: center; font-size: 11px; color: #6b7280; }
    /* status banner */
    #status textarea { font-size: 13px !important; color: #1d4ed8 !important;
                       background: #eff6ff !important; border-color: #bfdbfe !important; }
    /* chatbot bubbles */
    #chatbox { border-radius: 10px; }
    """,
) as demo:

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    with gr.Row(elem_id="header"):
        gr.Markdown(
            "## RAG Document Q&A\n"
            "Upload a PDF, pick a chunk size, then ask questions or run semantic search."
        )

    # ------------------------------------------------------------------
    # Main layout: sidebar | chat
    # ------------------------------------------------------------------
    with gr.Row():

        # ---- LEFT SIDEBAR ----
        with gr.Column(scale=1, min_width=260):

            gr.Markdown("### Document")
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            gr.Markdown("### Chunk size")
            chunk_radio = gr.Radio(
                choices=["Small", "Medium", "Large"],
                value="Medium",
                label="",
                info="Small=300 chars · Medium=1 000 · Large=2 000",
            )
            rebuild_btn = gr.Button("Rebuild index", size="sm", variant="secondary")

            gr.Markdown("### Index stats")
            with gr.Row():
                stat_chunks = gr.Textbox(label="Chunks", elem_classes="stat-box",
                                         interactive=False, max_lines=1, value="—")
                stat_dims   = gr.Textbox(label="Dims",   elem_classes="stat-box",
                                         interactive=False, max_lines=1, value="—")
            with gr.Row():
                stat_csize  = gr.Textbox(label="Chunk size", elem_classes="stat-box",
                                         interactive=False, max_lines=1, value="—")
                stat_avg    = gr.Textbox(label="Avg chars",  elem_classes="stat-box",
                                         interactive=False, max_lines=1, value="—")

            status_box = gr.Textbox(
                label="Status",
                value="No index yet — upload a PDF.",
                interactive=False,
                max_lines=2,
                elem_id="status",
            )

            gr.Markdown("### Mode")
            mode_radio = gr.Radio(
                choices=["Chat", "Search"],
                value="Chat",
                label="",
            )

            gr.Markdown("### Chunk comparison")
            comparison_table = gr.Dataframe(
                headers=["Preset", "Chunk size", "Total chunks", "Avg chars"],
                datatype=["str", "number", "number", "number"],
                interactive=False,
                wrap=True,
            )

        # ---- MAIN CHAT AREA ----
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbox",
                height=450,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question… (Search mode: enter a query)",
                    label="",
                    scale=5,
                    show_label=False,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    STAT_OUTPUTS = [stat_chunks, stat_dims, stat_csize, stat_avg,
                    status_box, comparison_table]

    pdf_input.change(
        fn=process_pdf,
        inputs=[pdf_input, chunk_radio],
        outputs=STAT_OUTPUTS,
    )

    rebuild_btn.click(
        fn=rebuild_index,
        inputs=[chunk_radio],
        outputs=STAT_OUTPUTS,
    )

    send_btn.click(
        fn=chat_respond,
        inputs=[msg_box, chatbot, mode_radio],
        outputs=[chatbot, msg_box],
    )

    msg_box.submit(
        fn=chat_respond,
        inputs=[msg_box, chatbot, mode_radio],
        outputs=[chatbot, msg_box],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_box])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()