import gradio as gr
import concurrent.futures
from src.rag import RAGPipeline
from src.agent.agent import Agent
from src.evaluator.judge import Judge
from src.hallucination.detector import HallucinationDetector
from src.reranker.reranker import Reranker
from src.guardrails.guardrails import Guardrails
from src.multiagent.orchestrator import Orchestrator

# Single global pipeline orchestrator
pipeline = RAGPipeline()
agent = Agent(pipeline)
judge = Judge()
detector = HallucinationDetector()
reranker = Reranker()
guardrails = Guardrails()
orchestrator = Orchestrator()

# --------------------------------------------------------------------------- #
# Custom CSS — modern dark AI chat UI                                          #
# --------------------------------------------------------------------------- #
custom_css = """
.gradio-container {
    background-color: #0b0d0e !important;
    color: white !important;
}
.main-container {
    max-width: 900px !important;
    margin: auto !important;
    padding-top: 2rem !important;
}
#chatbot-container {
    border: none !important;
    background: transparent !important;
}
.message.user {
    background-color: #2f2f2f !important;
    border-radius: 20px !important;
    border: none !important;
}
.message.bot {
    background-color: transparent !important;
    border: none !important;
}
#input-row {
    background: #212121;
    border-radius: 28px;
    padding: 8px 16px;
    border: 1px solid #3d3d3d;
    align-items: center;
    position: sticky;
    bottom: 20px;
}
#input-row:focus-within {
    border-color: #666;
}
#msg-box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    padding: 10px !important;
}
#upload-btn {
    min-width: 40px !important;
    background: transparent !important;
    border: none !important;
    font-size: 20px !important;
}
#upload-btn .file-preview {
    display: none !important;
}
#send-btn {
    background: #ffffff !important;
    color: black !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    font-weight: bold !important;
}
.file-status {
    font-size: 0.8rem;
    color: #888;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 4px;
}
"""


# --------------------------------------------------------------------------- #
# Callbacks                                                                    #
# --------------------------------------------------------------------------- #

def handle_upload(file) -> str:
    if file is None:
        return "No file selected."
    try:
        status = pipeline.load_pdf(file.name)
        filename = file.name.split("/")[-1]
        return f"📎 Attached: {filename}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat(user_message, history, use_agent, eval_mode, detect_hallucination, show_reranker, enable_guardrails, multi_agent_mode):
    if not user_message.strip():
        yield ("", history, 
                gr.update(), "No evaluation yet.",
                gr.update(), "No hallucination check yet.",
                gr.update(), "No re-ranking yet.",
                gr.update(), "No guardrails check yet.",
                gr.update(), "No multi-agent run yet.")
        return

    context = ""
    tool_used = None
    clean_answer = ""
    original_results = []
    reranked_results = []
    
    multi_report = "No multi-agent run yet."
    multi_accordion = gr.update(open=False)

    if multi_agent_mode:
        use_agent = False  # Multi-agent takes priority
        
        if pipeline.vector_store.is_empty:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "⚠️ Please attach a PDF first using the ➕ icon."})
            yield ("", history, 
                    gr.update(), "No evaluation yet.",
                    gr.update(), "No hallucination check yet.",
                    gr.update(), "No re-ranking yet.",
                    gr.update(), "No guardrails check yet.",
                    gr.update(), "No multi-agent run yet.")
            return

        tool_used = "rag_search" # Treat as document grounding logic for other evaluations
        
        original_strings = pipeline.retriever.retrieve(user_message, top_k=10)
        original_results = [{"text": text, "score": 0.0, "index": i} for i, text in enumerate(original_strings)]
        reranked_results = reranker.rerank(user_message, list(original_results), top_k=3)
        context = "\n\n---\n\n".join([r["text"] for r in reranked_results])
        
        # UX: Show the user we are running the orchestrator (takes a few seconds)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "..."})
        yield ("", history, 
                gr.update(open=False), "No evaluation yet.",
                gr.update(open=False), "No hallucination check yet.",
                gr.update(open=show_reranker), reranker.format_report(original_results, reranked_results) if show_reranker else "No re-ranking yet.",
                gr.update(open=False), "No guardrails check yet.",
                gr.update(open=True), "⏳ Multi-Agent pipeline running (Researcher -> Summarizer -> Critic)...")
        
        multi_result = orchestrator.run(user_message, context)
        answer = multi_result["final_answer"]
        clean_answer = answer
        multi_report = orchestrator.format_report(multi_result)
        multi_accordion = gr.update(open=True)
        
        history[-1]["content"] = answer
        pipeline.memory.add("user", user_message)
        pipeline.memory.add("assistant", answer)

    elif use_agent:
        result = agent.run(user_message)
        tool_used = result['tool_used']
        clean_answer = result['result']
        answer = f"🛠️ Tool: {tool_used}\n\n{clean_answer}"
        
        if tool_used == "rag_search":
            original_strings = pipeline.retriever.retrieve(user_message, top_k=5)
            context = "\n\n---\n\n".join(original_strings)
            
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
    else:
        if pipeline.vector_store.is_empty:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "⚠️ Please attach a PDF first using the ➕ icon."})
            yield ("", history, 
                    gr.update(), "No evaluation yet.",
                    gr.update(), "No hallucination check yet.",
                    gr.update(), "No re-ranking yet.",
                    gr.update(), "No guardrails check yet.",
                    gr.update(), "No multi-agent run yet.")
            return

        original_strings = pipeline.retriever.retrieve(user_message, top_k=10)
        original_results = [{"text": text, "score": 0.0, "index": i} for i, text in enumerate(original_strings)]
        
        reranked_results = reranker.rerank(user_message, list(original_results), top_k=3)

        context = "\n\n---\n\n".join([r["text"] for r in reranked_results])

        answer = pipeline.llm.generate(
            user_message=user_message,
            context=context,
            history=pipeline.memory.get_history()
        )
        clean_answer = answer
        pipeline.memory.add("user", user_message)
        pipeline.memory.add("assistant", answer)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

    # Reranking display (runs synchronously since no API call)
    rerank_report = "No re-ranking yet."
    rerank_accordion = gr.update(open=False)
    if show_reranker and (not use_agent or multi_agent_mode):
        rerank_report = reranker.format_comparison(original_results, reranked_results)
        rerank_accordion = gr.update(open=True)

    # 🚀 UX FIX: Yield the UI right now so the user instantly sees the answer!
    # Show "Processing..." in the accordions while the concurrent threads run.
    should_eval = eval_mode and (not use_agent or tool_used == "rag_search")
    should_hal = detect_hallucination and (not use_agent or tool_used == "rag_search")
    should_guard = enable_guardrails and not use_agent

    yield ("", history,
           gr.update(open=should_eval), "⏳ Running evaluator..." if should_eval else "No evaluation yet.",
           gr.update(open=should_hal), "⏳ Checking for hallucinations..." if should_hal else "No hallucination check yet.",
           rerank_accordion, rerank_report,
           gr.update(open=should_guard), "⏳ Checking guardrails..." if should_guard else "No guardrails check yet.",
           multi_accordion, multi_report)

    # Define concurrent tasks
    def run_eval():
        if should_eval:
            evaluation = judge.evaluate(user_message, context, clean_answer)
            return judge.format_report(evaluation), gr.update(open=True)
        return "No evaluation yet.", gr.update(open=False)

    def run_hal():
        if should_hal:
            hal_result = detector.detect(user_message, context, clean_answer)
            return detector.format_report(hal_result), gr.update(open=True)
        return "No hallucination check yet.", gr.update(open=False)

    def run_guard():
        if should_guard:
            guard_result = guardrails.run(user_message, answer, context)
            safe = guardrails.safe_answer(answer, guard_result)
            return guardrails.format_report(guard_result), gr.update(open=True), safe
        return "No guardrails check yet.", gr.update(open=False), answer

    # Execute concurrent modules
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_eval = executor.submit(run_eval)
        f_hal = executor.submit(run_hal)
        f_guard = executor.submit(run_guard)

    eval_report, eval_accordion = f_eval.result()
    hal_report, hal_accordion = f_hal.result()
    guard_report, guard_accordion, safe_answer_text = f_guard.result()

    # Update history with safe text if guardrails were active
    if should_guard:
        history[-1]["content"] = safe_answer_text

    # Yield final state!
    yield ("", history,
            eval_accordion, eval_report,
            hal_accordion, hal_report,
            rerank_accordion, rerank_report,
            guard_accordion, guard_report,
            multi_accordion, multi_report)


def reset_chat() -> tuple[list, str]:
    pipeline.vector_store.reset()
    pipeline.memory.clear()
    return [], "No documents indexed yet."


# --------------------------------------------------------------------------- #
# UI builder                                                                   #
# --------------------------------------------------------------------------- #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG PDF Chat") as demo:
        with gr.Column(elem_classes="main-container"):

            # Header
            gr.Markdown(
                "<h1 style='text-align:center; margin-bottom:0;'>📄 PDF Chat</h1>"
            )
            gr.Markdown(
                "<p style='text-align:center; color:#888;'>"
                "Upload a PDF and ask anything about it."
                "</p>"
            )

            # Chat display
            chatbot = gr.Chatbot(
                show_label=False,
                height=550,
                elem_id="chatbot-container"
            )
            
            with gr.Accordion("📊 Evaluation Report", open=False) as eval_box:
                eval_output = gr.Markdown("No evaluation yet.")
                
            with gr.Accordion("🔍 Hallucination Report", open=False) as hal_box:
                hal_output = gr.Markdown("No hallucination check yet.")
                
            with gr.Accordion("🔀 Re-ranker Report", open=False) as rerank_box:
                rerank_output = gr.Markdown("No re-ranking yet.")
                
            with gr.Accordion("🛡️ Guardrails Report", open=False) as guard_box:
                guard_output = gr.Markdown("No guardrails check yet.")
                
            with gr.Accordion("🤝 Multi-Agent Report", open=False) as multi_agent_box:
                multi_agent_output = gr.Markdown("No multi-agent run yet.")

            # File status indicator (centered, subtle)
            file_status = gr.Markdown(
                "No documents indexed yet.",
                elem_classes="file-status",
            )
            
            # Modes
            with gr.Row():
                use_agent = gr.Checkbox(label="🧮 Enable Calculator & PDF Search", value=False)
                eval_mode = gr.Checkbox(label="📊 Evaluate Answer", value=False)
                detect_hallucination = gr.Checkbox(label="🔍 Detect Hallucinations", value=False)
                show_reranker = gr.Checkbox(label="🔀 Show Re-ranker", value=False)
                enable_guardrails = gr.Checkbox(label="🛡️ Enable Guardrails", value=False)
                multi_agent_mode = gr.Checkbox(label="🤝 Multi-Agent Mode", value=False)

            # Floating input bar
            with gr.Row(elem_id="input-row"):
                upload_btn = gr.UploadButton(
                    "➕",
                    file_types=[".pdf"],
                    elem_id="upload-btn",
                    scale=0,
                )
                msg_box = gr.Textbox(
                    placeholder="Ask about your PDF...",
                    container=False,
                    elem_id="msg-box",
                    scale=7,
                    autofocus=True,
                    show_label=False,
                )
                send_btn = gr.Button("↑", elem_id="send-btn", scale=0)

            # Clear chat button (subtle, below input bar)
            with gr.Row():
                reset_btn = gr.Button(
                    "Clear Chat",
                    variant="secondary",
                    size="sm",
                )

        # ------------------------------------------------------------------- #
        # Event wiring                                                         #
        # ------------------------------------------------------------------- #
        upload_btn.upload(
            fn=handle_upload,
            inputs=[upload_btn],
            outputs=[file_status],
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot, use_agent, eval_mode, detect_hallucination, show_reranker, enable_guardrails, multi_agent_mode],
            outputs=[msg_box, chatbot, eval_box, eval_output, hal_box, hal_output, rerank_box, rerank_output, guard_box, guard_output, multi_agent_box, multi_agent_output],
        )

        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot, use_agent, eval_mode, detect_hallucination, show_reranker, enable_guardrails, multi_agent_mode],
            outputs=[msg_box, chatbot, eval_box, eval_output, hal_box, hal_output, rerank_box, rerank_output, guard_box, guard_output, multi_agent_box, multi_agent_output],
        )

        reset_btn.click(
            fn=reset_chat,
            inputs=[],
            outputs=[chatbot, file_status],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()