import os
import uuid
import streamlit as st

st.set_page_config(
    page_title="AmazonHelp Support AI",
    page_icon="💬",
    layout="wide",
)

# Configure Gemini before importing the core pipeline.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

@st.cache_resource(show_spinner="Loading the ML + RAG pipeline...")
def load_core():
    import pipeline as core
    return core

core = load_core()

st.title("AmazonHelp Support AI")
st.caption("Hiver SDE Intern — Classical ML + RAG + LangGraph HITL")

with st.sidebar:
    st.subheader("System")
    st.success("AmazonHelp")
    st.write("Classical ML → LLM Judge → RAG / HITL")
    st.divider()
    st.write("The core logic is implemented in `pipeline.py`. This Streamlit layer provides the interactive UI.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex}"
if "pending" not in st.session_state:
    st.session_state.pending = None
if "result" not in st.session_state:
    st.session_state.result = None

query = st.text_area(
    "Customer message",
    placeholder="Example: I was charged twice on my card this month, help!",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Run support pipeline", type="primary", use_container_width=True)
with col2:
    reset = st.button("New conversation", use_container_width=True)

if reset:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex}"
    st.session_state.pending = None
    st.session_state.result = None
    st.rerun()

if run:
    if not query.strip():
        st.warning("Enter a customer message first.")
    else:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            with st.spinner("Running classifier, retrieval and routing..."):
                result = core.app.invoke({"query": query.strip()}, config=config)
            snapshot = core.app.get_state(config)

            if snapshot.next:
                interrupt_payload = snapshot.tasks[0].interrupts[0].value
                st.session_state.pending = interrupt_payload
                st.session_state.result = result
                st.rerun()
            else:
                st.session_state.pending = None
                st.session_state.result = result
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")

result = st.session_state.result
pending = st.session_state.pending

if result:
    st.divider()
    st.subheader("Pipeline result")

    a, b, c = st.columns(3)
    a.metric("Intent", result.get("predicted_intent", "—"))
    b.metric("Judge", result.get("judge_decision", "—"))
    c.metric("HITL", "Required" if pending else "Not required")

    if pending:
        st.warning("Human intervention required")
        st.write("The graph is paused. Review the context below and provide the final customer response.")

        with st.expander("Customer / routing context", expanded=True):
            st.write("**Customer query:**", pending.get("customer_query", query))
            st.write("**Predicted intent:**", pending.get("predicted_intent", "—"))
            st.write("**Judge reason:**", pending.get("judge_reason", "—"))
            historical = pending.get("historical_response")
            if historical:
                st.write("**Historical support evidence:**")
                st.info(historical)

        human_response = st.text_area(
            "Human final response",
            height=180,
            placeholder="Write the response you want to send to the customer...",
            key="human_response_box",
        )

        if st.button("Submit human response", type="primary"):
            if not human_response.strip():
                st.warning("Enter the human response before submitting.")
            else:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    with st.spinner("Resuming LangGraph..."):
                        final = core.app.invoke(
                            core.Command(resume={"human_response": human_response.strip()}),
                            config=config,
                        )
                    st.session_state.pending = None
                    st.session_state.result = final
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not resume HITL: {exc}")
    else:
        st.success("Automated response generated")
        st.subheader("Final response")
        st.write(result.get("final_response", "No final response returned."))

        if result.get("judge_reason"):
            with st.expander("Routing explanation"):
                st.write(result["judge_reason"])

st.divider()
st.caption("Demo UI around the supplied core pipeline. For the Hiver evaluation submission, keep the notebook/evaluation artifacts separate from this Streamlit demo.")
