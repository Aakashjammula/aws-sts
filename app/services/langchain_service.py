# app/services/langchain_service.py

from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    HumanMessage,
)
from langchain_aws import ChatBedrockConverse

# App modules
from services.prompt import SYSTEM_PROMPT
from services.tools import all_tools
from core.utils import printer
from config.settings import config


# -----------------------------
# State schema
# -----------------------------
class GraphState(TypedDict, total=False):
    # Persistent conversation (without the system prompt)
    messages_history: List[AnyMessage]
    # Working buffer (this turn)
    turn: List[AnyMessage]
    # Most recent model message (could contain tool_calls)
    model_msg: Optional[AnyMessage]
    # Final output text to stream/say
    final_output: str
    # Input of this turn
    user_text: str
    # Loop control
    done: bool


# -----------------------------
# LLM and helpers
# -----------------------------
def build_llm():
    # Allow tool calling
    return ChatBedrockConverse(
        model_id=config["model_id"],
        region_name=config["region"],
    ).bind_tools(all_tools)


SYSTEM_MSG = SystemMessage(content=SYSTEM_PROMPT)


def safe_text(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out += item.get("text", "")
        return out
    return str(content)


def tool_map() -> Dict[str, Any]:
    return {t.name: t for t in all_tools}


def normalize_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    if isinstance(tool_calls, dict):
        return [tool_calls]
    return list(tool_calls)


def get_args(call: Dict[str, Any]) -> Dict[str, Any]:
    if "args" in call and isinstance(call["args"], dict):
        return call["args"]
    if "arguments" in call and isinstance(call["arguments"], dict):
        return call["arguments"]
    return {}


def get_call_id(call: Dict[str, Any]) -> str:
    return call.get("id") or call.get("call_id") or "unknown"


# -----------------------------
# Nodes
# -----------------------------
def agent(state: GraphState) -> GraphState:
    """
    Core agent step:
    - Takes messages_history + turn (latest HumanMessage and any ToolMessages)
    - Calls LLM once
    - If tool_calls present: append AI message and signal to go to tools
    - Else: finalize answer and mark done
    """
    llm = build_llm()
    history = state.get("messages_history", [])
    turn = state.get("turn", [])
    messages_for_llm = [SYSTEM_MSG] + history + turn

    printer("[LLM] agent step...", "info")
    model_msg = llm.invoke(messages_for_llm)

    # Append model message to turn buffer
    new_turn = list(turn) + [model_msg]

    # Check for tool calls
    calls = normalize_calls(getattr(model_msg, "tool_calls", None))
    if calls:
        # Will route to tools node
        return {
            "turn": new_turn,
            "model_msg": model_msg,
            "done": False,
        }

    # No tools: finalize immediately using the first response content
    text = safe_text(getattr(model_msg, "content", ""))
    new_turn.append(AIMessage(content=text))
    return {
        "turn": new_turn,
        "model_msg": model_msg,
        "final_output": text,
        "done": True,
    }


def tools(state: GraphState) -> GraphState:
    """
    Execute all requested tools from the last model_msg.
    Append ToolMessages to turn, then loop back to agent.
    """
    model_msg = state.get("model_msg")
    if not model_msg:
        # Nothing to do; go back to agent
        return {}

    tmap = tool_map()
    calls = normalize_calls(getattr(model_msg, "tool_calls", None))
    if not calls:
        return {}

    names = [c.get("name") for c in calls]
    printer(f"[LLM] tools step: {names}", "info")

    tool_msgs: List[ToolMessage] = []
    for c in calls:
        name = c["name"]
        args = get_args(c)
        fn = tmap[name]
        out = fn.invoke(args)
        tool_msgs.append(ToolMessage(content=str(out), tool_call_id=get_call_id(c)))

    return {"turn": list(state.get("turn", [])) + tool_msgs}


def end(state: GraphState) -> GraphState:
    """
    Commit the turn into persistent history and clear working buffers.
    """
    history = list(state.get("messages_history", []))
    turn = list(state.get("turn", []))
    return {
        "messages_history": history + turn,
        "turn": [],
        "model_msg": None,
    }


# -----------------------------
# Router
# -----------------------------
def route_from_agent(state: GraphState) -> str:
    # If agent set done=True, end; else if model asked for tools, go to tools
    if state.get("done"):
        return "end"
    model_msg = state.get("model_msg")
    calls = normalize_calls(getattr(model_msg, "tool_calls", None)) if model_msg else []
    return "tools" if calls else "end"


# -----------------------------
# Build graph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_node("end", end)

# Entry is agent; before calling, we must seed the user message into turn
builder.set_entry_point("agent")

# agent routes either to tools or end
builder.add_conditional_edges("agent", route_from_agent, {"tools": "tools", "end": "end"})
# After tools, we must call agent again (loop)
builder.add_edge("tools", "agent")
# end goes to terminal
builder.add_edge("end", END)

# Durable memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# -----------------------------
# Service wrapper (drop-in replacement for LangchainService)
# -----------------------------
class LangGraphService:
    def __init__(self, thread_id: str = "default"):
        self.thread_id = thread_id

    def _initial_state(self) -> GraphState:
        return {
            "messages_history": [],
            "turn": [],
            "model_msg": None,
            "final_output": "",
            "user_text": "",
            "done": False,
        }

    def _ensure_initialized(self):
        cfg = {"configurable": {"thread_id": self.thread_id}}
        snap = graph.get_state(cfg)
        if snap is None or not snap.values:
            graph.update_state(cfg, self._initial_state())

    def run_turn(self, user_text: str) -> str:
        """
        Synchronous helper that:
        - Seeds the user message into state.turn
        - Invokes the graph until end
        - Returns final_output (the last assistant content produced)
        """
        self._ensure_initialized()
        cfg = {"configurable": {"thread_id": self.thread_id}}

        # Seed the user message into the working turn
        graph.update_state(
            cfg,
            {
                "turn": [HumanMessage(content=user_text)],
                "user_text": user_text,
                "final_output": "",
                "done": False,
            },
        )

        # Run to completion
        final_state = graph.invoke({}, cfg)

        # If there was no no-tool finalize text, synthesize final text from the last AI message
        if not final_state.get("final_output"):
            # Extract last AI message from messages_history
            msgs = final_state.get("messages_history", [])
            last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            final_text = safe_text(getattr(last_ai, "content", "")) if last_ai else ""
            return final_text

        return final_state["final_output"]

    async def get_response_stream(self, text: str):
        """
        Async generator mimicking your previous interface.
        For now, we chunk the final answer after the run completes.
        If you want token-level streaming, we can expose events via graph.stream.
        """
        try:
            output = self.run_turn(text)
            if not output:
                output = "..."

            chunk_size = 160
            for i in range(0, len(output), chunk_size):
                yield output[i : i + chunk_size]
        except Exception as e:
            printer(f"[ERROR] Error in LangGraphService: {e}", "error")
            import traceback
            printer(f"[ERROR] Full traceback: {traceback.format_exc()}", "debug")
            yield "I'm sorry, I encountered an error. Please try again."
