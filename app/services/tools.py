# app/services/tools.py

from langchain_core.tools import tool
import datetime


# --- NEW TOOL ---
@tool
def get_available_tools() -> str:
    """
    Use this tool to get a list of all available tools and their descriptions.
    This is useful when the user asks "what can you do?" or "what tools do you have?".
    """
    descriptions = []
    for t in all_tools:
        # We don't want to list the tool that lists tools.
        if t.name != "get_available_tools":
            descriptions.append(f"- **{t.name}**: {t.description}")
   
    if not descriptions:
        return "I do not have any tools available at the moment."
       
    return "I have the following tools available:\n" + "\n".join(descriptions)


@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """
    A simple calculator that evaluates a mathematical expression.
    Example: "2+2" -> "4"
    """
    try:
        # Use a safe eval method if you plan to extend this
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression - {e}"


# Add the new tool to the list
all_tools = [get_available_tools, get_current_time, calculator]
