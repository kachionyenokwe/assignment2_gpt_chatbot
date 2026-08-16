import json
import os
from typing import Dict, Any, List, Union

# Resolve absolute path to local knowledge base markdown file
KB_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base.md"
)


# -------------------------------------------------------------------
# Python Native Tool Implementations
# -------------------------------------------------------------------

def get_weather(city: str) -> Dict[str, Any]:
    """
    Returns simulated environmental and weather metrics for supported cities.
    """

    city_clean = city.strip().title()

    weather_db = {
        "Metrocity": {
            "temperature": "28°C",
            "condition": "Heavy Rain",
            "humidity": "88%",
            "wind_speed": "22 km/h",
            "traffic_impact": "High (Visibility Reduced)",
            "grid_load_level": "Moderate"
        },
        "Seattle": {
            "temperature": "14°C",
            "condition": "Overcast",
            "humidity": "75%",
            "wind_speed": "10 km/h",
            "traffic_impact": "Low",
            "grid_load_level": "Normal"
        },
        "Tokyo": {
            "temperature": "31°C",
            "condition": "Clear / Extreme Heat",
            "humidity": "65%",
            "wind_speed": "8 km/h",
            "traffic_impact": "Moderate (AC Energy Surge)",
            "grid_load_level": "High (Peak AC Draw)"
        }
    }

    if city_clean in weather_db:
        return {"status": "success", "city": city_clean, **weather_db[city_clean]}

    return {
        "status": "not_found",
        "city": city_clean,
        "message": f"No weather data available for {city_clean}."
    }

def lookup_kb(query: str) -> Dict[str, Any]:
    """
    Searches the local Markdown Knowledge Base for smart city policies,
    traffic signal rules, and grid management procedures.
    """
    if not os.path.exists(KB_FILE_PATH):
        return {"status": "error", "message": f"Knowledge Base file not found at {KB_FILE_PATH}"}

    try:
        with open(KB_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {"status": "error", "message": f"Failed to read Knowledge Base: {str(e)}"}

    # Keyword-matching search algorithm over KB sections
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    sections = content.split("## ")
    
    matching_sections = []
    for sec in sections:
        if not sec.strip():
            continue
        sec_text = "## " + sec
        sec_lower = sec_text.lower()
        
        # Count keyword matches
        score = sum(1 for term in query_terms if term in sec_lower)
        if score > 0 or not query_terms:
            matching_sections.append((score, sec_text.strip()))

    # Sort matches by term frequency score
    matching_sections.sort(key=lambda x: x[0], reverse=True)

    if matching_sections:
        top_results = [sec[1] for sec in matching_sections[:2]]
        return {
            "status": "success",
            "query": query,
            "results_found": len(top_results),
            "excerpts": "\n\n---\n\n".join(top_results)
        }
    
    return {
        "status": "not_found",
        "query": query,
        "message": "No specific KB policy section matched your query."
    }


# -------------------------------------------------------------------
# Tool Execution Registry Router
# -------------------------------------------------------------------

TOOL_FUNCTION_MAP = {
    "get_weather": get_weather,
    "lookup_kb": lookup_kb
}


def execute_tool_call(tool_name: str, arguments_json: Union[str, Dict[str, Any]]) -> str:
    """
    Executes a tool requested by the GPT model and returns a JSON string result.
    """
    if tool_name not in TOOL_FUNCTION_MAP:
        return json.dumps({"error": f"Tool '{tool_name}' is not recognized."})

    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
        result = TOOL_FUNCTION_MAP[tool_name](**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Failed to execute {tool_name}: {str(e)}"})


# -------------------------------------------------------------------
# OpenAI Tool JSON Schemas (Compatible with OpenAI Python SDK v3.x)
# -------------------------------------------------------------------

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real-time environmental weather and traffic/grid impact metrics for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the target city (e.g., Metrocity, Tokyo, Seattle)."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_kb",
            "description": "Search the MetroCity Smart Infrastructure Knowledge Base for policies, traffic light rules, EV load shedding, and incident reporting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or question regarding traffic signals, emergency overrides, peak power hours, or 311 support."
                    }
                },
                "required": ["query"]
            }
        }
    }
]