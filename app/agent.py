import json
import os
from typing import AsyncGenerator, Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from app.tools import TOOLS_SCHEMA, execute_tool_call
from app.memory import memory_manager
from app.telemetry import TelemetryTracker
from app.safety import safety_guard

# Load variables from .env file into os.environ
load_dotenv()

# Read the key directly from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is missing. Check your .env file!")

# Initialize AsyncOpenAI client with your active key
openai_client = AsyncOpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-4o-mini"


async def stream_chat_response(
    conversation_id: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2
) -> AsyncGenerator[str, None]:
    """
    Core LLM Orchestration Engine:
    1. Validates input safety and updates conversation memory.
    2. Sends request with tools schema to OpenAI API.
    3. Handles tool execution handshakes if requested by model.
    4. Streams text response tokens back to client using SSE format.
    5. Calculates and yields final performance telemetry metrics.
    """
    telemetry = TelemetryTracker(model_name=model)
    telemetry.start_turn()

    # 1. Safety Guardrail Validation
    is_valid, err_msg = safety_guard.validate_input(user_message)
    if not is_valid:
        error_payload = {
            "type": "error",
            "content": err_msg or "Security policy violation."
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        return

    # 2. Append User Message to Memory History
    memory_manager.add_message(conversation_id, {"role": "user", "content": user_message})
    history = memory_manager.get_history(conversation_id)

    try:
        # 3. First-pass API Call (Checking for Tool Calls vs. Immediate Response)
        response = await openai_client.chat.completions.create(
            model=model,
            messages=history,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=temperature,
            stream=True
        )

        # Buffers for assembling tool call arguments across streamed chunks
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        text_content_buffer = ""
        has_tool_calls = False

        async for chunk in response:
            telemetry.record_first_token()
            delta = chunk.choices[0].delta if chunk.choices else None

            if not delta:
                continue

            # Stream regular text tokens immediately if no tool call is being constructed
            if delta.content:
                text_content_buffer += delta.content
                token_payload = {"type": "token", "content": delta.content}
                yield f"data: {json.dumps(token_payload)}\n\n"

            # Accumulate tool call fragments if model decides to invoke a tool
            if delta.tool_calls:
                has_tool_calls = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments": ""
                        }
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_buffer[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc.function.arguments

        # 4. Handle Tool Execution Handshake if Tool Calls Were Received
        if has_tool_calls and tool_calls_buffer:
            # Build assistant message containing tool calls for history
            assistant_tool_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_data["id"],
                        "type": "function",
                        "function": {
                            "name": tool_data["name"],
                            "arguments": tool_data["arguments"]
                        }
                    }
                    for tool_data in tool_calls_buffer.values()
                ]
            }
            memory_manager.add_message(conversation_id, assistant_tool_msg)

            # Execute each requested tool locally
            for tool_data in tool_calls_buffer.values():
                telemetry.record_tool_call()
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]
                tool_call_id = tool_data["id"]

                # Notify client that tool is executing
                status_payload = {
                    "type": "status",
                    "content": f"Executing tool `{tool_name}` with args: {tool_args}"
                }
                yield f"data: {json.dumps(status_payload)}\n\n"

                # Execute function
                tool_result_json = execute_tool_call(tool_name, tool_args)

                # Append tool result to conversation history
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result_json
                }
                memory_manager.add_message(conversation_id, tool_result_msg)

            # 5. Second-pass API Call to synthesize final answer using tool output
            updated_history = memory_manager.get_history(conversation_id)
            second_response = await openai_client.chat.completions.create(
                model=model,
                messages=updated_history,
                temperature=temperature,
                stream=True
            )

            final_text_buffer = ""
            async for chunk in second_response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    final_text_buffer += delta.content
                    token_payload = {"type": "token", "content": delta.content}
                    yield f"data: {json.dumps(token_payload)}\n\n"

            # Save final assistant text response to memory
            memory_manager.add_message(
                conversation_id,
                {"role": "assistant", "content": final_text_buffer}
            )

        elif text_content_buffer:
            # Save direct assistant text response to memory
            memory_manager.add_message(
                conversation_id,
                {"role": "assistant", "content": text_content_buffer}
            )

    except Exception as e:
        error_payload = {
            "type": "error",
            "content": f"API Execution Error: {str(e)}"
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        return

    # 6. Estimate Token Usage & Yield Final Telemetry Metrics Payload
    # Rough estimate calculation: 1 token ~ 4 characters
    full_history_text = "".join([str(m.get("content", "")) for m in history])
    estimated_prompt_tokens = max(10, len(full_history_text) // 4)
    estimated_completion_tokens = max(5, len(text_content_buffer or "done") // 4)

    metrics = telemetry.end_turn(
        prompt_tokens=estimated_prompt_tokens,
        completion_tokens=estimated_completion_tokens
    )

    metrics_payload = {"type": "telemetry", "data": metrics}
    yield f"data: {json.dumps(metrics_payload)}\n\n"