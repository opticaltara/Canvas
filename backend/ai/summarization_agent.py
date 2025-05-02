"""
Summarization Agent Service

Agent for summarizing text content using an LLM, producing Markdown output.
"""

import logging
from typing import Optional, AsyncGenerator, Union, Dict, Any
import os

from pydantic_ai import Agent, UnexpectedModelBehavior

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from backend.config import get_settings
# Need to define SummarizationQueryResult later in backend/core/query_result.py
from backend.core.query_result import SummarizationQueryResult
from backend.ai.events import (
    EventType,
    AgentType,
    StatusType,
    StatusUpdateEvent,
    SummaryUpdateEvent # Use this for intermediate updates if needed
)

summarization_agent_logger = logging.getLogger("ai.summarization_agent")

class SummarizationAgent:
    """Agent for generating text summaries in Markdown format."""
    def __init__(self, notebook_id: str):
        summarization_agent_logger.info(f"Initializing SummarizationAgent for notebook_id: {notebook_id}")
        self.settings = get_settings()
        self.model = OpenAIModel(
                self.settings.ai_model,
                provider=OpenAIProvider(
                    base_url='https://openrouter.ai/api/v1',
                    api_key=self.settings.openrouter_api_key,
                ),
        )
        self.notebook_id = notebook_id
        self.agent: Optional[Agent] = None # Agent instance created lazily or during run
        summarization_agent_logger.info(f"SummarizationAgent initialized successfully.")

    # Removed _get_stdio_server method as MCP is not needed

    def _read_system_prompt(self) -> str:
        """Reads the system prompt from the dedicated file."""
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", "summarization_agent_system_prompt.txt")
        try:
            with open(prompt_file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            summarization_agent_logger.error(f"System prompt file not found at: {prompt_file_path}")
            # Return a default minimal prompt as fallback
            return "You are a helpful AI assistant skilled at summarizing text into concise Markdown."
        except Exception as e:
            summarization_agent_logger.error(f"Error reading system prompt file {prompt_file_path}: {e}", exc_info=True)
            return "You are a helpful AI assistant skilled at summarizing text into concise Markdown." # Fallback

    def _initialize_agent(self) -> Agent:
        """Initializes the Pydantic AI Agent for summarization."""
        system_prompt = self._read_system_prompt()

        agent = Agent(
            self.model,
            # Use the defined SummarizationQueryResult type
            output_type=SummarizationQueryResult, 
            # No mcp_servers needed
            # mcp_servers=[], 
            system_prompt=system_prompt,
        )
        summarization_agent_logger.info(f"Summarization Agent instance created.")
        return agent # type: ignore

    async def run_summarization(
        self,
        text_to_summarize: str,
        original_request: Optional[str] = None
    ) -> AsyncGenerator[Union[StatusUpdateEvent, SummaryUpdateEvent, SummarizationQueryResult], None]: # Updated yield type
        """Generate a summary for the provided text, yielding status updates."""
        query_description = original_request or f"Summarize the following text: {text_to_summarize[:100]}..."
        summarization_agent_logger.info(f"Running summarization for request: '{query_description}'")
        # Yield StatusUpdateEvent
        yield StatusUpdateEvent(status=StatusType.STARTING, agent_type=AgentType.SUMMARIZATION, message="Initializing summarization agent...", attempt=None, max_attempts=None, reason=None, step_id=None, original_plan_step_id=None)
        
        try:
            if not self.agent:
                 self.agent = self._initialize_agent()
            # Yield StatusUpdateEvent
            yield StatusUpdateEvent(status=StatusType.AGENT_CREATED, agent_type=AgentType.SUMMARIZATION, message="Summarization agent ready.", attempt=None, max_attempts=None, reason=None, step_id=None, original_plan_step_id=None)
        except Exception as init_err:
            error_msg = f"Failed to initialize Summarization Agent: {init_err}"
            summarization_agent_logger.error(error_msg, exc_info=True)
            # Yield StatusUpdateEvent with error status
            yield StatusUpdateEvent(status=StatusType.ERROR, agent_type=AgentType.SUMMARIZATION, message=error_msg, reason="Initialization failed", attempt=None, max_attempts=None, step_id=None, original_plan_step_id=None)
            yield SummarizationQueryResult(query=query_description, data="", error=error_msg)
            return

        max_attempts = 3 # Reduced attempts for simpler task
        last_error = None
        final_result_data: Optional[SummarizationQueryResult] = None
        
        input_prompt = f"""Please summarize the following text into clear and concise Markdown:

---
{text_to_summarize}
---"""

        # Yield StatusUpdateEvent
        yield StatusUpdateEvent(status=StatusType.STARTING_ATTEMPTS, agent_type=AgentType.SUMMARIZATION, message=f"Attempting summarization (max {max_attempts} attempts)...", max_attempts=max_attempts, attempt=None, reason=None, step_id=None, original_plan_step_id=None)

        for attempt in range(max_attempts):
            current_attempt = attempt + 1
            summarization_agent_logger.info(f"Summarization Attempt {current_attempt}/{max_attempts}")
            # Yield StatusUpdateEvent
            yield StatusUpdateEvent(status=StatusType.ATTEMPT_START, agent_type=AgentType.SUMMARIZATION, attempt=current_attempt, max_attempts=max_attempts, reason=None, step_id=None, original_plan_step_id=None, message=f"Starting attempt {current_attempt}")

            try:
                summarization_agent_logger.info(f"Running agent with prompt: {input_prompt[:200]}...")
                # Use agent.run directly as no complex iteration or tool calls are expected
                run_result_obj = await self.agent.run(input_prompt)
                summarization_agent_logger.info(f"Attempt {current_attempt}: Agent run finished. Raw result: {run_result_obj}")
                
                # Access the output Pydantic model
                run_result = run_result_obj.output

                # Process the result
                if isinstance(run_result, SummarizationQueryResult):
                    final_result_data = run_result
                    # Ensure the original query context is preserved if available
                    final_result_data.query = query_description 
                    summarization_agent_logger.info(f"Successfully processed result on attempt {current_attempt}.")
                    # Use SummaryUpdateEvent for success
                    yield SummaryUpdateEvent(update_info={"message": f"Processing successful on attempt {current_attempt}"})
                    yield final_result_data
                    return # Success
                elif run_result and hasattr(run_result, 'data'):
                    summarization_agent_logger.warning(f"Agent returned unexpected type {type(run_result)}, but extracting data.")
                    # Attempt to construct the result object if possible
                    final_result_data = SummarizationQueryResult(
                        query=query_description,
                        data=str(run_result),
                        error=None
                    )
                    # Use SummaryUpdateEvent for success
                    yield SummaryUpdateEvent(update_info={"message": f"Processing successful (fallback) on attempt {current_attempt}"})
                    yield final_result_data
                    return # Success with fallback
                else:
                    summarization_agent_logger.error(f"Agent run attempt {current_attempt} produced invalid result: {run_result}")
                    last_error = f"Agent run produced invalid result type: {type(run_result)}"
                    # Use SummaryUpdateEvent for error
                    yield SummaryUpdateEvent(update_info={"error": last_error, "attempt": current_attempt})

            except UnexpectedModelBehavior as e:
                error_str = f"Unexpected model behavior: {str(e)}"
                summarization_agent_logger.error(f"UnexpectedModelBehavior during summarization attempt {current_attempt}: {e}", exc_info=True)
                # Use SummaryUpdateEvent for error
                yield SummaryUpdateEvent(update_info={"error": error_str, "attempt": current_attempt})
                last_error = f"Summarization failed due to unexpected model behavior: {str(e)}"
                # Potentially break if it's a model issue unlikely to resolve on retry
                # break 
            
            except Exception as e:
                error_str = f"General error: {str(e)}"
                summarization_agent_logger.error(f"General error during summarization attempt {current_attempt}: {e}", exc_info=True)
                # Use SummaryUpdateEvent for error
                yield SummaryUpdateEvent(update_info={"error": error_str, "attempt": current_attempt})
                last_error = f"Summarization failed unexpectedly: {str(e)}"
                # break # Probably break on general errors

            # If loop continues (error occurred and didn't return/break), prepare for next attempt or final failure
            if attempt < max_attempts - 1:
                 # Yield StatusUpdateEvent
                 yield StatusUpdateEvent(status=StatusType.RETRYING, agent_type=AgentType.SUMMARIZATION, attempt=current_attempt, reason=last_error or "unknown", max_attempts=max_attempts, step_id=None, original_plan_step_id=None, message=f"Retrying after error on attempt {current_attempt}")
            else:
                 summarization_agent_logger.error(f"Summarization failed after {max_attempts} attempts.")
                 break # Exit loop after final attempt failure

        # If loop finished without returning successfully
        if final_result_data is None:
            final_error_msg = f"Summarization failed after {max_attempts} attempts. Last error: {last_error or 'Unknown failure'}"
            summarization_agent_logger.error(final_error_msg)
            # Yield StatusUpdateEvent with error
            yield StatusUpdateEvent(status=StatusType.FINISHED_ERROR, agent_type=AgentType.SUMMARIZATION, message=final_error_msg, attempt=current_attempt, max_attempts=max_attempts, reason=last_error or 'Unknown failure', step_id=None, original_plan_step_id=None)
            yield SummarizationQueryResult(query=query_description, data="", error=final_error_msg)

# Example usage (for testing, might be removed later)
async def main():
    import asyncio
    logging.basicConfig(level=logging.INFO)
    agent = SummarizationAgent(notebook_id="test-notebook")
    text = """
    Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand, generate, and interact with human language. 
    Based on deep learning architectures, particularly transformers, LLMs are trained on vast amounts of text data, enabling them to perform a wide range of natural language processing tasks. 
    These tasks include text generation, translation, summarization, question answering, and sentiment analysis. 
    Key characteristics of LLMs include their massive size (billions of parameters), emergent abilities (capabilities not explicitly trained for), and their potential to revolutionize various industries. 
    However, challenges such as computational cost, potential biases in training data, and ethical considerations surrounding their use remain significant areas of research and discussion.
    """
    async for result in agent.run_summarization(text_to_summarize=text):
        print(result)

if __name__ == "__main__":
    # Running the example requires setting OPENROUTER_API_KEY environment variable
    # and potentially installing pydantic-ai, openai etc.
    # Example: export OPENROUTER_API_KEY='your_key_here'
    #          python -m backend.ai.summarization_agent
    
    # Check if settings are available before running main
    try:
        get_settings() # This will raise if env vars are missing
        import asyncio
        # asyncio.run(main()) # Commented out to prevent execution without setup
        print("To run the example main function, ensure OPENROUTER_API_KEY is set and uncomment 'asyncio.run(main())'.")
    except Exception as e:
        print(f"Could not run main due to missing settings or dependencies: {e}")
        print("Ensure OPENROUTER_API_KEY environment variable is set.") 