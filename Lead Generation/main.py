import asyncio
import json
import pathlib
import re
from typing_extensions import TypedDict
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from LLMConfig import llm
from mcp_utils import get_mcp_clients
from roleAgents import ApolloAgent, ResearchAgent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from roleFunctions import email_generator
import pandas as pd

# Initialize Azure model from LLMConfig
azure_model = llm()

# Initialize MCP clients for interacting with web automation tools
mcp_clients = get_mcp_clients()


# Define the structure of the GraphState dictionary
class GraphState(TypedDict):
    subgraph_messages: list  # List of messages exchanged so far
    next_agent: str          # Name of the next agent to execute
    information_list: any    # Collected information / data

def extract_json(text: str) -> str:
    """Extract JSON content from a string using regex."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return text


def supervisor_node_factory(model, prompt: str, log_path: str = None) -> RunnableLambda:
    """Creates a RunnableLambda node for the Supervisor agent."""
    system_prompt = prompt

    async def supervisor_node(state: GraphState) -> GraphState:
        # Extract existing state information
        try:
            messages = state["subgraph_messages"]
            information_list = state["information_list"]
        except Exception as e:
            print(f"Error reading state in supervisor_node: {e}")
            return {"subgraph_messages": [], "next_agent": "end", "information_list": []}

        # Prepare system message for the Supervisor agent
        system_message = SystemMessage(content=system_prompt)

        try:
            # Call the model with the system prompt and recent agent updates
            response = await model.ainvoke(
                system_message.content
                + "\n\nLast agent update:\n" + str(messages)
            )
        except Exception as e:
            print(f"Error invoking supervisor model: {e}")
            return {"subgraph_messages": messages, "next_agent": "end", "information_list": information_list}

        # Parse the response as JSON
        try:
            content = extract_json(response.content)
            parsed = json.loads(content)
            print("response ->", parsed, "\n")
            next_agent = parsed.get("next_agent", "end")
            assistant_msg = parsed.get("message", "")
            updated_state = parsed.get("updated_state", {})
        except Exception as e:
            print(f"Error parsing supervisor response: {e} \n{response}")
            next_agent = "end"
            assistant_msg = "Could not parse response, ending."
            updated_state = information_list

        # Append AI response to messages
        try:
            messages.append(AIMessage(content=assistant_msg))
        except Exception as e:
            print(f"Error appending AIMessage: {e}")
            messages = [AIMessage(content=assistant_msg)]

        # Return updated state for next execution
        return {"subgraph_messages": messages, "next_agent": next_agent, "information_list": updated_state}

    return RunnableLambda(supervisor_node)


def agent_node(agent: RunnableLambda) -> RunnableLambda:
    """Wraps an agent node so it can be executed in the workflow graph."""
    async def run_agent(state: GraphState) -> GraphState:
        # Extract current state information
        try:
            messages = state.get("subgraph_messages", [])
            information_list = state.get("information_list", [])
        except Exception as e:
            print(f"Error reading state in agent_node: {e}")
            return {"subgraph_messages": [], "next_agent": "supervisor", "information_list": []}

        # Execute the agent and append its response
        try:
            response = await agent.ainvoke({"messages": messages})
            updated_messages = messages + [response["messages"][-1]]
            messages = updated_messages[-10:]  # Keep last 10 messages for context
            last_response = response["messages"][-1]
        except Exception as e:
            print(f"Error running agent: {e}")
            last_response = AIMessage(content="Agent failed")
            messages.append(last_response)

        return {"subgraph_messages": messages, "next_agent": "supervisor", "information_list": information_list}

    return RunnableLambda(run_agent)


async def run_agent(prompt: str):
    """Main async function to run the agent workflow."""
    try:
        # Start a Playwright session via MCP client
        async with mcp_clients.session("playwright") as session:
            # Load tools for agents
            try:
                all_tools = await load_mcp_tools(session)
            except Exception as e:
                print(f"Error loading MCP tools: {e}")
                all_tools = []

            # Initialize agents
            try:
                Apollo_agent = ApolloAgent(tools=all_tools).agent
                Research_agent = ResearchAgent(tools=all_tools).agent
            except Exception as e:
                print(f"Error initializing agents: {e}")
                return

            # Supervisor system prompt instructions
            try:
                supervisor_system_prompt = (
                    "You are the Supervisor Agent. Your role is to **direct and monitor agents** to find, research, and contact business clients.\n\n"
                    "... [prompt truncated for brevity, full content remains in code] ..."
                )

                # Build the workflow graph
                workflow = StateGraph(GraphState)
                workflow.add_node("supervisor", supervisor_node_factory(azure_model, supervisor_system_prompt))
                workflow.add_node("ApolloAgent", agent_node(Apollo_agent))
                workflow.add_node("ResearchAgent", agent_node(Research_agent))
                workflow.add_node("EmailGenerator", email_generator())

                # Set workflow entry point
                workflow.set_entry_point("supervisor")

                # Define conditional transitions based on supervisor's decision
                workflow.add_conditional_edges(
                    "supervisor",
                    lambda state: state["next_agent"],
                    {
                        "ApolloAgent": "ApolloAgent",
                        "ResearchAgent": "ResearchAgent",
                        "EmailGenerator": "EmailGenerator",
                        "end": END,
                    }
                )

                # Add edges for each agent back to supervisor
                workflow.add_edge("ApolloAgent", "supervisor")
                workflow.add_edge("ResearchAgent", "supervisor")
                workflow.add_edge("EmailGenerator", "supervisor")

                app = workflow.compile()
            except Exception as e:
                print(f"Error setting up workflow: {e}")
                return

            # Initialize empty records for 5 leads
            try:
                initial_list = [{
                    "name": "",
                    "designation": "",
                    "employee count": "",
                    "email": "",
                    "linkedIn link": "",
                    "company name": "",
                    "company website link": "",
                    "company details": "",
                    "company type": "",
                    "personalized email body": "",
                    "personalized email subject": ""
                } for _ in range(5)]

                # Initialize the workflow state
                state: GraphState = {
                    "subgraph_messages": [HumanMessage(content=prompt)],
                    "next_agent": "supervisor",
                    "information_list": initial_list,
                    "long_term_summary": 'Start from logging in to the website.And ensuring that all the filters are applied and reflect on it.'
                }

                last_state = None
                # Stream workflow execution asynchronously
                async for cur_state in app.astream(state, {"recursion_limit": 1000}):
                    print("Execution in progress...\n")
                    last_state = cur_state  # store the last state


                return last_state  # Return final state after execution

            except Exception as e:
                print(f"Error running workflow: {e}")
                return None

    except Exception as e:
        print(f"Error starting Playwright MCP session: {e}")
        return None


# Load human instructions from JSON file
try:
    json_path = pathlib.Path(__file__).parent / "overall_executional_steps.json"
    human_instructions = json_path.read_text()
except Exception as e:
    print(f"Error reading human instructions: {e}")
    human_instructions = ""


if __name__ == "__main__":
    try:
        # Run the agent workflow with human instructions
        final_state = asyncio.run(run_agent(human_instructions))
        print("final state ->", final_state, "\n")

        # Extract final information list from supervisor
        final_values = final_state.get("supervisor", {}).get(
            "information_list", {}).get("information_list", [])
        print("final data ->", final_values, "\n")

        # Convert all values to strings for consistency
        for record in final_values:
            for key, value in record.items():
                record[key] = str(value) if value is not None else ""

        # Convert to pandas DataFrame
        df = pd.DataFrame(final_values)

        # Save the leads data to an Excel file
        excel_file = "leads_data.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"✅ Excel file '{excel_file}' created with all values as strings!")
    except Exception as e:
        print(f"Error running main async function: {e}")
