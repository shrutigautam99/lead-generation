from langchain_core.runnables import RunnableLambda
from LLMConfig import llm
from langchain_core.messages import AIMessage, HumanMessage
import json
import re


def email_generator() -> RunnableLambda:
    """Creates a RunnableLambda node for generating personalized outreach emails."""
    
    # Initialize the LLM model
    azure_model = llm()

    def extract_json(text: str) -> str:
        """Extract JSON content from a string using regex."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return match.group(0) if match else text
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return text

    async def run_agent(state):
        """Async function to generate personalized emails for each company."""
        
        # Extract current workflow state
        try:
            messages = state.get("subgraph_messages", [])
            information_list = state.get("information_list", [])
        except Exception as e:
            print(f"Error reading state in agent_node: {e}")
            return {"subgraph_messages": [], "next_agent": "supervisor", "information_list": []}

        # Prepare the prompt for the email generation model
        prompt = f"""
            You are **EmailAgent**, an AI sales outreach specialist representing a **hardware computer store**.

            **Use Case:**
            As the founder of a hardware computer store, your goal is to send them compelling, personalized outreach emails showing how our products and solutions can help their business perform better, scale faster, and stay ahead technologically.

            **Goal:**
            For each company in `information_list`, generate a personalized, professional B2B outreach email that highlights how our hardware expertise can help them achieve higher efficiency and reliability.

            **Instructions:**
            - Skip companies where `company_details` is missing or `website_inaccessible` is true.
            - Carefully read `company_details` to understand the company’s domain, pain points, and context before drafting the email.
            - Tailor each email to reflect how our hardware solutions or computer systems can **help their business rise higher** (e.g., improved performance, better scalability, or reliability).
            - Keep each email **concise (120–150 words)**, **persuasive**, and **goal-driven**.
            - Maintain a **professional yet approachable tone**.
            - End with a **clear call-to-action**, such as scheduling a short call, demo, or free consultation.
            - Do **not fabricate** details or make unrealistic promises.

            **Output Requirements:**
            For each valid company, add or update:
            - "personalized_email_subject": "<short, compelling subject line>"
            - "personalized_email_body": "<customized, persuasive outreach email>"

            All other fields must remain unchanged.

            **Output Format:**
            {{
            "next_agent": "Supervisor",
            "message": "Personalized outreach emails have been generated successfully.",
            "updated_state": {{
                "information_list": [<updated information_list attributes with the values>]
            }}
            }}
            """

        try:
            # Call the LLM with the prepared messages and prompt
            response = await azure_model.ainvoke(messages + [HumanMessage(content=prompt)])

            # Default fallback for updated information_list
            updated_info_list = information_list

            try:
                # Extract JSON from response and parse
                content = extract_json(response.content)
                parsed = json.loads(content)
                next_agent = parsed.get("next_agent", "Supervisor")
                assistant_msg = parsed.get("message", "")
                updated_info_list = parsed.get("updated_state", {}).get(
                    "information_list", information_list)
            except Exception as e:
                print(f"Error parsing updated_state from model output: {e}")

            # Keep last AI message in the message history
            last_response = AIMessage(content=response.content)
            updated_messages = messages + [last_response]
            updated_messages = updated_messages[-10:]  # keep only the last 10 messages for context

            # Return the updated workflow state
            return {
                "subgraph_messages": updated_messages,
                "next_agent": next_agent,
                "information_list": updated_info_list
            }

        except Exception as e:
            print(f"Error running email_generator: {e}")
            # Return fallback state in case of failure
            return {
                "subgraph_messages": messages,
                "next_agent": "supervisor",
                "information_list": state.get("information_list", [])
            }

    # Return a RunnableLambda wrapping the async agent function
    return RunnableLambda(run_agent)
