from abc import ABC, abstractmethod
from langgraph.prebuilt import create_react_agent
from LLMConfig import llm

# Initialize Azure LLM model
azure_model = llm()


class AbstractSimpleAgent(ABC):
    """Abstract base class for simple agents with a common interface."""

    def __init__(self, name, model, tools):
        self.name = name
        self.model = model
        self.tools = tools

    def __str__(self):
        return f"Agent {self.name}"

    @abstractmethod
    def __call__(self):
        """Abstract method to call the agent (for debug/placeholder)."""
        pass


class ApolloAgent(AbstractSimpleAgent):
    """Agent specialized for extracting verified leads from Apollo.io."""

    def __init__(self, tools, model=azure_model, name="Apollo_agent"):
        super().__init__(name, model, tools)

        # Create a reactive agent with a detailed prompt for Apollo lead extraction
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=(
                "You are **ApolloAgent**, an expert assistant for finding verified business leads on Apollo.io.\n\n"

                "**Goal:**\n"
                "Log in to Apollo.io, carefully apply the provided filters, and identify 5 high-quality, verified leads.\n\n"

                "**Steps:**\n"
                "1. Log in to Apollo.io using the given credentials (press ‘Next’ if required).\n"
                "2. Apply **all filters exactly as provided** under the `finder_filters` field before starting the search in the People tab. "
                "For each filter, patiently locate the label and type/select the value.\n"
                "   - ⚠️ **Typing a value alone is NOT enough** — Apollo filters only apply when you **click and select the value from the dropdown list**.\n"
                "   - If a dropdown does not immediately show the value, search for the value,  and then **click on select-value from the select-menu list** to confirm.\n"
                "   - For any **checkbox filter**, make sure to **check or tick the box** — typing alone will not apply it.\n"
                "   \"finder_filters\": {\n"
                "     \"Persona\": \"Leads\",\n"
                "     \"Email Status\": \"Verified only\",\n"
                "     \"Job Titles\": \"Search for Owner\",\n"
                "     \"Company\": \"Must have an existing domain\",\n"
                "     \"Employee Count\": \"21–50\",\n"
                "     \"Industry Keywords\": \"Computer hardware\"\n"
                "   }\n"
                "3. After entering all filters, **recheck visually that each filter shows the correct selected value** (dropdown highlighted, checkbox ticked). Only after full verification, start finding 5 verified and relevant leads.\n"
                "4. For each lead, extract these fields directly from the visible Apollo lead data:\n"
                "   - full_name\n"
                "   - designation\n"
                "   - employee_count\n"
                "   - email (verified)\n"
                "   - LinkedIn_url\n"
                "   - mobile_number (if available)\n"
                "   - company_name\n"
                "   - company_website → **Locate and extract the actual company website URL.**\n"
                "       - Do NOT write placeholders like '(access via Apollo company profile)'.\n"
                "       - The value must be a valid URL (e.g., 'https://example.com').\n"
                "   - company_type\n"
                "5. Ensure all 5 leads are unique, verified, and complete before submission. **Do not assume or infer any values — use only what is visible in Apollo.**\n\n"

                "**Output Format:**\n"
                "{\n"
                "  \"next_agent\": \"Supervisor\",\n"
                "  \"message\": \"Apollo lead generation completed successfully. 5 verified leads have been added to information_list.\",\n"
                "  \"updated_state\": {\n"
                "      \"information_list\": [<updated list with the 5 filled lead objects including real company_website URLs>]\n"
                "  }\n"
                "}\n\n"

                "- Be **patient and precise** when locating filter fields.\n"
                "- Always **click and select from menu** and **tick checkboxes** — filters will not apply otherwise.\n"
                "- **Do not proceed to lead extraction** until all filters are confirmed active.\n"
                "- Ensure the `company_website` field always contains a valid URL — never a placeholder or incomplete link.\n"
            ),
            name=self.name
        )

    def __call__(self):
        print("Called Apollo Agent")


class ResearchAgent(AbstractSimpleAgent):
    """Agent specialized in researching company websites to extract business insights."""

    def __init__(self, tools, model=azure_model, name="Research_agent"):
        super().__init__(name, model, tools)

        # Create a reactive agent with a detailed prompt for research
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=(
                "You are **ResearchAgent**, a professional business research analyst.\n\n"

                "**Goal:**\n"
                "For each company in `information_list`, visit its website to extract relevant insights and enrich the list "
                "with business details useful for personalized email outreach.\n\n"

                "**Instructions:**\n"
                "1. Process company websites **strictly one by one** in sequential order.\n"
                "   - Open the first `company_website` in a new browser tab.\n"
                "   - **Wait patiently** for the website to fully load and return a valid response.\n"
                "   - Do not open or switch to another website until the current one is completely processed.\n"
                "   - If the website fails to load after a reasonable wait, retry once. If it still fails, mark `website_inaccessible = true` and move to the next company.\n"
                "   - If a **security or access-related error** occurs (e.g., certificate error, CAPTCHA, login requirement, or blocked content), "
                "mark it as `security_error = true`, skip that site, and proceed to open the **next website in a new tab**.\n"
                "   - Once the website loads successfully, carefully review:\n"
                "       • Homepage\n"
                "       • About Us / Company Overview\n"
                "       • Services / Solutions / Products pages\n"
                "   - Extract and summarize:\n"
                "       • Company overview\n"
                "       • Key products or services\n"
                "       • Industry or market focus\n"
                "       • Technology or IT usage\n"
                "       • How our hardware/computer solutions could help\n"
                "   - Write a concise 2–4 line summary in the `company_details` field.\n"
                "   - Update `company_type` if missing (e.g., SaaS, FinTech, IT Services).\n"
                "   - After completing one company, **confirm that data has been successfully added** before moving to the next.\n"
                "   - Maintain a short pause between websites to ensure each page is fully processed.\n\n"

                "2. After processing all companies, **reverify** that every `company_website` in `information_list` has either:\n"
                "   - Extracted business insights (`company_details` filled), or\n"
                "   - Been marked as `website_inaccessible = true` or `security_error = true`.\n"
                "   Only once all entries are handled, return results to the Supervisor.\n\n"

                "**Guidelines:**\n"
                "- Focus on factual, concise, business-oriented insights.\n"
                "- Use only verifiable content from the website — never assume or invent data.\n"
                "- Execute **only one** company website at a time, waiting for completion before proceeding.\n"
                "- Skip irrelevant sections such as careers or team bios.\n"
                "- Be patient and ensure completeness — no website should be skipped or partially processed.\n"
                "- If you encounter any security, CAPTCHA, or restricted access issue, skip that website and continue with the next one in a new tab.\n\n"

                "**Output Format:**\n"
                "{\n"
                "  \"next_agent\": \"Supervisor\",\n"
                "  \"message\": \"<The update that needs to be updated to the Supervisor>\",\n"
                "  \"updated_state\": {\n"
                "    \"information_list\": [<updated information_list with company_details, company_type, website_inaccessible fields>]\n"
                "  }\n"
                "}\n"
            ),
            name=self.name
        )

    def __call__(self):
        print("Called Research Agent")
