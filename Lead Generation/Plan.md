# Section 1
- Lead generation framework's comparision
1) Apollo.io -> large contact database + integrated outreach (multiple features-automation) + LangChain/LangGraph support
2) Hunter.io -> smaller contact database + fewer outreach features
3) Clearbit -> high quality dataset, but pricy

----------------------------------------------------------------------------------------------------------------------------------------------
# Section 2
- Basic structure of the approach
1) Search potential leads - companies  
2) Analyze/research about the potential leads via web-scraping
3) Generate the personalized outreach messages for the shortlisted leads

------------------------------------------------------------------------------------------------------------------------------------------------
# Section 2
- Technologies/Frameworks used
1) LangChain
2) LangGraph
3) Apollo
4) OpenAI
5) Playwright MCP

--------------------------------------------------------------------------------------------------------------------------------------------------
# Section 2
# Executional approach (Approach 1) - Agentic (modern)
1) Search potential leads - ReAct agent (agentic web-scraping)
2) Analyze/Research about potential leads - ReAct agent (agentic web-scraping)
3) Monitor and Correction/Re-execution - ReAct agent
4) Personalized outreach message - LLM call 
5) Store results in a Google Sheet - Python function

# Executional approach (Approach 2) - API call + Web-scraping (traditional)
1) Utilize API of a lead generation website - API call
2) Perform web scraping on the website links - Web-scraping
3) Personalized outreach message - LLM call
4) Store results in a Google Sheet - Python function

# We are going to follow approach 1 -> show my capabilities while achieving the goal of assignment.

Step 1 - START
Step 2 - Call supervisor to start execution
Step 3 - Apollo agent -> browse Apollo -> apply filters and search leads -> select 5 leads -> update the state with the required information
Step 4 - Supervisor analyze the results and take decision to move forward or re-execute Apollo agent
Step 5 - Research agent -> browse selected company websites sequentially -> analyze the website content -> summarize the content based on our context (use-case) 
Step 6 - Supervisor analyze the results and take decision to move forward or re-execute Research agent
Step 7 - Generate Personalized outreach message 
Step 8 - Store results in an Excel sheet
Step 10 - END

----------------------------------------------------------------------------------------------------------------------------------------------------
# Section 2
# Output format (state)

[...,{
    "name": "",
    "designation": "",
    "employee count": "",
    "email": "",
    "linkedin_link": "",
    "mobile number": "",
    "company name": "",
    "company website link": "",
    "company details": "",
    "company type": "",
    "personalized email body": "",
    "personalized email subject": ""
},...]

# Section 3
# Web scraping
- Implementing agentic web scraping
- Enable to monitor as it happens
- Agent will refer the HTML DOM structure

# Personalized email generation
- Utilize OpenAI
- Sync between prompt generation and email formation
- Better performance - based on my individual experience

---------------------------------------------------------------------------------------------------------------------------------------------------
# Section 4
# Edge cases
- Since agentic, if stuck the agent will figure out the resolution steps
- Implemented Supervisor will monitor each agent's results at each step
- If web scraping is blocked, the agent can still go through HTML DOM structure and extract the relevant details
- AI agent's prompt can be strictified not to add make-up data
- Research Agent will verify the scraped information, can be prompted to have a boolean flag `isAuthentic`
- LangGraph offers human-in-loop (HIL), human intervention can be implemented to verify or give a go-ahead
- If execution stops in between, LangGraph offers time-travel, that stores and resumes the execution from where it stopped

---------------------------------------------------------------------------------------------------------------------------------------------------

# Files summary
- overall_executional_steps.json - contains human instructions for the overall execution - filters, keywords etc
- graphStructure.txt - contains the graph structure
- workflow.txt - contains the flow diagram of the overall project
- main.py - contains the main executable code
- role_agents - contains the prompts of the Apollo and Research agent (ReAct agents)
- role_functions - contains the email generation code (LLM call)
- mcp_utils - contains the code to connect with Playwright MCP
