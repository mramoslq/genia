import getpass
import os
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set up environment variables first
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Initialize the LLM with your desired parameters
llm = ChatGroq(
    model="llama3-8b-8192",  # Updated to correct model name
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 1. Define your tool using the @tool decorator
@tool
def get_population(location: str) -> str:
    """Get the current population in a given location."""
    if location.lower() == "madrid, spain":
        return "The population of Madrid, Spain is approximately 3.3 million."
    return f"Sorry, I don't have population data for {location}."

# 2. Register your tools
tools = [get_population]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 3. Create an agent with the registered tools
agent = create_tool_calling_agent(llm, tools, prompt=prompt)

# 4. Create the executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Invoke the agent with a query
response = agent_executor.invoke({"input": "What is the population of Barcelona, Spain?"})
print(response["output"])