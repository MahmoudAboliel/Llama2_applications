from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import Replicate
from langchain_community.tools import WikipediaQueryRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools
import wikipediaapi
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#===========================================SET API===========================================#
import dotenv
dotenv.load_dotenv()
#===========================================INIT MODEL===========================================#
llm = Replicate(
        model="meta/meta-llama-3-8b-instruct",
        model_kwargs={"temperature": 0.2, "max_length": 500, "top_p": 1},
        )

#===========================================INIT TOOLS===========================================#
#########################################################################
# Method One To Import Wikipedia Tool
# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
# wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
# wikipedia_tool.description = (
#             "Use this tool to search for detailed information about specific historical events, "
#             "scientific topics, or notable public figures. "
#             "Input should be a search query."
# )
#########################################################################
# Method Two To Import Wikipedia Tool
def search_wikipedia(query):
    user_agent = "nativebot/1.0 (mahmoudabulail1998@gmail.com)"

    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return "No results found"
wikipedia_tool = Tool(
    name="Wikipedia",
    func=search_wikipedia,
    description="Use this tool to search for detailed information about specific historical events, scientific topics, or notable public figures. Input should be a search query."
)
# print(wikipedia_tool.description)
#########################################################################
# Method Three To Import Wikipedia Tool
tools = load_tools(["wikipedia"], llm=llm)
#########################################################################
# Import tavily search engin Tool
tavily_tool = TavilySearchResults(max_results=1)
tavily_tool.description = ("A search engine optimized for comprehensive, accurate, and trusted results."
                           "Useful for when you need to answer questions about current events. Input should be a search query."
                           "Use this tool only when you do not find results on Wikipedia.")

# for one and two method
# tools = [wikipedia_tool, tavily_tool]

# for three method
tools.append(tavily_tool)
# print(wikipedia_tool.run("what is langchain"))
# print(tavily_tool.run("what is langchin"))
##########################################################################################################
###################################INIT PROMPT############################################################
template = """
You are Nemo, a helpful and respectful assistant designed to aid English learning at the {level} level 
If the user makes any mistakes in the language, rephrase his message correctly and then respond to it, But uppercase and lowercase letters or punctuation marks are not considered errors 
Your focus is on interactive, motivating conversations 
Keep responses concise and engaging, encouraging user interaction 
Aim to keep answers around 20 words whenever possible 


TOOLS:

------

Nemo has access to the following tools:

{tools}

If the user's message contains a question about updatable information, detailed information about specific historical events, scientific topics, or notable public figures. don't response directly but use the [{tool_names}] for search after that response
To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action, for wikipedia should be general title

Observation: the result of the action
    ...(Thought/Action/Observation can't never repeat just one time)
Thought: If the user makes any mistakes in the language, But uppercase and lowercase letters or punctuation marks are not considered mistakes
Action: rephrase his message correctly
Final Answer: [your response here], and then correct user message (if any)

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No
Thought: If the user makes any mistakes in the language, But uppercase and lowercase letters or punctuation marks are not considered mistakes 
Action: rephrase his message correctly
Action: Reply to user message
Final Answer: [your response here], and then the correct user message (if any)

```

Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
# prompt = hub.pull("hwchase17/react")
####################################################################################################
agent = create_react_agent(llm, tools, prompt)


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# print()
def agent(level, prompt, session_id):
    response = agent_with_chat_history.invoke(
                {"input": prompt, "level": level},
                {"configurable": {"session_id": session_id}})['output']
    return response

while True:
        prompt = input("User: ")
        response = agent_with_chat_history.invoke(
                {"input": prompt, "level": "beginner"},
                {"configurable": {"session_id": "<foo>"}},
        )['output']

        print(f"Bot: {response}")
