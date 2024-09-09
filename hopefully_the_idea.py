import autogen
import tempfile
import shutil
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

config_list = [
    {
        'model':'gpt-4',
        'api_key': ''
    }
]

llm_config={
    "seed": 45, 
    "config_list": config_list,
    "temperature": 0
}

executor = LocalCommandLineCodeExecutor(
    timeout=100,  # Timeout for each code execution in seconds.
    work_dir='your_dir',  # Use the temporary directory to store the code files.
)

code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor},  # Use the local command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
)

programmer = autogen.AssistantAgent(
    name="programmer", 
    llm_config=llm_config,
    system_message= "You can write python and shell code, make sure to save all code to disk",
    code_execution_config={"work_dir":"your_dir"}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy", 
    human_input_mode = "TERMINATE",
    max_consecutive_auto_reply=20,
    is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    llm_config=llm_config,
    system_message="""reply terminate if solved, else reply continue"""
)

task = """Complete the Kaggle Titanic Competition"""

group_chat = autogen.GroupChat(
    agents=[code_executor_agent, programmer, user_proxy],
    messages=[]
)

manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""Complete the Kaggle Titanic Competition"""
)