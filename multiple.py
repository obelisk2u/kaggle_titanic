import autogen

config_list = [
    {
        'model':'gpt-4',
        'api_key': ''
    }
]

llm_config={
    "cache_seed": 42, 
    "config_list": config_list,
    "temperature": 0
}

user_proxy = autogen.UserProxyAgent(
    name="Bossman", 
    human_input_mode = "TERMINATE",
    system_message="""Interact with planner to approve the plan""",
    code_execution_config={
        "work_dir":"out"}
)

programmer = autogen.AssistantAgent(
    name="Programmer",
    llm_config=llm_config,
    system_message="""Follow approved plan and make sure to save all code you generate to disk. You can write python and shell code.""",
    code_execution_config={
        "work_dir":"out"
    }
)

planner = autogen.AssistantAgent(
    name = "Planner",
    system_message="""You suggest a plan and work with Bossman on whether the plan needs any revisions""",
    llm_config=llm_config
)

finalcut = autogen.AssistantAgent(
    name = "Critic",
    system_message="Double check the plan and code from other agents and provide feedback",
    llm_config=llm_config
)

group_chat = autogen.GroupChat(
    agents=[finalcut, planner, programmer, user_proxy],
    messages=[]
)

manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""Save to disk a python file that will complete the Kaggle Titanic Competition"""
)