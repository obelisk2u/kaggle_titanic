import autogen

config_list = [
    {
        'model':'gpt-4',
        'api_key': ''
    }
]

llm_config={
    #"request_timeout": 600, 
    "seed": 42, 
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="assistant", 
    llm_config=llm_config,
    system_message= "My First Assistant",
    code_execution_config={"work_dir":"out"}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy", 
    human_input_mode = "TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    
    llm_config=llm_config,
    system_message="""reply terminate if solved, else reply continue"""
)

task = """
Solve the Kaggle Titanic competition using train.csv in my working directory as the training data, output the code used into a file called sub.py"""
user_proxy.initiate_chat(
    assistant,
    message=task
)