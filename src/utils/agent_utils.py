from crewai import Agent, Task
from langchain_community.llms import LlamaCpp
import yaml
import os


def create_agent_from_yaml(agent_yaml_file: str, llm: LlamaCpp):
    with open(agent_yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    agent = Agent(
        role=config['agent']['role'],
        goal=config['agent']['goal'],
        backstory=config['agent']['backstory'],
        allow_delegation=config['agent']['allow_delegation'],
        verbose=config['agent']['verbose'],
        llm=llm
    )

    return agent


def create_task_from_yaml(task_yaml_file: str, agent: Agent, context: str):
    with open(task_yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    task = Task(
        name=config['task']['name'],
        description=config['task']['description'],
        expected_output=config['task']['expected_output'],
        agent=agent,
    )

    return task


def get_context(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    return "Generate a story"  # Default prompt if the file is not found
