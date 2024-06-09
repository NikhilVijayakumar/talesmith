# src/main.py
from src.usecase.story_usecase import generate_story
from src.configuration.environment import Environment
from src.configuration.config import Config


def main():
    Config.load_environment(Environment.XFORGE_OPENCHAT)
    generate_story()


main()