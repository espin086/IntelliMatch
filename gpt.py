import logging
import openai
from openai import OpenAI
import streamlit as st

"""
This is a GPT text generation module. It is used to generate text based on a prompt.
"""

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = st.secrets["openai"]["key"]


class GPTModelHandler:
    """Handles GPT Model Operations."""

    def __init__(
        self,
        api_key: str = API_KEY,
        model: str = "gpt-4o",
        prompt: str = "This is a test",
    ) -> None:
        """
        Initialize the GPTModelHandler.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The GPT model to use. Default is 'gpt-3.5-turbo'.
        """

        assert isinstance(api_key, str), "api_key must be a string."
        assert isinstance(model, str), "model must be a string."
        assert isinstance(prompt, str), "prompt must be a string."
        


        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.prompt = prompt

    def generate_response(self, system_role:str="system"):
        """
        Generate response from GPT model.

        Args:
            prompt (str): The prompt for the GPT model.
            system_role (str): The role for the prompt. Default is 'system'.

        Returns:
            str: The generated response.
        """

        assert isinstance(system_role, str), "system_role must be a string."    

        messages = [{"role": system_role, "content": self.prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        logging.info("Generated GPT response")
        return response.choices[0].message.content


class PromptOptimizer(GPTModelHandler):
    """This optimizes promps that come in"""

    def __init__(self, prompt:str="this is a test") -> None:
        super().__init__()

        assert isinstance(prompt, str), "prompt must be a string."

        self.prompt = f""" You are a world class prompt engineer. Please improve and enhance this prompt: {prompt}.   Guidelines:
        - Use your analytics prowness and creative imagination to return a stellar prompt for me.
        - Output should not include any explanation on the changes, just the revied prompts"""
        logging.info("Updated prompt for optimization")

    def response(self) -> str:
        """Generate response for the prompt optimizer."""
        return self.generate_response(self.prompt)


class TOCD(GPTModelHandler):
    """Template: Task, Output, Context, Data"""

    def __init__(self, task:str, output:str, context:str, data:str) -> None:
        """Initializing Class"""
        super().__init__()

        assert isinstance(task, str), "task must be a string."
        assert isinstance(output, str), "output must be a string."
        assert isinstance(context, str), "context must be a string."
        assert isinstance(data, str), "data must be a string."

        self.task = task
        self.output = output
        self.context = context
        self.data = data

        self.prompt = f""" 
        
        Complete this Task:{self.task}.

        Format for Output: {self.output}

        To Ensure Relevance Remember: {self.context}

        Here is Data you Need for Your Response: {self.data}
        """
        logging.info(f"instance of TOCD class created using model: {self.model}")

    def response(self) -> str:
        """Generate response for the TOCD template."""
        return self.generate_response(self.prompt)


class RTAO(GPTModelHandler):
    """Role, Task, Audience, Output"""

    def __init__(self, role:str, task:str, audience:str, output:str) -> None:
        super().__init__()

        assert isinstance(role, str), "role must be a string."
        assert isinstance(task, str), "task must be a string."
        assert isinstance(audience, str), "audience must be a string."
        assert isinstance(output, str), "output must be a string."

        self.role = role
        self.task = task
        self.audience = audience
        self.output = output

        self.prompt = f"""

        Act as a {self.role}.

        Accomplish this task: {task}.

        The target audience/demographic for your response is: {self.audience}.

        Your response should confirm to this output: {self.output}
        """

    def response(self) -> str:
        """Generate response for the RTAO template."""
        return self.generate_response(self.prompt)


class Ultimate(GPTModelHandler):
    "The ultimate prompt template"

    def __init__(self, role, behavior, task, structure, constraints, data) -> None:
        super().__init__()

        assert isinstance(role, str), "role must be a string."
        assert isinstance(behavior, str), "behavior must be a string."
        assert isinstance(task, str), "task must be a string."
        assert isinstance(structure, str), "structure must be a string."
        assert isinstance(constraints, str), "constraints must be a string."
        assert isinstance(data, str), "data must be a string."


        self.role = role
        self.task = task
        self.behavior = behavior
        self.structure = structure
        self.constraints = constraints
        self.data = data
        self.prompt = f"""

        Act as a {self.role}. Your key traits are {self.behavior}. 

        Help me with this task: {self.task}.

        Your response/output should be formatted like this: {self.structure}.

        Note: here are the relevant constaints: {self.constraints}

        You can reference this data to help you with your response: {self.data}
        """

    def response(self) -> str:
        """Generate response for the Ultimate template."""
        return self.generate_response(self.prompt)


if __name__ == "__main__":
    TASK = (
        "give me ideas on how to write python to speed up my work as a data scientist"
    )
    STRUCTURE = "In the output use citatios from Minto Pyramid in your response"
    OUTPUT = "use minto pyramid to respond, write it like a mckinsey consultant, Make the output in markdown"
    CONTEXT = "you are an independent contractor trying to get more business"
    DATA = "You made $300K last year you want to make $600 this year"
    AUDIENCE = (
        "Your auddience is colleage students at UCS in the computer science department"
    )
    ROLE = "you are a very successful start up founder who does hands on coding"
    BEHAVIOR = "You have a coaching program for young programmers"
    CONSTRAINTS = (
        "you can't use any cloud technology, all must be locally written and deployed"
    )

    oTOCD = TOCD(
        task=TASK,
        output=OUTPUT,
        context=CONTEXT,
        data=DATA,
    )
    # logging.info(oTOCD.response())

    oRTAO = RTAO(role=ROLE, task=TASK, audience=AUDIENCE, output=OUTPUT)
    # logging.info(oRTAO.response())

    oUltimate = Ultimate(
        role=ROLE,
        behavior=BEHAVIOR,
        task=TASK,
        structure=STRUCTURE,
        constraints=CONSTRAINTS,
        data=DATA,
    )
    print(oUltimate.response())

    # oPromptOptimizer = PromptOptimizer(
    #     "Please write code to pull video game data in Python"
    # )

    # print(oPromptOptimizer.response())
