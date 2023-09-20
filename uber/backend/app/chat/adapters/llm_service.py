from langchain import (
    LLMMathChain,
    SerpAPIWrapper,
    GoogleSerperAPIWrapper,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

now = datetime.now()

temperature = 0

prev_response = ""


class LLMService:
    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.temperature = temperature
        self.llm = ChatOpenAI(
            temperature=temperature,
            model="gpt-3.5-turbo-0613",
            openai_api_key=OPENAI_API_KEY,
        )
        self.search = GoogleSerperAPIWrapper()
        self.search_photo = GoogleSerperAPIWrapper(type="images")

        self.llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
        self.prev_response = prev_response
        self.tools = [
            Tool(
                name="Search",
                func=self.search.run,
                description="useful when you need to get an information about your exact properties, events, papers, news and other information you do not have. For example, if you are being asked for your radius use this tool.  Another example: if you are being asked for an astronomical event, use this tool. Another example: if you are being asked a paper use this tool. Note that you may use the current date in the search",
            ),
            Tool(
                name="Photo",
                func=self.search_photo.results,
                description="useful when you need to search images of astronomical objects.",
            ),
            Tool(
                name="Calculator",
                func=self.llm_math_chain.run,
                description="useful when need to perform calculations with numbers that are not represented by symbols. DO NOT perform 'radius' * 2 operation, becuase 'radius' is not a number but rather a word. Instead in such case look up for the radius value using search tool yourslef and calculate the numbers. For example: 17 * 10 ",
            ),
        ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

    def tst(self):
        data = self.search_photo.results("sun")
        logging.info(f"server length {data}")

    def chat_with_object(self, prompt, space_object, context, history):
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S

        final_prompt = f"""
            You are a celestial object in triple backticks below.

            '''Name: {space_object}

            Constellation: {context}'''

            We are having a conversation about space, astronomy, astronomical calculations and your physical properties as a '''{space_object}'''.

            Answer as if you were a {space_object}.

            Note that today is: {dt_string}

            In case the conversation topic became related to the astronomical phenomenas, suggest to send a photo of you to me.
            If you will invoke the 'Photo' tool add 'Astronomy' prefix before the thing you want to get a photo of. For example: 'Astronomy {space_object}'
            
            Our previous conversation is below in triple backticks:

            '''
            {history}
            '''

            Do not include <{space_object}> in the beginning of your answer. 
            
            Continue the conversation.

            Our current conversation is below in triple backticks:

            '''
            <Me> {prompt}

            <{space_object}> 
            '''
            
            """

        final_answer: str = self.agent.run(input=final_prompt)

        return final_answer
