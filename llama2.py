from langchain_community.llms import Replicate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import Tool

import dotenv
from sqlalchemy.sql.functions import session_user

dotenv.load_dotenv()

class Llama2Chat:
    def __init__(self):
        self.model = Replicate(
            model="meta/meta-llama-3-8b-instruct",
            model_kwargs={"temperature": 0.2, "max_length": 500, "top_p": 1}
        )
        self.system_prompt = "You are Nemo, a helpful and respectful assistant designed to aid English learning at the {level} level. If the user makes any mistakes in the language, rephrase his message correctly and then respond to it. Your focus is on interactive, motivating conversations. Keep responses concise and engaging, encouraging user interaction. Aim to keep answers around 20 words whenever possible."
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ]
        )
        self.chain = self.prompt | self.model
        # self.chain = LLMChain(llm=self.model, prompt=self.prompt)
        self.chat_history = ChatMessageHistory()
        self.chain_with_message_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def generate_response(self, level, input_text, session_id="001"):
        response = self.chain_with_message_history.invoke(
            {"level": level, "input": input_text},
            {"configurable": {"session_id": session_id}}
        )
        return response



def llama2_tool():
    llama2 = Llama2Chat()
    return Tool(name="llama2_tool", func=llama2.chain.run, description="Generate responses for general messages.")


level = input("select level: ")
llm = Llama2Chat()
while True:
    prompt = input("User: ")
    response = llm.generate_response(level=level, input_text=prompt, session_id="001")
    print(f"Bot: {response}")
