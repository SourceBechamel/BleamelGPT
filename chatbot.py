import typing
from typing import Iterable

import pydantic
from langchain import memory, schema
from langchain.prompts import chat
from langchain.schema import runnable
from langchain_core import language_models


class ChatChainInput(typing.TypedDict):
    history: typing.List[schema.BaseMessage]
    question: str


class Chatbot(pydantic.v1.BaseModel):
    chain: runnable.RunnableSequence[ChatChainInput, str]
    history: memory.ChatMessageHistory

    @property
    def messages(self) -> typing.List[schema.BaseMessage]:
        return self.history.messages

    def clear_messages(self):
        self.history.clear()

    def invoke(self, question: str) -> str:
        answer = self.chain.invoke({'question': question,
                                    'history': self.history.messages})
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        return answer

    def stream(self, question: str) -> Iterable[str]:
        tokens = []
        for token in self.chain.stream({'question': question,
                                        'history': self.history.messages}):
            yield token
            tokens.append(token)
        answer = ''.join(tokens)
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)

    @classmethod
    def from_chat_model(cls, chat_model: language_models.BaseChatModel,
                        history: memory.ChatMessageHistory | None = None):
        template = chat.ChatPromptTemplate.from_messages([
            chat.MessagesPlaceholder('history'),
            chat.HumanMessagePromptTemplate.from_template('{question}')
        ])
        return cls(
            chain=template | chat_model | schema.StrOutputParser(),
            history=history or memory.ChatMessageHistory()
        )
