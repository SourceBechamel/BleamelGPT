import typing
from typing import Iterable

import pydantic
from langchain import schema, prompts
from langchain.prompts import chat
from langchain.schema import runnable
from langchain_core import language_models, chat_history


class ChatChainInput(typing.TypedDict):
    history: typing.List[schema.BaseMessage]
    question: str


class Chatbot(pydantic.BaseModel, arbitrary_types_allowed=True):
    chain: runnable.Runnable[ChatChainInput, str]
    history: chat_history.InMemoryChatMessageHistory

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
    def from_chat_model(cls,
                        chat_model: language_models.BaseChatModel,
                        system_message: str | schema.SystemMessage | prompts.SystemMessagePromptTemplate):
        template = chat.ChatPromptTemplate.from_messages([
            schema.SystemMessage(content=system_message) if isinstance(system_message, str) else system_message,
            chat.MessagesPlaceholder('history'),
            chat.HumanMessagePromptTemplate.from_template('{question}')
        ])

        history = chat_history.InMemoryChatMessageHistory()

        return cls(
            chain=template | chat_model | schema.StrOutputParser(),
            history=history
        )
