import abc
import typing

from langchain import schema


class ChatbotProtocol(typing.Protocol):
    def invoke(self, question: str) -> str:
        """
        Invokes the chatbot to answer the given question.

        Args:
            question: Question to answer.

        Returns:
            The chatbots response to the question.
        """
        pass

    def stream(self, question: str) -> typing.Iterable[str]:
        """
        Invokes the chatbot to stream the answer for the given question.

        Args:
            question: Question to answer.

        Returns:
            An iterator for the chatbots response to the question.
        """
        pass

    @property
    @abc.abstractmethod
    def messages(self) -> typing.List[schema.BaseMessage]:
        """
        Returns the conversation history sofar
        """
        pass

    def clear_messages(self) -> None:
        """
        Clears the conversation history.
        """
        pass
