import pytest
from langchain_community import chat_models

from bleamel import chatbots


@pytest.fixture
def system_message() -> str:
    return 'You are an expert when it comes to flowers.'


@pytest.fixture
def question() -> str:
    return 'How much water do roses need?'


@pytest.fixture
def answer() -> str:
    return 'Like a lot'


@pytest.fixture
def fake_llm(answer):
    return chat_models.FakeListChatModel(
        responses=[
            answer
        ]
    )


class TestChatbot:

    def test_invoke(self, system_message, question, answer, fake_llm):
        sut = chatbots.Chatbot.from_chat_model(chat_model=fake_llm, system_message=system_message)
        response = sut.invoke(question)
        assert response == answer

    def test_stream(self, system_message, question, answer, fake_llm):
        sut = chatbots.Chatbot.from_chat_model(chat_model=fake_llm, system_message=system_message)
        chunks = list(sut.invoke(question))
        assert len(chunks) > 1
        assert ''.join(chunks) == answer

    def test_update_history__invoke(self, system_message, question, answer, fake_llm):
        sut = chatbots.Chatbot.from_chat_model(chat_model=fake_llm, system_message=system_message)
        sut.invoke(question)
        messages = sut.messages
        assert len(messages) == 2
        assert messages[0].content == question
        assert messages[1].content == answer

    def test_update_history__stream(self, system_message, question, answer, fake_llm):
        sut = chatbots.Chatbot.from_chat_model(chat_model=fake_llm, system_message=system_message)
        list(sut.stream(question))
        messages = sut.messages
        assert len(messages) == 2
        assert messages[0].content == question
        assert messages[1].content == answer
