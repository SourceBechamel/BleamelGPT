import langchain_ollama

import chatbot


class FlowerChatbotFactory:

    @staticmethod
    def create_basic_chatbot():
        system_message = 'You are a helpful chatbot with an expert knowledge of flowers and plants'
        chat_model = langchain_ollama.ChatOllama(model='llama3.1')
        return chatbot.Chatbot.from_chat_model(
            chat_model=chat_model,
            system_message=system_message
        )
