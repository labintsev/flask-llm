from flask_sqlalchemy import SQLAlchemy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import dotenv

env = dotenv.dotenv_values(".env")

db = SQLAlchemy()


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.Text, nullable=False)
    llm_reply = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())


def dummy_llm_service(user_message):
    return f"Вы сказали: {user_message}, но я пока не могу ответить на это."


class LLMService:
    def __init__(self, sys_prompt):
        try:
            # Создаем клиент с вашим токеном
            self.client = GigaChat(
                credentials=env['GIGA_KEY'], 
                model='GigaChat-2', 
                verify_ssl_certs=False)
            self.sys_prompt = sys_prompt
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

    def chat(self, message):
        try:
            # Обращаемся к API
            messages = [
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=message)
                ]
            # Запрос к модели
            response = self.client.invoke(messages)

            # Возвращаем ответ
            return response.content

        except Exception as e:
            return f"Произошла ошибка: {str(e)}"
