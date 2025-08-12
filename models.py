from flask_sqlalchemy import SQLAlchemy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import dotenv
import logging

# Настройка логгирования
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из файла .env
try:
    env = dotenv.dotenv_values(".env")
    API_KEY = env["GIGA_KEY"]
except FileNotFoundError:
    raise FileNotFoundError("Файл .env не найден. Убедитесь, что он существует в корневой директории проекта.")
except KeyError as e:
    raise KeyError(f"Переменная окружения {str(e)} не найдена в файле .env. Проверьте его содержимое.")


# Инициализируем SQLAlchemy для работы с базой данных через Flask
db = SQLAlchemy()


class ChatHistory(db.Model):
    """
    Модель SQLAlchemy для хранения истории общения пользователя с LLM.

    Атрибуты:
        id (int): Уникальный идентификатор записи.
        user_message (str): Сообщение пользователя.
        llm_reply (str): Ответ языковой модели.
        timestamp (datetime): Время создания записи.
    """
    id = db.Column(db.Integer, primary_key=True)        # Уникальный идентификатор
    user_message = db.Column(db.Text, nullable=False)   # Текст сообщения пользователя
    llm_reply = db.Column(db.Text, nullable=False)      # Ответ модели
    timestamp = db.Column(db.DateTime, server_default=db.func.now())  # Временная метка создания записи


class LLMService:
    """
    Класс для взаимодействия с внешней языковой моделью (например, YandexGPT).

    Атрибуты:
        sys_prompt (str): Системный промпт для LLM.
        client: Клиент OpenAI для обращения к API Yandex.
        model (str): Идентификатор используемой LLM модели.
    """
    def __init__(self, prompt_file):
        """
        Инициализация сервиса LLM.

        Аргументы:
            prompt_file (str): Путь к файлу с системным промптом для LLM.
        """
        # Читаем системный промпт из файла и сохраняем в атрибут sys_prompt
        with open(prompt_file, encoding='utf-8') as f:
            self.sys_prompt = f.read()
                
        try:
            # Создаем клиент с вашим токеном
            self.client = GigaChat(
                credentials=API_KEY, 
                model='GigaChat-2', 
                verify_ssl_certs=False)
        except Exception as e:
            logger.error(f"Ошибка при авторизации модели. Проверьте настройки аккаунта и область действия ключа API. {str(e)}")

    def chat(self, message):
        """
        Отправляет сообщение к языковой модели и возвращает её ответ.

        Аргументы:
            message (str): Сообщение пользователя.

        Возвращает:
            str: Ответ языковой модели или сообщение об ошибке.
        """
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
            # В случае ошибки возвращаем её описание
            logger.error(f"Произошла ошибка: {str(e)}")
            return f"Произошла ошибка: {str(e)}"


llm_1 = LLMService('prompts/prompt_1.txt')


def chat_with_llm(user_message):
    """
    Чат с использованием сервиса LLM.

    Аргументы:
        user_message (str): Сообщение пользователя.

    Возвращает:
        str: Ответ LLM.
    """
    response = llm_1.chat(user_message)
    return response
