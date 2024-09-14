from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Generic, List, TYPE_CHECKING, TypeVar
from uuid import UUID

from flask_login import UserMixin

if TYPE_CHECKING:
    from .database import Database


T = TypeVar('T')


class DatabaseObject(Generic[T]):
    def __init__(self) -> None:
        super().__init__()

        self.id: UUID

    def to_dict(self) -> Dict[str, Any]:
        dict_ = self.__dict__.copy()

        # Ignore keys marked as priv
        for key in self.__dict__:
            if key.startswith("_"):
                dict_.pop(key)

        return dict_


class Conversation(DatabaseObject["Conversation"]):
    def __init__(self, data: Dict[str, Any], db: Database) -> Conversation:
        super().__init__()

        self._db = db

        self.id = data["conversation_id"]
        self.name: str = data["conversation_name"]
        self.user_id: UUID = data["user_id"]
        self.bot_user_id: UUID = data["bot_user_id"]
        self.created_at: datetime = data["created_at"]

    def get_messages(self, before: datetime, after: datetime, limit = 100) -> List[Message]:
        return self._db.get_conversation_messages(self.id, before, after, limit=limit)

    def get_user(self) -> User | None:
        return self._db.get_conversation_user(self.id)

    def get_bot_user(self) -> User | None:
        return self._db.get_conversation_bot_user(self.id)

    def send_message(self, sent_by_bot: bool, content: str) -> Message:
        return self._db.send_message(
            self.bot_user_id if sent_by_bot else self.user_id,
            self.id,
            content
        )


class Message(DatabaseObject["Message"]):
    def __init__(self, data: Dict[str, Any], db: Database) -> None:
        super().__init__()

        self._db = db

        self.id = data["message_id"]
        self.user_id: UUID = data["user_id"]
        self.conversation_id: UUID = data["conversation_id"]
        self.content: str = data["content"]
        self.created_at: datetime = data["created_at"]

    def get_user(self) -> User | None:
        return self._db.get_user(self.user_id)

    def get_conversation(self) -> Conversation | None:
        return self._db.get_conversation(self.conversation_id)


class BotUser(DatabaseObject["BotUser"]):
    def __init__(self, data: Dict[str, Any], db: Database) -> None:
        super().__init__()

        self._db = db

        self.id = data["user_id"]
        self.creator_id = data["creator_id"]
        self.last_updated: str = data["last_updated"]
        self.text_gen_model_name: str = data["text_gen_model_name"]
        self.text_gen_starting_context: str = data["text_gen_starting_context"]
        self.tts_model_name: str = data["tts_model_name"]
        self.tts_speaker_name: str = data["tts_speaker_name"]

    def get_user(self) -> User | None:
        return self._db.get_user(self.id)


class User(DatabaseObject["User"], UserMixin):
    def __init__(self, data: Dict[str, Any], db: Database) -> None:
        super().__init__()

        self._db = db

        self.id = data["user_id"]
        self.username: str = data["username"]
        self.is_bot: bool = data["is_bot"]
        self.created_at: datetime = data["created_at"]

    def get_conversations(self, limit = 100) -> List[Conversation]:
        return self._db.get_user_conversations(self.id, limit=limit)

    def get_user_credentials(self) -> UserCredential | None:
        return self._db.get_user_credentials(self.id)

    def get_bot_user(self) -> BotUser:
        return self._db.get_bot_user(self.id)


# Does not inherit from DatabaseObject so as to make accidental sending of its data harder
class UserCredential:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.id = data["user_id"]
        self.password = data["password"]
