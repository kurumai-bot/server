from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Callable, List, ParamSpec, TypeVar
from uuid import UUID, uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from utils import Cache
from .models import BotUser, Conversation, Message, User, UserCredential


# TODO: update queries to not be broken with the db changes
get_user_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s;"
).format(
    sql.Identifier("usr"),
    sql.Identifier("user_id")
)

get_user_from_username_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s"
).format(
    sql.Identifier("usr"),
    sql.Identifier("username")
)

get_user_conversations_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s LIMIT %s;"
).format(
    sql.Identifier("conversation"),
    sql.Identifier("user_id")
)

get_user_credentials_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s"
).format(
    sql.Identifier("user_credential"),
    sql.Identifier("user_id")
)

get_bot_user_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s;"
).format(
    sql.Identifier("bot_user"),
    sql.Identifier("user_id")
)

create_bot_user_query = sql.SQL(
    "INSERT INTO {} ({}, {}, {}, {}, {}, {}, {}) VALUES (%s, %s, %s, %s, %s, %s, %s)"
).format(
    sql.Identifier("bot_user"),
    sql.Identifier("user_id"),
    sql.Identifier("creator_id"),
    sql.Identifier("last_modified"),
    sql.Identifier("text_gen_model_name"),
    sql.Identifier("text_gen_starting_context"),
    sql.Identifier("tts_model_name"),
    sql.Identifier("tts_speaker_name"),
)

create_user_query = sql.SQL(
    "INSERT INTO {} ({}, {}, {}, {}) VALUES (%s, %s, %s, %s)"
).format(
    sql.Identifier("usr"),
    sql.Identifier("user_id"),
    sql.Identifier("username"),
    sql.Identifier("is_bot"),
    sql.Identifier("created_at")
)

create_user_credentials_query = sql.SQL(
    "INSERT INTO {} ({}, {}) VALUES (%s, %s)"
).format(
    sql.Identifier("user_credential"),
    sql.Identifier("user_id"),
    sql.Identifier("password"),
)

get_conversation_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s;"
).format(
    sql.Identifier("conversation"),
    sql.Identifier("conversation_id")
)

get_conversation_messages_query = sql.SQL(
    "SELECT * FROM {0} WHERE {1}=%s AND {2}<=%s AND {2}>=%s ORDER BY {2} DESC LIMIT %s;"
).format(
    sql.Identifier("message"),
    sql.Identifier("conversation_id"),
    sql.Identifier("created_at")
)

get_conversation_user_query = sql.SQL(
    "SELECT {1}.* FROM {0} INNER JOIN {1} ON {0}.{2}={1}.{2} WHERE {0}.{3}=%s;"
).format(
    sql.Identifier("conversation"),
    sql.Identifier("usr"),
    sql.Identifier("user_id"),
    sql.Identifier("conversation_id")
)

get_conversation_bot_user_query = sql.SQL(
    "SELECT {1}.* FROM {0} INNER JOIN {1} ON {0}.{2}={1}.{3} WHERE {0}.{4}=%s;"
).format(
    sql.Identifier("conversation"),
    sql.Identifier("usr"),
    sql.Identifier("bot_user_id"),
    sql.Identifier("user_id"),
    sql.Identifier("conversation_id")
)

create_conversation_query = sql.SQL(
    "INSERT INTO {} ({}, {}, {}, {}, {}) VALUES (%s, %s, %s, %s, %s)"
).format(
    sql.Identifier("conversation"),
    sql.Identifier("conversation_id"),
    sql.Identifier("user_id"),
    sql.Identifier("bot_user_id"),
    sql.Identifier("conversation_name"),
    sql.Identifier("created_at")
)

get_message_query = sql.SQL(
    "SELECT * FROM {} WHERE {}=%s;"
).format(
    sql.Identifier("message"),
    sql.Identifier("message_id")
)

send_message_query = sql.SQL(
    "INSERT INTO {} ({}, {}, {}, {}, {}) VALUES (%s, %s, %s, %s, %s)"
).format(
    sql.Identifier("message"),
    sql.Identifier("message_id"),
    sql.Identifier("user_id"),
    sql.Identifier("conversation_id"),
    sql.Identifier("content"),
    sql.Identifier("created_at")
)


RT = TypeVar("RT")
P = ParamSpec("P")
GetCacheFunc = Callable[["Database", UUID], RT]


# A decorator that adds the return object of a function to the cache of the db
def _get_cache(ttl: timedelta = None):
    def get_cache_decorator(func: Callable[P, RT]) -> Callable[P, RT]:
        def wrapper(db: "Database", id: UUID, *args, **kwargs) -> RT:
            # The cache key is just the func name minus the `get_`, `send_`, etc. prefix
            cache_key = f"{re.sub('^.*_', '', func.__name__)}_{str(id)}"

            # Check if requested item is alr in cache
            if cache_key in db.cache:
                return db.cache[cache_key][1]

            # If not add it to cache and return it
            value = func(db, id, *args, **kwargs)
            db.cache.add(cache_key, value, ttl)
            return value

        return wrapper
    return get_cache_decorator


# TODO: implement logging for the entire db lib
# TODO: Figure out how to handle deletion of users and stuff
# TODO: Figure out how to handle duplicate keys
class Database:
    def __init__(
        self,
        connection_url: str,
        logger: logging.Logger = None,
        cache_logger: logging.Logger = None
    ) -> None:
        self.logger = logger or logging.getLogger("database")

        self.conn = psycopg.connect(connection_url, row_factory=dict_row)
        self.cursor = self.conn.cursor()
        self.cache = Cache(logger=cache_logger)
        self.cache.start()

        self.logger.info("Database connection established succesfully.")

    @_get_cache()
    def get_user(self, id: UUID) -> User | None:
        cursor = self.cursor.execute(get_user_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return User(data, self)

    # No caching since username lookup is only for login
    def get_user_from_username(self, username: str) -> User | None:
        cursor = self.cursor.execute(get_user_from_username_query, (username,))
        data = cursor.fetchone()
        if data is None:
            return None
        return User(data, self)

    # No caching since credential lookup is only for login
    def get_user_credentials(self, id: UUID) -> UserCredential | None:
        cursor = self.cursor.execute(get_user_credentials_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return UserCredential(data)

    def get_user_conversations(self, id: UUID, limit = 100) -> List[Conversation]:
        cursor = self.cursor.execute(get_user_conversations_query, (id, limit))
        data = cursor.fetchall()
        if data is None:
            return []
        return [Conversation(dict_, self) for dict_ in data]

    def create_user(
        self,
        username: str,
        is_bot: bool,
        password: str = None,
        created_at: datetime = None
    ) -> User:
        data = {
            "user_id": uuid4(),
            "username": username,
            "is_bot": is_bot,
            "created_at": created_at or datetime.now(timezone.utc)
        }
        user = User(data, self)

        # Add to db
        self.cursor.execute(
            create_user_query,
            (
                user.id,
                user.username,
                user.is_bot,
                user.created_at
            )
        )

        if password is not None:
            credential_data = {
                "user_id": user.id,
                "password": password
            }
            user_credential = UserCredential(credential_data)

            self.cursor.execute(
                create_user_credentials_query,
                (
                    user_credential.id,
                    user_credential.password
                )
            )

        # Add to cache
        # No need to add credentials to cache since it's a rarer operation
        self.cache.add(f"user_{user.id}", user)

        self.conn.commit()
        self.logger.debug("User created with the following parameters: %s", data)
        return user

    @_get_cache()
    def get_conversation(self, id: UUID) -> Conversation | None:
        cursor = self.cursor.execute(get_conversation_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return Conversation(data, self)

    # TODO: Figure out a good way to cache this
    def get_conversation_messages(
        self,
        id: UUID,
        before: datetime,
        after: datetime,
        limit = 100
    ) -> List[Message]:
        cursor = self.cursor.execute(get_conversation_messages_query, (id, before, after, limit))
        data = cursor.fetchall()
        if data is None:
            return []
        return [Message(dict_, self) for dict_ in data]

    @_get_cache()
    def get_conversation_user(self, id: UUID) -> User | None:
        cursor = self.cursor.execute(get_conversation_user_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return User(data, self)

    @_get_cache()
    def get_conversation_bot_user(self, id: UUID) -> User | None:
        cursor = self.cursor.execute(get_conversation_bot_user_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return User(data, self)

    def create_conversation(
        self,
        user_id: UUID,
        bot_user_id: UUID,
        name: str = None,
        created_at: datetime = None
    ) -> Conversation:
        data = {
            "conversation_id": uuid4(),
            "user_id": user_id,
            "bot_user_id": bot_user_id,
            "conversation_name": name,
            "created_at": created_at or datetime.now(timezone.utc)
        }
        conversation = Conversation(data, self)

        # Add to db
        self.cursor.execute(
            create_conversation_query,
            (
                conversation.id,
                conversation.user_id,
                conversation.bot_user_id,
                conversation.name,
                conversation.created_at
            )
        )

        # Add to cache
        self.cache.add(f"conversation_{conversation.id}", conversation)

        self.conn.commit()
        self.logger.debug("Conversation created with the following parameters: %s", data)
        return conversation

    @_get_cache()
    def get_message(self, id: UUID) -> Message | None:
        cursor = self.cursor.execute(get_message_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return Message(data, self)

    def send_message(
        self,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
        created_at: datetime = None
    ) -> Message:
        data = {
            "message_id": uuid4(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "content": content,
            "created_at": created_at or datetime.now(timezone.utc)
        }
        message = Message(data, self)

        # Add to db
        self.cursor.execute(
            send_message_query,
            (
                message.id,
                message.user_id,
                message.conversation_id,
                message.content,
                message.created_at
            )
        )

        # Add to cache
        self.cache.add(f"message_{message.id}", message)

        self.conn.commit()
        self.logger.debug("Message created with the following parameters: %s", data)
        return message

    @_get_cache()
    def get_bot_user(self, id: UUID) -> BotUser | None:
        cursor = self.cursor.execute(get_bot_user_query, (id,))
        data = cursor.fetchone()
        if data is None:
            return None
        return BotUser(data, self)

    def create_bot_user(
        self,
        username: str,
        creator_id: UUID,
        last_modified: datetime = None,
        text_gen_model_name: str = None,
        tts_model_name: str = None,
        text_gen_starting_context: str = None,
        tts_speaker_name: str = None,
        created_at: datetime = None,
    ) -> BotUser:
        user = self.create_user(username, True, created_at=created_at)

        data = {
            "user_id": user.id,
            "creator_id": creator_id,
            "last_modified": last_modified or datetime.now(timezone.utc),
            "text_gen_model_name": text_gen_model_name,
            "text_gen_starting_context": text_gen_starting_context,
            "tts_model_name": tts_model_name,
            "tts_speaker_name": tts_speaker_name
        }
        bot_user = BotUser(data, self)

        # Add to db
        self.cursor.execute(
            create_bot_user_query,
            (
                bot_user.id,
                bot_user.creator_id,
                bot_user.last_modified,
                bot_user.text_gen_model_name,
                bot_user.text_gen_starting_context,
                bot_user.tts_model_name,
                bot_user.tts_speaker_name
            )
        )

        # Add to cache
        self.cache.add(f"bot_user_{bot_user.id}", bot_user)

        self.conn.commit()
        self.logger.debug("BotUser created with the following parameters: %s", data)
        return bot_user

    def close(self):
        self.cursor.close()
        self.conn.close()
        self.cache.close()
