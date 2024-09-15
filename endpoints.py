from datetime import datetime
from uuid import UUID, uuid4

from flask import request
from flask.views import MethodView
from flask_login import current_user, login_required, login_user, logout_user
from flask_socketio import emit
import orjson
from werkzeug.security import check_password_hash, generate_password_hash

from constants import AIINTERFACE, DB
import db
from utils import abort_if_none, list_to_dict, get_session_data


current_user: db.User


class Conversation(MethodView):
    @login_required
    def get(self, conversation_id):
        conversation = abort_if_none(DB.get_conversation(conversation_id))
        if conversation.user_id != current_user.id:
            return "User does not have access to that resource.", 403

        return conversation.to_dict()


class Conversations(MethodView):
    @login_required
    def get(self):
        return list_to_dict(DB.get_user_conversations(current_user.id))

    @login_required
    def post(self):
        try:
            name: str = request.json["name"]

            if not isinstance(name, str):
                raise ValueError()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        name = name.strip()
        if len(name) > 32:
            return "Conversation name may not exceed 32 characters.", 400

        # TODO: [User Management] Temp!!!
        conversation = DB.create_conversation(current_user.id, UUID("10a8827c-9187-4dc6-9c48-c883af32ebc4"), name)

        # Tell other connected clients that a channel was made
        session_data = get_session_data()
        if session_data is not None:
            socket_event = {
                "event": "create_channel",
                "id": str(uuid4()),
                "conversation": {
                    "id": str(conversation.id),
                    "name": conversation.name,
                    "user_id": str(conversation.user_id),
                    "bot_user_id": str(conversation.bot_user_id),
                    "created_at": conversation.created_at
                }
            }
            emit(
                "create_channel",
                orjson.dumps(socket_event),
                namespace="/",
                to=current_user.id,
            )

        return conversation.to_dict()


class Login(MethodView):
    @login_required
    def get(self):
        return current_user.to_dict()

    def post(self):
        try:
            username: str = request.json["username"]
            password: str = request.json["password"]

            if not isinstance(username, str) or not isinstance(password, str):
                raise ValueError()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        if len(username) > 32:
            return "Username may not exceed 32 characters", 400

        if len(password) > 128:
            return "Password may not exceed 128 characters", 400

        user = DB.get_user_from_username(username)
        if user is None:
            return "Username or password is incorrect", 401

        credentials = user.get_user_credentials()
        if check_password_hash(credentials.password, password):
            login_user(user, remember=True)
            return user.to_dict()
        return "Username or password is incorrect", 401


class Logout(MethodView):
    @login_required
    def post(self):
        logout_user()
        return "", 200


class Message(MethodView):
    @login_required
    def get(self, conversation_id: UUID, message_id: UUID):
        conversation = abort_if_none(DB.get_conversation(conversation_id))
        if conversation.user_id != current_user.id:
            return "User does not have access to that resource.", 403

        message = abort_if_none(DB.get_message(message_id))
        if message.conversation_id != conversation.id:
            return "User does not have access to that resource.", 403

        return message.to_dict()


class Messages(MethodView):
    @login_required
    def get(self, conversation_id: UUID):
        conversation = abort_if_none(DB.get_conversation(conversation_id))
        if conversation.user_id != current_user.id:
            return "User does not have access to that resource.", 403

        try:
            before = request.args.get("before", datetime.max, datetime.fromisoformat)
        except ValueError:
            return "\"before\" timestamp is formatted incorrectly.", 400

        try:
            after = request.args.get("after", datetime.min, datetime.fromisoformat)
        except ValueError:
            return "\"after\" timestamp is formatted incorrectly.", 400

        try:
            limit = min(100, request.args.get("limit", 100, int))
        except ValueError:
            return "\"limit\" integer is formatted incorrectly."

        return list_to_dict(conversation.get_messages(before, after, limit=limit))

    @login_required
    def post(self, conversation_id: UUID):
        conversation = abort_if_none(DB.get_conversation(conversation_id))
        if conversation.user_id != current_user.id:
            return "User does not have access to that resource.", 403

        try:
            content = str(request.json["content"]).strip()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        if len(content) > 4096:
            return "Message content may not exceed 4096 characters.", 400

        message = conversation.send_message(False, content)

        session_data = get_session_data()
        if session_data is not None:
            AIINTERFACE.send_text_data(
                str(message.user_id) + str(message.conversation_id),
                message.content
            )

            # Tell other connected clients that a message was sent
            emit(
                "10",
                orjson.dumps({ "message": message.to_dict() }),
                namespace="/",
                to=current_user.id,
            )

        return message.to_dict()


class Signup(MethodView):
    def post(self):
        try:
            username: str = request.json["username"]
            password: str = request.json["password"]

            if not isinstance(username, str) or not isinstance(password, str):
                raise ValueError()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        if len(username) > 32:
            return "Username may not exceed 32 characters", 400

        if len(password) > 128:
            return "Password may not exceed 128 characters", 400

        existing_user = DB.get_user_from_username(username)
        if existing_user is not None:
            return "A user with that username already exists", 400

        hashed_pw = generate_password_hash(password)

        return DB.create_user(username, False, hashed_pw).to_dict()


class User(MethodView):
    # TODO: [User Management] kek
    @login_required
    def get(self, user_id):
        return user_id

    @login_required
    def put(self, user_id):
        return user_id
