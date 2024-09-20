from datetime import datetime
from uuid import UUID

from flask import request
from flask.views import MethodView
from flask_login import current_user, login_required, login_user, logout_user
from flask_socketio import emit
import orjson
from werkzeug.security import check_password_hash, generate_password_hash

from constants import AIINTERFACE, AVAILABLE_MODELS, DB
import db
from utils import abort_if_none, list_to_dict, get_session_data


current_user: db.User


# TODO: Pydantic could clean this up
# TODO: Add rate limits
class BotUser(MethodView):
    @login_required
    def get(self, bot_user_id):
        bot_user = abort_if_none(DB.get_bot_user(bot_user_id))
        user = bot_user.get_user()
        ret = bot_user.to_dict()
        ret["username"] = user.username
        ret["created_at"] = user.created_at

        if bot_user.creator_id != current_user.id:
            del ret["creator_id"]

        return ret

    @login_required
    def patch(self, bot_user_id):
        bot_user = abort_if_none(DB.get_bot_user(bot_user_id))

        if bot_user.creator_id != current_user.id:
            return "User cannot edit a bot it does not own."

        try:
            json = request.json
            if len(json) == 0:
                raise ValueError()
            username: str = json.pop("username", None)
            text_gen_model_name: str = json.pop("text_gen_model_name", None)
            text_gen_starting_context: str = json.pop("text_gen_starting_context", None)
            tts_model_name: str = json.pop("tts_model_name", None)
            tts_speaker_name: str = json.pop("tts_speaker_name", None)

            if ((username is not None and not isinstance(username, str))
                    or (text_gen_model_name is not None
                        and not isinstance(text_gen_model_name, str))
                    or (text_gen_starting_context is not None
                        and not isinstance(text_gen_starting_context, str))
                    or (tts_model_name is not None and not isinstance(tts_model_name, str))
                    or (tts_speaker_name is not None and not isinstance(tts_speaker_name, str))):
                raise ValueError()
            if len(json) > 0:
                raise ValueError()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        if username is not None and len(username) > 32:
            return "`username` cannot be longer than 32 characters.", 400
        if text_gen_model_name is not None:
            if len(text_gen_model_name) > 128:
                return "`text_gen_model_name` cannot be longer than 128 characters.", 400
            elif text_gen_model_name not in AVAILABLE_MODELS["text_gen"]:
                return f"\"{text_gen_model_name}\" is not an allowed text gen model.", 400
        if text_gen_starting_context is not None and len(text_gen_starting_context) > 4096:
            return "`text_gen_starting_context` cannot be longer than 4096 characters.", 400
        if tts_model_name is not None:
            if len(tts_model_name) > 128:
                return "`tts_model_name` cannot be longer than 128 characters.", 400
            if tts_model_name not in AVAILABLE_MODELS["tts"]:
                return f"\"{tts_model_name}\" is not an allowed tts model.", 400
            if tts_speaker_name is None:
                return "`tts_speaker_name` must be specified if `tts_model_name` is non null.", 400
        if tts_speaker_name is not None:
            if len(tts_speaker_name) > 128:
                return "`tts_speaker_name` cannot be longer than 128 characters.", 400
            curr_tts_model = tts_model_name or bot_user.tts_model_name
            if tts_speaker_name not in AVAILABLE_MODELS["tts"][curr_tts_model]:
                return (
                    f"\"{tts_speaker_name}\" is not an allowed speaker on the \"{curr_tts_model}\" "
                    + "tts model.",
                    400
                )

        bot_user, user = DB.update_bot_user(
            bot_user_id,
            username,
            None,
            text_gen_model_name,
            text_gen_starting_context,
            tts_model_name,
            tts_speaker_name
        )

        ret = bot_user.to_dict()
        ret["username"] = user.username
        ret["created_at"] = user.created_at

        session_data = get_session_data()
        if session_data is not None:
            emit(
                str(13),
                orjson.dumps({ "bot_user": ret }),
                namespace="/",
                to=str(bot_user.id),
            )
        # TODO: Handle updating the ai interface with new bot user stuff

        return ret


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
            bot_user_id: str = request.json["bot_user_id"]

            if not isinstance(name, str) or not isinstance(bot_user_id, str):
                raise ValueError()
        except (ValueError, KeyError):
            return "Post data does not contain valid Json data.", 400

        name = name.strip()
        if len(name) > 32:
            return "Conversation name may not exceed 32 characters.", 400

        try:
            bot_user_uuid = UUID(bot_user_id)
        except ValueError:
            return "Bot userid is not a valid UUID.", 400

        # TODO: Add accesstypes to bot users so that there can be more flexibility with who can use
        # which bot users
        abort_if_none(DB.get_bot_user(bot_user_uuid))

        conversation = DB.create_conversation(current_user.id, bot_user_uuid, name)

        # Tell other connected clients that a channel was made
        session_data = get_session_data()
        if session_data is not None:
            socket_event = {
                "conversation": {
                    "id": str(conversation.id),
                    "name": conversation.name,
                    "user_id": str(conversation.user_id),
                    "bot_user_id": str(conversation.bot_user_id),
                    "created_at": conversation.created_at
                }
            }
            emit(
                str(12),
                orjson.dumps(socket_event),
                namespace="/",
                to=str(current_user.id),
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
                to=str(current_user.id),
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


class UserAvailableModels(MethodView):
    @login_required
    def get(self):
        return AVAILABLE_MODELS
