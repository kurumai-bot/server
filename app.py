import logging
from uuid import UUID

from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from flask_socketio import SocketIO

from constants import AIINTERFACE, DB, HOST, LOGGER, PORT, SECRET_KEY
from db import User
import endpoints
import sockets
from utils import ORJSONProvider, add_url_rule_view


app = Flask(__name__)
app.json = ORJSONProvider(app)
app.secret_key = SECRET_KEY
# This won't scale horizontally, but will work fine for the time being
app.extensions["user_sessions"] = {}
login_manager = LoginManager(app)
cors = CORS(app, supports_credentials=True)

socketio_logger = LOGGER.getChild("socketio")
socketio_logger.setLevel(logging.ERROR)
engineio_logger = LOGGER.getChild("engineio")
engineio_logger.setLevel(logging.ERROR)
socketio = SocketIO(
    app,
    json=app.json,
    cors_allowed_origins="*",
    logger=socketio_logger,
    engineio_logger=engineio_logger
)

app.config["REMEMBER_COOKIE_SECURE"] = True
app.config["REMEMBER_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"

@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    return DB.get_user(UUID(user_id))


add_url_rule_view(app, "/bot_user/<uuid:bot_user_id>", view=endpoints.BotUser)
add_url_rule_view(app, "/conversations/<uuid:conversation_id>", view=endpoints.Conversation)
add_url_rule_view(app, "/conversations", view=endpoints.Conversations)
add_url_rule_view(app, "/auth", view=endpoints.Login)
add_url_rule_view(app, "/auth/logout", view=endpoints.Logout)
add_url_rule_view(app, "/auth/signup", view=endpoints.Signup)
add_url_rule_view(
    app,
    "/conversations/<uuid:conversation_id>/messages/<uuid:message_id>",
    view=endpoints.Message
)
add_url_rule_view(app, "/conversations/<uuid:conversation_id>/messages", view=endpoints.Messages)
add_url_rule_view(app, "/user/<uuid:user_id>", view=endpoints.User)
add_url_rule_view(app, "/user/available_models", view=endpoints.UserAvailableModels)

socketio.on_event("connect", sockets.handle_connect)
socketio.on_event("disconnect", sockets.handle_disconnect)
socketio.on_event("3", sockets.handle_mic_packet)
socketio.on_event("11", sockets.handle_set_conversation)

socketio.start_background_task(sockets.poll_pipeline_loop(app))


if __name__ == "__main__":
    LOGGER.info("Starting listener server.")
    try:
        socketio.run(app, host=HOST, port=PORT)
    finally:
        LOGGER.info("Releasing DB resources.")
        DB.close()
        LOGGER.info("Releasing AI Interface resources")
        AIINTERFACE.close()

# TODO: consider making a base class for processors that has like a __call__ or process() function
# that could return values by blocking, then create a thread per connection to handle all the funky
# bits.
# TODO: Figure out when to use .get vs .pop
# TODO: Add facial expressions and stuff. Probably just ask AI to add an expression from a list
# given as its first prompt
# TODO: Implement logging
# TODO: place kwargs stuff at the top
