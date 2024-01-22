-- All Timestamps are UTC

-- Because uuids don't have to be unique across tables, there's a tiny
-- chance that there's overlap. But because this is a small app, the chance
-- is small enough to not worry about.

CREATE TABLE usr (
    user_id uuid PRIMARY KEY DEFAULT gen_random_uuid()
  , username varchar(32) NOT NULL UNIQUE
  , is_bot bool NOT NULL DEFAULT false
  , created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX user_username_idx ON usr (username);

CREATE TABLE user_credential (
    user_id uuid PRIMARY KEY REFERENCES usr (user_id)
  , password text NOT NULL
);

CREATE TABLE conversation (
    conversation_id uuid PRIMARY KEY DEFAULT gen_random_uuid()
  , user_id uuid NOT NULL REFERENCES usr (user_id) ON UPDATE CASCADE ON DELETE CASCADE
  , bot_user_id uuid NOT NULL REFERENCES usr (user_id) ON UPDATE CASCADE
  , conversation_name varchar(32) NOT NULL DEFAULT 'Conversation'
  , created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- No need for a bot user index since accessing the conversations the bot is in
-- shouldn't ever be needed.
CREATE INDEX conversation_user_idx ON conversation (user_id);

CREATE TABLE message (
    message_id uuid PRIMARY KEY DEFAULT gen_random_uuid()
  , user_id uuid NOT NULL REFERENCES usr (user_id) ON UPDATE CASCADE
  , conversation_id uuid NOT NULL REFERENCES conversation (conversation_id) ON UPDATE CASCADE ON DELETE CASCADE
  , content varchar(4096) NOT NULL DEFAULT ''
  , created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX message_conversation_idx ON message (conversation_id);
CREATE INDEX message_created_at_idx ON message (created_at);

CREATE TABLE model_preset (
    model_preset_id uuid PRIMARY KEY DEFAULT gen_random_uuid()
  , user_id uuid NOT NULL REFERENCES usr (user_id) ON UPDATE CASCADE
  , model_preset_name varchar(32) NOT NULL DEFAULT ''
  , text_gen_model_name varchar(128) NOT NULL DEFAULT ''
  , text_gen_starting_context varchar(4096) NOT NULL DEFAULT ''
  , tts_model_name varchar(128) NOT NULL DEFAULT ''
  , tts_speaker_name varchar(128) NOT NULL DEFAULT ''
  , created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX model_preset_user_idx ON model_preset (user_id);
