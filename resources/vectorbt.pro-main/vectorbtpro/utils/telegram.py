# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing messaging functionality using Python Telegram Bot.

!!! info
    For default settings, see `vectorbtpro._settings.telegram`.
"""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("telegram")

import inspect
import logging
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import Configured, merge_dicts
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.requests_ import text_to_giphy_url
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "TelegramBot",
]

logger = logging.getLogger(__name__)


def send_action(action: str) -> tp.Callable:
    """Decorator factory that sends a specified Telegram chat action during command execution.

    This decorator is intended for bound callback methods that accept `self`, `update`, `context`,
    and optionally additional arguments.

    Args:
        action (str): Telegram chat action to send (e.g., 'typing', 'upload_photo').

    Returns:
        Callable: Decorator that wraps the target command function.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def command_func(
            self, update: Update, context: CallbackContext, *args, **kwargs
        ) -> tp.Callable:
            if update.effective_chat:
                context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


def self_decorator(self, func: tp.Callable) -> tp.Callable:
    """Decorator that injects the bot instance as the first argument to a command function.

    Args:
        self: Bot instance to pass to the command.
        func (Callable): Command function to wrap.

    Returns:
        Callable: Decorated command function with the bot instance provided.
    """

    def _command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)

    return _command_func


try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0)

if __version_info__ < (20, 0):
    from telegram import Update
    from telegram.error import ChatMigrated, Unauthorized
    from telegram.ext import (
        CallbackContext,
        CommandHandler,
        Defaults,
        Dispatcher,
        Filters,
        Handler,
        MessageHandler,
        PicklePersistence,
        Updater,
    )
    from telegram.utils.helpers import effective_message_type

    class LogHandler(Handler, Base):
        """Telegram update handler that logs user messages.

        If an update contains an effective message, logs its text content for text messages
        or the message type for non-text messages.
        """

        def check_update(self, update: object) -> tp.Optional[tp.Union[bool, object]]:
            if isinstance(update, Update) and update.effective_message:
                message = update.effective_message
                message_type = effective_message_type(message)
                if message_type is not None:
                    if message_type == "text":
                        logger.info(f'{message.chat_id} - User: "%s"', message.text)
                    else:
                        logger.info(f"{message.chat_id} - User: %s", message_type)
                return False
            return None

    if Handler.check_update.__doc__:
        LogHandler.check_update.__doc__ = f"""Docstring of `telegram.ext.Handler.check_update`:

```text
{inspect.cleandoc(Handler.check_update.__doc__)}
```
"""

    class TelegramBot(Configured):
        """Telegram bot class integrating with the python-telegram-bot (PTB) library.

        See https://github.com/python-telegram-bot/python-telegram-bot/wiki/ to get started.

        Args:
            giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                These settings are merged with the default configuration obtained from
                `vectorbtpro._settings.telegram` under the giphy key.
                See `vectorbtpro.utils.requests_.text_to_giphy_url`.
            **kwargs: Keyword arguments for configuring the updater;
                they override settings for the bot.

        !!! info
            For default settings, see `vectorbtpro._settings.telegram`.

        Examples:
            Let's extend `TelegramBot` to track cryptocurrency prices:

            ```python
            import ccxt
            import logging
            from vectorbtpro import *

            from telegram.ext import CommandHandler
            from telegram import __version__ as TG_VER

            try:
                from telegram import __version_info__
            except ImportError:
                __version_info__ = (0, 0)

            if __version_info__ >= (20, 0):
                raise RuntimeError(f"This implementation is not compatible with telegram version {TG_VER}")

            # Enable logging
            logging.basicConfig(
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
            )
            logger = logging.getLogger(__name__)


            class MyTelegramBot(vbt.TelegramBot):
                @property
                def custom_handlers(self):
                    return (CommandHandler('get', self.get),)

                @property
                def help_message(self):
                    return "Type /get [symbol] [exchange id (optional)] to get the latest price."

                def get(self, update, context):
                    chat_id = update.effective_chat.id

                    if len(context.args) == 1:
                        symbol = context.args[0]
                        exchange = 'binance'
                    elif len(context.args) == 2:
                        symbol = context.args[0]
                        exchange = context.args[1]
                    else:
                        self.send_message(chat_id, "This command requires symbol and optionally exchange id.")
                        return
                    try:
                        ticker = getattr(ccxt, exchange)().fetchTicker(symbol)
                    except Exception as e:
                        self.send_message(chat_id, str(e))
                        return
                    self.send_message(chat_id, str(ticker['last']))


            if __name__ == "__main__":
                bot = MyTelegramBot(token='YOUR_TOKEN')
                bot.start()
            ```
        """

        _expected_keys_mode: tp.ExpectedKeysMode = "disable"

        def __init__(self, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            from vectorbtpro._settings import settings

            bot_cfg = settings["telegram"]["bot"]
            giphy_cfg = settings["telegram"]["giphy"]

            Configured.__init__(self, giphy_kwargs=giphy_kwargs, **kwargs)

            giphy_kwargs = merge_dicts(giphy_cfg, giphy_kwargs)
            self.giphy_kwargs = giphy_kwargs
            default_kwargs = dict()
            passed_kwargs = dict()
            for k in get_func_kwargs(Updater.__init__):
                if k in bot_cfg:
                    default_kwargs[k] = bot_cfg[k]
                if k in kwargs:
                    passed_kwargs[k] = kwargs.pop(k)
            updater_kwargs = merge_dicts(default_kwargs, passed_kwargs)
            persistence = updater_kwargs.pop("persistence", None)
            if persistence is not None:
                if isinstance(persistence, bool):
                    if persistence:
                        persistence = "telegram_bot.pickle"
                    else:
                        persistence = None
                if persistence is not None:
                    if isinstance(persistence, str):
                        persistence = PicklePersistence(persistence)
            defaults = updater_kwargs.pop("defaults", None)
            if defaults is not None:
                if isinstance(defaults, dict):
                    defaults = Defaults(**defaults)

            logger.info("Initializing bot")
            self._updater = Updater(persistence=persistence, defaults=defaults, **updater_kwargs)

            self._dispatcher = self.updater.dispatcher

            self.dispatcher.add_handler(self.log_handler)
            self.dispatcher.add_handler(CommandHandler("start", self.start_callback))
            self.dispatcher.add_handler(CommandHandler("help", self.help_callback))
            for handler in self.custom_handlers:
                self.dispatcher.add_handler(handler)
            self.dispatcher.add_handler(
                MessageHandler(Filters.status_update.migrate, self.chat_migration_callback)
            )
            self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown_callback))
            self.dispatcher.add_error_handler(self_decorator(self, type(self).error_callback))

            if "chat_ids" not in self.dispatcher.bot_data:
                self.dispatcher.bot_data["chat_ids"] = []
            else:
                logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data["chat_ids"]))

        @property
        def updater(self) -> Updater:
            """Telegram updater instance used to poll messages and handle updates.

            Returns:
                Updater: `Updater` instance responsible for polling messages.
            """
            return self._updater

        @property
        def dispatcher(self) -> Dispatcher:
            """Telegram dispatcher instance for registering handlers.

            Returns:
                Dispatcher: `telegram.ext.Dispatcher` instance managing update handlers.
            """
            return self._dispatcher

        @property
        def log_handler(self) -> LogHandler:
            """Telegram log handler instance for logging incoming user updates.

            Returns:
                LogHandler: Instance of `LogHandler` set up for logging updates.
            """
            return LogHandler(lambda update, context: None)

        @property
        def custom_handlers(self) -> tp.Iterable[Handler]:
            """Custom command handlers.

            Override this property to provide additional command handlers.
            The order of handlers is significant.

            Returns:
                Iterable[Handler]: Iterable collection of additional command `telegram.extHandler` instances.
            """
            return ()

        @property
        def chat_ids(self) -> tp.List[int]:
            """List of chat IDs that have interacted with the bot.

            A chat ID is added when the `/start` command is received.

            Returns:
                List[int]: List of chat IDs for all chats currently tracked by the bot.
            """
            return self.dispatcher.bot_data["chat_ids"]

        def start(self, in_background: bool = False, **kwargs) -> None:
            """Start the Telegram bot.

            Keyword arguments are passed to `telegram.ext.updater.Updater.start_polling`
            to override the default bot settings from `vectorbtpro._settings.telegram`.

            Args:
                in_background (bool): Run the bot in the background if True; otherwise, block the main thread.
                **kwargs: Keyword arguments for `Updater.start_polling`.

            Returns:
                None

            !!! info
                For default settings, see `bot` in `vectorbtpro._settings.telegram`.
            """
            from vectorbtpro._settings import settings

            bot_cfg = settings["telegram"]["bot"]

            default_kwargs = dict()
            passed_kwargs = dict()
            for k in get_func_kwargs(self.updater.start_polling):
                if k in bot_cfg:
                    default_kwargs[k] = bot_cfg[k]
                if k in kwargs:
                    passed_kwargs[k] = kwargs.pop(k)
            polling_kwargs = merge_dicts(default_kwargs, passed_kwargs)

            logger.info("Running bot %s", str(self.updater.bot.get_me().username))
            self.updater.start_polling(**polling_kwargs)
            self.started_callback()

            if not in_background:
                self.updater.idle()

        def started_callback(self) -> None:
            """Callback executed after the bot has started.

            Override this method to execute custom commands once the bot is online.

            Returns:
                None
            """
            self.send_message_to_all("I'm back online!")

        def send(
            self, kind: str, chat_id: int, *args, log_msg: tp.Optional[str] = None, **kwargs
        ) -> None:
            """Send a message of a specified kind to a target chat ID.

            Args:
                kind (str): Type of message to send (e.g., "message", "animation").
                chat_id (int): Unique identifier of the target chat.
                *args: Positional arguments for `TelegramBot.send`.
                log_msg (Optional[str]): Message description for logging; if not provided,
                    defaults to the value of `kind`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            try:
                getattr(self.updater.bot, "send_" + kind)(chat_id, *args, **kwargs)
                if log_msg is None:
                    log_msg = kind
                logger.info(f"{chat_id} - Bot: %s", log_msg)
            except ChatMigrated as e:
                new_id = e.new_chat_id
                if chat_id in self.chat_ids:
                    self.chat_ids.remove(chat_id)
                self.chat_ids.append(new_id)
                self.send(kind, new_id, *args, log_msg=log_msg, **kwargs)
            except Unauthorized:
                logger.info(f"{chat_id} - Unauthorized to send the %s", kind)

        def send_to_all(self, kind: str, *args, **kwargs) -> None:
            """Send a message of a specified kind to all chat IDs in `TelegramBot.chat_ids`.

            Args:
                kind (str): Type of message to send (e.g., "message", "animation").
                *args: Positional arguments for `TelegramBot.send`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            for chat_id in self.chat_ids:
                self.send(kind, chat_id, *args, **kwargs)

        def send_message(self, chat_id: int, text: str, *args, **kwargs) -> None:
            """Send a text message to the specified chat.

            Args:
                chat_id (int): Unique identifier of the target chat.
                text (str): Content of the message to send.
                *args: Positional arguments for `TelegramBot.send`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            log_msg = '"%s"' % text
            self.send("message", chat_id, text, *args, log_msg=log_msg, **kwargs)

        def send_message_to_all(self, text: str, *args, **kwargs) -> None:
            """Send a text message to all chats in `TelegramBot.chat_ids`.

            Args:
                text (str): Content of the message to send.
                *args: Positional arguments for `TelegramBot.send_to_all`.
                **kwargs: Keyword arguments for `TelegramBot.send_to_all`.

            Returns:
                None
            """
            log_msg = '"%s"' % text
            self.send_to_all("message", text, *args, log_msg=log_msg, **kwargs)

        def send_giphy(
            self, chat_id: int, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs
        ) -> None:
            """Send a GIPHY animation generated from the provided text to the specified chat.

            Args:
                chat_id (int): Unique identifier of the target chat.
                text (str): Text from which to generate a GIPHY URL.
                *args: Positional arguments for `TelegramBot.send`.
                giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                    If not provided, defaults to `TelegramBot.giphy_kwargs`.
                    See `vectorbtpro.utils.requests_.text_to_giphy_url`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send("animation", chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

        def send_giphy_to_all(
            self, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs
        ) -> None:
            """Send a GIPHY animation generated from the provided text to all chats in `TelegramBot.chat_ids`.

            Args:
                text (str): Text from which to generate a GIPHY URL.
                *args: Positional arguments for `TelegramBot.send_to_all`.
                giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                    If not provided, defaults to `TelegramBot.giphy_kwargs`.
                    See `vectorbtpro.utils.requests_.text_to_giphy_url`.
                **kwargs: Keyword arguments for `TelegramBot.send_to_all`.

            Returns:
                None
            """
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send_to_all("animation", gif_url, *args, log_msg=log_msg, **kwargs)

        @property
        def start_message(self) -> str:
            """Message to be sent in response to the `/start` command.

            Override this property to define a custom start message.

            Returns:
                str: Message.
            """
            return "Hello!"

        def start_callback(self, update: object, context: CallbackContext) -> None:
            """Handle the `/start` command callback.

            Args:
                update (object): Incoming update.
                context (CallbackContext): Callback context containing additional data.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                if chat_id not in self.chat_ids:
                    self.chat_ids.append(chat_id)
                self.send_message(chat_id, self.start_message)

        @property
        def help_message(self) -> str:
            """Message to be sent in response to the `/help` command.

            Override this property to define a custom help message.

            Returns:
                str: Message.
            """
            return "Can't help you here, buddy."

        def help_callback(self, update: object, context: CallbackContext) -> None:
            """Handle the `/help` command callback.

            Args:
                update (object): Incoming update.
                context (CallbackContext): Callback context containing additional data.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, self.help_message)

        def chat_migration_callback(self, update: object, context: CallbackContext) -> None:
            """Handle chat migration events by updating the internal chat IDs.

            Args:
                update (object): Incoming update.
                context (CallbackContext): Callback context containing additional data.

            Returns:
                None
            """
            if isinstance(update, Update) and update.message:
                old_id = update.message.migrate_from_chat_id or update.message.chat_id
                new_id = update.message.migrate_to_chat_id or update.message.chat_id
                if old_id in self.chat_ids:
                    self.chat_ids.remove(old_id)
                self.chat_ids.append(new_id)
                logger.info(f"{old_id} - Chat migrated to {new_id}")

        def unknown_callback(self, update: object, context: CallbackContext) -> None:
            """Handle unknown commands by notifying the user.

            Args:
                update (object): Incoming update.
                context (CallbackContext): Callback context containing additional data.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                logger.info(f'{chat_id} - Unknown command "{update.message}"')
                self.send_message(chat_id, "Sorry, I didn't understand that command.")

        def error_callback(self, update: object, context: CallbackContext, *args) -> None:
            """Handle errors that occur during update processing.

            Args:
                update (object): Update that triggered the error.
                context (CallbackContext): Callback context containing additional data.
                *args: Additional positional arguments.

            Returns:
                None
            """
            logger.error(
                'Exception while handling an update "%s": ', update, exc_info=context.error
            )
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, "Sorry, an error happened.")

        def stop(self) -> None:
            """Stop the bot.

            Logs an informational message and stops the updater.

            Returns:
                None
            """
            logger.info("Stopping bot")
            self.updater.stop()

        @property
        def running(self) -> bool:
            """Indicate whether the bot is currently running.

            Returns:
                bool: True if the bot is running, False otherwise.
            """
            return self.updater.running

else:
    import asyncio
    import platform
    import signal

    from telegram import Update
    from telegram.error import ChatMigrated, Forbidden, TelegramError
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        BaseHandler,
        CommandHandler,
        ContextTypes,
        Defaults,
        MessageHandler,
        PicklePersistence,
        filters,
    )
    from telegram.helpers import effective_message_type
    from telegram.request import BaseRequest

    class LogHandler(BaseHandler, Base):
        """Log user updates by recording message content and type."""

        def check_update(self, update: object) -> tp.Optional[tp.Union[bool, object]]:
            if isinstance(update, Update) and update.effective_message:
                message = update.effective_message
                message_type = effective_message_type(message)
                if message_type is not None:
                    if message_type == "text":
                        logger.info(f'{message.chat_id} - User: "%s"', message.text)
                    else:
                        logger.info(f"{message.chat_id} - User: %s", message_type)
                return False
            return None

    class TelegramBot(Configured):
        """Telegram bot class integrating with the python-telegram-bot library.

        See https://github.com/python-telegram-bot/python-telegram-bot/wiki/ to get started.

        Args:
            giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                These settings are merged with the default configuration obtained from
                `vectorbtpro._settings.telegram` under the giphy key.
                See `vectorbtpro.utils.requests_.text_to_giphy_url`.
            **kwargs: Keyword arguments for configuring the updater;
                they override settings for the bot.

        !!! info
            For default settings, see `vectorbtpro._settings.telegram`.

        !!! note
            If you get "RuntimeError: Cannot close a running event loop" when running in Jupyter,
            you might need to patch `asyncio` using https://github.com/erdewit/nest_asyncio, like this
            (right before `bot.start()` in each new runtime)

            ```pycon
            >>> !pip install nest_asyncio
            >>> import nest_asyncio
            >>> nest_asyncio.apply()
            ```

        Examples:
            Let's extend `TelegramBot` to track cryptocurrency prices, as a Python script:

            ```python
            import ccxt
            import logging
            from vectorbtpro import *

            from telegram.ext import CommandHandler
            from telegram import __version__ as TG_VER

            try:
                from telegram import __version_info__
            except ImportError:
                __version_info__ = (0, 0)

            if __version_info__ < (20, 0):
                raise RuntimeError(f"This implementation is not compatible with telegram version {TG_VER}")

            # Enable logging
            logging.basicConfig(
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
            )
            logger = logging.getLogger(__name__)


            class MyTelegramBot(vbt.TelegramBot):
                @property
                def custom_handlers(self):
                    return (CommandHandler('get', self.get),)

                @property
                def help_message(self):
                    return "Type /get [symbol] [exchange id (optional)] to get the latest price."

                async def get(self, update, context):
                    chat_id = update.effective_chat.id

                    if len(context.args) == 1:
                        symbol = context.args[0]
                        exchange = 'binance'
                    elif len(context.args) == 2:
                        symbol = context.args[0]
                        exchange = context.args[1]
                    else:
                        await self.send_message(chat_id, "This command requires symbol and optionally exchange id.")
                        return
                    try:
                        ticker = getattr(ccxt, exchange)().fetchTicker(symbol)
                    except Exception as e:
                        await self.send_message(chat_id, str(e))
                        return
                    await self.send_message(chat_id, str(ticker['last']))


            if __name__ == "__main__":
                bot = MyTelegramBot(token='YOUR_TOKEN')
                bot.start()
            ```
        """

        _expected_keys_mode: tp.ExpectedKeysMode = "disable"

        def __init__(self, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            from vectorbtpro._settings import settings

            giphy_cfg = settings["telegram"]["giphy"]

            Configured.__init__(self, giphy_kwargs=giphy_kwargs, **kwargs)

            giphy_kwargs = merge_dicts(giphy_cfg, giphy_kwargs)
            self._giphy_kwargs = giphy_kwargs
            self._application = self.build_application(**kwargs)
            self.register_handlers()
            self._loop = None

        def build_application(self, **kwargs) -> Application:
            """Build the application.

            Override bot settings from `vectorbtpro._settings.telegram` using additional
            keyword arguments. For each key in `**kwargs` that matches an attribute of
            `telegram.ext._applicationbuilder.ApplicationBuilder`, call the corresponding
            builder method. If a value is a dict, it is unpacked to provide multiple keyword
            arguments (except for `defaults`, which is passed as a single argument).

            Returns:
                Application: Constructed application instance.

            !!! info
                For default settings, see `bot` in `vectorbtpro._settings.telegram`.
            """
            from vectorbtpro._settings import settings

            bot_cfg = dict(settings["telegram"]["bot"])
            bot_cfg = merge_dicts(bot_cfg, kwargs)

            builder = ApplicationBuilder()

            persistence = bot_cfg.pop("persistence", None)
            if persistence is not None:
                if isinstance(persistence, bool):
                    if persistence:
                        persistence = "telegram_bot.pickle"
                    else:
                        persistence = None
                if persistence is not None:
                    if isinstance(persistence, str):
                        persistence = PicklePersistence(persistence)
                    builder.persistence(persistence)
            defaults = bot_cfg.pop("defaults", None)
            if defaults is not None:
                if isinstance(defaults, dict):
                    defaults = Defaults(**defaults)
                builder.defaults(defaults)
            for k, v in bot_cfg.items():
                if hasattr(builder, k):
                    if isinstance(v, dict):
                        getattr(builder, k)(**v)
                    else:
                        getattr(builder, k)(v)

            return builder.build()

        @property
        def custom_handlers(self) -> tp.Iterable[BaseHandler]:
            """Custom handlers to add.

            Override this property to provide additional custom handlers.
            The order of the handlers determines the registration sequence.

            Returns:
                Iterable[BaseHandler]: Iterable of custom handlers.
            """
            return ()

        def register_handlers(self) -> None:
            """Register bot handlers.

            Register the log handler, start command, help command, custom handlers, migration handler,
            unknown command handler, and error handler with the application.

            Returns:
                None
            """
            self.application.add_handler(self.log_handler)
            self.application.add_handler(CommandHandler("start", self.start_callback))
            self.application.add_handler(CommandHandler("help", self.help_callback))
            for handler in self.custom_handlers:
                self.application.add_handler(handler)
            self.application.add_handler(
                MessageHandler(filters.StatusUpdate.MIGRATE, self.chat_migration_callback)
            )
            self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_callback))
            self.application.add_error_handler(self_decorator(self, type(self).error_callback))

        @property
        def giphy_kwargs(self) -> tp.Kwargs:
            """Keyword arguments for GIPHY.

            Returns:
                Kwargs: Dictionary of keyword arguments for configuring GIPHY.
            """
            return self._giphy_kwargs

        @property
        def application(self) -> Application:
            """Application instance.

            Returns:
                Application: Current application instance.
            """
            return self._application

        @property
        def chat_ids(self) -> tp.Set[int]:
            """Chat IDs that have interacted with the bot.

            Chat IDs are added upon receiving the `/start` command.

            Returns:
                Set[int]: Set of chat IDs.
            """
            return self.application.bot_data.setdefault("chat_ids", set())

        async def log_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Log callback.

            Args:
                update (Update): Update from Telegram.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.

            Returns:
                None
            """
            pass

        @property
        def log_handler(self) -> LogHandler:
            """Log handler instance.

            Returns:
                LogHandler: Instance of the log handler.
            """
            return LogHandler(self.log_callback)

        async def send(
            self, kind: str, chat_id: int, *args, log_msg: tp.Optional[str] = None, **kwargs
        ) -> None:
            """Send a message of a specified kind to a chat.

            Args:
                kind (str): Type of message to send (e.g., "message", "animation").
                chat_id (int): Unique identifier of the target chat.
                *args: Positional arguments for sending the message.
                log_msg (Optional[str]): Log message for tracking the message.

                    Defaults to the message kind if not provided.
                **kwargs: Keyword arguments for sending the message.

            Returns:
                None
            """
            try:
                await getattr(self.application.bot, "send_" + kind)(chat_id, *args, **kwargs)
                if log_msg is None:
                    log_msg = kind
                logger.info(f"{chat_id} - Bot: %s", log_msg)
            except ChatMigrated as e:
                new_id = e.new_chat_id
                if chat_id in self.chat_ids:
                    self.chat_ids.remove(chat_id)
                self.chat_ids.add(new_id)
                await self.send(kind, new_id, *args, log_msg=log_msg, **kwargs)
            except Forbidden:
                logger.info(f"{chat_id} - Forbidden to send the %s", kind)

        async def send_to_all(self, kind: str, *args, **kwargs) -> None:
            """Send a message of a specified kind to all chats.

            Args:
                kind (str): Type of message to send (e.g., "message", "animation").
                *args: Positional arguments for `TelegramBot.send`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            for chat_id in self.chat_ids:
                await self.send(kind, chat_id, *args, **kwargs)

        async def send_message(self, chat_id: int, text: str, *args, **kwargs) -> None:
            """Send a text message to a chat.

            Args:
                chat_id (int): Unique identifier of the target chat.
                text (str): Text message to send.
                *args: Positional arguments for `TelegramBot.send`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            log_msg = '"%s"' % text
            await self.send("message", chat_id, text, *args, log_msg=log_msg, **kwargs)

        async def send_message_to_all(self, text: str, *args, **kwargs) -> None:
            """Send a text message to all chats.

            Args:
                text (str): Text message to send.
                *args: Positional arguments for `TelegramBot.send_to_all`.
                **kwargs: Keyword arguments for `TelegramBot.send_to_all`.

            Returns:
                None
            """
            log_msg = '"%s"' % text
            await self.send_to_all("message", text, *args, log_msg=log_msg, **kwargs)

        async def send_giphy(
            self, chat_id: int, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs
        ) -> None:
            """Send a GIPHY animation generated from text to a chat.

            Args:
                chat_id (int): Unique identifier of the target chat.
                text (str): Text to convert into a GIPHY URL.
                *args: Positional arguments for `TelegramBot.send`.
                giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                    If not provided, defaults to `TelegramBot.giphy_kwargs`.
                    See `vectorbtpro.utils.requests_.text_to_giphy_url`.
                **kwargs: Keyword arguments for `TelegramBot.send`.

            Returns:
                None
            """
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            await self.send("animation", chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

        async def send_giphy_to_all(
            self, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs
        ) -> None:
            """Send a GIPHY animation generated from text to all chats.

            Args:
                text (str): Text to convert into a GIPHY URL.
                *args: Positional arguments for `TelegramBot.send_to_all`.
                giphy_kwargs (KwargsLike): Keyword arguments for generating the GIPHY URL.

                    If not provided, defaults to `TelegramBot.giphy_kwargs`.
                    See `vectorbtpro.utils.requests_.text_to_giphy_url`.
                **kwargs: Keyword arguments for `TelegramBot.send_to_all`.

            Returns:
                None
            """
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            await self.send_to_all("animation", gif_url, *args, log_msg=log_msg, **kwargs)

        async def post_start_callback(self) -> None:
            """Execute post-start actions.

            Send a message to all chats indicating that the bot is active.

            Returns:
                None
            """
            await self.send_message_to_all("I'm back to life!")

        async def pre_stop_callback(self) -> None:
            """Execute pre-stop actions.

            Send a message to all chats indicating that the bot is shutting down.

            Returns:
                None
            """
            await self.send_message_to_all("Bye!")

        @property
        def start_message(self) -> str:
            """Start command message.

            Message sent to a chat when the `/start` command is received.

            Returns:
                str: Start message.
            """
            return "Hello!"

        async def start_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle the `/start` command.

            Adds the chat ID to the chat IDs set and sends the start message.

            Args:
                update (Update): Update triggered by the `/start` command.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                if chat_id not in self.chat_ids:
                    self.chat_ids.add(chat_id)
                await self.send_message(chat_id, self.start_message)

        @property
        def help_message(self) -> str:
            """Help command message.

            Message sent to a chat when the `/help` command is received.

            Returns:
                str: Help message.
            """
            return "Can't help you here, buddy."

        async def help_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle the `/help` command.

            Sends the help message to the chat.

            Args:
                update (Update): Update triggered by the `/help` command.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                await self.send_message(chat_id, self.help_message)

        async def chat_migration_callback(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle chat migration updates.

            Updates the chat IDs set when a chat migrates and logs the migration event.

            Args:
                update (Update): Update containing migration information.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.

            Returns:
                None
            """
            if isinstance(update, Update) and update.message:
                old_id = update.message.migrate_from_chat_id or update.message.chat_id
                new_id = update.message.migrate_to_chat_id or update.message.chat_id
                if old_id in self.chat_ids:
                    self.chat_ids.remove(old_id)
                self.chat_ids.add(new_id)
                logger.info(f"{old_id} - Chat migrated to {new_id}")

        async def unknown_callback(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle unknown commands.

            Logs an unknown command event and informs the user that the command was not understood.

            Args:
                update (Update): Update containing the unknown command.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.

            Returns:
                None
            """
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                logger.info(f'{chat_id} - Unknown command "{update.message}"')
                await self.send_message(chat_id, "Sorry, I didn't understand that command.")

        async def error_callback(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
        ) -> None:
            """Handle errors during update processing.

            Logs the error and sends an error message to the user.

            Args:
                update (Update): Update that caused the error.
                context (ContextTypes.DEFAULT_TYPE): Context associated with the update.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.

            Returns:
                None
            """
            logger.error(
                'Exception while handling an update "%s": ', update, exc_info=context.error
            )
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                await self.send_message(chat_id, "Sorry, an error happened.")

        @property
        def loop(self) -> tp.Optional[asyncio.BaseEventLoop]:
            """Current event loop.

            Returns:
                Optional[asyncio.BaseEventLoop]: Event loop instance, if available.
            """
            return self._loop

        def start(
            self,
            close_loop: bool = True,
            stop_signals: tp.Sequence[int] = BaseRequest.DEFAULT_NONE,
            **kwargs,
        ) -> None:
            """Start the bot.

            Args:
                close_loop (bool): Whether to close the event loop after stopping the bot.
                stop_signals (Sequence[int]): Signals used to stop the event loop.
                **kwargs: Keyword arguments that override the `bot` settings from
                    `vectorbtpro._settings.telegram`.

                    Only keys accepted by `telegram.ext._updater.Updater.start_polling` are passed.

            Returns:
                None

            !!! info
                For default settings, see `bot` in `vectorbtpro._settings.telegram`.
            """
            from vectorbtpro._settings import settings

            bot_cfg = settings["telegram"]["bot"]

            if not self.application.updater:
                raise RuntimeError(
                    "Application.run_polling is only available if the application has an Updater."
                )

            def error_callback(exc: TelegramError) -> None:
                self.application.create_task(self.application.process_error(error=exc, update=None))

            default_kwargs = dict()
            passed_kwargs = dict()
            for k in get_func_kwargs(self.application.updater.start_polling):
                if k in bot_cfg:
                    default_kwargs[k] = bot_cfg[k]
                if k in kwargs:
                    passed_kwargs[k] = kwargs.pop(k)
            polling_kwargs = merge_dicts(default_kwargs, passed_kwargs)

            updater_coroutine = self.application.updater.start_polling(**polling_kwargs)
            self._loop = asyncio.get_event_loop()

            if stop_signals is BaseRequest.DEFAULT_NONE and platform.system() != "Windows":
                stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGABRT)

            def _raise_system_exit() -> None:
                raise SystemExit

            try:
                if stop_signals is not BaseRequest.DEFAULT_NONE:
                    for sig in stop_signals or []:
                        self.loop.add_signal_handler(sig, _raise_system_exit)
            except NotImplementedError as exc:
                warn(
                    f"Could not add signal handlers for the stop signals {stop_signals} due to "
                    f"exception `{exc!r}`. If your event loop does not implement `add_signal_handler`,"
                    " please pass `stop_signals=None`."
                )

            try:
                self.loop.run_until_complete(self.application.initialize())
                if self.application.post_init is not None:
                    self.loop.run_until_complete(self.application.post_init(self.application))
                self.loop.run_until_complete(
                    updater_coroutine
                )  # one of updater.start_webhook/polling
                self.loop.run_until_complete(self.application.start())
                self.loop.run_until_complete(self.post_start_callback())
                self.loop.run_forever()
            except (KeyboardInterrupt, SystemExit):
                pass
            except Exception as exc:
                updater_coroutine.close()
                raise exc
            finally:
                self.stop(close_loop=close_loop)

        def stop(self, close_loop: bool = True) -> None:
            """Stop the bot.

            Args:
                close_loop (bool): Whether to close the event loop if it is not running after stopping.

            Returns:
                None
            """
            if self.loop is None:
                raise RuntimeError("There is no event loop running this Application")
            try:
                self.loop.run_until_complete(self.pre_stop_callback())
                if self.application.updater.running:
                    self.loop.run_until_complete(self.application.updater.stop())
                if self.application.running:
                    self.loop.run_until_complete(self.application.stop())
                self.loop.run_until_complete(self.application.shutdown())
                if self.application.post_shutdown is not None:
                    self.loop.run_until_complete(self.application.post_shutdown(self.application))
            finally:
                if close_loop and not self.loop.is_running():
                    self.loop.close()

        @property
        def running(self) -> bool:
            """Bot running state.

            Returns:
                bool: True if the bot is running, False otherwise.
            """
            return self.application.running
