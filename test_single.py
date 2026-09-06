"""Compatibility entrypoint; production runner lives in duplexchat_pipe.single_audio."""
from duplexchat_pipe.single_audio import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
