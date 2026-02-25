"""
Blueprint to Furnished Plan - Application Entry Point.
Launches the Gradio server.
"""
from config import AppConfig
from src.ui.gradio_app import create_app


def main():
    config = AppConfig()
    app = create_app(config)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
