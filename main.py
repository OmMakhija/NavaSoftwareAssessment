from dotenv import load_dotenv
from app.ui import build_ui

load_dotenv()

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
