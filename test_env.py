import os

from dotenv import load_dotenv

load_dotenv()

has_key = bool(os.getenv("OPENAI_API_KEY"))
print("OPENAI_API_KEY configured:", has_key)
