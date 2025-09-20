# my_local_agent/agent.py
import os
from google.adk.agents import LlmAgent
from typing import Annotated

# 定義一個簡單的函式工具，用來處理輸入資料
def get_hello_message(city: Annotated) -> str:
    """
    Retrieves a greeting message for a specified city.
    """
    return f"你好，來自 {city} 的使用者！很高興能為您服務。"

# 建立一個 ADK 代理實例
hello_world_agent = LlmAgent(
    name="hello_world_agent",
    model="gemini-2.5-flash",
    description="An agent that provides a greeting for a city.",
    instruction="""
        你是一個專門提供城市問候語的助理。
        當使用者詢問關於一個城市時，請使用 `get_hello_message` 工具來獲取訊息，然後以友善的語氣回覆。
    """,
    tools=[get_hello_message] # 將我們的函式註冊為一個工具
)

root_agent = hello_world_agent