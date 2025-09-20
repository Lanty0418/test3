# app.py

import asyncio
import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI
from pydantic import BaseModel
from google.adk.agents import Agent,SequentialAgent,LlmAgent
from google.adk.app import App
from google.adk.events import Event
from google.genai.types import Content, Part

# --- 你的工具函式 ---
def get_weather(city: str) -> dict:
    if city.lower() == "new york":
        return {"status": "success",
                "report": "The weather in New York is sunny, 25°C (77°F)."}
    else:
        return {"status": "error",
                "error_message": f"Weather info for {city} not available."}

def get_current_time(city: str) -> dict:
    if city.lower() == "new york":
        tz = ZoneInfo("America/New_York")
        now = datetime.datetime.now(tz)
        return {"status": "success",
                "report": f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S')}"}
    else:
        return {"status": "error",
                "error_message": f"Timezone for {city} not available."}

# --- 建立 agents ---
weather_agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash",
    description="Get weather info",
    instruction="I can provide weather information for a city.",
    tools=[get_weather]
)

time_agent = LlmAgent(
    name="time_agent",
    model="gemini-2.0-flash",
    description="I can provide current time information for a city.",
    tools=[get_current_time]
)

# 使用 LlmAgent 來協調兩個工具
# LlmAgent 會根據使用者輸入，自動判斷要呼叫 get_weather 或 get_current_time
llm_classifier_agent = LlmAgent(
    name="llm_classifier_agent",
    model="gemini-2.0-flash",
    instruction="根據使用者的輸入，判斷並呼叫合適的工具來回覆。",
    tools=[get_weather, get_current_time]
)

# 為了簡化，直接將 llm_classifier_agent 設定為根代理
# 你也可以使用 SequentialAgent 來編排，但對於工具選擇，LlmAgent 更適合
root_agent = llm_classifier_agent

# 建立 ADK App 物件，這是執行代理的入口
adk_app = App(
    name="my_adk_app",
    root_agent=root_agent
)

# --- 建立 FastAPI 應用程式 ---
api_app = FastAPI()

# 定義 API 請求的輸入格式
class AskRequest(BaseModel):
    message: str

# 建立 API 路由
@api_app.post("/ask")
async def ask_agent(request: AskRequest):
    try:
        # 將輸入訊息轉換成 ADK 框架期望的格式
        user_input_content = Content(parts=[Part(text=request.message)])

        # 呼叫 ADK 的非同步 run_async() 方法來執行代理
        async_generator = adk_app.run_async(user_input_content)

        # 逐一處理非同步生成器回傳的事件
        full_response = ""
        async for event in async_generator:
            if isinstance(event, Event) and event.content:
                for part in event.content.parts:
                    if part.text:
                        full_response += part.text
        
        return {"response": full_response}
    except Exception as e:
        return {"error": str(e)}