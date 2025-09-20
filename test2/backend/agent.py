# agents.py
from google.adk.agents import Agent,SequentialAgent,LlmAgent
import datetime
from zoneinfo import ZoneInfo

# 天氣工具
def get_weather(city: str) -> dict:
    if city.lower() == "new york":
        return {"status": "success",
                "report": "The weather in New York is sunny, 25°C (77°F)."}
    else:
        return {"status": "error",
                "error_message": f"Weather info for {city} not available."}

# 時間工具
def get_current_time(city: str) -> dict:
    if city.lower() == "new york":
        tz = ZoneInfo("America/New_York")
        now = datetime.datetime.now(tz)
        return {"status": "success",
                "report": f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S')}"}
    else:
        return {"status": "error",
                "error_message": f"Timezone for {city} not available."}

# 建立 agents
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
    description="Get time info",
    instruction="I can provide current time information for a city.",
    tools=[get_current_time]
)

classifier_agent = SequentialAgent(
    name="bert_classifier_agent",
    sub_agents=[weather_agent, time_agent],
)