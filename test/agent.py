from google.adk.agents import Agent

root_agent = Agent(
    name="hello_agent",
    model="gemini-2.5-flash",
    instruction="你是一個友善的助理，回答用戶問題。"
)