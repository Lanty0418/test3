# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .agent import classifier_agent, get_weather, get_current_time
from google.genai.types import Content, Part

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask_agent(query: Query):
    try:
        print(f"收到查詢: {query.message}")
        
        # 創建用戶輸入
        user_input = Content(parts=[Part(text=query.message)])
        
        # 嘗試使用 run_live 方法並正確處理異步生成器
        if hasattr(classifier_agent, 'run_live'):
            try:
                print("使用 run_live 方法...")
                
                # run_live 返回異步生成器，需要迭代處理
                async_gen = classifier_agent.run_live(user_input)
                
                result = []
                event_count = 0
                
                # 迭代異步生成器
                async for event in async_gen:
                    event_count += 1
                    print(f"Event {event_count}: {type(event)} - {event}")
                    
                    # 提取事件內容
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    result.append(part.text)
                                    print(f"提取到文本: {part.text}")
                    
                    # 嘗試 stringify_content 方法
                    elif hasattr(event, 'stringify_content'):
                        try:
                            content_str = event.stringify_content()
                            if content_str:
                                result.append(content_str)
                                print(f"字符串化內容: {content_str}")
                        except Exception as e:
                            print(f"stringify_content 失敗: {e}")
                    
                    # 檢查事件是否有 text 屬性
                    elif hasattr(event, 'text'):
                        result.append(event.text)
                        print(f"事件文本: {event.text}")
                    
                    else:
                        # 記錄未知事件格式
                        event_str = str(event)
                        print(f"未知事件格式: {event_str}")
                        result.append(event_str)
                
                final_result = "".join(result) if result else f"處理了 {event_count} 個事件，但沒有提取到文本"
                print(f"最終結果: {final_result}")
                
                return {
                    "result": final_result,
                    "method": "run_live_processed",
                    "event_count": event_count
                }
                
            except Exception as e:
                print(f"run_live 處理失敗: {e}")
                # 繼續到備用方法
        
        # 備用方法：直接使用工具函數
        print("使用備用工具函數方法...")
        message = query.message.lower()
        
        if any(keyword in message for keyword in ["weather", "天氣", "溫度"]):
            if "new york" in message or "紐約" in message:
                result = get_weather("New York")
                return {
                    "result": result["report"] if result["status"] == "success" else result["error_message"],
                    "method": "direct_weather_tool"
                }
            else:
                return {"result": "請詢問 New York 的天氣", "method": "weather_help"}
        
        elif any(keyword in message for keyword in ["time", "時間", "幾點"]):
            if "new york" in message or "紐約" in message:
                result = get_current_time("New York")
                return {
                    "result": result["report"] if result["status"] == "success" else result["error_message"],
                    "method": "direct_time_tool"
                }
            else:
                return {"result": "請詢問 New York 的時間", "method": "time_help"}
        
        else:
            return {
                "result": "請詢問 New York 的天氣或時間。例如：'New York 天氣如何？' 或 'New York 現在幾點？'",
                "method": "help"
            }
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "ADK Agent API is running!", "agent": classifier_agent.name}