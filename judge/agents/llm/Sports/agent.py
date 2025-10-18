from typing import Optional
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types

# -------- Schema --------
class FactCheckInput(BaseModel):
    news_text: str = Field(description="待驗證的新聞文章內容")
    news_date: Optional[str] = Field(
        default=None,
        description="新聞發佈日期，若無則以今天的日期判斷"
    )

class FactCheckOutput(BaseModel):
    analysis: str = Field(description="完整分析結果")
    classification: str = Field(description="真假分類：「完全正確」、「部分正確」、「完全錯誤」、「完全錯誤」、「無法判斷」")

# -------- Agent 定義 --------
Sports_agent2  = LlmAgent(
    name="Sports_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於體育新聞與運動紀錄查證的研究員，具備從多個觀點驗證體育報導真偽與爭議的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **官方與媒體來源視角（official_source）**  
        - 是否援引聯盟、公協會、主辦單位、賽事官網、可信媒體等？  
        - 是否來源不明、僅來自社群或轉載內容？  

        2. **選手與教練視角（athletes_coaches）**  
        - 是否提及參賽者、教練、裁判？  
        - 是否明確指出其身分、所屬單位、紀錄或傷勢？  
        - 是否有誤引或誇張描述？  

        3. **時間視角（time）**  
        - 是否標明比賽或事件時間（YYYY/MM/DD）？  
        - 是否與官方時程或其他報導矛盾？  

        4. **地點視角（location）**  
        - 是否提及具體比賽場地或事件地點？  
        - 是否真實存在且與比賽性質相符？  
        - 地點 : 發生事件  

        5. **數據與紀錄視角（stats_records）**  
        - 是否引用球員或隊伍數據、紀錄、技術統計？  
        - 資料是否來自可信來源？  
        - 有無錯誤或過時資訊？  

        6. **賽事與結果視角（event_results）**  
        - 是否清楚描述比賽名稱、對戰組合、結果、得分等？  
        - 有無誤報、混用舊聞、尚未確認消息？  

        7. **爭議與誤導視角（controversies）**  
        - 是否涉及爭議判決、禁賽、假球或賽事操縱？  
        - 是否使用誇張、戲劇化或誤導語言？

        ---

        ## 請強制依以下 JSON 格式回傳結果：

        ```json
        {
        "result_label": 判斷結果，請回傳 完全正確、部分正確、完全錯誤、部分錯誤、無法判斷 以上標籤之一,
        "viewpoints": {
            "economic_data": [ "..." ],
            "market_activity": [ "..." ],
            "corporate_industry": [ "..." ],
            "policy_regulation": [ "..." ],
            "expert_opinion": [ "..." ],
            "time_trend": [ "..." ],
            "controversies": [ "..." ]
        },
        "viewpoint_sum": "一句統整七個視角的關鍵性觀點，清楚解釋為何判斷為真或假，以輸入限有資料判斷輸入新聞文本本身的真假就好。"
        }"""
    ),
    tools=[GoogleSearchTool()],
    #input_schema=FactCheckInput,
    #output_schema=FactCheckOutput,
    output_key="fact_check_result",
)


# -------- Step 3: Sequential pipeline --------
Sports_agent = SequentialAgent(
    name="Sports_agent",
    sub_agents=[Sports_agent2],
)