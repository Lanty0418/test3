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
Politics_agent2  = LlmAgent(
    name="Politics_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於政府政策與政治新聞查證的研究員，具備從多個視角驗證新聞內容真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **政策視角（policy）**
        - 是否提及具體政策內容？
        - 是否明確指出執行單位與執行時間？
        - 是否有政策名稱或改革方向？

        2. **法案視角（legislation）**
        - 是否提及具體法案？
        - 是否說明法案當前狀態（通過、修法中、草案階段）？
        - 是否引用條文正確？

        3. **發言視角（statement）**
        - 是否提及政治人物的公開發言或主張？
        - 是否清楚指出發言者、時間、場合？
        - 是否存在斷章取義？

        4. **政府立場視角（official_position）**
        - 是否提及行政或立法機關、政黨立場或評論？
        - 是否引用正式聲明或公告？

        5. **時間視角（time）**
        - 是否指出事件發生時間（YYYY/MM/DD）- 對應事件?
        - 是否有時序錯置或模糊？

        6. **地點視角（location）**
        - 是否指出具體地點？
        - 是否與事件性質（如選舉、立法院會期）相關？
        - 地點 : 發生事件

        7. **爭議與矛盾視角（controversies）**
        - 是否出現矛盾觀點、不一致陳述？
        - 是否有引用錯誤、缺乏來源或情緒化用語？

        ---
        現在請判斷以下新聞文本的真偽，並依照上述七個視角進行分析。
        判斷文本為:
        {_init_session}

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
Politics_agent = SequentialAgent(
    name="Politics_agent",
    sub_agents=[Politics_agent2],
)