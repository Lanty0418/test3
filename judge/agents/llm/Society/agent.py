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
Society_agent2  = LlmAgent(
    name="Society_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於社會事件與公共議題查證的研究員，具備從多個視角驗證新聞內容真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **當事人視角（people_involved）**
        - 是否提及受害者、加害者、目擊者或執法人員？
        - 是否明確指出身分、年齡、職業或機構隸屬？

        2. **事件內容視角（event_description）**
        - 是否明確描述事件經過（誰、做了什麼、造成何結果）？
        - 是否存在簡化、誇張或缺乏背景補充？

        3. **時間視角（time）**
        - 是否標註事件發生時間（YYYY/MM/DD）？
        - 是否合理、明確，是否與其他資訊矛盾？

        4. **地點視角（location）**
        - 是否標示具體地點？
        - 是否與事件性質相符？
        - 地點 : 發生事件

        5. **官方回應視角（official_response）**
        - 是否引述警消、政府、地方機關等正式說法？
        - 是否有回應來源、聲明或數據？

        6. **社會反應視角（public_reaction）**
        - 是否提及群眾反應、社群討論、輿論或倡議團體聲明？
        - 是否偏向單一立場？

        7. **爭議與矛盾視角（controversies）**
        - 是否出現不同說法、矛盾或缺乏佐證？
        - 是否有誤導性數據或情緒化描述？

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
Society_agent = SequentialAgent(
    name="Society_agent",
    sub_agents=[Society_agent2],
)