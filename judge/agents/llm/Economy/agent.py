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
Economy_agent2  = LlmAgent(
    name="Economy_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於經濟與財經新聞查證的研究員，具備從多個觀點驗證市場訊息與數據主張真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **經濟數據視角（economic_data）**  
        - 是否提及具體的經濟數據（GDP、CPI、失業率、出口數字等）？  
        - 是否數據有明確來源、合理性？  

        2. **金融市場動態視角（market_activity）**  
        - 是否涉及股市、匯市、債市或大宗商品波動？  
        - 是否具體指出市場指數或價格走勢？  

        3. **企業與產業動態視角（corporate_industry）**  
        - 是否涉及特定企業、產業的動態或財報？  
        - 是否包含併購、破產、投資、裁員等訊息？  

        4. **政策與監管視角（policy_regulation）**  
        - 是否提及政府、央行或金融監管機構的政策？  
        - 是否涉及利率調整、稅制改革、監管措施？  

        5. **專家與研究機構觀點（expert_opinion）**  
        - 是否引用分析師、學者或研究機構的評論？  
        - 是否有數據支撐或只是推測？  

        6. **時間與趨勢視角（time_trend）**  
        - 是否標註具體日期（YYYY/MM/DD）或時間範圍？  
        - 是否指出經濟趨勢（上升、下降、持平、預測）？  

        7. **爭議與矛盾視角（controversies）**  
        - 是否存在不同解讀、矛盾數據或誇張描述？  
        - 是否缺乏來源或帶有情緒化語氣？  

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
Economy_agent = SequentialAgent(
    name="Economy_agent",
    sub_agents=[Economy_agent2],
)