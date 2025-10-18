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
Entertainment_agent2  = LlmAgent(
    name="Entertainment_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於國際新聞與地緣政治查證的研究員，具備從多國觀點與跨文化脈絡判斷新聞內容真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **人物與關係視角（people_relationship）**  
        - 是否提及藝人、經紀人、合作對象等？  
        - 是否誤植、虛構或未經確認？  

        2. **事件與情節視角（event_description）**  
        - 是否清楚描述事件經過、衝突起因與後果？  
        - 是否存在誤導或情節誇大？  

        3. **來源與爆料視角（source_validity）**  
        - 是否指出新聞來源（媒體、報導、社群）？  
        - 是否可信？是否為匿名或未經證實的資訊？  

        4. **發言與回應視角（statement_response）**  
        - 是否引述當事人、經紀公司、主辦單位等回應？  
        - 有無失真或語境錯誤？  

        5. **時間與背景視角（time_context）**  
        - 是否標示具體時間、活動、地點？  
        - 是否為舊聞誤傳或時間錯置？  

        6. **誇張與炒作視角（sensationalism）**  
        - 是否使用情緒化詞語如「爆料」、「決裂」、「鬧翻」等？  
        - 是否標題黨或語意模糊？  

        7. **爭議與矛盾視角（contradictions）**  
        - 是否出現不同單位、藝人對事件說法相矛盾？  
        - 是否有誤導敘述或未提供佐證？

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
Entertainment_agent = SequentialAgent(
    name="Entertainment_agent",
    sub_agents=[Entertainment_agent2],
)