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
Lifestyle_agent2  = LlmAgent(
    name="Lifestyle_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於民生議題與消費新聞查證的研究員，具備從多個視角驗證新聞內容真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **商品與服務視角（product_service）**  
        - 是否介紹或推薦特定產品、品牌、服務？  
        - 是否聲稱具有效能、優惠、限量等吸引消費的特徵？  

        2. **價格與折扣視角（pricing_discount）**  
        - 是否提到商品價格、折扣、補助、退稅、限時優惠等？  
        - 是否標明優惠條件、期間、是否誤導？  

        3. **健康與安全視角（health_safety）**  
        - 是否與健康、醫療、食品安全有關？  
        - 是否提及科學依據、檢驗單位、認證文件等？  

        4. **來源與品牌視角（source_brand）**  
        - 是否標明品牌、製造商、原廠授權？  
        - 是否涉及冒用、仿冒或來源不明？  

        5. **時間與適用性視角（time_applicability）**  
        - 訊息是否仍有效？  
        - 有無標明時間、地點、對象？  
        - 是否為過期資訊或錯置背景？  

        6. **政策與規範視角（policy_regulation）**  
        - 是否提及政府補貼、消費補助、法規限制？  
        - 是否清楚標示公告來源與施行狀況？  

        7. **誇張與誤導視角（misleading_claims）**  
        - 是否使用誇張語彙、標題黨、偽科學詞彙或情緒化描述？  
        - 是否斷章取義或有誤導成分？

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
Lifestyle_agent = SequentialAgent(
    name="Lifestyle_agent",
    sub_agents=[Lifestyle_agent2],
)