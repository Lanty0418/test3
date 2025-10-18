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
International_agent2  = LlmAgent(
    name="International_agent2",
    model="gemini-2.0-flash",
    instruction=(
        """您是一位專精於國際新聞與地緣政治查證的研究員，具備從多國觀點與跨文化脈絡判斷新聞內容真偽的能力。請根據以下新聞內容與整理後資訊，依照指定格式進行查核推理。
        【注意】
        根據以下規則進行文本真為判斷。
        result_label請傳回以下標籤內容之一：完全正確、部分正確、完全錯誤、部分錯誤、無法判斷。
        判斷文本真假時，如果文本中有提及時間點的話，就請以文本的時間作為基準來判斷真假；如果文本中沒有提及時間點，則請以今天的日期作為基準來判斷真假。
        【規則】
        
        請依下列七個推理視角輸出資訊，每個視角請以條列方式表述。  

        1. **消息來源與引用視角（source_attribution）**  
        - 是否明確指出消息來源（如路透社、官媒、外交部記者會）？  
        - 是否為一手消息或二手轉述？  
        - 來源是否可信、具可查性？  

        2. **時間與地點視角（time_location）**  
        - 是否標示具體事件的時間與發生地點？  
        - 與當時地區狀況是否相符？  
        - 是否為舊聞誤傳？  

        3. **數據與證據視角（data_evidence）**  
        - 是否引用照片、影片、報告、調查結果？  
        - 資料來源是否可信？  
        - 是否有雙方證據或背景說明？  

        4. **國家與立場視角（national_stance）**  
        - 是否涉及國際爭議、制裁、外交立場或軍事對抗？  
        - 是否偏向特定立場、陣營語氣或宣傳內容？  

        5. **文化與翻譯視角（cultural_translation）**  
        - 是否因翻譯錯誤或文化背景不符造成誤解？  
        - 是否引用語意失真、與文化、宗教、歷史脈絡不符？  

        6. **國際組織與第三方觀點視角（multilateral_validation）**  
        - 是否提及聯合國、WHO、NATO 等中立組織觀點？  
        - 是否引用第三方或其他國家反應佐證？  

        7. **誇張與誤導視角（misleading_claims）**  
        - 是否出現誇張標題、情緒化語言（如「全球震驚」、「全面毀滅」）？  
        - 是否與內容不符或有誤導疑慮？

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
International_agent = SequentialAgent(
    name="International_agent",
    sub_agents=[International_agent2],
)