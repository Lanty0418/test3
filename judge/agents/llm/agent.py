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
from .tools import (
    Politics_tool,
    Society_tool,
    Economy_tool,
    Lifestyle_tool,
    International_tool,
    Technology_tool,
    Entertainment_tool,
    Sports_tool,
    Local_tool,
)

# -------- Agent 定義 --------
text_classfier_tool_agent  = LlmAgent(
    name="text_classfier_agent",
    model="gemini-2.5-flash",
    instruction=(
        """請幫我針對state['_init_session']的內容進行分類
        請幫我分成以下類別
        政治 (Politics): 報導國內外政府、政黨、選舉、政策、法案、政治人物動態等相關消息。
        社會 (Society): 報導社會上發生的事件、公共議題、犯罪、交通事故、天災人禍、弱勢群體、社會福利、倫理道德、人情趣味等。
        經濟 (Economy): 報導產業經濟、商業動態、投資理財、金融市場、股市匯市、企業經營、貿易、就業等相關消息。
        生活 (Lifestyle): 報導與民眾日常生活息息相關的資訊，如購物、食品安全、健康醫療、氣象交通、旅遊資訊、消費趨勢等。
        國際 (International): 報導國外發生的重要事件、國際關係、戰爭衝突、外交談判、國際組織動態、全球性議題等。
        科技 (Technology): 報導科學研究、技術發展、網路趨勢、電子產品、人工智慧、太空探索等相關消息。
        娛樂 (Entertainment): 報導演藝圈動態、明星八卦、電影戲劇、音樂、頒獎典禮、文化藝術活動等。
        體育 (Sports): 報導國內外體育賽事、運動員動態、體育新聞、健身保健等。
        地方 (Local): 報導特定地區發生的新聞事件、地方政策、社區活動、風土民情、地方產業等。
        
        請只回傳分類結果的中文名稱，例如「政治」、「社會」、「經濟」、「生活」、「國際」、「科技」、「娛樂」、「體育」、「地方」中的一個。
        請將結果存入 state['news_category'] 中。"""

    ),
    tools=[GoogleSearchTool()],
    #input_schema=FactCheckInput,
    #output_schema=FactCheckOutput,
    output_key="news_category",
)

fact_check_agent = LlmAgent(
    name="fact_check_agent",
    model="gemini-2.5-flash",
    instruction=(
        """你負責呼叫相對應的工具來進行判斷，切勿自行乎做判斷。
        請依照 state['news_category'] 的結果，來判斷要叫哪一個工具。
        對照表如下：
        政治 (Politics) -> Politics_tool
        社會 (Society) -> Society_tool
        經濟 (Economy) -> Economy_tool
        生活 (Lifestyle) -> Lifestyle_tool
        國際 (International) -> International_tool
        科技 (Technology) -> Technology_tool
        娛樂 (Entertainment) -> Entertainment_tool
        體育 (Sports) -> Sports_tool
        地方 (Local) -> Local_tool
        
        請將呼叫工具輸出結果直接存入 state['fact_check_result'] 中。"""
    ),
    #input_schema=FactCheckInput,
    tools= [Politics_tool,Society_tool,Economy_tool,Lifestyle_tool,International_tool,Technology_tool,Entertainment_tool,Sports_tool,Local_tool],
    output_key="fact_check_result_json",
    generate_content_config=types.GenerateContentConfig(temperature=0.4),
)

# -------- Step 3: Sequential pipeline --------
llm_agent = SequentialAgent(
    name="fact_check_agent",
    sub_agents=[text_classfier_tool_agent, fact_check_agent],
)