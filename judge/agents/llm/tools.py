"""主持人相關工具：提供退出迴圈與統計指標"""

from typing import Any
from pydantic import BaseModel
from google.adk.tools.agent_tool import AgentTool

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from .Politics import Politics_agent
from .Society import Society_agent
from .Economy import Economy_agent
from .Lifestyle import Lifestyle_agent
from .International import International_agent
from .Technology import Technology_agent
from .Entertainment import Entertainment_agent
from .Sports import Sports_agent
from .Local import Local_agent




Politics_tool = AgentTool(Politics_agent)
Politics_tool.name = "call_Politics"
Economy_tool = AgentTool(Economy_agent)
Economy_tool.name = "call_Economy"
Society_tool = AgentTool(Society_agent)
Society_tool.name = "call_Society"

Lifestyle_tool = AgentTool(Lifestyle_agent)
Lifestyle_tool.name = "call_Lifestyle"
International_tool = AgentTool(International_agent)
International_tool.name = "call_International"
Technology_tool = AgentTool(Technology_agent)
Technology_tool.name = "call_Technology"
Entertainment_tool = AgentTool(Entertainment_agent)
Entertainment_tool.name = "call_Entertainment"
Society_tool = AgentTool(Society_agent)
Society_tool.name = "call_Society"
Sports_tool = AgentTool(Sports_agent)
Sports_tool.name = "call_Sports"
Local_tool = AgentTool(Local_agent)
Local_tool.name = "call_Local"



