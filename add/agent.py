from google.adk.agents import SequentialAgent


# === 匯入子代理 ===
from add.agents.llm.agent import llm_agent
from add.agents.classifier.agent import classifier_agent
from add.agents.weight.agent import weight_agent

# =============== Root Pipeline ===============
# 固定順序：Curator → Historian → 主持人回合制（正/反/極端）→ Social → Evidence → Jury → Synthesizer(JSON)





root_agent = SequentialAgent(
    name="root_pipeline",
    sub_agents=[
        llm_agent,
        classifier_agent,
        weight_agent
    ],
)
