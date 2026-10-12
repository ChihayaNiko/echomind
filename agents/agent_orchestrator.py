import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 账单/退款
    ESCALATION = "escalation" # 人工升级（占位）

@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)

@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            messages.append({"role": "user", "content": f"[结构化实体]\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return extract_text_content(resp.content)

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)

class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    system_prompt = (
        "你是 EchoMind 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    system_prompt = (
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    system_prompt = (
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    )

class AgentOrchestrator:
    """
        多 Agent 编排器。

        路由逻辑（三层）：
          1. 意图 → Agent 类型映射
          2. 同类多实例时按 routing_score() 选最优
          3. 专属 Agent 失败时降级到 GeneralAgent
        """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
        IntentCategory.BILLING: AgentType.BILLING,
        IntentCategory.REFUND: AgentType.BILLING,
        IntentCategory.INVOICE: AgentType.BILLING,
        IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
        IntentCategory.ACCOUNT: AgentType.BILLING,
        IntentCategory.ACCOUNT_SECURITY: AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL:   [GeneralAgent(client, model, skill_manager)],
            AgentType.TECHNICAL: [TechnicalAgent(client, model, skill_manager)],
            AgentType.BILLING:   [BillingAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
