"""dsa-lite 核心模块。

模块分工：
    fetchers     多源行情获取（降级链）
    indicators   技术指标与市场快照（纯标准库）
    strategies   策略实现（规则 / LLM / 台账回放）
    signals      信号模型与台账、缓存
    llm          LLM 分析器（OpenAI 兼容）
    backtest     回测引擎（核心）
    validator    策略体检与过拟合检测
    report       报告渲染
    notify       多通道推送
"""

__version__ = "1.0.0"
