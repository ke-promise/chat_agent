"""兼容旧导入路径的 AgentLoop 模块。

当前项目真正的业务编排器位于 ``chat_agent.loop.AgentLoop``。保留这个文件是为了兼容
早期代码或外部脚本中 ``chat_agent.agent.loop`` 的导入方式。
"""

from chat_agent.loop import AgentLoop

__all__ = ["AgentLoop"]
