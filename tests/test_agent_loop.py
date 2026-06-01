from macro_ds import agent
from macro_ds.schema import Message, ToolCall


class _ScriptedDriver:
    def __init__(self, scripted):
        self.scripted = scripted
        self.i = 0
        self.usage = {}

    def step(self, messages, tool_schemas):
        m = self.scripted[self.i]
        self.i += 1
        return m


def test_loop_executes_tools_then_terminates(mocker):
    scripted = [
        Message(
            role="assistant",
            content="Thought: search.",
            tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "x"})],
        ),
        Message(role="assistant", content="Final report: yields up. [http://u]"),
    ]
    mocker.patch.object(agent, "execute_tool", return_value="RESULT: http://u")
    trace = agent.run_agent(
        "Where are yields headed?", asof_date="2026-06-01", driver=_ScriptedDriver(scripted), max_steps=12
    )
    assert trace.final_report.startswith("Final report")
    assert trace.steps == 2
    assert trace.completed is True
    # the tool result was appended to the transcript with the right call id
    tool_msgs = [m for m in trace.messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].tool_call_id == "1"


def test_loop_force_stops_at_max_steps(mocker):
    class LoopDriver:
        usage = {}

        def step(self, messages, tool_schemas):
            return Message(
                role="assistant",
                content="Thought: again.",
                tool_calls=[ToolCall(id="x", name="web_search", arguments={"query": "x"})],
            )

    mocker.patch.object(agent, "execute_tool", return_value="RESULT")
    trace = agent.run_agent("q", asof_date="2026-06-01", driver=LoopDriver(), max_steps=3)
    assert trace.final_report is None
    assert trace.completed is False
    assert trace.steps == 3


def test_tool_error_increments_counter(mocker):
    scripted = [
        Message(
            role="assistant",
            content="Thought.",
            tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "x"})],
        ),
        Message(role="assistant", content="Final report: done. [http://u]"),
    ]
    mocker.patch.object(agent, "execute_tool", return_value="ERROR: boom")
    trace = agent.run_agent("q", asof_date="2026-06-01", driver=_ScriptedDriver(scripted), max_steps=12)
    assert trace.tool_errors == 1
