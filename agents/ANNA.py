import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from utils.llm_init import LLM_initializer
class ANNA():
    def __init__(self):
        llm = LLM_initializer()
        self.memory = InMemorySaver()
        self.prompt = "None"
        self.agent = create_agent(llm.model, system_prompt=self.prompt, checkpointer=self.memory)


print("--- Chat with the agent (type 'exit' to stop) ---")
evaluator = ANNA()

while True:
    user_input = input("You: ")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("Exiting...")
        break

    try:
        # stream agent response
        for event in evaluator.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"}},  
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()  # prints token-by-token streaming

    except Exception as e:
        print(f"Error: {e}")
