from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from app.agents.retriever import RetrieverAgent
from app.agents.memory import MemoryAgent
from app.agents.responder import ResponderAgent


# Define the state schema
class BotState(TypedDict):
    user_id: str
    query: str
    docs: List
    answer: str
    history: str


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline using LangGraph.
    Connects retriever, memory, and responder agents in a graph to process queries.
    """
    def __init__(self):
        self.retriever_agent = RetrieverAgent()
        self.memory_agent = MemoryAgent()
        self.responder_agent = ResponderAgent()

        # New StateGraph constructor with state_schema
      
        self.graph = StateGraph(
            state_schema=BotState,
            input_keys=["user_id", "query"],
            output_keys=["answer"]
        )

        # Add nodes to the graph

        self.graph.add_node("retriever", self.retriever_agent)
        self.graph.add_node("memory", self.memory_agent)
        self.graph.add_node("responder", self.responder_agent)

        self.graph.set_entry_point("retriever")
        self.graph.add_edge("retriever", "memory")
        self.graph.add_edge("memory", "responder")
        self.graph.add_edge("responder", END)

        self.app = self.graph.compile()

    def index_documents(self, documents: list[str]):
        """
        Indexes a list of documents using the retriever agent.
        """
        self.retriever_agent.index_documents(documents)

    def query(self, user_id: str, query: str):
        """
        Processes a user query through the RAG pipeline and returns the answer.
        """
        state = {"user_id": user_id, "query": query}
        final_state = self.app.invoke(state)
        self.memory_agent.update_answer(user_id, final_state["answer"])
        return final_state["answer"]
