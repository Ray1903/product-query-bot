import google.generativeai as genai
from app.config import GEMINI_API_KEY
from langchain.prompts import PromptTemplate

class ResponderAgent:
    """
    Agent that generates answers using the Gemini model.
    Formats a prompt with conversation history, product context, and the user query.
    """
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")  

        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "query"],
            template=(
                "You are a helpful product assistant.\n"
                "Use the conversation history and product context to answer.\n"
                "If the answer isn't clear, say 'I don't know'.\n\n"
                "Conversation History:\n{history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n"
                "Answer:"
            ),
        )

    def __call__(self, state):
        """
        Generates an answer using the Gemini model, given the docs, history, and query in the state.
        Genera una respuesta usando Gemini, dado el contexto y el historial.
        """
        docs = state.get("docs", [])
        context = "\n".join([doc.page_content for doc in docs])
        history = state.get("history", "")
        prompt = self.prompt_template.format(
            history=history,
            context=context,
            query=state["query"]
        )

        # Call Gemini model
        # Llamada a Gemini
        response = self.model.generate_content(prompt)
        answer = response.text

        state["answer"] = answer
        return state
