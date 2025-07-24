class MemoryAgent:
    """
    Agent that manages in-memory conversation history for each user.
    Stores a list of queries and answers per user_id, and provides conversation history as context.
    """
    def __init__(self):
        # In-memory dictionary: {user_id: [history of queries and answers]}
        # Diccionario en memoria: {user_id: [historial de queries y respuestas]}
        self.memory = {}

    def __call__(self, state):
        """
        Adds the current query to the user's history and attaches the conversation history to the state.
        Agrega el query actual al historial y pasa el historial como contexto adicional.
        """
        user_id = state["user_id"]
        query = state["query"]

        if user_id not in self.memory:
            self.memory[user_id] = []

        # Add the current query to the history
        # Agregamos el query actual al historial
        self.memory[user_id].append({"query": query})

        # Pass the history as additional context
        # Pasamos el historial como contexto adicional
        state["history"] = "\n".join(
            [f"Q: {m['query']} A: {m.get('answer', '')}" for m in self.memory[user_id] if m.get("answer")]
        )
        return state

    def update_answer(self, user_id, answer):
        """
        Updates the last answer in the user's history.
        Actualiza la última respuesta en el historial.
        """
        if user_id in self.memory and self.memory[user_id]:
            self.memory[user_id][-1]["answer"] = answer
