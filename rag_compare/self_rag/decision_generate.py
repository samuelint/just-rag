class DecideToGenerate:
    def __call__(self, state):
        filtered_documents = state["documents"]

        if not filtered_documents:
            return "all_documents_not_relevant_to_question"
        else:
            return "relevant_documents"
