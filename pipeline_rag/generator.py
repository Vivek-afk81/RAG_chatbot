from huggingface_hub import InferenceClient

def answer_question(question, context, api_key):

    client = InferenceClient(
        api_key=api_key
    )

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    return response