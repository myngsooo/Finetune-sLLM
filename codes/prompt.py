def prompt():
    prompt_template="""
### Context:
{context}

### Question:
{query}

### Answer:
{answer}
    """
    return prompt_template