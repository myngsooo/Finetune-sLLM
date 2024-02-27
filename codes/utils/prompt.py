def basic_prompt():
    prompt_template="""
### Context:
{context}

### Question:
{query}

### Answer:
{answer}
    """
    return prompt_template

def prompt_test():
    prompt_template="""
### Context: 
### Question:
### Answer: 
"""
    return prompt_template