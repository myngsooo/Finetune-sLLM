def get_prompt(mode='test'):
    if mode=='test':
        prompt_template="""
### System
You are an AI Assistant. You must give the answer by following Context. While performing the task think step-by-step and justify your steps. Think like you are answering to a five-year-old.

### Context:
{context}

### Question:
{query}

### Answer
"""
    elif mode=='train':
        prompt_template="""
### System
You are an AI Assistant. You must give the answer by following Context. While performing the task think step-by-step and justify your steps. Think like you are answering to a five-year-old.

### Context:
{context}

### Question:
{query}

### Answer:
{answer}
"""
    return prompt_template
