def get_prompt(mode='test'):
    if mode=='test':
        prompt_template="""
### Context:
{context}

### Question:
{query}
"""
    elif mode=='train':
        prompt_template="""
### Context:
{context}

### Question:
{query}

### Answer:
{answer}
"""
    return prompt_template