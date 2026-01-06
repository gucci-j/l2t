import random


def _predict_context_given_one_word(
    text: str, 
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    include_starting_word: bool = False,
) -> tuple:
    """
    Generate prompt-completion pair for predicting the context given one word.
    
    Args:
        text (str): Input text to be manipulated.
        no_instruction_variation (bool): If True, the function will not use different instruction variations.
        no_instruction (bool): If True, the function will not use any instruction.
        include_starting_word (bool): If True, completion will include the starting word.
        
    Returns:
        tuple: A tuple containing:
            - String with the prompt
            - String with the completion
    
    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_second_half(text)
        >>> print(result)
        ('Generate the continuation of the text based on the beginning word: The", "cat sat on the mat')
    """
    instruction_variants = (
        "Generate the text starting with this word.",
        "Complete a passage that begins with this word.",
        "Write a text that starts with the following word.",
        "Create content continuing from this initial word.",
        "Compose text using this word as the beginning.",
        "Develop text that follows from this starting word.",
        "Expand this word into a complete passage.",
        "Form a coherent text starting with this word.",
        "Continue writing from this first word.",
        "Construct a text using this word as your starting point.",
        "Build a narrative around this opening word.",
        "Start with this word and create a coherent text.",
        "Use this word as the first in your text.",
        "Begin a passage with the provided word.",
        "Take this word and extend it into a full text.",
        "Craft a story starting with this word.",
        "Using this first word, write a complete passage.",
        "From this initial word, develop a full text.",
        "This word is your starting point - continue the text.",
        "Extend this opening word into a full passage."
    )
    if no_instruction_variation:
        instruction = "Generate the text starting with this word."
    else:
        instruction = random.choice(instruction_variants)
    words = text.split(" ")
    if include_starting_word:
        completion = " ".join(words)
    else:
        completion = " ".join(words[1:])
    prompt = words[0]
    if no_instruction:
        return prompt, completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, prompt = prompt, instruction
    return f"{instruction}\n\n{prompt}", completion


def generate_one_word_completion_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for the task of predicting the context given one word.
    """
    task_funcs = {
        "predict_context_given_one_word": _predict_context_given_one_word,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "predict_context_given_one_word"
            results = [
                task_funcs[task_variant](text, kwargs["no_instruction_variation"], kwargs["no_instruction"], kwargs["include_starting_word"])
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text, kwargs["no_instruction_variation"], kwargs["no_instruction"], kwargs["include_starting_word"])
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        if use_only_primary:
            task_variant = "predict_context_given_one_word"
        else:
            task_variant = random.choice(task_variants)
        prompt, completion = task_funcs[task_variant](example["text"], kwargs["no_instruction_variation"], kwargs["no_instruction"], kwargs["include_starting_word"])
        return {"prompt": prompt, "completion": completion}
