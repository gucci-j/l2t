import random


def _predict_second_half(
    text: str, 
    no_instruction_variation: bool = False, 
    no_instruction: bool = False,
) -> tuple:
    """
    Generate prompt-completion pair for predicting the second half of the text.
    
    Args:
        text (str): Input text to be manipulated.
        no_instruction_variation (bool): If True, the function will not use different instruction variations.
        no_instruction (bool): If True, the function will not use any instruction.
        
    Returns:
        tuple: A tuple containing:
            - String with the prompt
            - String with the completion
    
    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_second_half(text)
        >>> print(result)
        ('Given the first half of the text, complete the second half: \n\nThe cat sat on', ' the mat')
    """
    instruction_variants = (
        "Continue this text:",
        "Given this beginning, how would the text continue?",
        "Complete this text by continuing from where it left off:",
        "Here's the start of a passage. What comes next?",
        "Finish this text in the most natural way possible:",
        "How would you continue this text from where it ends?",
        "This text needs an ending. How would you complete it?",
        "Read this opening and write its continuation:",
        "Given the opening part, finish this text coherently:",
        "What's the most logical way to finish this text?",
        "Can you complete this passage?",
        "Add the missing conclusion to this text:",
        "Where does this text go from here?",
        "Extend this text to its natural conclusion:",
        "Following this introduction, what happens next?",
        "Write the rest of this passage:",
        "Take this beginning and craft an ending:",
        "Continue where this text leaves off:",
        "How does the text end?",
        "Finish writing what was started here:"
    )
    if no_instruction_variation:
        instruction = "Continue this text:"
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    half_idx = len(words) // 2
    prompt = " ".join(words[:half_idx])
    completion = " " + " ".join(words[half_idx:])
    if no_instruction:
        return prompt, completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, prompt = prompt, instruction
    return f"{instruction}\n\n{prompt}", completion


def _predict_second_half_with_hint(text: str, no_instruction_variation: bool = False) -> tuple:
    """
    Generate prompt-completion pair for predicting the second half of the text with a hint.
    
    Args:
        text (str): Input text to be manipulated.
        no_instruction_variation (bool): If True, the function will not use different instruction variations.
        
    Returns:
        tuple: A tuple containing:
            - String with the prompt
            - String with the completion
    
    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_second_half_with_hint(text)
        >>> print(result)
        ('Given the first half of the text, complete the second half: \n\nThe cat sat on\n\nHint: The number of words in the second half of the text is 2.', ' the mat')
    """
    instruction_variants = (
        "Looking at this beginning and using the hint below, how would the text continue?",
        "I'll show you the first part of a text and a helpful hint. How would you continue it?", 
        "With the aid of a hint, complete this text based on its opening segment:",
        "Here's the start of a passage and a clue. How should it end?",
        "Read this beginning and the provided hint. What's the most logical continuation?",
        "Using this opening part and the hint below, craft a suitable ending:",
        "The text starts like this - use the hint to complete it coherently:",
        "Consider this first half and the hint - what would be the best conclusion?",
        "With this beginning and a helpful clue, how should the text conclude?",
        "Starting from this opening and guided by the hint, complete the text:",
    )
    if no_instruction_variation:
        instruction = "Given the first half of the text, complete the second half:"
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    half_idx = len(words) // 2
    prompt = " ".join(words[:half_idx])
    completion = " " + " ".join(words[half_idx:])
    hint = f"Hint: The number of words in the second half of the text is {len(words) - half_idx}."
    return f"{instruction}\n\n{prompt}\n\n{hint}", completion


def _predict_first_half(text: str, no_instruction_variation: bool = False) -> tuple:
    """
    Generate prompt-completion pair for predicting the first half of the text.
    
    Args:
        text (str): Input text to be manipulated.
        no_instruction_variation (bool): If True, the function will not use different instruction variations.
        
    Returns:
        tuple: A tuple containing:
            - String with the prompt
            - String with the completion
    
    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_first_half(text)
        >>> print(result)
        ('Given the second half of the text, complete the first half: \n\nthe mat', 'The cat sat on')
    """
    instruction_variants = (
        "Looking at this ending, how do you think the text began?",
        "What would be the most natural opening for this text ending?",
        "Based on this conclusion, reconstruct how the text started:",
        "Here's the latter part of a text. How did it begin?",
        "Read this ending and imagine its perfect beginning:",
        "Given this final segment, craft a fitting introduction:",
        "The text ends like this - how would you write its start?",
        "Consider this ending - what opening would flow into it best?",
        "Using this conclusion as context, how should the text begin?",
        "Working backwards from this ending, compose its beginning:",
    )
    if no_instruction_variation:
        instruction = "Given the second half of the text, complete the first half:"
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    half_idx = len(words) // 2
    prompt = " ".join(words[half_idx:])
    completion = " " + " ".join(words[:half_idx])
    return f"{instruction}\n\n{prompt}", completion


def _predict_first_half_with_hint(text: str, no_instruction_variation: bool = False) -> tuple:
    """
    Generate prompt-completion pair for predicting the first half of the text with a hint.
    
    Args:
        text (str): Input text to be manipulated.
        no_instruction_variation (bool): If True, the function will not use different instruction variations.
        
    Returns:
        tuple: A tuple containing:
            - String with the prompt
            - String with the completion
    
    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_first_half_with_hint(text)
        >>> print(result)
        ('Given the second half of the text, complete the first half: \n\nthe mat\n\nHint: The number of words in the first half of the text is 4.', 'The cat sat on')
    """
    instruction_variants = (
        "Here's a challenge: This is the ending of a text. With the hint below, figure out how it began.",
        "Time for some creative writing: Using the hint provided, write the opening that leads to this ending.",
        "Let's work backwards: Given this ending and the hint below, reconstruct the text's beginning.",
        "Writing puzzle: This is how the text ends. Use the hint to discover its starting point.",
        "Detective work: With the clue provided, find the perfect beginning for this ending.",
        "Reconstruction task: Based on this ending and the helpful hint, compose its opening.",
        "Creative challenge: This text needs its beginning. Use the hint to piece it together.",
        "Text completion exercise: With the hint's guidance, write what comes before this ending.",
        "Literary puzzle: Find the missing beginning of this text using the provided hint.",
        "Writing backwards: This is the end - use the hint to create its perfect beginning.",
    )
    if no_instruction_variation:
        instruction = "Given the second half of the text, complete the first half:"
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    half_idx = len(words) // 2
    prompt = " ".join(words[half_idx:])
    completion = " " + " ".join(words[:half_idx])
    hint = f"Hint: The number of words in the first half of the text is {half_idx}."
    return f"{instruction}\n\n{prompt}\n\n{hint}", completion


def generate_half_completion_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for half text completion task.
    """
    task_funcs = {
        "predict_second_half": _predict_second_half,
        "predict_second_half_with_hint": _predict_second_half_with_hint,
        "predict_first_half": _predict_first_half,
        "predict_first_half_with_hint": _predict_first_half_with_hint,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "predict_second_half"
            results = [
                task_funcs[task_variant](text, kwargs["no_instruction_variation"], kwargs["no_instruction"])
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text, kwargs["no_instruction_variation"])
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
            task_variant = "predict_second_half"
        else:
            task_variant = random.choice(task_variants)
        prompt, completion = task_funcs[task_variant](example["text"], kwargs["no_instruction_variation"])
        return {"prompt": prompt, "completion": completion}
