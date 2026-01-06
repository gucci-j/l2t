import random


def _generate_coherent_paragraph(
    text: list[str],
    prev_text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple[str, str]:
    """
    Generate prompt and expected completion for coherent paragraph tasks.

    Args:
        text (list[str]): Sentences forming the correct passage.
        prev_text (str): An extra sentence to be inserted randomly.
        no_instruction_variation (bool): If True, the instruction will not be varied.
        no_instruction (bool): If True, the instruction will not be included.

    Returns:
        tuple[str, str]: A tuple containing the instruction prompt and the correct passage.

    Example:
        >>> text = ["This is the first sentence.", "This is the second sentence."]
        >>> prev_text = "This is the extraneous sentence."
        >>> prompt, completion = _generate_coherent_paragraph(text, prev_text)
        >>> print(prompt)
        "Remove the sentence that disrupts the passage coherence.\n\nPassage: This is the first sentence. This is the extraneous sentence. This is the second sentence.\n\n"
        >>> print(completion)
        "This is the first sentence. This is the second sentence."
    """
    instructions = [
        "Remove the sentence that breaks the flow of the passage.",
        "Omit the sentence that seems out of place.",
        "Delete the sentence that doesn't belong in this passage.",
        "Remove the sentence that disrupts the coherence of this text.",
        "Eliminate the sentence that doesn't fit with the rest of the passage.",
        "Get rid of the sentence that breaks the logical connection in this text.",
        "Take out the sentence that interrupts the natural flow.",
        "Remove the out-of-place sentence from this passage.",
        "Delete the sentence that feels disconnected from the main idea.",
        "Exclude the sentence that doesn't maintain the passage's continuity.",
        "Remove the sentence that doesn't contribute to the passage's theme.",
        "Delete the sentence that stands out as irrelevant.",
        "Take out the sentence that disrupts the narrative consistency.",
        "Eliminate the sentence that breaks the passage's coherence.",
        "Identify and remove the sentence that doesn't belong.",
        "Remove the sentence that seems unrelated to the context.",
        "Delete the intrusive sentence from this passage.",
        "Omit the sentence that detracts from the passage's unity.",
        "Remove the misplaced sentence in this text.",
        "Discard the sentence that interrupts the flow of ideas."
    ]
    if no_instruction_variation:
        instruction = "Remove the out-of-place sentence from this passage."
    else:
        instruction = random.choice(instructions)
    shuffled_passage = text.copy()
    shuffled_passage.insert(random.randint(0, len(text)), prev_text)
    if no_instruction:
        prompt = f"{' '.join(shuffled_passage)}"
    else:
        # 50% chance to swap the instruction and the shuffled text.
        if random.random() < 0.5:
            prompt = f"{instruction}\n\n{' '.join(shuffled_passage)}"
        else:
            prompt = f"{' '.join(shuffled_passage)}\n\n{instruction}"
    return prompt, " ".join(text)


def _generate_deletion(
    text: list[str],
    prev_text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple[str, str]:
    """
    Generate prompt and expected completion for sentence deletion tasks.

    Args:
        text (list[str]): Sentences forming the correct passage.
        prev_text (str): An extra sentence to be inserted randomly.
        no_instruction_variation (bool): If True, the instruction will not be varied.
        no_instruction (bool): If True, the instruction will not be included.

    Returns:
        tuple[str, str]: A tuple containing the instruction prompt and the correct passage.

    Example:
        >>> text = ["This is the first sentence.", "This is the second sentence."]
        >>> prev_text = "This is the extraneous sentence."
        >>> prompt, completion = _generate_deletion(text, prev_text)
        >>> print(prompt)
        "Identify the sentence that disrupts the passage coherence.\n\nPassage: This is the first sentence. This is the extraneous sentence. This is the second sentence.\n\n"
        >>> print(completion)
        "This is the extraneous sentence."
    """
    instructions = [
        "Find the out-of-place sentence in this passage.",
        "Identify the sentence that doesn't belong in this passage.",
        "Which sentence disrupts the flow of this text?",
        "Point out the sentence that breaks the coherence of this passage.",
        "Spot the sentence that seems disconnected from the main idea.",
        "Determine which sentence is unrelated to the passage theme.",
        "Locate the sentence that interrupts the natural flow of ideas.",
        "Which sentence would you remove to maintain coherence?",
        "Highlight the sentence that feels out of context.",
        "Identify which sentence doesn't fit with the rest of the content.",
        "Which sentence stands out as irrelevant in this text?",
        "Find the sentence that breaks the logical progression.",
        "Identify the sentence that disrupts narrative cohesion.",
        "Which sentence seems misplaced in this passage?",
        "Pick out the sentence that doesn't follow the topic.",
        "What sentence would you delete to improve flow?",
        "Which sentence contains information unrelated to the passage?",
        "Find the sentence that creates a disconnect in the text.",
        "Which sentence interrupts the passage's consistency?",
        "Identify the odd sentence in this paragraph.",
    ]
    if no_instruction_variation:
        instruction = "Find the out-of-place sentence in this passage."
    else:
        instruction = random.choice(instructions)
    shuffled_passage = text.copy()
    shuffled_passage.insert(random.randint(0, len(text)), prev_text)
    if no_instruction:
        prompt = f"{' '.join(shuffled_passage)}"
    else:
        # 50% chance to swap the instruction and the shuffled text.
        if random.random() < 0.5:
            prompt = f"{instruction}\n\n{' '.join(shuffled_passage)}"
        else:
            prompt = f"{' '.join(shuffled_passage)}\n\n{instruction}"
    return prompt, prev_text


def _choose_deletion(
    text: list[str],
    prev_text: str
) -> tuple[str, str]:
    """
    Generate prompt and expected completion for sentence deletion tasks.

    Args:
        text (list[str]): Sentences forming the correct passage.
        prev_text (str): An extra sentence to be inserted randomly.

    Returns:
        tuple[str, str]: A tuple containing the instruction prompt and the correct passage.

    Example:
        >>> text = ["This is the first sentence.", "This is the second sentence."]
        >>> prev_text = "This is the extraneous sentence."
        >>> prompt, completion = _choose_deletion(text, prev_text)
        >>> print(prompt)
        "Choose the index of the sentence that disrupts the passage coherence.\n\nPassage: 1. This is the first sentence. 2. This is the extraneous sentence. 3. This is the second sentence.\n\n"
        >>> print(completion)
        "2"
    """
    instructions = [
        "Which numbered sentence breaks the flow of the passage?",
        "Enter the number of the sentence that seems out of place.",
        "Which sentence number disrupts the passage's coherence?",
        "Type the number of the sentence that doesn't fit with the rest.",
        "What is the number of the sentence that feels disconnected?",
        "Identify the sentence number that breaks the logical flow.",
        "Which numbered sentence would you remove to improve coherence?",
        "Enter the position number of the misplaced sentence.",
        "Which sentence number appears to be unrelated to the main theme?",
        "What is the number of the sentence that interrupts the narrative?"
    ]
    instruction = random.choice(instructions)
    shuffled_passage = text.copy()
    shuffled_passage.insert(random.randint(0, len(text)), prev_text)
    numbered_passage = [f"{i+1}. {sent}" for i, sent in enumerate(shuffled_passage)]
    prompt = f"{instruction}\n\nPassage: {' '.join(numbered_passage)}\n\n"
    return prompt, str(shuffled_passage.index(prev_text) + 1)


def generate_sentence_deletion_instruction_data(
    example: dict,
    batched: bool = False,
    prev_example: dict = None,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for sentence deletion tasks.
    """
    task_funcs = {
        "generate_coherent_paragraph": _generate_coherent_paragraph,
        "generate_deletion": _generate_deletion,
        "choose_deletion": _choose_deletion,
    }
    task_variants = list(task_funcs.keys())
    
    if batched:
        if prev_example is None:
            raise ValueError("Previous example is required when batched is True.")
        if use_only_primary:
            task_variants = ["generate_coherent_paragraph", "generate_deletion"]
            results = [
                task_funcs[random.choice(task_variants)](
                    text, random.choice(prev_example["text"]), kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False)
                )
                for text in example["text"]
                if len(' '.join(text).split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text, random.choice(prev_example["text"]))
                for text in example["text"]
                if len(' '.join(text).split()) >= min_num_words
            ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        raise ValueError("Batched mode is required for this task type.")
