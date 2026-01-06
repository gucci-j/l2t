import random
from itertools import permutations


def _generate_full_paragraph(
    text: list[str],
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple:
    """
    Generate prompt and completion for generating a full paragraph.
    
    Args:
        text: List of sentences.
        no_instruction_variation: If True, the instruction will not be varied.
        no_instruction: If True, no instruction will be added.
        
    Returns:
        Tuple of prompt and completion.
        
    Example:
        >>> result = _generate_full_paragraph(["This is the first sentence.", "This is the second sentence.", "This is the third sentence."])
        >>> result
        ("Reorder the following sentences to form a coherent paragraph.\n\nThis is the second sentence. This is the first sentence. This is the third sentence.", "This is the first sentence. This is the second sentence. This is the third sentence.")
    """
    instruction_variants = (
        "Reorder the following sentences.",
        "Put these sentences in a logical order.",
        "Arrange these sentences.",
        "Reorder these sentences.",
        "Create a proper paragraph.",
        "Rearrange these sentences.",
        "Arrange these sentences properly.",
        "Sort these sentences.",
        "Order these sentences coherently.",
        "Organize these sentences into a paragraph.",
        "Fix the order of these sentences.",
        "Make these sentences flow logically.",
        "Put these sentences in the correct sequence.",
        "Reconstruct this paragraph correctly.",
        "Restore the proper sentence order.",
        "Form a coherent paragraph from these sentences.",
        "Place these sentences in sensible order.",
        "Correct the order of these sentences.",
        "Structure these sentences properly.",
        "Arrange this text into a logical paragraph.",
    )
    if no_instruction_variation:
        instruction = "Reorder the following sentences."
    else:
        instruction = random.choice(instruction_variants)
    shuffled_text = text.copy()
    random.shuffle(shuffled_text)
    context = ' '.join(shuffled_text)
    if no_instruction:
        prompt = context
    else:
        # 50% chance to swap the instruction and the shuffled text.
        if random.random() < 0.5:
            prompt = f"{instruction}\n\n{context}"
        else:
            prompt = f"{context}\n\n{instruction}"
    completion = " ".join(text)
    return prompt, completion


def _generate_ordering(text: list[str]) -> tuple:
    """
    Generate prompt and completion for ordering sentences.

    Args:
        text: List of sentences.

    Returns:
        Tuple of prompt and completion.

    Example:
        >>> result = _generate_ordering(["This is the first sentence.", "This is the second sentence.", "This is the third sentence."])
        >>> result
        ("Order the following sentences and indicate the sequence using numbers and arrows (e.g., 1 -> 2 -> 3) to form a coherent paragraph.\n\n1. This is the second sentence.\n2. This is the first sentence.\n3. This is the third sentence.", "1 -> 2 -> 3")
    """
    instruction_variants = (
        "Arrange these sentences in the right order and show their sequence using numbers (like 1 -> 2 -> 3).",
        "What's the correct order of these sentences? Express it as a sequence (e.g., 1 -> 2 -> 3).",
        "Number these sentences in order to create a logical flow (format: 1 -> 2 -> 3).",
        "How should these sentences be ordered? Write the sequence as numbers (1 -> 2 -> 3).",
        "Put these sentences in order and show your answer as a numbered sequence (1 -> 2 -> 3).",
        "These sentences need to be reordered. Show the correct sequence using numbers (1 -> 2 -> 3).",
        "What's the proper sequence of these sentences? Use numbers to show order (1 -> 2 -> 3).",
        "Create a logical sequence from these sentences using numbers (format: 1 -> 2 -> 3).",
        "Show the correct order of these sentences using a numbered sequence (1 -> 2 -> 3).",
        "Indicate the proper arrangement of these sentences with numbers (like 1 -> 2 -> 3)."
    )
    instruction = random.choice(instruction_variants)
    shuffled_text = text.copy()
    random.shuffle(shuffled_text)
    prompt = f"{instruction}\n\n" + "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(shuffled_text)])
    completion = " -> ".join([str(text.index(sentence) + 1) for sentence in shuffled_text])
    return prompt, completion


def _choose_ordering(text: list[str]) -> tuple:
    """
    Generate prompt and completion for choosing the correct ordering of sentences.
    Assumes that text contains exactly three sentences.

    Args:
        text: List of three sentences.

    Returns:
        Tuple of prompt and completion letter.

    Example:
        >>> result = _choose_ordering(["This is the first sentence.", "This is the second sentence.", "This is the third sentence."])
        >>> result
        ("Choose the correct ordering of the following sentences to form a coherent paragraph.\n\n1. This is the second sentence.\n2. This is the first sentence.\n3. This is the third sentence.\n\n(a) 1 -> 2 -> 3\n(b) 2 -> 1 -> 3\n(c) 3 -> 1 -> 2\n(d) 2 -> 3 -> 1\n(e) 1 -> 3 -> 2\n(f) 3 -> 2 -> 1", "a")
    """
    instruction_variants = (
        "Looking at these sentences, which sequence creates the most logical paragraph?",
        "From the options below, select the arrangement that makes the most sense.",
        "Which of these possible orderings creates the most coherent narrative?",
        "Read these sentences carefully. Which sequence flows most naturally?",
        "Identify the most logical arrangement of these sentences from the given options.",
        "Among these possible sequences, which one forms the most coherent paragraph?",
        "Based on the content, which ordering creates the clearest progression of ideas?",
        "Choose the sequence that best connects these sentences together.",
        "Which arrangement of these sentences tells the story most effectively?",
        "Select the ordering that produces the most natural flow of information."
    )
    instruction = random.choice(instruction_variants)
    # Create a list of tuples (original_index, sentence) and shuffle it to preserve original order.
    sentence_tuples = list(enumerate(text, start=1))
    random.shuffle(sentence_tuples)
    prompt_lines = [f"{i+1}. {sentence}" for i, (_, sentence) in enumerate(sentence_tuples)]
    prompt = f"{instruction}\n\n" + "\n".join(prompt_lines) + "\n\n"
    # Determine the correct ordering using the original indices.
    correct_order_numbers = [str(orig_index) for orig_index, sentence in sentence_tuples]
    correct_order = " -> ".join(correct_order_numbers)
    # Generate all possible ordering options for three sentences.
    all_orderings = [" -> ".join(p) for p in permutations(["1", "2", "3"], 3)]
    random.shuffle(all_orderings)
    # Label the options with letters (a through f).
    option_labels = [f"{chr(65 + i)}. {option}" for i, option in enumerate(all_orderings)]
    prompt += "\n".join(option_labels)
    # Find the letter corresponding to the correct ordering.
    correct_letter = chr(65 + all_orderings.index(correct_order))
    return prompt, correct_letter


def generate_sentence_reordering_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for sentence reordering task.
    """
    task_funcs = {
        "generate_full_paragraph": _generate_full_paragraph,
        "generate_ordering": _generate_ordering,
        "choose_ordering": _choose_ordering,
    }
    task_variants = list(task_funcs.keys())

    if not batched:
        raise ValueError("Sentence reordering task requires batched data.")

    prompts = []
    completions = []
    if use_only_primary:
        task_variant = "generate_full_paragraph"
        results = [
            task_funcs[task_variant](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
            for text in example["text"]
            if len(' '.join(text).split()) >= min_num_words
        ]
    else:
        results = [
            task_funcs[random.choice(task_variants)](text)
            for text in example["text"]
            if len(' '.join(text).split()) >= min_num_words
        ]
    if results:
        prompts, completions = zip(*results)
    else:
        prompts, completions = [], []
    return {"prompt": list(prompts), "completion": list(completions)}
