import random
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))
MASK_TOKEN_VARIATIONS = (
    "[MASK]",
    "___",
    "@@@",
    "###",
    "+++",
    "<<>>",
    "(())",
    "$$$",
)

def _predict_last_phrase(text: list[str]) -> tuple[str, str]:
    """
    Generate prompt and completion for predicting the last phrase.
    
    Args:
        text: List of sentences.
        
    Returns:
        Tuple of prompt and completion.

    Example:
    >>> _predict_last_phrase(["The quick brown fox jumps over the lazy dog.", "The lazy dog is sleeping."])
    ("Predict the last phrase of the following text.\n\nPassage: The quick brown fox jumps over the lazy dog.\n\nQuestion: The lazy dog ___?", "is sleeping.")
    """
    instruction_variants = (
        "Complete the last phrase of this passage.",
        "What comes next in this text?",
        "Fill in the missing ending of this passage.",
        "How does this text end?",
        "What's the concluding phrase of this text?",
        "Can you complete the final part of this text?",
        "What words complete this passage?",
        "Finish this text by adding the missing phrase.",
        "What's the natural ending to this passage?",
        "Complete this text with the appropriate ending.",
    )
    instruction = random.choice(instruction_variants)
    passage = text[:-1]  # Exclude the last sentence
    words = text[-1].split()
    # Find the first stop word encountered from the end
    last_stop_index = -1
    for offset, token in enumerate(reversed(words)):
        if token.lower() in STOP_WORDS:
            last_stop_index = len(words) - 1 - offset
            break
    phrase = " ".join(words[last_stop_index:])
    base_words = words[:last_stop_index]
    question = " ".join(base_words) + " ___?"
    prompt = f"{instruction}\n\nPassage: {' '.join(passage)}\n\nQuestion: {question}\n"
    return prompt, phrase


def _predict_last_phrase_with_hint(text: list[str]) -> tuple[str, str]:
    """
    Generate prompt and completion for predicting the last phrase, including a word count hint.
    
    Args:
        text: List of sentences.
        
    Returns:
        Tuple of prompt and completion.
        
    Example:
    >>> _predict_last_phrase_with_hint(["The quick brown fox jumps over the lazy dog.", "The lazy dog is sleeping."])
    ("Looking at this passage, what phrase completes it? (I'll give you a hint about the number of words)\n\nPassage: The quick brown fox jumps over the lazy dog.\n\nQuestion: The lazy dog ___? (Hint: 2 words)\n", "is sleeping.")
    """
    instruction_variants = (
        "Looking at this passage, what phrase completes it? (I'll give you a hint about the number of words)",
        "Based on the context, how does this passage end? (With a hint for word count)",
        "What's the missing phrase at the end of this text? (I'll hint at the length)",
        "Can you figure out the final phrase of this passage? (Word count provided)",
        "Using the given context, complete this passage. (With a helpful word count hint)",
        "What words would naturally conclude this text? (Number of words hinted)",
        "Read the passage and predict its ending phrase. (Word count included)",
        "How would you complete this passage? (Length hint provided)",
        "Considering the context, what's the final phrase? (With word count)",
        "What phrase brings this passage to its conclusion? (Number of words given)",
    )
    instruction = random.choice(instruction_variants)
    passage = text[:-1]  # Exclude the last sentence
    words = text[-1].split()
    # Find the first stop word encountered from the end
    last_stop_index = -1
    for offset, token in enumerate(reversed(words)):
        if token.lower() in STOP_WORDS:
            last_stop_index = len(words) - 1 - offset
            break
    phrase = " ".join(words[last_stop_index:])
    base_words = words[:last_stop_index]
    question = " ".join(base_words) + " ___?"
    num_words = len(words) - last_stop_index
    hint = f"(Hint: {num_words} word{'s' * (num_words != 1)})"
    prompt = f"{instruction}\n\nPassage: {' '.join(passage)}\n\nQuestion: {question} {hint}\n"
    return prompt, phrase


def _classify_last_phrase(text: list[str], prev_text: str) -> tuple[str, str]:
    """
    Generate prompt and completion for classifying the last phrase.
    
    Args:
        text: List of sentences.
        prev_text: List of sentences from the previous example.
        
    Returns:
        Tuple of prompt and completion.

    Example:
    >>> _classify_last_phrase(["The quick brown fox jumps over the lazy dog.", "The lazy dog is sleeping."], "The cat sat on the mat.")
    ("Choose the last phrase of the following text from the options below.\n\nPassage: The quick brown fox jumps over the lazy dog.\n\nQuestion: The lazy dog ___?\n\nOptions:\nA. is sleeping.\nB. the mat.\n", "A")
    """
    instruction_variants = (
        "Based on the passage, which option best completes the text?",
        "From the given choices, select the most appropriate ending.",
        "Read the passage and choose the correct final phrase.",
        "Which of these options naturally concludes the passage?",
        "Looking at the context, pick the right ending from below.",
        "Select the ending that best fits the passage.",
        "Choose the most suitable conclusion from the options.",
        "Among these choices, which one completes the text?",
        "Pick the ending that makes the most sense.",
        "Which option provides the correct ending to this passage?",
    )
    instruction = random.choice(instruction_variants)
    passage = text[:-1]  # Exclude the last sentence
    words = text[-1].split()
    # Find the first stop word encountered from the end
    last_stop_index = -1
    for offset, token in enumerate(reversed(words)):
        if token.lower() in STOP_WORDS:
            last_stop_index = len(words) - 1 - offset
            break
    phrase = " ".join(words[last_stop_index:])
    base_words = words[:last_stop_index]
    question = " ".join(base_words) + " ___?"
    prev_words = prev_text.split()
    last_stop_index_prev = -1
    for offset, token in enumerate(reversed(prev_words)):
        if token.lower() in STOP_WORDS:
            last_stop_index_prev = len(prev_words) - 1 - offset
            break
    prev_phrase = " ".join(prev_words[last_stop_index_prev:])
    options = [f"{phrase}"] + [f"{prev_phrase}"]  # Include the last phrase from both current and previous example
    random.shuffle(options)
    options_joined = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
    prompt = (
        f"{instruction}\n\n"
        f"Passage: {' '.join(passage)}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_joined}\n"
    )
    completion = chr(65 + options.index(f"{phrase}"))
    return prompt, completion


def _classify_last_phrase_with_generation(
    text: list[str],
    prev_text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple[str, str]:
    """
    Generate prompt and completion for classifying the last phrase. The completion should include the full phrase.
    
    Args:
        text: List of sentences.
        prev_text: List of sentences from the previous example.
        no_instruction_variation: Whether to exclude instruction variations.
        no_instruction: Whether to exclude instruction.
        
    Returns:
        Tuple of prompt and completion.

    Example:
    >>> _classify_last_phrase_with_generation(["The quick brown fox jumps over the lazy dog.", "The lazy dog is sleeping."])
    ("Pick the right ending from below.\n\nThe quick brown fox jumps over the lazy dog. The lazy dog ___?\n\nOptions:\nA. is sleeping.\nB. the mat.\n", "A. is sleeping.")
    """
    mask_variant = random.choice(MASK_TOKEN_VARIATIONS)
    instruction_variants = (
        "Which option best completes the text?",
        "Select the most appropriate ending.",
        "Choose the correct final phrase.",
        "Which of these options naturally concludes the passage?",
        "Pick the right ending.",
        "Select the ending that best fits the passage.",
        "Choose the most suitable conclusion.",
        "Which one completes the text?",
        "Pick the ending that makes the most sense.",
        "Which option provides the correct ending?",
        "Which ending belongs to this text?",
        "Find the correct conclusion below.",
        "Identify the proper ending.",
        "Select the ending that flows naturally.",
        "Which option completes the passage correctly?",
        "Choose the ending that fits best.",
        "Which is the real ending?",
        "Pick the authentic conclusion.",
        "Select the true ending from the options.",
        "Which ending continues the passage properly?",
    )
    if no_instruction_variation:
        instruction = "Pick the right ending."
    else:
        instruction = random.choice(instruction_variants)
    passage = text[:-1]  # Exclude the last sentence
    words = text[-1].split()
    # Find the first stop word encountered from the end
    last_stop_index = -1
    for offset, token in enumerate(reversed(words)):
        if token.lower() in STOP_WORDS:
            last_stop_index = len(words) - 1 - offset
            break
    phrase = " ".join(words[last_stop_index:])
    base_words = words[:last_stop_index]
    # 50% chance of adding ? at the end
    if random.random() < 0.5:
        question = " ".join(base_words) + f" {mask_variant}?"
    else:
        question = " ".join(base_words) + f" {mask_variant}"
    # Generate options
    prev_words = prev_text.split()
    last_stop_index_prev = -1
    for offset, token in enumerate(reversed(prev_words)):
        if token.lower() in STOP_WORDS:
            last_stop_index_prev = len(prev_words) - 1 - offset
            break
    prev_phrase = " ".join(prev_words[last_stop_index_prev:])
    options = [f"{phrase}"] + [f"{prev_phrase}"]  # Include the last phrase from both current and previous example
    random.shuffle(options)
    # Generate prompt and completion
    anchor_variants = (
        ("[a] ", "[b] "),
        ("[A] ", "[B] "),
        ("1. ", "2. "),
        ("A. ", "B. "),
        ("a. ", "b. "),
        ("i. ", "ii. "),
        ("1: ", "2: "),
        ("A: ", "B: "),
        ("a: ", "b: "),
        ("i: ", "ii: "),
        ("(1) ", "(2) "),
        ("(A) ", "(B) "),
        ("(a) ", "(b) "),
        ("(i) ", "(ii) "),
        ("1) ", "2) "),
        ("A) ", "B) "),
        ("a) ", "b) "),
        ("i) ", "ii) "),
    )
    # 50% chance not to use an anchor
    if random.random() < 0.5:
        anchor = ("", "")
    else:
        anchor = random.choice(anchor_variants)
    # 50% chance to combine the passage and question
    if random.random() < 0.5:
        context = f"{' '.join(passage)} {question}"
    else:
        context = f"{' '.join(passage)}\n\nQuestion: {question}"
    # 50% chance to insert "Options:" before the options
    if random.random() < 0.5:
        context += f"\n\nOptions:\n{anchor[0]}{options[0]}\n{anchor[1]}{options[1]}"
    else:
        context += f"\n\n{anchor[0]}{options[0]}\n{anchor[1]}{options[1]}"
    if no_instruction:
        prompt = context
    else:
        # 50% chance to swap the instruction and context
        if random.random() < 0.5:
            prompt = f"{instruction}\n\n{context}"
        else:
            prompt = f"{context}\n\n{instruction}"
    completion = f"{anchor[options.index(phrase)]}{phrase}"
    return prompt, completion


def generate_last_phrase_prediction_instruction_data(
    example: dict,
    batched: bool = False,
    prev_example: dict = None,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for last phrase prediction task.
    """
    task_funcs = {
        "predict_last_phrase": _predict_last_phrase,
        "predict_last_phrase_with_hint": _predict_last_phrase_with_hint,
        "classify_last_phrase": _classify_last_phrase,
        "classify_last_phrase_with_generation": _classify_last_phrase_with_generation,
    }
    task_variants = list(task_funcs.keys())

    if not batched:
        raise ValueError("Batched must be True for last phrase prediction task.")
    
    if prev_example is None:
        raise ValueError("Previous example must be provided for last phrase prediction task.")
    
    prompts = []
    completions = []
    for text in example["text"]:
        # Skip examples where the total number of words is less than the minimum
        if len(" ".join(text).split()) < min_num_words:
            continue
        if use_only_primary:
            task_type = "classify_last_phrase_with_generation"
        else:
            task_type = random.choice(task_variants)
        if task_type == "classify_last_phrase":
            prompt, completion = task_funcs[task_type](text, random.choice(prev_example["text"]))
        elif task_type == "classify_last_phrase_with_generation":
            prompt, completion = task_funcs[task_type](text, random.choice(prev_example["text"]), kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
        else:
            prompt, completion = task_funcs[task_type](text)
        prompts.append(prompt)
        completions.append(completion)
    return {"prompt": prompts, "completion": completions}
