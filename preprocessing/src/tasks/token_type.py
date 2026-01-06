import re
import random
from nltk.corpus import stopwords

# Cache resources globally to avoid repeated creation
STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def get_token_type(word: str) -> str:
    stripped = re.sub(r'[^\w\s]', '', word)
    if stripped.lower() in STOP_WORDS:
        return "stopword"
    elif stripped.isdigit():
        return "digit"
    return "content"

def _identify_single_token_type(
    text: str
) -> tuple:
    """
    Generate instruction data for identifying the token type of a single token from the input text.
    
    This function selects a valid alphanumeric word from the given text, determines its type based on
    whether it is a stopword, a digit, or regular content, and constructs a prompt with a randomly chosen
    instruction variant asking for its type. If no valid word is found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the token type,
               which can be "stopword", "digit", or "content". In the absence of valid words, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the token type of the word \"fox\" in the following text: \n\nThe quick brown fox jumps over the lazy dog.",
            "content"
        )
    """
    words = text.split()
    valid_words = [word for word in words if word.isalnum()]
    if not valid_words:
        return None, None
    word = random.choice(valid_words)
    completion = get_token_type(word)
    word = re.sub(r'[^\w\s]', '', word)
    instruction_variants = (
        f"Looking at the text below, what type of token is '{word}'?",
        f"In the following text, classify '{word}' as a token type.",
        f"Analyze the word '{word}' and determine its token type in this text:",
        f"What category of token does '{word}' fall into in the text below?",
        f"Examine the following text and identify the token type of '{word}'.",
        f"Please categorize '{word}' into its appropriate token type:",
        f"In the passage below, what token type best describes '{word}'?",
        f"Taking a look at this text, how would you classify '{word}' as a token?",
        f"From the following passage, identify what type of token '{word}' is.",
        f"Based on the text below, determine the token type of '{word}'.",
    )
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{text}"
    return prompt, completion


def _identify_multiple_token_types_with_masks(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for identifying the token types of multiple tokens from the input text.
    
    This function selects a set of valid alphanumeric words from the given text, determines each word's type based on
    whether it is a stopword, a digit, or regular content, and masks the selected words in the text. It then constructs a prompt
    with a randomly chosen instruction variant. If no valid words are found, a message is returned.
    
    Args:
        text (str): The input text to process.
        **kwargs: Additional arguments. Recognizes "mask_ratio" (default 0.2) to control the proportion of words to mask.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is a list of token types
               corresponding to each masked word (each can be "stopword", "digit", or "content"). In the absence of valid words,
               a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the token types of the masked words in the following text: \n\nThe [MASK] brown fox jumps [MASK] the lazy dog.",
            "["content", "stopword"]"
        )
    """
    instruction_variants = (
        "For each [MASK] in the text below, provide its token type as a list",
        "Looking at the text below, what are the token types of the masked words? Return as a list",
        "Analyze the [MASK] tokens in this text and provide their types as a list",
        "Return a list of token types for each masked word in the following text",
        "What token types correspond to the [MASK] tokens? Provide answer as a list",
        "Examine the text and list the token types of all masked words",
        "Create a list of token types for each [MASK] in the following text",
        "Identify and list the token types of each masked word in order",
        "For the text below, what type is each masked token? Return as a list",
        "Generate a list of token types for the [MASK] tokens in this text"
    )
    instruction = random.choice(instruction_variants)
    tokens = text.split()
    
    # Find indices of valid tokens (alphanumeric)
    valid_indices = [i for i, token in enumerate(tokens) if token.isalnum()]
    if not valid_indices:
        return None, None
    mask_ratio = random.uniform(0.1, 0.3)
    num_masked = max(1, int(len(valid_indices) * mask_ratio))
    num_masked = min(num_masked, len(valid_indices))
    
    # Select unique indices to mask
    indices_to_mask = random.sample(valid_indices, num_masked)
    # To maintain the order as in the text
    indices_to_mask.sort()
    
    masked_words = []
    for idx in indices_to_mask:
        token = tokens[idx]
        # Remove punctuation from the token for type checking.
        stripped_token = re.sub(r'[^\w\s]', '', token)
        masked_words.append(get_token_type(stripped_token))
        tokens[idx] = "[MASK]"
    
    masked_text = " ".join(tokens)
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, str(masked_words)
    
    
def _count_content_words(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
) -> tuple:
    """
    Generate instruction data for counting the number of content words in the input text.
    
    This function counts the number of content words in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the count. If no content words are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        no_instruction_variation (bool): If True, the instruction variation is not applied.
        no_instruction (bool): If True, the instruction is not included in the output.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the count of content words.
               In the absence of content words, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Count the number of content words in the following text: \n\nThe quick brown fox jumps over the lazy dog.",
            "5"
        )
    """
    instruction_variants = (
        "Count the number of content words in this text.",
        "How many content words are present in this passage?",
        "Calculate the total count of content words in the following text.",
        "Please count the content words present in this text.",
        "Determine the number of content words in this passage",
        "What is the total count of content words in this text?",
        "Count the content words in the following passage.",
        "Identify the number of content words in this text.",
        "How many content words can you find in this text?",
        "Calculate the total number of content words in this passage.",
        "Find the total number of content words in this text.",
        "How many non-stopwords are in this passage?",
        "Count only the content words in the following text.",
        "Tell me the number of content words in this passage.",
        "Give me a count of content words from this text.",
        "Excluding stopwords, how many words are in this text?",
        "What's the count of meaningful words in this passage?",
        "How many content-bearing words does this text contain?",
        "Tally the content words in the following passage.",
        "Report the number of content words in this text.",
    )
    if no_instruction_variation:
        instruction = "Count the number of content words in this text."
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    content_words = [word for word in words if word.isalnum() and word.lower() not in STOP_WORDS]
    if not content_words:
        # 50% chance to swap the instruction and the text.
        if random.random() < 0.5:
            return f"{instruction}\n\n{text}", "0"
        else:
            return f"{text}\n\n{instruction}", "0"
    count = str(len(content_words))
    # 50% chance to swap the instruction and the text.
    if random.random() < 0.5:
        return f"{instruction}\n\n{text}", count
    else:
        return f"{text}\n\n{instruction}", count


def _count_stopwords(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
) -> tuple:
    """
    Generate instruction data for counting the number of stopwords in the input text.
    
    This function counts the number of stopwords in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the count. If no stopwords are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        no_instruction_variation (bool): If True, the instruction variation is not applied.
        no_instruction (bool): If True, the instruction is not included in the output.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the count of stopwords.
               In the absence of stopwords, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Count the number of stopwords in the following text: \n\nThe quick brown fox jumps over the lazy dog.",
            "3"
        )
    """
    instruction_variants = (
        "Count the number of stopwords in this text.",
        "How many stopwords are present in this passage?",
        "Calculate the total count of stopwords in the following text.",
        "Please count the stopwords present in this text.",
        "Determine the number of stopwords in this passage.",
        "What is the total count of stopwords in this text?",
        "Count the stopwords in the following passage.",
        "Identify the number of stopwords in this text.",
        "How many stopwords can you find in this text?",
        "Calculate the total number of stopwords in this passage.",
        "Find the number of stopwords in this text.",
        "How many stopwords does this passage contain?",
        "Tally up all stopwords in the following text.",
        "Count stopwords only in this passage.",
        "Give me the stopword count for this text.",
        "How many common stopwords are in this text?",
        "Report the total stopwords found in this passage.",
        "Enumerate the stopwords in this text.",
        "What is the frequency of stopwords in this passage?",
        "Tell me how many stopwords appear in this text."
    )
    if no_instruction_variation:
        instruction = "Count the number of stopwords in this text."
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    stopwords_count = len([word for word in words if word.lower() in STOP_WORDS])
    if not stopwords_count:
        # 50% chance to swap the instruction and the text.
        if random.random() < 0.5:
            return f"{instruction}\n\n{text}", "0"
        else:
            return f"{text}\n\n{instruction}", "0"
    # 50% chance to swap the instruction and the text.
    if random.random() < 0.5:
        return f"{instruction}\n\n{text}", str(stopwords_count)
    else:
        return f"{text}\n\n{instruction}", str(stopwords_count)


def _count_digits(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
) -> tuple:
    """
    Generate instruction data for counting the number of digits in the input text.
    
    This function counts the number of digits in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the count. If no digits are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        no_instruction_variation (bool): If True, the instruction variation is not applied.
        no_instruction (bool): If True, the instruction is not included in the output.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the count of digits.
               In the absence of digits, a message is returned.
    
    Example:
        Input: "The quick brown 2 cats jumps over the lazy dog."
        Output: (
            "Count the number of digits in the following text: \n\nThe quick brown 2 cats jumps over the lazy dog.",
            "1"
        )
    """
    instruction_variants = (
        "Count the number of digits in this text.",
        "How many digits are present in this passage?",
        "Calculate the total count of digits in the following text.",
        "Please count the digits present in this text.",
        "Determine the number of digits in this passage.",
        "What is the total count of digits in this text?",
        "Count the digits in the following passage.",
        "Identify the number of digits in this text.",
        "How many digits can you find in this text?",
        "Calculate the total number of digits in this passage.",
        "Find the total number of digits in this text.",
        "How many numeric characters are in this passage?",
        "Count only the digits in the following text.",
        "Tell me the number of digits in this passage.",
        "Give me a count of digits from this text.",
        "How many numbers appear in this text?",
        "What's the count of numerical digits in this passage?",
        "How many numerical characters does this text contain?",
        "Tally the digits in the following passage.",
        "Report the number of digits in this text.",
    )
    if no_instruction_variation:
        instruction = "Count the number of digits in this text."
    else:
        instruction = random.choice(instruction_variants)
    digits = [char for char in text if char.isdigit()]
    digits_count = len(digits)
    if not digits_count:
        # 50% chance to swap the instruction and the text.
        if random.random() < 0.5:
            return f"{instruction}\n\n{text}", "0"
        else:
            return f"{text}\n\n{instruction}", "0"
    # 50% chance to swap the instruction and the text.
    if random.random() < 0.5:
        return f"{instruction}\n\n{text}", str(digits_count)
    else:
        return f"{text}\n\n{instruction}", str(digits_count)


def _count_punctuation_marks(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
) -> tuple:
    """
    Generate instruction data for counting the number of punctuation marks in the input text.
    
    This function counts the number of punctuation marks in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the count. If no punctuation marks are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        no_instruction_variation (bool): If True, the instruction variation is not applied.
        no_instruction (bool): If True, the instruction is not included in the output.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the count of punctuation marks.
               In the absence of punctuation marks, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog!!"
        Output: (
            "Count the number of punctuation marks in the following text: \n\nThe quick brown fox jumps over the lazy dog!!",
            "2"
        )
    """
    instruction_variants = (
        "Count the number of punctuation marks in this text.",
        "How many punctuation marks are present in this passage?",
        "Calculate the total count of punctuation marks in the following text.",
        "Please count the punctuation marks present in this text.",
        "Determine the number of punctuation marks in this passage.",
        "What is the total count of punctuation marks in this text?",
        "Count the punctuation marks in the following passage.",
        "Identify the number of punctuation marks in this text.",
        "How many punctuation marks can you find in this text?",
        "Calculate the total number of punctuation marks in this passage.",
        "Find the total number of punctuation marks in this text.",
        "How many punctuation symbols are in this passage?",
        "Count only the punctuation marks in the following text.",
        "Tell me the number of punctuation marks in this passage.",
        "Give me a count of punctuation marks from this text.",
        "How many punctuation characters appear in this text?",
        "What's the count of punctuation symbols in this passage?",
        "How many punctuation signs does this text contain?",
        "Tally the punctuation marks in the following passage.",
        "Report the number of punctuation marks in this text.",
    )
    if no_instruction_variation:
        instruction = "Count the number of punctuation marks in this text."
    else:
        instruction = random.choice(instruction_variants)
    punctuation_marks = [char for char in text if char in PUNCTUATION]
    punctuation_marks_count = len(punctuation_marks)
    if not punctuation_marks_count:
        # 50% chance to swap the instruction and the text.
        if random.random() < 0.5:
            return f"{instruction}\n\n{text}", "0"
        else:
            return f"{text}\n\n{instruction}", "0"
    # 50% chance to swap the instruction and the text.
    if random.random() < 0.5:
        return f"{instruction}\n\n{text}", str(punctuation_marks_count)
    else:
        return f"{text}\n\n{instruction}", str(punctuation_marks_count)


def _generate_content_word_list(
    text: str
) -> tuple:
    """
    Generate instruction data for listing all content words in the input text.
    
    This function lists all content words in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the list. If no content words are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the list of content words.
               In the absence of content words, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "List all the content words in the following text: \n\nThe quick brown fox jumps over the lazy dog.",
            "["quick", "brown", "fox", "jumps", "lazy", "dog"]"
        )
    """
    instruction_variants = (
        "Extract all meaningful content words from this text",
        "What are all the content words present in this passage?",
        "Create a list of content words from the following text",
        "Identify and list every content word in this text",
        "Which words in this text are content words? List them all",
        "Provide a list of all content words found in this passage",
        "What content words appear in the following text?",
        "Extract and list the content words from this text",
        "Show me all content words contained in this passage",
        "Generate a list of all content words from the text below",
    )
    instruction = random.choice(instruction_variants)
    words = text.split()
    content_words = [word for word in words if word.isalnum() and word.lower() not in STOP_WORDS]
    if not content_words:
        return f"{instruction}: \n\n{text}", "No content words found."
    return f"{instruction}: \n\n{text}", str(content_words)


def _generate_stopword_list(
    text: str
) -> tuple:
    """
    Generate instruction data for listing all stopwords in the input text.
    
    This function lists all stopwords in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the list. If no stopwords are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the list of stopwords.
               In the absence of stopwords, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "List all the stopwords in the following text: \n\nThe quick brown fox jumps over the lazy dog.",
            "["The", "over", "the"]"
        )
    """
    instruction_variants = (
        "What stopwords appear in this passage? Please list them all.",
        "Extract and list every stopword from the following text.",
        "Could you identify all stopwords present in this text?", 
        "Make a list of all stopwords that occur in the passage below.",
        "Which words in this text are considered stopwords? List them.",
        "Provide a complete list of stopwords found in this text.",
        "What are all the stopwords used in the following passage?",
        "Review this text and list out all stopwords you find.",
        "Create a comprehensive list of stopwords from this passage.",
        "Looking at the text below, what stopwords can you identify?"
    )
    instruction = random.choice(instruction_variants)
    words = text.split()
    stopwords_list = [word for word in words if word.lower() in STOP_WORDS]
    if not stopwords_list:
        return f"{instruction}\n\n{text}", "No stopwords found."  
    return f"{instruction}\n\n{text}", str(stopwords_list)


def _generate_digit_list(
    text: str
) -> tuple:
    """
    Generate instruction data for listing all digits in the input text.
    
    This function lists all digits in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the list. If no digits are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the list of digits.
               In the absence of digits, a message is returned.
    
    Example:
        Input: "The quick brown 2 cats jumps over the lazy dog."
        Output: (
            "List all the digits in the following text: \n\nThe quick brown 2 cats jumps over the lazy dog.",
            "['2']"
        )
    """
    instruction_variants = (
        "What numerical digits appear in this text? Please list them all.",
        "Extract and list every numerical digit from the following passage.",
        "Could you identify all numbers present in this text?",
        "Make a list of all digits that occur in the text below.",
        "Which characters in this text are digits? List them out.",
        "Provide a complete list of numerical digits found in this passage.",
        "What are all the numbers used in the following text?",
        "Review this passage and list out all digits you find.",
        "Create a comprehensive list of numerical digits from this text.",
        "Looking at the passage below, what digits can you identify?"
    )
    instruction = random.choice(instruction_variants)
    digits = [char for char in text if char.isdigit()]
    if not digits:
        return f"{instruction}\n\n{text}", "No digits found."
    return f"{instruction}\n\n{text}", str(digits)


def _generate_punctuation_mark_list(
    text: str
) -> tuple:
    """
    Generate instruction data for listing all punctuation marks in the input text.
    
    This function lists all punctuation marks in the given text and constructs a prompt with a randomly chosen
    instruction variant asking for the list. If no punctuation marks are found, it returns an appropriate message.
    
    Args:
        text (str): The input text to process.
        
    Returns:
        tuple: A tuple where the first element is the prompt string and the second element is the list of punctuation marks.
               In the absence of punctuation marks, a message is returned.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog!!"
        Output: (
            "List all the punctuation marks in the following text: \n\nThe quick brown fox jumps over the lazy dog!!",
            "['!', '!']"
        )
    """
    instruction_variants = (
        "What punctuation marks appear in this text? Please list them all.",
        "Extract and list every punctuation mark from the following passage.",
        "Could you identify all punctuation marks present in this text?",
        "Make a list of all punctuation marks that occur in the text below.", 
        "Which characters in this text are punctuation marks? List them out.",
        "Provide a complete list of punctuation marks found in this passage.",
        "What are all the punctuation marks used in the following text?",
        "Review this passage and list out all punctuation marks you find.",
        "Create a comprehensive list of punctuation marks from this text.",
        "Looking at the text below, what punctuation marks can you identify?"
    )
    instruction = random.choice(instruction_variants)
    punctuation_marks = [char for char in text if char in PUNCTUATION]
    if not punctuation_marks:
        return f"{instruction}\n\n{text}", "No punctuation marks found."
    return f"{instruction}\n\n{text}", str(punctuation_marks)


def generate_token_type_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for token type identification task.
    """
    task_funcs = {
        "identify_single_token_type": _identify_single_token_type,
        "identify_multiple_token_types_with_masks": _identify_multiple_token_types_with_masks,
        "count_content_words": _count_content_words,
        "count_stopwords": _count_stopwords,
        "count_digits": _count_digits,
        "count_punctuation_marks": _count_punctuation_marks,
        "generate_content_word_list": _generate_content_word_list,
        "generate_stopword_list": _generate_stopword_list,
        "generate_digit_list": _generate_digit_list,
        "generate_punctuation_mark_list": _generate_punctuation_mark_list,
    }
    task_variants = list(task_funcs.keys())
    
    if batched:
        if use_only_primary:
            task_variants = [
                "count_content_words",
                "count_stopwords",
                "count_digits",
                "count_punctuation_marks",
            ]
            results = [
                task_funcs[random.choice(task_variants)](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text)
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        results = [r for r in results if r[0] is not None and r[1] is not None]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    
    else:
        if use_only_primary:
            task_variants = [
                "count_content_words",
                "count_stopwords",
                "count_digits",
                "count_punctuation_marks",
            ]
            prompt, completion = task_funcs[task_variant](example["text"], kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
        else:
            task_variant = random.choice(task_variants)
            prompt, completion = task_funcs[task_variant](example["text"])
        if prompt is not None and completion is not None:        
            return {"prompt": prompt, "completion": completion}
        return None
