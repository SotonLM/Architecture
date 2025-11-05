def get_tokens(text):
    punctuation = "!?.,:;'\"-()[]{}*/\\"
    remove_punctuation = str.maketrans('', '', punctuation)
    text = (text.translate(remove_punctuation)).lower()
    tokens = text.split()

    return tokens