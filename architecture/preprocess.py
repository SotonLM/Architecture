def get_tokens(text):
    punctuation = "!?.:;,'\"-()[]{}/\\|" # 'punctuation' the program will remove
    remove_punct = str.maketrans('', '', punctuation)
    text = (text.lower()).translate(remove_punct).split()
    return text

if __name__ == '__main__': # added this for testing, shouldn't execute if we import it
    text = input('enter dtad ')
    print(get_tok
