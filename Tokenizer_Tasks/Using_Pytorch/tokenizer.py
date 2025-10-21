class tokemizer:
    def __init__(self):
        pass
    def tokenize(self,text):
        token=text.lower().replace('.',' .').split()
        return token
    def detokenize(self,tokens):
        text=' '.join(tokens).replace(' .','.')
        return text
tokenizer_instance=tokemizer()
text="Sandeep Muhal have a bike."
tokenized_text=tokenizer_instance.tokenize(text)
print(tokenized_text)
detokenized_text=tokenizer_instance.detokenize(tokenized_text)
print(detokenized_text)