from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelWithLMHead
import nltk
from nltk import tokenize
import random
from wonderwords import RandomWord
import re
from word2number import w2n
from num2words import num2words

# accepted tags when generating random choices
accepted = ['NNS', 'NN', 'JJS', 'JJ', 'NNP', 'NNPS', 'VB', 'VBD']

# models that will be used
questioning_model = "ThomasSimonini/t5-end2end-question-generation"
answering_model = "MaRiOrOsSi/t5-base-finetuned-question-answering"

QuestionModel, AnswerModel = None, None
QuestionTokenizer, AnswerTokenizer = None, None

Words = None

# load models to memory
def initialize():
    global QuestionModel, AnswerModel, QuestionTokenizer, AnswerTokenizer
    global Words

    print('Initializing t5 Models...')
    QuestionModel = T5ForConditionalGeneration.from_pretrained(questioning_model)
    QuestionTokenizer = T5Tokenizer.from_pretrained(questioning_model)

    AnswerModel = AutoModelWithLMHead.from_pretrained(answering_model)
    AnswerTokenizer = AutoTokenizer.from_pretrained(answering_model)

    # check if nltk resources is available else download the resources
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('words')

    Words = set(nltk.corpus.words.words())

# generate question using ThomasSimonini/t5-end2end-question-generation model
def getQuestions(input_string, **generator_args):
    generator_args = {
        "max_length": 256,
        "num_beams": 3,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = QuestionTokenizer.encode(input_string, return_tensors="pt")
    res = QuestionModel.generate(input_ids, **generator_args)
    output = QuestionTokenizer.batch_decode(res, skip_special_tokens=True)
    output = [item.split("<extra_id_-1>") for item in output]
    return output[0]

# generate answer using MaRiOrOsSi/t5-base-finetuned-question-answering model
def getAnswer(question, context):
    input = f"question: {question} context: {context}"
    encoded_input = AnswerTokenizer([input],
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)
    output = AnswerModel.generate(input_ids = encoded_input.input_ids,
                                    attention_mask = encoded_input.attention_mask)
    output = AnswerTokenizer.decode(output[0], skip_special_tokens=True)

    return output

# get multiple choices based on the answer
# Check if answer contains number. 
# If true, Extract the number (digit) using regex expression ('\d+').
#          Check if num's first digit is 0 then change to 1 (ex. 099 to 199).
#          Weights for getting the range of the random number depends on the number of 
#          digits of the current number (ex. 12 has 2 digits therefore the val(weights) = 5, 
#          hence the random number would be between 12-5 and 12+5)
# Else, Tag each words from the answer using NLTK
#       When the picked word is accepted by its type (can be noun, pronouns, verb, adjective),
#       random words based on that type are generated.
#       While if the word is a cardinal number, it will be converted to number format (int),
#       then generate a random number between the current number plus/minus the weight.
def getChoices(answer):
    choices = []
    choices.append(answer)

    # get random numbers between the current number plus/minus a value
    def getRandom(choice, num, value):
        rand = random.randint(abs(int(num)-value), int(num)+value)

        ch = choice.replace(str(num), str(rand))
        if ch in choices:
            ch = getRandom(choice, num, value)

        return ch
        
    # check if answer contains numbers
    if any(i.isdigit() for i in answer):
        nums = re.findall(r'\d+', answer)
        for i in range(3):
            choice = answer
            for num in nums:
                val = 10
                new = num

                # check if num's first digit is 0 then change to 1 (ex. 099 to 199)
                if len(num) > len(str(int(num))) and len(str(int(num)))>=2:
                    new = int(num) + pow(10, len(num)-1)

                # adjust value range
                if len(str(new)) == 2:
                    val = 5
                if len(str(new)) == 1:
                    val = 3

                choice = getRandom(choice, new, val)

            choices.append(choice)

    else:
        # split words from the answer (answers can be phrases)
        answer_tokenize = tokenize.word_tokenize(answer)

        # get part-of-speech tags of the words using nltk
        tokens = nltk.pos_tag(answer_tokenize)
        
        r = RandomWord()

        def getRandomWord():
            word = random.choice(tokens)
            
            # check if the tag of the random picked word is accepted(only be noun, pronouns, adjectives, and verb)
            if word[1] in accepted:
                new = r.word(include_parts_of_speech=["noun"])

                # get random words for respective POS tags using RnadomWord
                if word[1] == 'JJS' or word[1] == 'JJ':
                    new =  r.word(include_parts_of_speech=["adjectives"])
                
                if word[1] == 'NN' or word[1] == 'NNS' or word[1]=='NNP':
                    new =  r.word(include_parts_of_speech=["noun"])
                
                if word[1] == 'VB' or word[1] == 'VBD':
                    new = r.word(include_parts_of_speech=['verb'])
                
                # get the new choice by replacing the word picked with the new word 
                choice = answer.replace(word[0], new.capitalize() if word[0][0].isupper() else new)

                # if choice exists in list of newly created choices, create new one
                if choice in choices:
                    choice = getRandomWord()
            
            # check if the picked word is a cardinal number (ex. Seven, One, etc)
            elif word[1] == 'CD':
                weight = 3
                try:
                    # convert the num word (string) to number format (ex. Seven -> 7)
                    num = w2n.word_to_num(word[0])
                except ValueError:
                    choice = getRandomWord()

                # get random number between the current number plus/minus 3
                rand = random.randint(num - weight, num + weight)
                # convert the new number to word (string format)
                new = num2words(rand)

                # get the new choice by replacing the word picked with the new word 
                choice = answer.replace(word[0], new.capitalize() if word[0][0].isupper() else new)

                # if choice exists in list of newly created choices, create new one
                if choice in choices:
                    choice = getRandomWord()
                                   
            else:
                choice = getRandomWord()

            return choice

        for i in range(3):
            try:
                choices.append(getRandomWord())
            except RecursionError:
                return None
    
    random.shuffle(choices)
    return choices
    
def generate_quiz(context):
    qa = []
    
    questions = [question for question in getQuestions(context) if question]
    for question in questions:
        answer = getAnswer(question, context)

        if answer:
            choices = getChoices(answer)

            if choices:
                qa.append({'question': question, 'answer': answer, 'choices': choices })

    return qa