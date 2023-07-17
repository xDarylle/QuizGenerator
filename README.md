# Quiz Generator
The **quiz generator** can produce multiple choice quizzes from a given *context* or paragraph by using two **T5 models** to generate questions and their correct answers: 
[*t5-end2end-question-generation*](https://huggingface.co/ThomasSimonini/t5-end2end-question-generation) for generating questions 
and [*t5-base-finetuned-question-answering*](https://huggingface.co/MaRiOrOsSi/t5-base-finetuned-question-answering) for generating the correct answers. 
The resulting quiz includes a question, the correct answer, and four choices including the correct answer.


## Demo
Please try the Quiz Generator demo through [online Colab demo](https://colab.research.google.com/drive/118UmolMkHnkwQNt6-LSLKmOX2Va4Va3W?usp=sharing).

## Usage
1. Clone this repository
```
git clone https://github.com/xShiro2/QuizGenerator.git
cd QuizGenerator
```

2. Setup and activate virtual environment
```
python -m venv venv
venv/Scripts/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Use quiz generator
```
from generator import initialize, generate_quiz

# download and load models & additional nltk resources
initialize()

# set context
context = "Your context or paragraph here..."

# generate quiz
result = generate_quiz(context)
```
