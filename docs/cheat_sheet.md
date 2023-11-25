## regex

1. python regex: search: find something anywhere in the string and return a match object. match: find something at the beginning of the string and return a match object.
2. A greedy quantifier such as ?, *, +, and {m,n} matches as many characters as possible (longest match). A non-greedy quantifier such as ??, *?, +?, and {m,n}? matches as few characters as possible (shortest possible match).
3. \s will match \n, while '.' and ' ' will not.
4. lookahead (to the right of the pattern): positive ?= negative ?! (?=(<regex>)): capture character n-grams with step size = 1. 
5. (?:): non-capturing group.
6. []: "^": negate the characters following the caret. "-" means range. the usual meta characters are normal characters inside a character class. no need to be escaped. [more info](https://www.regular-expressions.info/charclass.html).
7. python double backslash: python interpret "\\\\" as single backslash "\\" .

## python GIL mechanisms
1. time out function on windows system:
```python
import concurrent.futures as futures

with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(<your_function_name>, *args, **kwargs)
    completion = future.result(<your_time_out_limit>)
    executor._threads.clear()
    futures.thread._threads_queues.clear()
```


## other
1. [metissa and exponent](https://www.storyofmathematics.com/glossary/mantissa/)
2. python type hinting: List, Dict, Any, etc.
3. kaggle API:
4. Decorator function with arguments: add a wrapper outside the decoration function (request retry).
   - can add double decoration function.
5. ignore warning context:

For fixed warning types:

```python
import warnings
from urllib3.exceptions import InsecureRequestWarning

def ignore_request_warning(func):
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=InsecureRequestWarning)
            return func(*args, **kwargs)
    return inner
```
For flexible warning types which could be input as arguments: 

```python
import warnings

def ignore_request_warning(warning):
    def decorator(function)
        def inner(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=warning)
                return func(*args, **kwargs)
        return inner
    return decorator
```

6. bitsandbytes might has some problem with running no V100. Set with `torch.autocast("cuda"):`{:.python} before training the trainer. [More info](https://github.com/TimDettmers/bitsandbytes/issues/240)
7. [Model save weight](https://github.com/huggingface/peft/issues/286#issuecomment-1501617281)
8. peft model merge and unload.
9. repeatition of data hurts naive bayes
why is naive bayes generative models? what is generative and discriminative model?
logistic classification can be viewed as nn, or directed graphical model! 
why crf is discriminative?
