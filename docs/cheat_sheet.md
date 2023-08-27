## regex

1. python regex: search: find something anywhere in the string and return a match object. match: find something at the beginning of the string and return a match object.
2. A greedy quantifier such as ?, *, +, and {m,n} matches as many characters as possible (longest match). A non-greedy quantifier such as ??, *?, +?, and {m,n}? matches as few characters as possible (shortest possible match).
3. \s will match \n, while '.' and ' ' will not.
4. lookahead (to the right of the pattern): positive ?= negative ?! (?=(<regex>)): capture character n-grams with step size = 1. 
5. (?:): non-capturing group.
6. []: "^": negate the characters following the caret. "-" means range. the usual meta characters are normal characters inside a character class. no need to be escaped. [more info](https://www.regular-expressions.info/charclass.html).
7. python double backslash: python interpret "\\" as "\".

## other
1. [metissa and exponent](https://www.storyofmathematics.com/glossary/mantissa/)
2. python type hinting: List, Dict, Any, etc.
3. kaggle API:
4. Decorator function with arguments: add a wrapper outside the decoration function (request retry).
   - can add double decoration function.
