1. python regex: search: find something anywhere in the string and return a match object. match: find something at the beginning of the string and return a match object.
2. A greedy quantifier such as ?, *, +, and {m,n} matches as many characters as possible (longest match). A non-greedy quantifier such as ??, *?, +?, and {m,n}? matches as few characters as possible (shortest possible match).
3. \s will match \n, while '.' and ' ' will not.
4. lookahead: ?= ?!
5. markdown table:


||Eventid|Alert Description|Issueid|Root Cause Alert Name|Incident Name|Incident Details|Is Root Cause|Time Of Occurrence|Time Of Arrival|Device Name|Event Name|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|data type|object|object|object|object|object|object|object|datetime64[ns]|datetime64[ns]|object|object|
|count|10|10|10|10|10|10|10|10|10|10|10|
|unique|10|10|1|1|1|1|1|7|5|5|6|

