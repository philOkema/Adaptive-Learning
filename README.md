(b) Initial Machine Learning aims to understand pattern classification well. 
Often many algorithms are used. Using a commented code, write a 
program to implement MNIST Dataset learning of digits 0 – 9. 

Question Two 

Day	Outlook	Temperature	Humidity	Wind	Play Tennis
D1	Sunny	Hot	High	Weak	No
D2	Sunny	Hot	High	Strong	No
D3	Overcast	Hot	High	Weak	Yes
D4	Rain	Mild	High	Weak	Yes
D5	Rain	Cool	Normal	Weak	Yes
D6	Rain	Cool	Normal	Strong	No
D7	Overcast	Cool	Normal	Strong	Yes
D8	Sunny	Mild	High	Weak	No
D9	Sunny	Cool	Normal	Weak	Yes
D10	Rain	Mild	Normal	Weak	Yes
D11	Sunny	Mild	Normal	Strong	Yes
D12	Overcast	Mild	High	Strong	Yes
D13	Overcast	Hot	Normal	Weak	Yes
D14	Rain	Mild	High	Strong	No

(a) Consider the above dataset of playing decision. Calculate information 
gain of each feature. 
(b)  Identify the most feasible to split. 
(c)  Write a python code we can use to calculate entropy and information 
gain. 

Question Two Answer
(a) Calculate information gain of each feature.
Dataset Summary:

Total instances: 14
Play Tennis = Yes: 9
Play Tennis = No: 5

Total Entropy (S) = 0.9403 bits
Information Gain (IG) for each feature:

Outlook: 0.2467 (Highest)
Humidity: 0.1518
Wind: 0.0481
Temperature: 0.0292 (Lowest)

(b) Identify the most feasible to split.
The most feasible attribute to split on first is Outlook, because it has the highest Information Gain (0.2467).
This means splitting on Outlook reduces uncertainty the most.
