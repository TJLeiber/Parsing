# BiaffineGraphBasedParser
A semantic dependency parsing module for english which leverages a biaffine graph-based parser from [Dozat & Manning 2018](https://aclanthology.org/P18-2077/)

## Requirements
Transformers (version >= 4.41.2 recommended)<br>
NumPy (version >= 1.26.4 recommended)<br>
Pytorch (version >= 2.2.0.post100 recommended)<br>
NLTK (version >= 3.8.1 recommended)<br>
gensim (version >= 4.3.0 recommended)<br>
Python3<br>

## Usage
There are two main ways of interacting with the library besides writing your own script:<br> 
(1) running the main.py script from command line, (2) through direct interaction with the Python interpreter<br>
<br>
-------------------------- (1) --------------------------
<br>
This intro assumes you navigated to ~/path/to/directory/BiaffineGraphBasedParser/Projet<br>
runnign the main.py script allows you to parse sentences from a source txt file to a target txt file.
Information on syntax can be found via the -h, --help options when running the script
<br>
<br>
![help option on main.py](https://github.com/TJLeiber/BiaffineGraphBasedParser/blob/main/misc/Screenshot%202024-06-15%20at%2010.23.56.png)
<br>
<br>
Given this syntax the most simple way of parsing an unstructured text is to simple hand the source file to the parser:
<br>
<br>
![parsing options](https://github.com/TJLeiber/BiaffineGraphBasedParser/blob/main/misc/Screenshot%202024-06-15%20at%2010.49.12.png)
<br>
<br>
The command should have alloed you to go from a file like the following:
![source.txt](https://github.com/TJLeiber/BiaffineGraphBasedParser/blob/main/misc/Screenshot%202024-06-15%20at%2010.41.01.png)
<br>
<br>
To the following target file containing parsed sentences:

