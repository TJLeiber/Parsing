# Parsing
A semantic dependency parsing module for english which leverages a simplified biaffine graph-based parser from [Dozat & Manning 2018](https://aclanthology.org/P18-2077/)

## Requirements
Transformers (version >= 4.41.2 recommended)<br>
Pandas (version >= 2.2.2 recommended)<br>
NumPy (version >= 1.26.4 recommended)<br>
Pytorch (version >= 2.2.0.post100 recommended)<br>
NLTK (version >= 3.8.1 recommended)<br>
gensim (version >= 4.3.0 recommended)<br>
--> scipy (gensim import without manually installing scipy version ~ 1.11.4 can cause errors. We recommend version <= 1.13)<br>
Python3 (We used version 3.11.9)<br>

Hence ideally follow these steps to run the program:<br>
(1) create a new virtual environment to run this program (test was done with a fresh conda environment)

(2) conda activate MyNewEnv

(3) pip install scipy==1.11.4

(4) pip install gensim

(4) pip install numpy

(5) pip install torch

(6) pip install nltk

(7) pip install pandas

(8) pip install transformers==4.41.2

## Known Errors
(a) An error that might occur when installing gensim without specifying the scipy version is ImportError: cannot import name 'triu' from 'scipy.linalg' since triu from linalg has become deprecated in scipy 1.13

(b) Another error concerning attributes of the loaded BERT model may occur when running a transformers version older than the recommended one<br>
This error may occur when using -c, --concat option without the correct transformers version.

(c) When running the script you may be asked to first run the following:
"import nltk<br>
nltk.download("punkt")"<br>

## Usage
There are two main ways of interacting with the library besides writing your own script:<br> 
(1) running the main.py script from command line, (2) through direct interaction with the Python interpreter<br>
<br>
### (1) Running main.py from command line

This intro assumes you navigated to ~/path/to/directory/Parsing/Projet<br>
runnign the main.py script allows you to parse sentences from a source txt file to a target txt file.
Information on syntax can be found via the -h, --help options when running the script
<br>
<br>
![help option on main.py](misc/Screenshot%202024-06-15%20at%2010.23.56.png)
<br>
<br>
Given this syntax the most simple way of parsing an unstructured text is to handand unstructured source file to the parser indicating the target files name:
<br>
<br>
![parsing options](misc/Screenshot%202024-06-15%20at%2010.49.12.png)
<br>
<br>
The command should have allowed you to go from a file like the following:
<br>
<br>
![source.txt](misc/Screenshot%202024-06-15%20at%2010.41.01.png)
<br>
<br>
To the following target file containing parsed sentences:
<br><br>
![target.txt](misc/Screenshot%202024-06-15%20at%2010.56.04.png)
<br>
<br>
Each sentence is preceded by a number. The grid following a number is organized s.t. for an entry i,j=1 means word i is a head of word j.<br>
The very first row indicates which word is root.
<br>
<br> Argument -c, --concat changes the underlying parser that is used, i.e. a parser using BERT + GloVe word encodings which can boost performance at the cost of inference speed.<br>
Using -g, --gpu will attempt to perform inference on an available gpu<br>
In order to use the -i, --individual_sentences option the source file must contain text s.t. each line contains a single sentence. This effectively allows the user to parse specific sentences.<br>Here is an example on the required structure of a source file when using -i:<br><br>
![structured_source.txt](misc/Screenshot%202024-06-15%20at%2011.08.30.png)
<br>
<br>
Hence when using the following command in terminal:
<br>
<br>
![command_structured](misc/Screenshot%202024-06-15%20at%2011.15.56.png)
<br>
<br>
We get the following file as output:
<br>
<br>
![out_structured.txt](misc/Screenshot%202024-06-15%20at%2011.17.24.png)
### (2) Using the parser through direct interaction with the python interpreter
Use the python command to interact with your interpreter then make the following two imports (again we assume you have navigated to ~/path/to/directory/BiaffineGraphBasedParser/Projet)
<br>
<br>
![commands](misc/Screenshot%202024-06-15%20at%2012.03.53.png)
<br>
<br>
A parser is initialized with two kwargs model="fast" | "default" (fast only using GloVe embeddings) and device="cpu" | "gpu".<br><br>
![init](misc/Screenshot%202024-06-15%20at%2012.22.21.png)
<br><br>

The parser basically allows three output types for sentences that are handed to it, pandas dataframe, torch tensor, target txt file to write into.<br>
Both former ones will be outputted in a dictionary, the latter option will either create a new file at the specified location or the parser sentences will be appended to the file if it already exists.
<br>
another optional argument is_split_into_words can be used to indicate that the sentences in the list object handed to the parser have already been tokenized.
<br>
Here the same example from before, with a list containing whole sentences. As output types we chose both the torch tensor and the list of pandas dataframes.
<br>
<br>
![parsing_commands](misc/Screenshot%202024-06-15%20at%2012.34.18.png)
<br>
<br>
Here the output as torch tensor:
<br>
<br>
![parsing_commands](misc/Screenshot%202024-06-15%20at%2012.35.19.png)
