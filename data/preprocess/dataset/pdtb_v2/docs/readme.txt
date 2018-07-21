1. Publication Title
The Penn Discourse Treebank (PDTB) version 2.0


2. Authors
Rashmi Prasad (rjprasad@seas.upenn.edu)
Alan Lee (aleewk@seas.upenn.edu)
Nikhil Dinesh (nikhild@seas.upenn.edu)
Eleni Miltsakaki (elenimi@seas.upenn.edu)
Geraud Campion (geraud@seas.upenn.edu)
Livio Robaldo (livio.robaldo@di.unito.it)
Aravind Joshi (joshi@seas.upenn.edu)
Bonnie Webber (bonnie@inf.ed.ac.uk)

Contact email: pdtb-request@seas.upenn.edu


3. Data Type
Text


4. Data Sources
Newswire - the Penn Treebank Wall Street Journal text corpus


5. Project
Penn Discourse Treebank (PDTB) project
The goal of the PDTB project is to develop a large scale corpus annotated with information related to discourse structure.  While there are many aspects of discourse that are crucial to a complete understanding of natural language, the Penn Discourse Treebank (PDTB) focuses on encoding discourse relations associated with discourse connectives, adopting a lexically grounded approach for the annotation. The corpus provides annotations for the argument structure of Explicit and Implicit connectives, the senses of connectives and the attribution of connectives and their arguments. The lexically grounded approach exposes a clearly defined level of discourse structure which will support the extraction of a range of inferences associated with discourse connectives.

To date, the PDTB group has carried out various experiments on the corpus, particularly examining the following issues:
- the alignment between syntax and discourse, particularly with regards to attribution
- sense disambiguation of discourse connectives
- complexity of dependencies in discourse

After the release of the final corpus, the PDTB group will continue exploring the above issues as well as focus on more extended projects such as discourse parsing, automatic summarization, and natural language generation. Further work will also explore foundational issues in discourse.


6. Applications
Discourse analysis, information retrieval, information extraction, sense disambiguation, language generation, discourse parsing, automatic summarization, subjectivity analysis


7. Languages
English


8. Special License
n/a


9. Grant Number and Funding Agency
The project is partially supported by NSF Grant: Research Resources, EIA 02-24417 awarded to the University of Pennsylvania (PI: Aravind Joshi)


10. Copyright
Copyright, the Penn Discourse Treebank Group, 2008


11. Description of the Corpus Structure and Data Attributes
The corpus consists entirely of ASCII-encoded text files.  The files are annotated using a project-specific file format and its full description is given in the annotation manual.  They are partitioned into 25 sections numbered 00 to 24, mirroring the directory structure of the Penn Treebank.

Filename convention: wsj_XXYY.pdtb where XX refers to the section number and YY the file number.
Total files: 2159
Size of the data (non-compressed): 32344K (/doc/pdtb2-0)
Size of the data (compressed); 5544K (/doc/pdtb2-0.zip)
Data compression is done using zip.

The corpus annotates 40600 discourse relations, distributed into the following five types:
18459 Explicit Relations
16053 Implicit Relations
624 Alternative Lexicalizations
5210 Entity Relations
254 No Relations

The explicits, implicits and alternative lexicalizations are also annotated for attribution and semantic classes.

The project provides a number of tool support:
- a Java-based API that will enable users to manipulate and query the data from the corpus.  The latest version of the API is available at the project webpage: http://www.seas.upenn.edu/~pdtb.  
- a corpus browser built using the API - the PDTB Browser - is also provided and can be downloaded via Java WebStart or as a .jar file from the project webpage.  The Browser links the PDTB annotations to the Wall Street Journal raw text as well as the syntactic annotations of the Penn Treebank.  Both corpora are required for lauching the browser.
- a simple Perl script is provided with this distribution (util/convert.pl) that converts the file format into a 48-column pipe-delimited flat file.  This is intended particularly for users who may not want to use the API or program in Java.  The pipe-delimited file may be uploded into a spreadsheet or easily processed using scripting languages.


12. Quality Control
A user can verify the integrity of a PDTB distribution by launching the corpus using the PDTB Browser.  The browser provides a Query option which can iterate through the entire corpus.  A successful full-corpus iteration should return 40600 discourse relations. 




