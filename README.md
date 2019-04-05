# paragraph_implicit_discourse_relations
Note: I am working on upgrading the Pytorch version and using more user-friendly I/O files. I expect to release [new version of code](https://github.com/ZeyuDai/easy-to-use_discourse_parser) by end of May.

Code for the NAACL 2018 paper ["Improving Implicit Discourse Relation Classification by Modeling Inter-dependencies of Discourse Units in a Paragraph"](http://www.aclweb.org/anthology/N18-1013)

```
@inproceedings{dai2018improving,
  title={Improving Implicit Discourse Relation Classification by Modeling Inter-dependencies of Discourse Units in a Paragraph},
  author={Dai, Zeyu and Huang, Ruihong},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  volume={1},
  pages={141--151},
  year={2018}
}
```

To run the code:
1. Download preprocessed [pdtb v2.0 data](https://drive.google.com/open?id=1ZBLoysAkbu73bt8RttJLYCRjuuMyLKMw) in .pt format (All the Words/POS/NER/label(both implict&explicit) and discourse unit (DU) boundary information are already transformed to Pytorch vector format) and put it in folder ./data/ <br/>
2. For the model without CRF, run ```python run_discourse_parsing.py``` <br/>
3. For the model with CRF, run ```python run_CRF_discourse_parsing.py``` <br/>
4. For binary classification, run ```python run_binary_target_discourse_parsing.py``` <br/>
5. You can change the hyperparameters in .py file before the main() function (I am sorry that I didn't write code for config). Feel free to contact me if you need pretrained model file.<br/>

--------------------------------------------------------------------
About preprocessing:
1. Download both Google [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and preprocessed [POS/NER file](https://drive.google.com/open?id=1_X7DZhxw4GKaCZ8_sfgrJcoPrSVe4DLq) (You can also generate them by yourself by downloading Standford [CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/) and put them in ./data/resource) <br/>
2. The PDTB v2.0 dataset raw files are already in the ./data/preprocess/dataset/ <br/>
3. run ```python pdtb_preprocess_moreexpimp_paragraph.py``` <br/> 

--------------------------------------------------------------------
Package Version:<br/>
python == 2.7.10<br/>
torch == 0.3.0<br/>
nltk >= 3.2.2<br/>
gensim >= 0.13.2<br/>
numpy >= 1.13.1<br/>
