# paragraph_implicit_discourse_relations
Here is the code for the NAACL2018 paper ["Improving Implicit Discourse Relation Classification by Modeling Inter-dependencies of Discourse Units in a Paragraph"](http://www.aclweb.org/anthology/N18-1013)

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

To run my code:
1. Download preprocessed [pdtb v2.0 data file](https://drive.google.com/open?id=1ZBLoysAkbu73bt8RttJLYCRjuuMyLKMw) in .py format (All the Words/POS/NER/label and discourse unit (DU) boundary information are already transformed to vector format) and put in folder ./data <br/>
2. For the basic system without CRF, run ```python run_discourse_parsing.py```
3. For the model with CRF, run ```python run_CRF_discourse_parsing.py```

