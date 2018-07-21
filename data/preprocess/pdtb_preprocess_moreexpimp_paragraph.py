import os 
import sys
import re
import itertools
import cPickle
import copy
import numpy as np

import torch
from torch.autograd import Variable

import gensim
import nltk
from nltk.tag import StanfordPOSTagger,StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import StanfordTokenizer


# Load Google pretrained word2vec
model = gensim.models.Word2Vec.load_word2vec_format('../resource/GoogleNews-vectors-negative300.bin', binary=True)
# Load Glove pretrained word2vec
#model = gensim.models.Word2Vec.load_word2vec_format('../resource/glove.840B.300d.w2vformat.txt', binary=False)

# Load stored pos/ner parsing sentence
sentence_pos_ner_dict = {}
with open('../resource/pdtb_sentence_pos_ner_dict.pkl','r') as f:
	sentence_pos_ner_dict = cPickle.load(f)
	f.close()

stanford_dir = '../resource/stanford-postagger-2016-10-31/'
modelfile = stanford_dir + 'models/english-left3words-distsim.tagger'
jarfile = stanford_dir + 'stanford-postagger.jar'
pos_tager = StanfordPOSTagger(modelfile, jarfile, encoding='utf8')
#print pos_tager.tag("Brack Obama lives in New York .".split())

st = StanfordTokenizer(jarfile, encoding='utf8')
#print st.tokenize('Among 33 men who worked closely with the substance, 28 have died -- more than three times the expected number. Four of the five surviving workers have asbestos-related diseases, including three with recently diagnosed cancer.')

stanford_dir = '../resource/stanford-ner-2016-10-31/'
modelfile = stanford_dir + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
jarfile = stanford_dir + 'stanford-ner.jar'
ner_tager = StanfordNERTagger(modelfile, jarfile, encoding='utf8')
#print ner_tager.tag("In Jan. 5, Brack Obama lives in New York at 5:20 .".split())
#print ner_tager.tag(nltk.word_tokenize("Assets of the 400 taxable funds grew by $1.5 billion during the latest week, to $352.7 billion."))


vocab={}
def unknown_words(word,k=300):
	if word == '' or word in ['<SOS>','<EOS>','<ParaBoundary>']:
		return torch.zeros(k)
	if word not in vocab:
		vocab[word] = torch.rand(k)/2 - 0.25 
	return vocab[word] 

NER_LIST = ['ORGANIZATION','LOCATION','PERSON','MONEY','PERCENT','DATE','TIME']
PEN_TREEBANK_POS_LIST = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
def tansfer_word2vec(input_list,posner_flag=True,k=300):
	if posner_flag:
		pos_list,ner_list = input_list[0],input_list[1]
		embedding = torch.zeros(len(pos_list),k+len(PEN_TREEBANK_POS_LIST)+len(NER_LIST))

		for i in range(len(pos_list)):
			word,pos,ner = pos_list[i][0],pos_list[i][1],ner_list[i][1]

			if word in model:
				embedding[i,:k] = torch.from_numpy(model[word])
			#elif word.lower() in model:
			#	embedding[i,:k] = torch.from_numpy(model[word.lower()])
			else:
				embedding[i,:k] = unknown_words(word)

			if pos in PEN_TREEBANK_POS_LIST:
				embedding[i,k+PEN_TREEBANK_POS_LIST.index(pos)] = 1
			if ner in NER_LIST:
				embedding[i,k+len(PEN_TREEBANK_POS_LIST)+NER_LIST.index(ner)] = 1

		return embedding
	else:
		word_list = input_list
		embedding = torch.zeros(len(word_list),k)
		for i in range(len(word_list)):
			word = word_list[i]

			if word in model:
				embedding[i,:] = torch.from_numpy(model[word])
			#elif word.lower() in model:
			#	embedding[i,:] = torch.from_numpy(model[word.lower()])
			else:
				embedding[i,:] = unknown_words(word)
		return embedding

discourse_sense_list = ['Temporal','Comparison','Contingency','Expansion']
def process_discourse_relation_label(discourse_label, discourse_type):
	y = torch.zeros(len(discourse_sense_list))

	for label in discourse_label.split('|'):
		level1_sense = label.split('.')[0]

		if discourse_type == 'Explicit':
			y[discourse_sense_list.index(level1_sense)] = -1
		#elif discourse_type == 'EntRel':
		#	label[discourse_sense_list.index('Expansion')] = 1
		else:
			if level1_sense != '':
				y[discourse_sense_list.index(level1_sense)] = 1

	return y

'''def process_discourse_relation_label_8way(discourse_label, discourse_type):
	y = torch.zeros(len(discourse_sense_list)*2)

	for label in discourse_label.split('|'):
		level1_sense = label.split('.')[0]

		if discourse_type == 'Explicit':
			y[len(discourse_sense_list)+discourse_sense_list.index(level1_sense)] = 1
		#elif discourse_type == 'EntRel':
		#	label[discourse_sense_list.index('Expansion')] = 1
		else:
			if level1_sense != '':
				y[discourse_sense_list.index(level1_sense)] = 1

	return y'''

def process_doc_paras_labels(doc_sentence_list, doc_discourse_dict):
	sentence_index_list = []
	discourse_dict = {}
	for argpair in doc_discourse_dict:
		arg1_index,arg2_index = argpair[0],argpair[1]
		discourse_type,discourse_label = doc_discourse_dict[argpair][0],doc_discourse_dict[argpair][1]

		if discourse_type in ['Implicit','Explicit']:
			discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
			if arg1_index not in sentence_index_list:
				sentence_index_list.append(arg1_index)
			if arg2_index not in sentence_index_list:
				sentence_index_list.append(arg2_index)

	if len(sentence_index_list) <= 1:
		return [], []

	sentence_index_list.sort()
	#print sentence_index_list
	#print discourse_dict
	
	paras_sentence_list = []
	paras_y_list = []

	sentence_index = 0
	para_sentence_list = []
	discourse_list = []
	while(sentence_index <= sentence_index_list[-1]):
		if (sentence_index,sentence_index+1) in discourse_dict:
			discourse_list.append((sentence_index,sentence_index+1))
			if len(para_sentence_list) == 0:
				para_sentence_list.append(doc_sentence_list[sentence_index])
			para_sentence_list.append(doc_sentence_list[sentence_index+1])
		else:
			if len(discourse_list) != 0:
				para_y = torch.zeros(len(discourse_list),len(discourse_sense_list))
				for i in range(len(discourse_list)):
					discourse = discourse_list[i]
					discourse_type,discourse_label = discourse_dict[discourse][0],discourse_dict[discourse][1]
					para_y[i,:] = process_discourse_relation_label(discourse_label, discourse_type)
					#para_y[i,:] = process_discourse_relation_label_8way(discourse_label, discourse_type)

				if torch.sum(para_y.abs()) > 0:
					paras_sentence_list.append(para_sentence_list)
					paras_y_list.append(para_y)
			para_sentence_list = []
			discourse_list = []

		sentence_index += 1

	return paras_sentence_list,paras_y_list

def process_sentence(sentence, posner_flag = True, sentencemarker = False, paramarker = False):
	if posner_flag:
		word_list = nltk.word_tokenize(sentence)
		if sentence not in sentence_pos_ner_dict:
			pos_list = pos_tager.tag(word_list)
			ner_list = ner_tager.tag(word_list)
			sentence_pos_ner_dict[sentence] = (copy.deepcopy(pos_list),copy.deepcopy(ner_list))
		else:
			pos_list = copy.deepcopy(sentence_pos_ner_dict[sentence][0])
			ner_list = copy.deepcopy(sentence_pos_ner_dict[sentence][1])
			assert len(pos_list) == len(word_list)

		if sentencemarker:
			pos_list.insert(0,('<SOS>',''))
			ner_list.insert(0,('<SOS>',''))

			pos_list.append(('<EOS>',''))
			ner_list.append(('<EOS>',''))

		if paramarker:
			pos_list.insert(0,('<ParaBoundary>',''))
			ner_list.insert(0,('<ParaBoundary>',''))
		
		return tansfer_word2vec((pos_list,ner_list), posner_flag = True)
	else:
		word_list = nltk.word_tokenize(sentence)
		#word_list = st.tokenize(sentence)
		
		if sentencemarker:
			word_list.insert(0,'<SOS>')
			word_list.append('<EOS>')
		
		if paramarker:
			word_list.insert(0,'<ParaBoundary>')

		return tansfer_word2vec(word_list, posner_flag = False)

def fold_word2vec(fold_discourse_relation_list, posner_flag = True, sentencemarker = False, connectivemarker = False):
	global para_length_list
	print "total number of documents:" + str(len(fold_discourse_relation_list))
	y_total = torch.zeros(len(discourse_sense_list))
	y_explicit =  torch.zeros(len(discourse_sense_list))
	y_implicit =  torch.zeros(len(discourse_sense_list))

	#para_sentence_lists = []
	para_embedding_list = []
	para_label_length_list = []
	eos_position_lists = []
	connective_position_lists = []
	y_list = []

	for i in range(len(fold_discourse_relation_list)):
		if i % 10 == 0:
			print i

		doc_sentence_list,doc_discourse_dict = fold_discourse_relation_list[i][0],fold_discourse_relation_list[i][1]
		paras_sentence_list,paras_y_list= process_doc_paras_labels(doc_sentence_list, doc_discourse_dict)
		
		if len(paras_sentence_list) == 0:
			continue

		for para_sentence_list, y in zip(paras_sentence_list, paras_y_list):
			print para_sentence_list
			print y

			para_length_list.append(len(para_sentence_list))
			#para_sentence_lists.append(para_sentence_list)

			y_total = y_total + torch.sum(y.abs(),0)
			y_explicit = y_explicit + torch.sum(y.clamp(-1,0).abs(),0)
			y_implicit = y_implicit + torch.sum(y.clamp(0,1),0)
			para_label_length_list.append(torch.sum(y.abs()))

			sentence_embedding_list = []
			eos_position_list = []
			connective_position_list = []
			para_length = 0
			for sentence in para_sentence_list:
				sentence_embedding = process_sentence(sentence, posner_flag = posner_flag, sentencemarker = sentencemarker)
				sentence_embedding_list.append(sentence_embedding)

				if connectivemarker:
					if sentence_startwith_connective(sentence):
						if sentence.strip()[0] == '"':
							connective_position_list.append(para_length+1)
						else:
							connective_position_list.append(para_length)
					else:
						connective_position_list.append(-1)

				para_length = para_length + sentence_embedding.size(0)
				eos_position_list.append(para_length)


			assert len(eos_position_list) - 1 == y.size(0)
			para_embedding = torch.cat(sentence_embedding_list)
			para_embedding = para_embedding.view(1,-1, para_embedding.size(-1))

			para_embedding = Variable(para_embedding, requires_grad = False)
			y = Variable(y, requires_grad = False)

			para_embedding_list.append(para_embedding)
			eos_position_lists.append(eos_position_list)
			connective_position_lists.append(connective_position_list)
			y_list.append(y)

	'''with open('./data/pdtb_implicit_moreexplicit_discourse_paragraph_multilabel_devdata.pkl','w+') as f:
		cPickle.dump([para_sentence_lists,y_list],f)
		f.close()'''

	print 'Discourse relation distribution'
	print y_total
	print 'Explicit discourse relation distribution'
	print y_explicit
	print 'Implicit discourse relation distribution'
	print y_implicit
	
	if connectivemarker:
		return (para_embedding_list,para_label_length_list,eos_position_lists,connective_position_lists),y_list
	else:
		return (para_embedding_list,para_label_length_list,eos_position_lists),y_list

end_list = ('...', 'a.k.a.', 'A.E.', 'A.L.', 'A.P.', 'Alex.', 'Ark.)', '(c.i.f.)', '(f.o.b.)', 'C.B.', 'C.J.', 'C.J.B.', 'C.R.', 'C.W.', 'Del.', 'D.N.', 'D.S.', 'D.T.', 'E.C.', 'E.E.', 'E.W.', 'F.A.O.', 'F.C.', 'F.E.', 'F.H.', 'F.W.', 'G.O.', 'G.m.b.', 'G.m.b.H.', 'Gov.', 'H.G.', 'H.H.', 'H.L.', 'H.R.', 'I.E.P.', 'I.M.', 'I.W.', 'Ind.', 'Ind.)', 'J.', 'J.D.', 'J.E.', 'J.F.', 'J.L.', 'J.M.', 'J.V.', 'J.V.)', 'J.X.', 'Jos.A.', 'L.J.', 'L.L.', 'L.M.', 'M.A.', 'M.R.', 'Messrs.', 'Mr.', 'Mrs.', 'Ms.', 'Mass.)', 'Md.)', 'Miss.)', 'Mo.)', 'Mont.)', 'Neb.', 'No.', 'Nos.', 'R.D.', 'R.L.', 'R.P.', 'R.R.', 'Reps.', 'Sen.', 'Sens.', 'T.D.', 'T.T.', 'U.S.', 'V.H.', 'W.A.', 'W.D.', 'W.G.', 'W.I.', 'W.N.', 'W.T.', 'labs at the U.')
def sentence_tokenizer(sentence):
	if '. . .' in sentence:
		sentence = sentence.replace('. . .','...')

	sentences_list = nltk.sent_tokenize(sentence)
	corrected_sentences_list = []

	i = -1
	while i < len(sentences_list) - 1:
		i = i + 1
		if sentences_list[i].endswith(end_list) and i+1 < len(sentences_list):
			if sentences_list[i+1].endswith(end_list) and i+2 < len(sentences_list):
				if sentences_list[i+2].endswith(end_list) and i+3 < len(sentences_list):
					corrected_sentences_list.append(sentences_list[i] + ' ' + sentences_list[i+1]+ ' ' + sentences_list[i+2]+ ' ' + sentences_list[i+3] )
					i = i + 3
				else:
					corrected_sentences_list.append(sentences_list[i] + ' ' + sentences_list[i+1]+ ' ' + sentences_list[i+2] )
					i = i + 2
			else:
				corrected_sentences_list.append(sentences_list[i] + ' ' + sentences_list[i+1])
				i = i + 1
		else:
			corrected_sentences_list.append(sentences_list[i])
	return corrected_sentences_list

def clean_arg(arg):
	newarg = ''

	for i in range(len(arg)):
		newarg = newarg + arg[i]
		if arg[i] == '.' and i - 2>= 0 and i + 2< len(arg) and '.' not in arg[i-2:i] and '.' not in arg[i+1:i+3]:
			if arg[i+1].isalpha() and arg[i+1:i+3] not in ['Va']  and i + 4 < len(arg) and arg[i+1:i+5] not in ['Cal-']: 
										     #special case of wsj_0328                           #special case of wsj_1495
				newarg = newarg + ' '

		if arg[i] == '?' or arg[i] == '!':
			if i + 1< len(arg) and arg[i+1].isalpha():
				newarg = newarg + ' '
	
	return sentence_tokenizer(newarg)

def split_arg(arg,doc_sentence_list):
	for i in range(1,len(arg)):
		if arg[i]== ' ':
			if search_sentence_index(doc_sentence_list,arg[i+1:],0) > search_sentence_index(doc_sentence_list,arg[:i],0):
				arg = [arg[:i], arg[i:]]
				break
	return arg

def sentence_contain_arg(sentence,arg):
	sentence = re.sub('[^\w\s]','',sentence).split()
	arg = re.sub('[^\w\s]','',arg).split()

	for word in arg:
		if word not in sentence:
			return False
		if arg.count(word) > 1 and sentence.count(word) < arg.count(word):
			return False
	
	return True

def search_sentence_index(doc_sentence_list,arg,start):
	for i in range(start,len(doc_sentence_list)):
		if arg in doc_sentence_list[i] and len(arg.split())>2:
			return i
		if sentence_contain_arg(doc_sentence_list[i],arg):
			return i

	return -1

def update_doc_discourse_dict(doc_discourse_dict, split_sentence_index):
	new_doc_discourse_dict = {}

	for argpair in doc_discourse_dict:
		arg1_index,arg2_index = argpair[0],argpair[1]
		discourse_type,discourse_label = doc_discourse_dict[argpair][0],doc_discourse_dict[argpair][1]

		if arg2_index <= split_sentence_index:
			new_doc_discourse_dict[argpair] = (discourse_type,discourse_label)
		else:
			new_doc_discourse_dict[(arg1_index+1,arg2_index+1)] = (discourse_type,discourse_label)

	return new_doc_discourse_dict

def extract_implicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict):
	global implicit_filter_count,double_label_count

	prev_index = 0
	for i in range(0,len(pipe_file_lines)):
		pipe_line = pipe_file_lines[i].split('|')
		discourse_type = pipe_line[0]
		if discourse_type not in ['Implicit','AltLex','EntRel']:
			continue

		discourse_label = pipe_line[11]
		#catch double label
		if pipe_line[12] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[12]
		if pipe_line[13] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[13]

		fake_connective = pipe_line[9]
		arg1 = pipe_line[24]
		arg2 = pipe_line[34]

		if len(arg1) <=2 and pipe_line[31] != '': #special case for wsj_1856
			arg1 = pipe_line[31]

		arg1 = clean_arg(arg1)[-1]
		arg2 = clean_arg(arg2)[0]

		arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
		if arg1_index != -1:
			prev_index = arg1_index
		arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1 in ['some structural damage to headquarters and no power']: #special case for wsj_1915
			arg1_index = search_sentence_index(doc_sentence_list,arg1, prev_index+1)
			prev_index = arg1_index

		# handle the special case that two arguments are seperated by ':' or ';'
		if arg1_index == -1 or arg2_index == -1:
			if arg1_index == -1:
				if ';' in arg1:
					arg1 = arg1.split(';')[-1]
				if ':' in arg1:
					arg1 = arg1.split(':')[-1]
				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			if arg2_index == -1:
				if ';' in arg2:
					arg2 = arg2.split(';')[0]
				if ':' in arg2:
					arg2 = arg2.split(':')[0]
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		# handle the special case that two arguments are seperated by ':' or ';'
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			for j in range(len(sentence)):
				if sentence[j] in [';',':','.'] or ( j < len(sentence)-1 and sentence[j:j+2] in ['--']):
					if sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,arg1_index+1)

			
		if arg2_index - arg1_index == 2 and arg1_index != -1 and arg2_index!=-1:
			if '"' in doc_sentence_list[arg1_index]:
				#print doc_sentence_list[arg1_index] + ' ' + doc_sentence_list[arg1_index+1]
				doc_sentence_list = doc_sentence_list[:arg1_index] + [doc_sentence_list[arg1_index] + ' ' + doc_sentence_list[arg1_index+1]] + doc_sentence_list[arg2_index:]

				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,arg1_index+1)

		if arg1_index == -1 or arg2_index == -1:
			if arg1_index == -1:
				arg1 = split_arg(arg1,doc_sentence_list)
				if type(arg1) == type([]):
					arg1 = arg1[-1]
				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)

			if arg2_index == -1:
				arg2 = split_arg(arg2,doc_sentence_list)			
				if type(arg2) == type([]):
					arg2 = arg2[0]
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		if arg1_index == -1 or arg2_index == -1:
			implicit_filter_count = implicit_filter_count + 1	
			#print arg1 if arg1_index == -1 else arg2
			#print prev_index
			#print arg1_index,arg1
			#print arg2_index,arg2
			continue

		if arg1_index - arg2_index == 1 or arg2_index - arg1_index == 1:
			prev_index = max(arg1_index,arg2_index)

			if arg1_index > arg2_index:
				tmp_index = arg1_index
				arg1_index = arg2_index
				arg2_index = tmp_index

			if (arg1_index,arg2_index) not in doc_discourse_dict:
				doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
			else:
				assert (discourse_type,discourse_label) == doc_discourse_dict[(arg1_index,arg2_index)]

		elif arg1_index == arg2_index:
			implicit_filter_count = implicit_filter_count + 1	
			#print arg1
			#print arg2
			#print doc_sentence_list[arg1_index]
		else:
			implicit_filter_count = implicit_filter_count + 1
			#print prev_index
			#print arg1_index, arg1
			#print doc_sentence_list[arg1_index]
			#print arg2_index, arg2
			#print doc_sentence_list[arg2_index]

	return doc_sentence_list, doc_discourse_dict

def extract_explicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict,doc_paragraph_first_sentence_list):
	global explicit_count,double_label_count,explicit_filter_count

	prev_index = 0
	for i in range(0,len(pipe_file_lines)):
		pipe_line = pipe_file_lines[i].split('|')
		discourse_type = pipe_line[0]
		if discourse_type not in ['Explicit']:
			continue

		discourse_label = pipe_line[11]
		#catch double label
		if pipe_line[12] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[12]
		if pipe_line[13] != '':
			double_label_count = double_label_count + 1
			discourse_label = discourse_label + '|' + pipe_line[13]

		connective = pipe_line[5]
		assert len(connective) > 0
		arg1 = pipe_line[24]
		arg2 = pipe_line[34]

		arg1 = clean_arg(arg1)[-1]
		arg2 = clean_arg(arg2)[0]

		arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
		arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1 or arg2_index == -1:
			if arg1_index == -1:
				arg1 = split_arg(arg1,doc_sentence_list)
				if type(arg1) == type([]):
					arg1 = arg1[-1]
				arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)

			if arg2_index == -1:
				arg2 = split_arg(arg2,doc_sentence_list)			
				if type(arg2) == type([]):
					arg2 = arg2[0]
				arg2_index =  search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)


		# catch the explicit discourse relations within sentence, split the sentence into two arguments
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			for j in range(len(sentence)):
				if sentence[j] in [',',':',';','.','?','!']:
					if (sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],arg1)):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			
			if arg1 in ['you give parties']: #special case of wsj_1367
				arg1_index = arg1_index+1

			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		if arg1_index == arg2_index and arg1_index != -1:
			for j in range(len(sentence)):
				if sentence[j] in [' ','-']:
					if (sentence_contain_arg(sentence[:j],arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],arg1)):
						doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
						doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
						break

			arg1_index =  search_sentence_index(doc_sentence_list,arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)

		# arg2, connective arg1, arg2
		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			flag = False

			for k in range(len(arg1.split())/2+1):
				tmp_arg1 = ' '.join(arg1.split(' ')[k:])
				for j in range(len(sentence)):
					if sentence[j] in [',',':',';','.','-','?','!',' ']:
						if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
							doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
							doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
							flag = True
							break
				if flag:
					break

			if not flag:			
				for k in range(1,len(arg1.split())/2+1):
					tmp_arg1 = ' '.join(arg1.split(' ')[:-k])
					for j in range(len(sentence)):
						if sentence[j] in [',',':',';','.','-','?','!',' ']:
							if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],arg2)) or (sentence_contain_arg(sentence[:j],arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
								doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
								doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
								flag = True
								break
					if flag:
						break

			arg1_index =  search_sentence_index(doc_sentence_list,tmp_arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,arg2,prev_index)

		if arg1_index == arg2_index and arg1_index != -1:
			sentence = doc_sentence_list[arg1_index]
			flag = False

			replace_arg1 = arg2
			replace_arg2 = arg1

			for k in range(len(replace_arg1.split())/2+1):
				tmp_arg1 = ' '.join(replace_arg1.split(' ')[k:])
				for j in range(len(sentence)):
					if sentence[j] in [',',':',';','.','-','?','!',' ']:
						if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],replace_arg2)) or (sentence_contain_arg(sentence[:j],replace_arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
							doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
							doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
							flag = True
							break
				if flag:
					break

			if not flag:			
				for k in range(1,len(replace_arg1.split())/2+1):
					tmp_arg1 = ' '.join(replace_arg1.split(' ')[:-k])
					for j in range(len(sentence)):
						if sentence[j] in [',',':',';','.','-','?','!',' ']:
							if (sentence_contain_arg(sentence[:j],tmp_arg1) and sentence_contain_arg(sentence[j+1:],replace_arg2)) or (sentence_contain_arg(sentence[:j],replace_arg2) and sentence_contain_arg(sentence[j+1:],tmp_arg1)):
								doc_sentence_list = doc_sentence_list[:arg1_index] + [sentence[:j+1],sentence[j+1:]] + doc_sentence_list[arg1_index+1:]
								doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg1_index)
								flag = True
								break
					if flag:
						break

			arg1_index =  search_sentence_index(doc_sentence_list,tmp_arg1,prev_index)
			arg2_index = search_sentence_index(doc_sentence_list,replace_arg2,prev_index)		

		if arg1_index == -1:
			arg1_index =  search_sentence_index(doc_sentence_list,arg1,0)
		if arg2_index == -1:
			arg2_index =  search_sentence_index(doc_sentence_list,arg2,0)
		

		# if arg2 is the first sentence of any paragraphs and arg1 locates more than one sentence from arg2, copy the arg1 in front of arg2 
		if arg2_index - arg1_index >= 2 and arg1_index != -1 and arg2_index != -1:
			if doc_sentence_list[arg2_index] in doc_paragraph_first_sentence_list and (arg2_index-1,arg2_index) not in doc_discourse_dict:
				doc_sentence_list = doc_sentence_list[:arg2_index] + [doc_sentence_list[arg1_index]] + doc_sentence_list[arg2_index:]
				doc_discourse_dict = update_doc_discourse_dict(doc_discourse_dict,arg2_index)

				arg1_index =  search_sentence_index(doc_sentence_list,arg1,arg1_index+1)
				arg2_index = search_sentence_index(doc_sentence_list,arg2,arg2_index+1)

		if arg1_index == -1 or arg2_index == -1:
			explicit_filter_count += 1
			#print arg1 if arg1_index == -1 else arg2
			#print prev_index
			#print arg1_index,arg1
			#print arg2_index,arg2
			#print doc_sentence_list
			continue

		if arg1_index - arg2_index == 1 or arg2_index - arg1_index == 1:
			#assert connective in doc_sentence_list[arg1_index]
			prev_index = max(arg1_index,arg2_index)

			if arg1_index > arg2_index:
				tmp_index = arg1_index
				arg1_index = arg2_index
				arg2_index = tmp_index

			if (arg1_index,arg2_index) not in doc_discourse_dict:
				explicit_count = explicit_count + 1
				doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
			else:
				if doc_discourse_dict[(arg1_index,arg2_index)][0] in ['EntRel','AltLex']:
					explicit_count = explicit_count + 1
					doc_discourse_dict[(arg1_index,arg2_index)] = (discourse_type,discourse_label)
				elif doc_discourse_dict[(arg1_index,arg2_index)][0] in ['Explicit']:
					double_label_count = double_label_count + 1
					explicit_count = explicit_count + 1
					prev_discourse_label = doc_discourse_dict[(arg1_index,arg2_index)][1]

					doc_discourse_dict[(arg1_index,arg2_index)]  = (discourse_type,prev_discourse_label + '|' + discourse_label)
					#print '-----------------------------------'
					#print arg1_index,arg1
					#print arg2_index,arg2
					#print doc_sentence_list[arg1_index]
					#print doc_sentence_list[arg2_index]
					#print prev_discourse_label
					#print discourse_label

				else:
					explicit_filter_count += 1
					#print '-----------------------------------'
					#print arg1_index,arg1
					#print arg2_index,arg2
					#print doc_sentence_list[arg1_index]
					#print doc_sentence_list[arg2_index]
					#print doc_discourse_dict[(arg1_index,arg2_index)]
					#print discourse_label
					pass
		elif arg1_index == arg2_index:
			explicit_filter_count += 1
			#print '------------------------'
			#print discourse_label
			#print arg1
			#print arg2
			#print doc_sentence_list[arg1_index]
			pass
		else:
			explicit_filter_count += 1
			#print '------------------------'
			#print discourse_label
			#print arg1_index, arg1
			#print doc_sentence_list[arg1_index]
			#print arg2_index, arg2
			#print doc_sentence_list[arg2_index]
			pass

	return doc_sentence_list,doc_discourse_dict

para_length_list = []
explicit_count = 0
implicit_filter_count = 0
explicit_filter_count = 0
double_label_count = 0
def process_doc(pipe_file_path,raw_file_path):
	pipe_file = open(pipe_file_path,'r')
	raw_file = open(raw_file_path,'r')

	pipe_file_lines = pipe_file.readlines()
	raw_file_lines =  raw_file.readlines()

	doc_paragraph_first_sentence_list = []
	doc_sentence_list = []
	for i in range(2,len(raw_file_lines)):
		line = raw_file_lines[i].replace('\n','').strip()
		if len(line) > 0:
			sentences_list = sentence_tokenizer(line)
			doc_sentence_list = doc_sentence_list + sentences_list
			doc_paragraph_first_sentence_list.append(sentences_list[0])

	doc_discourse_dict = {}
	doc_sentence_list, doc_discourse_dict = extract_implicit_relation(pipe_file_lines,doc_sentence_list, doc_discourse_dict)

	if len(doc_sentence_list) >= 2 and len(doc_discourse_dict) >= 1:
		doc_sentence_list, doc_discourse_dict = extract_explicit_relation(pipe_file_lines,doc_sentence_list,doc_discourse_dict,doc_paragraph_first_sentence_list)

	return doc_sentence_list,doc_discourse_dict

def process_fold(fold_list):
	fold_doc_list = []

	pipe_file_path = './dataset/pdtb_v2/data_t/pdtb/'
	raw_file_path = './dataset/pdtb_v2/data/raw/wsj/'

	for fold in fold_list:
		print 'fold: ' + str(fold)
		fold_pipe_file_path = os.path.join(pipe_file_path,fold)
		fold_raw_file_path = os.path.join(raw_file_path,fold)

		for fold_file in sorted(os.listdir(fold_pipe_file_path)):
			filename =  fold_file.split('.')[0]

			if len(filename) == 0:
				continue

			pipe_file = os.path.join(fold_pipe_file_path,filename+'.pipe')
			raw_file = os.path.join(fold_raw_file_path,filename)

			doc_sentence_list,doc_discourse_dict = process_doc(pipe_file,raw_file)

			if len(doc_sentence_list) >= 2 and len(doc_discourse_dict) >= 1:
				fold_doc_list.append((doc_sentence_list,doc_discourse_dict))

	X,Y = fold_word2vec(fold_doc_list, posner_flag = True, sentencemarker = False, connectivemarker = False)
	return X,Y

training_fold_list = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
dev_fold_list = ['00','01']
test_fold_list = ['21','22']

dev_X,dev_Y = process_fold(dev_fold_list)
train_X,train_Y = process_fold(training_fold_list)
test_X, test_Y = process_fold(test_fold_list)
print 'implicit filter count: ' + str(implicit_filter_count)
print 'explicit filter count: ' + str(explicit_filter_count)
print 'explicit count: ' + str(explicit_count)
print 'double label count: ' + str(double_label_count)
print 'average para length: ' + str(sum(para_length_list) / float(len(para_length_list)))
print 'para length distribution: ' + str(np.unique(para_length_list, return_counts=True))

'''with open('../resource/pdtb_sentence_pos_ner_dict.pkl','w') as f:
	cPickle.dump(sentence_pos_ner_dict,f)
	f.close()'''

pdtb_data = {}
pdtb_data['dev_X'] = dev_X
pdtb_data['dev_Y'] = dev_Y
pdtb_data['train_X'] = train_X
pdtb_data['train_Y'] = train_Y
pdtb_data['test_X'] = test_X
pdtb_data['test_Y'] = test_Y

outfile = open('data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt','w')
torch.save(pdtb_data,outfile)