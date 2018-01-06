#/usr/bin/env python
# -*- coding: utf-8 -*-

# Gymnasion 
# Kyle Booten, 2018

import random
import json
import gensim
from keras.models import load_model
from keras.preprocessing import sequence
from ascii_graph import Pyasciigraph
import pickle
import numpy as np
import gensim
from langdetect import detect
from unidecode import unidecode
import spacy
from nltk.corpus import wordnet as wn
from nltk import ngrams,tokenize
from collections import defaultdict,deque
import re



#######################
# loading models/data #
#######################

## load SpaCy model
print "Loading SpaCy model."
nlp = spacy.load('en')


## load word2vec model
print "Loading word2vec model."
word2vec_path = "/Users/kyle/Desktop/GoogleNews-vectors-negative300.bin" ## [kyle: remove]
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
model.init_sims(replace=True)


## word pairs mined from Project Gutenberg
with open('data/noun2adj.json','r') as f:
    noun2adj = json.load(f)
with open('data/noun2v_o.json','r') as f:
    noun2v_o = json.load(f)


## lstm authorship classifier
print "Loading keras model for authorship classification."
auth_model = load_model('data/auth_identification.h5') 
with open('data/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('data/num2auth.json','r') as f:
    num2auth = json.load(f)
authors =num2auth.values()


## quotes
print "Loading and processing quotes."
with open('data/gymnasion_quotes.json','r') as f: ## [kyle: remove]
    quotes = json.load(f)
quotes_processed = []
for q,author in quotes:
    try:
        quotes_processed.append((list(nlp(unicode(q)).sents)[0],author))
    except:
        pass

### parts of quotes
quote_stubs = []
for q,auth in quotes_processed:
    try:
        maximum = min(len(q)-3,14)
        quote_stubs.append(q[:random.randrange(6,maximum)])
    except:
        pass


## a bit of nltk stuff
### lemmatizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
### also stopwords
from nltk.corpus import stopwords
stops = stopwords.words('english')


## mapping between words
conversions={
    "a":"the",
    "my":"your",
    "your":"that",
}


stops = stops + ["whose","less","thee","thine","thy","thou","one"]
pluralnouns = ["people","children","men","women","feet","sheep","teeth","fish"]

class Gymnasion(object):
    
    def __init__(self):
        self.line = ""
        self.poem = ""
        self.to_imitate = ""
        self.imitation_attempt_counter = 0
        self.banished_words = []   
        self.used_quotes = []
        self.used_quote_stubs = []
        self.just_used_nouns = deque(maxlen=10)
        self.boredom = 0
        self.n_quotes_to_sample = len(quotes_processed)/2 
        self.n_quotes_stubs_to_sample = len(quote_stubs)/2
        self.gym_functions = [
        self.return_a_command_for_comparison,
        self.return_a_command_to_recall_earlier_noun,
        self.return_comment_about_ents,
        self.return_question_about_adjectives,
        self.return_question_about_object,
        self.return_question_about_verb,
        self.return_question_about_verb_and_object,
        self.return_quote_according_to_word_mover_distance,
        self.return_quote_stub_according_to_word_mover_distance,
        self.return_comment_about_authorial_imitation,
        self.return_repetition_judgment_syntax,
        self.return_word_banishment,
        self.return_check_words_in_sentence,
        self.return_word2vec_suggestions,
        ]
        
    ##############
    # adjectives #
    ##############
    
    
    def _format_remark_upon_adj_of_nouns(self,noun,adjectives,pos):
        a_string = "What sort of %s?" % noun.lower()
        adjectives = list(set(adjectives))
        for adj in adjectives:
            a_string+=" "+adj.title()+"?"
        return a_string
    
    def _format_remark_upon_adj_of_nouns2(self,noun,adjectives,pos):
        adjectives = list(set(adjectives))
        adj0,adj1 = adjectives[0],adjectives[1]
        if pos=="NN":
            a_string = "Ah yes...a %s.  But be specific: it was a %s %s, not a %s one." % (noun.lower(), adj0, noun, adj1)
            vowels = 'aeiou'
            if (adj0[0] in vowels)==True:
                a_string = a_string.replace("it was a","it was an")
            if (adj1[0] in vowels)==True:
                a_string = a_string.replace("not a","not an")
        elif pos=="NNS":
            a_string = "Now describe how %s these %s are, how %s..." % (adj0,noun, adj1)
        return a_string
    
    def return_question_about_adjectives(self, spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: comment on the adjectives -- "What sort of frog? Distant? Green?"
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        nouns = [token for token in lastsentence if token.tag_ in ["NN","NNS"]]  
        nouns = [n for n in nouns if n not in self.just_used_nouns] ## ref deque object
        random.shuffle(nouns)
        for n in nouns:
            adjs = noun2adj[n.lemma_]
            remark = random.choice([self._format_remark_upon_adj_of_nouns,self._format_remark_upon_adj_of_nouns2])
            returntext = remark(n.text,random.sample(adjs,2),n.tag_)
            self.just_used_nouns.append(n) ## add to deque object
            return returntext
  

    #################
    # verbs/objects #
    #################
    
    
    def _process_noun_and_verb_into_question(self,noun,verb):
        beginnings = ["What could","Why does","What shall","What did", "Why would", "For whom does"]
        return "%s the %s %s?" % (random.choice(beginnings),noun.lower(),verb)
        
    def _process_noun_and_object_into_question(self,noun,verb):
        beginnings = ["What could","What does","What shall","What did"]
        return "%s the %s do to the %s?" % (random.choice(beginnings),noun.lower(),verb)
        
    def return_question_about_verb(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: comment on the verb -- "What shall the wolf devour?"
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        nouns = [(token.text,token.lemma_)  for token in lastsentence if token.tag_ in ["NN","NNS"]]  
        a_noun = random.choice(nouns)
        a_verb = random.choice(noun2v_o[a_noun[1]])[0]
        to_return = self._process_noun_and_verb_into_question(a_noun[0],a_verb)
        if a_noun[0]!=a_noun[1]:
            to_return = to_return.replace(" does "," do ")
        return to_return

    def return_question_about_object(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: comment on the object -- "Who could the wolf injure?"
        rtype:string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        nouns = [token for token in lastsentence if token.tag_ in ["NN","NNS"]]  
        nouns = [n for n in nouns if n not in self.just_used_nouns]
        random.shuffle(nouns)
        for n in nouns:
            a_verb,an_object = random.choice(noun2v_o[n.lemma_])
            to_return = self._process_noun_and_object_into_question(n.text,an_object)
            if (n.text!=n.lemma_) or (n.text in pluralnouns):
                to_return = to_return.replace(" does "," do ")
            self.just_used_nouns.append(n)
            return to_return
    
    def return_question_about_verb_and_object(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: a comment on the verb and object -- "Why does the wolf injure the pig?"
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        nouns = [token for token in lastsentence if token.tag_ in ["NN","NNS"]]  
        nouns = [n for n in nouns if n not in self.just_used_nouns]     
        random.shuffle(nouns)
        for n in nouns:
            a_verb,an_object = random.choice(noun2v_o[n.lemma_])
            to_return = random.choice(["Which %s does the %s %s?" % (an_object,n.text.lower(),a_verb), "Why does the %s %s the %s?" % (n.text.lower(), a_verb, an_object)])
            if (n.text!=n.lemma_) or (n.text in pluralnouns):
                to_return = to_return.replace(" does "," do ")
            self.just_used_nouns.append(n)
            return to_return

    
    ######################
    # chattering archive #
    ######################
    
    
    def _format_chattering_archive(self,quote,author):
        a_string = random.choice(["Learn from this mastery","Now imitate this","Emulate this discourse","Listen and respond","No no no, not like that, like this","Student, learn from these words","Respond to a clearer voice","Absorb these words"])
        author = nlp(unicode(author))
        a_string+=": \n\""+quote.text+"\""
        ents = [ent.text for ent in author.ents if ent.label_=="PERSON"]
        if len(ents)==1:
            a_string+="\n   (%s)" % ents[0]
        return a_string
    
    def return_quote_according_to_word_mover_distance(self,spacynewtext,quotes_processed=quotes_processed,n=4):
        """
        param: spacynewtext: a SpaCy doc object
        param: quotes_processed: quotes to search
        param: n: number of top n quotes to random choose from
        returns: a quote 
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        stub = [t.text for t in lastsentence if t.tag_ in ["NN","NNS","JJ"]]  
        if len(stub)<1:
            return None  ## don't try if there are no nouns to match
        score_and_quote = []
        for q,author in random.sample(quotes_processed,self.n_quotes_to_sample):
            q_nouns = [t.text for t in q if t.tag_ in ["NN","NNS","JJ"]]  
            score_and_quote.append((model.wmdistance(stub,q_nouns),(q,author)))
        score_and_quote = sorted(score_and_quote, key=lambda x: x[0])
        score_and_quote = [(s,q) for s,q in score_and_quote if q[0] not in self.used_quotes]
        picked_quote = random.choice(score_and_quote[:n])[1]
        self.used_quotes.append(picked_quote[0])
        return self._format_chattering_archive(*picked_quote)

    def _format_chattering_archive_stub(self,quote):
        a_string = random.choice(["Now finish this true sentence:","Now follow this wordpath:","Let these words take you by the tongue:","Follow me:","Complete these words:"])
        a_string+=" \n\""+quote.text+"...\""
        return a_string
    
    def return_quote_stub_according_to_word_mover_distance(self,spacynewtext,quotes_processed=quote_stubs,n=4):
        """
        param: spacynewtext: a SpaCy doc object
        param: quotes_processed: quotes to search
        param: n: number of top n quotes to random choose from
        returns: part of a quote
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        stub = [t.lemma_.lower() for t in lastsentence if t.tag_ in ["NN","NNS","JJ"]]  
        if len(stub)<1:
            return None  ## don't try if there are no nouns to match
        score_and_quote = []
        for q in random.sample(quotes_processed,self.n_quotes_stubs_to_sample):
            q_stub = [t.lemma_.lower() for t in q if t.tag_ in ["NN","NNS","JJ"]]  
            if list(set(stub)&set(q_stub))==[]:
                score_and_quote.append((model.wmdistance(stub,q_stub),q))
        score_and_quote = sorted(score_and_quote, key=lambda x: x[0])
        score_and_quote = [(s,q) for s,q in score_and_quote if q not in self.used_quote_stubs]
        picked_quote = random.choice(score_and_quote[:n])[1]
        self.used_quote_stubs.append(picked_quote)
        return self._format_chattering_archive_stub(score_and_quote[0][1])
    
    
    
    #################################
    # word2vec word recommendations #
    #################################
    
    
    def _format_get_related_terms_by_wordnet(self,word,other_words):
        beginning = "You've sung me %s...now sing me " % word
        for o in other_words:
            o = o.replace("_"," ")
            beginning += "%s..." % o
        return beginning
    
    def _get_related_terms_by_word2vec(self,word,r=3,n=20):
        focalterm=word
        relatedwords = [word]
        relatedwords_lemmas = [wordnet_lemmatizer.lemmatize(word)]

        for i in range(r):
            topwords = [w for w,value in model.wv.most_similar(positive=[word],topn=n)]
            topwords = [w for w in topwords if wordnet_lemmatizer.lemmatize(w).lower() not in relatedwords_lemmas]
            topwords = [w for w in topwords if any(letter.isupper() for letter in w)==False]
            focalterm = random.choice(topwords)
            relatedwords.append(focalterm)
            relatedwords_lemmas.append(wordnet_lemmatizer.lemmatize(focalterm).lower())
        output_format = self._format_get_related_terms_by_wordnet
        return output_format(word,relatedwords[1:])
    
    def return_word2vec_suggestions(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns:  suggestion that the writer refer to nouns related to a noun in the input text
        rtype: string
        """
        lastsentence = list(spacynewtext.sents)[-1]
        nouns = [token.text for token in lastsentence if token.tag_ in ["NN","NNS"]]  
        return self._get_related_terms_by_word2vec(random.choice(nouns))
 

    ###################
    # word banishment #
    ###################
 

    def _format_word_repeat_violation(self,w1,w2):
        output = random.choice(["Are you even paying attention to yourself?","You forget my instructions.","How narrow your mind...","How disappointing.","Try harder."])
        output += "\n I said not to sing of %s, and yet you sing of %s..." % (w2,w1)
        return output
    
    def return_word_banishment(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: a declaration of word banishment
        rtype: string
        """
        ok_pos = ["NOUN"]
        lastsentence = list(spacynewtext.sents)[-1]
        words = [w.text for w in lastsentence if w.pos_ in ok_pos]
        b = random.choice([w for w in words if w not in self.banished_words])
        b = b.lower()
        relword = random.choice(["kin","brethren","kindred","neighbors"])
        self.banished_words.append(b)
        return "I forbid you from singing of %s or this word's %s." % (b,relword)
    
    def return_check_words_in_sentence(self,spacynewtext,min_distance=.3):
        """
        param: spacynewtext: SpaCy doc object
        param: min_distance: int that defines how similar words must to judge the banishment violated
        returns: if word banishment violated, chastisement
        rtype: a string
        """
        ok_pos = ["NOUN"] ## just nouns right now
        words = [w.text.lower() for w in spacynewtext if w.pos_ in ok_pos]
        random.shuffle(words)
        for w in words:
            for b in self.banished_words:
                if model.similarity(w,b)>=min_distance:
                    return self._format_word_repeat_violation(w,b)

                
    ###################
    # people & places #
    ###################
    
    
    def _format_comment_about_entity(self,ent,ent_type):
        if ent_type == "GPE":
            q = random.choice(["Show me %s...the taste of the people, the customs...",                               "Yes, now lure me to %s...",                              "Now describe the history of %s..."])
        elif ent_type == "PERSON":
            q = random.choice(["Now sing in praise of %s...",                              "Now sing me of the youth of %s...",                              "Now sing to me what we may learn from the sins of %s..."])
        return q % ent
    
    def return_comment_about_ents(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: comment on PERSON or GPE named entity within the sentence
        rtype: string
        """
        ents = list(spacynewtext.ents)
        random.shuffle(ents)
        return self._format_comment_about_entity(ents[0],ents[0].label_)
    
    
    ####################
    # neural imitation #
    ####################
    
    
    def _vectorize_new_sentence(self,asent):
        vec = tokenizer.texts_to_sequences([asent])
        return sequence.pad_sequences(vec,maxlen=300)[0]
    
    def _predict_sentence_author(self,asentence,model=auth_model):
        vectorized = self._vectorize_new_sentence(asentence)
        prediction= model.predict(np.array([vectorized]))[0]
        prediction_labeled = [(num2auth[str(n)],v) for n,v in enumerate(prediction)]
        prediction_labeled.append(("--Similarity Threshold--",.5))
        prediction_sorted = sorted(prediction_labeled,key=lambda x: x[1],reverse=True) 
        return prediction_sorted
    
    def return_comment_about_authorial_imitation(self,spacynewtext,given_author=None):
        """
        param: spacynewtext: SpaCy doc object
        param: given_author: a string naming a particular author to prompt (e.g. "dickinson")
        returns: a judgment of the degree to which the input text echoes one of several authors
        rtype: string
        """
        prediction_labeled = self._predict_sentence_author(unidecode(spacynewtext.text))
        graph = Pyasciigraph(float_format="{:,.3f}")
        graph_filled = "\n".join(graph.graph("# neural authorship classification #",prediction_labeled))
        if self.to_imitate=="":
            if given_author!=None:
                author=given_author
            else:
                author = random.choice([a for a in authors if a!=prediction_labeled[0][0]])
            self.to_imitate=author
            to_return = "Try that again, in the style of %s." % author.title()
            self.imitation_attempt_counter = 4
            return to_return
        else:
            top_author = prediction_labeled[0][0]
            if top_author==self.to_imitate:
                to_return = random.choice(["Good.","Satisfactory.","I hope you have learned by letting go of yourself."])
                to_return+="\n\n"
                to_return+= graph_filled
                self.to_imitate = ""
                self.imitation_attempt_counter = 0
                return to_return
            else:
                if self.imitation_attempt_counter == 1:
                    to_return = random.choice(["You are hopeless.","You lack charm, and also skill.","Keep going I suppose.  I'm starting to lose faith.", "We'll have you try again later, dear."])
                    to_return+= "\n\n"
                    to_return += graph_filled
                    self.to_imitate = ""
                    self.imitation_attempt_counter = 0
                    return to_return
                else:
                    if top_author=="--Similarity Threshold--":
                        to_return = random.choice(["Your style is muddled.","No, imitate a true style."])
                        to_return+= "\n\n"
                        to_return += graph_filled
                        self.imitation_attempt_counter-=1
                        return to_return
                    else:
                        to_return = random.choice(["No, not %s --- imitate %s."]) 
                        to_return = to_return % (top_author.title(),self.to_imitate.title())
                        to_return+= "\n\n"
                        to_return += graph_filled
                        self.imitation_attempt_counter-=1
                        return to_return
    
    
    ########################
    # syntactic repetition #
    ########################
    
    
    def _format_comment_about_repetition_of_syntax(self,repeats):
        beginning = random.choice(["Return to this rhythm:","Give me more of this wordfruit:","Eschew this tired syntax:","I grow weary of such wordstumps:","Are you trying to bore me?"])
        for r in repeats:
            beginning+="\n"
            beginning+=" "*random.randrange(3,6)
            beginning+="\""+" ".join(i.text for i in r)+"\""
        beginning = re.sub(r'\s([?!,.;])', r'\1', beginning)
        beginning = re.sub(r' n\'t',"n't",beginning)
        beginning = re.sub(r' \'t',"'t",beginning)
        return beginning
    
    
    def return_repetition_judgment_syntax(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: comment about the writer's frequently used syntax patterns
        rtype: string
        """
        limits = [(7,2),(6,2),(5,2),(4,3),(3,7)]
        for gram_number,count in limits:
            grams = defaultdict(list)
            for sent in nlp(unicode(self.poem)).sents:
                temp_grams = ngrams(sent,gram_number)
                for tg in temp_grams:
                    grams[tuple([t.pos_ for t in tg])].append(tg)
            repeated_grams = [gram for gram,examples in grams.items() if len(examples)>=count]
            ## messy way to make sure that syntax gram is in the last sentence too
            new_text_grams = list(ngrams([t.pos_ for t in spacynewtext],gram_number))
            repeated_grams = [rg for rg in repeated_grams if rg in new_text_grams]
            if repeated_grams!=[]:
                gramchoice = random.choice(repeated_grams)
                return self._format_comment_about_repetition_of_syntax(grams[gramchoice])
    
    
    ##############
    # re-turning #
    ##############

    
    def _convert_noun_chunk(self,nc,conversions=conversions):
        converted = []
        for token in nc:
            try: 
                converted.append(conversions[token.text.lower()])
            except:
                converted.append(token.text.lower())
        return " ".join(converted)
    
    def return_a_command_to_recall_earlier_noun(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: a string demands the writer return to an earlier noun chunk
        rtype: string
        """
        if self.boredom<5:
            self.boredom+=1
            return None
        self.boredom=0
        noun_chunks = list(nlp(unicode(self.poem)).noun_chunks)
        noun_chunks = [nc for nc in noun_chunks if (nc.text in spacynewtext.text)==False]
        noun_chunks = [nc for nc in noun_chunks if nc[-1].tag_ in ["NN","NNS"]]
        a_noun_chunk = random.choice(noun_chunks)
        a_noun_chunk_converted = self._convert_noun_chunk(a_noun_chunk)
        a_noun_chunk_recomposed = a_noun_chunk_converted
        beginning = random.choice(["I grow weary.","I grow bored.","You have lost the thread.","You must learn to focus.","You have strayed too far."])
        ending = random.choice(["Return to the part about %s.","Tell me more about %s."]) % a_noun_chunk_recomposed
        return beginning+" "+ending
    
    def return_a_command_for_comparison(self,spacynewtext):
        """
        param: spacynewtext: SpaCy doc object
        returns: a string that demands comparison of one noun chunk in input text and one from earlier text
        rtype: string
        """
        if self.boredom<5: 
            self.boredom+=1
            return None
        self.boredom=0
        lastsentence = list(spacynewtext.sents)[-1]
        noun_chunks_early = nlp(unicode(self.poem)).noun_chunks
        noun_chunks_early = [nc for nc in noun_chunks_early if nc[-1].tag_ in ["NN","NNS"]]
        noun_chunks_recent = [nc for nc in lastsentence.noun_chunks if nc[-1].tag_ in ["NN","NNS"]]
        noun_chunks_recent = [nc for nc in noun_chunks_recent if nc not in noun_chunks_early]
        beginning = "Now compare %s to %s." % (self._convert_noun_chunk(random.choice(noun_chunks_early)),self._convert_noun_chunk(random.choice(noun_chunks_recent)))
        ending = random.choice(["Which will last longer?","Which is deeper?","Which is more lovely?","Which is closer to the divine?","Which would you destroy, if you must destroy one?","Which is more virtuous?","Which is more sinful?"])
        return beginning + " " + ending
    
    
    #############
    # Phonaskos #
    #############

    
    def output_gym_line(self,gym_line):
        """
        Print Phonaskos's lines in gray
        """
        gym_line = ["      "+l for l in gym_line.split("\n")]
        gym_line = "\n".join(gym_line)
        return "\x1b[1;37m%s\x1b[0m" % gym_line
    
    def phonaskos(self):
        """
        Teacher of song.
        Interactive critique. 
        """       
        while True:
            temp_line = raw_input("\n")
            if temp_line.startswith("***"):
                break
            else:
                gym_line = None
                random.shuffle(self.gym_functions)

                ### prioritize checking banished words
                try:
                    self.gym_functions.insert(0, self.gym_functions.pop(self.gym_functions.index(self.return_check_words_in_sentence)))
                except:
                    pass

                ### check back in about imitation
                try:
                    if self.imitation_attempt_counter>0:
                        self.gym_functions.insert(0, self.gym_functions.pop(self.gym_functions.index(self.return_comment_about_authorial_imitation)))
                except:
                    pass

                for i in self.gym_functions:
                    try:
                        gym_line = i(nlp(unicode(temp_line)))
                    except:
                        pass

                    if gym_line!=None:
                        print
                        print self.output_gym_line(gym_line)
                        break
                        
            self.poem+="\n"+temp_line


def main():
    g = Gymnasion()
    g.phonaskos()


if __name__ == "__main__":
    main()
