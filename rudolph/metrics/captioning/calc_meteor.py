import nltk
import numpy as np
import wordtodigits as w2d
from ruwordnet import RuWordNet

from rudolph.metrics.captioning.utils import postprocess
from rudolph.metrics.captioning.rumeteor import meteor_score


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
wn = RuWordNet()


def get_meteor(pred, true):
    hypothesis = nltk.word_tokenize(w2d.convert(pred))
    reference = [nltk.word_tokenize(pred) for pred in true]
    return meteor_score(reference, hypothesis, wordnet=wn)


def calc_meteor(true_json, pred_json):
    scores = []
    for key in pred_json:
        hyps = pred_json[key]
        meteor = 0
        ## обрабатываем только первый текст, если пришла картинка пропускаем ее
        for hyp in hyps:
            if hyp['type'] == 'text':
                ## приходит строка
                hyp = hyp['content'] 
                for i in range(len(true_json[key][0]['content'])):
                    ## у gt несколько вариантов в списке, сравниваем с каждым
                    ref = true_json[key][0]['content'][i]
                    ref_p, hyp_p = postprocess(ref, hyp, language="ru", detokenize_after = True, tokenize_after=True, lower=True)
                    meteor_new = get_meteor(hyp_p, [ref_p]) 
                    if meteor_new > meteor:
                        meteor = meteor_new
                    print(ref_p, hyp_p)
                    print(meteor)
                break
        scores.append(meteor)
    return np.mean(np.array(scores))