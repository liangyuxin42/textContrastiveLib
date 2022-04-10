# PATH DEFINE
# download from :
# SentEval：https://github.com/princeton-nlp/SimCSE/tree/main/SentEval
# SentEval/data: https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar

import sys
PATH_TO_SENTEVAL = '/SentEval'
PATH_TO_DATA = '/SentEval/data'
wiki_for_sts = "/wiki_for_sts_32_norepeat"
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

