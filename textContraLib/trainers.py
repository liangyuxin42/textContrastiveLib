from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers.trainer import Trainer,TrainingArguments
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    BertTokenizer,
)
from tools import *
from PATH import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = load_from_disk(wiki_for_sts)

# override the evaluate method
class SimCSETrainer(Trainer):
    def __init__(self,**paraments):
        super().__init__(**paraments)
        self.best_sts = 0.0
        self.best_pool_sts = 0.0
        self.corr_record = []

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        self.model.eval()
        corr = evalModel_dev(self.model,tokenizer)
        self.corr_record.append(corr)

        if corr > self.best_sts:
            self.best_sts = corr
            evalModel_all(self.model,tokenizer)
            self.save_model(self.args.output_dir+"\\best-model")

        self.model.train()
        print('corr before pooler:',corr,'\n max corr ',self.best_sts)
        metrics = {'corr before pooler':corr}
        self.log(metrics)

        return metrics