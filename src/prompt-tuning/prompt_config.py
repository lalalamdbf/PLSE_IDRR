from prompt import load_plm
from prompt.manual_template import ManualTemplate
from prompt.manual_verbalizer import ManualVerbalizer
import torch

class PromptConfig(object):
    def __init__(self, args):
        
        plm, tokenizer, model_config, WrapperClass = load_plm(args.pretrain_file)
        self.plm = plm
        self.tokenizer = tokenizer
        self.wrapperclass = WrapperClass
        if args.use_pretrain:
            net_state = torch.load(args.plse_pretrain_file)['net']
            self.plm.load_state_dict(net_state, strict=False)
    
        self.promptTemplate = ManualTemplate(
            text='{"placeholder":"text_a"}{"special": "</s>"}{"mask"}{"special": "</s>"}{"placeholder":"text_b"}',
            tokenizer=tokenizer
        )

        if args.num_rels == 4:
            # pdtb2 4
            classes = ["Comparision", "Contingency", "Expansion", "Temporal"]
            label_words={
                "Comparision": ["but", "however","although","though"],
                "Contingency": ["because","so","thus","therefore", "consequently"],
                "Expansion": ["instead","rather","and","also","furthermore", "example","instance","fact","indeed","particular","specifically"],
                "Temporal": ["then", "before","after","meanwhile","when"],
            }
        elif args.num_rels == 11:
            # pdtb2 11
            classes = list(range(11))
            label_words={
                0: ['although','though','however'],
                1: ["but"],
                2: ["because",'so','thus','consequently',"therefore"],
                3: ['since','as'],
                4: ["instead",'rather'],
                5: ["and","also","furthermore",'fact'],
                6: ["example",'instance'],
                7: ["finally"],
                8: ["specifically",'particular','indeed'],
                9: ["then",'before','after'],
                10:["meanwhile","when"]
                }
        else: 
            # conll
            classes = list(range(14))
            label_words={
            0:  ['however','although'],
            1:  ["but"],
            2:  ["because"],
            3:  ['so','thus','consequently','therefore'],
            4:  ["if"],
            5:  ["unless"],
            6:  ["instead",'rather'],
            7:  ["and","also","furthermore",'fact'],
            8:  ["except"],
            9:  ["example",'instance'],
            10: ["specifically",'particular','indeed'],
            11: ["then",'before'],
            12: ['after'],
            13: ["meanwhile",'when']
             }
            
        self.promptVerbalizer = ManualVerbalizer(
            classes=classes,
            label_words=label_words,
            tokenizer=tokenizer,
            post_log_softmax= False, 
        )
    
    def get_plm(self):
        return self.plm
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_wrapperclass(self):
        return self.wrapperclass
    
    def get_template(self):
        return self.promptTemplate

    def get_verbalizer(self):
        return self.promptVerbalizer