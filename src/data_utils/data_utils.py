from collections import defaultdict
import re 


MAX_CONTEXT_LENGTH=350 # measured in words
WORDS_PER_EVENT=10 
MAX_LENGTH=512
MAX_TGT_LENGTH=40

def get_template(ex, index, ontology_dict, tokenizer):
    event_type = ex['event_mentions'][index]['event_type']

    role2arg = defaultdict(list)
    for argument in ex['event_mentions'][index]['arguments']:
        role2arg[argument['role']].append(argument)
    role2arg = dict(role2arg)

    arg_idx2text = defaultdict(list)
    for role in role2arg.keys():
        if role not in ontology_dict[event_type]:
            continue
        for i,argument in enumerate(role2arg[role]):
            arg_text = argument['text']
            if i < len(ontology_dict[event_type][role]):
                arg_idx = ontology_dict[event_type][role][i]
            else:
                arg_idx = ontology_dict[event_type][role][-1]
            arg_idx2text[arg_idx].append(arg_text)
    
    template = ontology_dict[event_type]['template']
    input_template = re.sub(r'<arg\d>', '<arg>',template)
    for arg_idx, text_list in arg_idx2text.items():
        text = ' and '.join(text_list)
        template = re.sub('<{}>'.format(arg_idx), text, template)
    output_template = re.sub(r'<arg\d>','<arg>', template)

    space_tokenized_input_template = input_template.split()
    tokenized_input_template = [] 
    for w in space_tokenized_input_template:
        tokenized_input_template.extend(tokenizer.tokenize(w, add_prefix_space=True))

    space_tokenized_output_template = output_template.split()
    tokenized_output_template = [] 
    for w in space_tokenized_output_template:
        tokenized_output_template.extend(tokenizer.tokenize(w, add_prefix_space=True))
    return tokenized_input_template, tokenized_output_template

def get_context_sentence_selection(ex, index, max_length, related_sent = None):
        '''
        RETURN:  
        context: part of the context with the center word and no more than max length.
        offset: the position of the first token of context in original document
        '''
        trigger = ex['event_mentions'][index]['trigger']
        center_sent = trigger['sent_idx']
        offset = sum([len(ex['sentences'][idx][0]) for idx in range(center_sent)])

        sents = list(related_sent[center_sent])

        def sents_words(sent_list):
            context_words = 0
            for sent in sent_list:
                context_words += len(ex['sentences'][sent][0])
            return context_words
        
        def update_sents(sent_list):
            if center_sent - sent_list[0] < sent_list[-1] - center_sent:
                return sent_list[:-1]
            return sent_list[1:]
                
        context = []
        selected_sentence = [center_sent]
        while True:
            if sents_words(sents) <= max_length:
                for sent in sents:
                    context.append([tup[0] for tup in ex['sentences'][sent][0]])
                selected_sentence = sents
                break
            elif len(sents) == 1:
                center_context = [tup[0] for tup in ex['sentences'][center_sent][0]]
                trigger_start = trigger['start'] - offset
                start_idx = max(0, trigger_start - max_length//2)
                end_idx = min(len(center_context), trigger_start + max_length//2)
                center_context = center_context[start_idx: end_idx]
                context = [center_context]
                offset += start_idx
                break
            else:
                sents = update_sents(sents)
        
        
        # print('context:',context)
        # print(center_sent, 'offset:', offset)
        # print(selected_sentence)
        # if len(context[0]) == 0:
        #     print(ex['sentences'][center_sent])
        #     assert 1==0
        
        return context, selected_sentence, offset


def get_context(ex, index, max_length):
        '''
        RETURN:  
        context: part of the context with the center word and no more than max length.
        offset: the position of the first token of context in original document
        '''
        trigger = ex['event_mentions'][index]['trigger']
        offset = 0
        context = ex["tokens"]
        center_sent = trigger['sent_idx']
        if len(context) > max_length:
            cur_len = len(ex['sentences'][center_sent][0])
            context = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > max_length:
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start - max_length//2)
                end_idx = min(len(context), trigger_start + max_length//2)
                context = context[start_idx: end_idx]
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(center_sent)]) + start_idx
            else:
                left = center_sent -1 
                right = center_sent +1 
                
                total_sents = len(ex['sentences'])
                prev_len = 0 
                while cur_len >  prev_len:
                    prev_len = cur_len 
                    # try expanding the sliding window 
                    if left >= 0:
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= max_length:
                            context = left_sent_tokens + context
                            left -=1 
                            cur_len += len(left_sent_tokens)
                    
                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= max_length:
                            context = context + right_sent_tokens
                            right +=1 
                            cur_len += len(right_sent_tokens)
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left+1)])
        
        assert len(context) <= max_length

        assert ex["tokens"][offset:offset + len(context)] == context

        return context, offset

def simple_type(type_name):
    if ':' in type_name:
        t1,t2 = type_name.split(':')
        return t2.lower()

    _,t1,t2 = type_name.split('.')
    if t2 == "Unspecified":
        return t1.lower()
    if len(t1) < len(t2):
        return t1.lower()
    return t2.lower()


def tokenize_with_labels(tokens, labels, tokenizer, type = 'bart'):
    '''
    tokens: a list of tokens
    labels: a list of labels, each of them matches with the token
    RETURN:
    tokenized_tokens: a list of tokenized tokens
    tokenized_labels
    '''
    assert len(tokens) == len(labels)

    if type == 'bart':
        tokenized_tokens = tokenizer.tokenize(' '.join(tokens), add_prefix_space=True)
        tokenized_labels = [0]* len(tokenized_tokens)
        ptr = 0
        for idx,token in enumerate(tokenized_tokens):
            tokenized_labels[idx] = labels[ptr]
            if idx+1<len(tokenized_tokens) and (tokenized_tokens[idx+1][0] == "Ġ" or tokenized_tokens[idx+1]==' <tgr>'):
                ptr += 1 
    else:
        tokenized_tokens = tokenizer.tokenize(' '.join(tokens))
        tokenized_labels = [0]* len(tokenized_tokens)
        ptr = 0
        current_word = ''
        for idx,token in enumerate(tokenized_tokens):
            if token.startswith('##'):
                current_word +=token[2:]
            else:
                current_word +=token
            tokenized_labels[idx] = labels[ptr]
            if current_word == tokens[ptr]:                
                ptr += 1
                current_word = ''
    # for i,_ in enumerate(zip(tokens, labels)):
    #     print(i, _)
    # for i,_ in enumerate(zip(tokenized_tokens, tokenized_labels)):
    #     print(i, _)
    # assert 1==0
    assert len(tokenized_tokens) == len(tokenized_labels)

    return tokenized_tokens, tokenized_labels


def create_instance_tag_other_event_sentence_selection(ex, ontology_dict, index=0, id2entity=None, tokenizer=None,related_sent = None, sent2args = None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, selected_sentence, offset = get_context_sentence_selection(ex, index, 300, related_sent = related_sent)

    

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    center_sent = trigger['sent_idx']

    context_tag_other_events = []

    for sidx, sent in enumerate(selected_sentence):
        add_tag = defaultdict(set)

        sent_offset = sum([len(ex['sentences'][idx][0]) for idx in range(sent)])
        if sent == center_sent:
            add_tag[(trigger_start, trigger_end)].add('trigger')
            sent_offset = offset
    
        for arg in sent2args[sent]:
            if arg['eid'] == index:
                continue
            arg_start = arg['start'] - sent_offset
            arg_end = arg['end'] - sent_offset
            add_tag[(arg_start,arg_end)].add(arg["role"])
        
        add_tag = list(add_tag.items())
        add_tag = sorted(add_tag, key=lambda x:x[0][0])

        sentence_tag_other_events = []
        curr = 0
        for arg in add_tag:
            pre_words = tokenizer.tokenize(' '.join(context[sidx][curr:arg[0][0]]), add_prefix_space=True)                
            arg_words = tokenizer.tokenize(' '.join(context[sidx][arg[0][0]:arg[0][1]]), add_prefix_space=True)
            if 'trigger' not in arg[1]:
                prefix = tokenizer.tokenize(' '.join(arg[1]), add_prefix_space=True)
                sentence_tag_other_events += pre_words + [" <tag>", ] + prefix + [' </tag>', ] + arg_words 
            else:
                sentence_tag_other_events += pre_words + [" <tgr>", ] + arg_words + [' <tgr>', ]
            curr = arg[0][1]

        suf_words = tokenizer.tokenize(' '.join(context[sidx][curr:]), add_prefix_space=True)
        sentence_tag_other_events += suf_words

        context_tag_other_events.extend(sentence_tag_other_events)
    

    if len(context_tag_other_events) > MAX_LENGTH:
        context_tag_other_events = context_tag_other_events[-440:]
    return input_template, output_template, context_tag_other_events

def create_instance_normal_sentence_selection(ex, ontology_dict, index=0, id2entity=None, tokenizer=None,related_sent = None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, selected_sentence, offset = get_context_sentence_selection(ex, index, MAX_CONTEXT_LENGTH, related_sent = related_sent)

    

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    center_sent = trigger['sent_idx']

    context_tag_trigger = []

    for idx,sent in enumerate(selected_sentence):
        if sent == center_sent:
            prefix = tokenizer.tokenize(' '.join(context[idx][:trigger_start]), add_prefix_space=True)
            tgt = tokenizer.tokenize(' '.join(context[idx][trigger_start: trigger_end]), add_prefix_space=True)
            suffix = tokenizer.tokenize(' '.join(context[idx][trigger_end:]), add_prefix_space=True)
            context_tag_trigger.extend(prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix)
        else:
            sent_tokenized = tokenizer.tokenize(' '.join(context[idx]), add_prefix_space=True)
            context_tag_trigger.extend(sent_tokenized)

    # print(context_tag_trigger)
    # print('-'*80)
    return input_template, output_template, context_tag_trigger

def create_instance_normal(ex, ontology_dict, index=0, id2entity=None, tokenizer=None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH)

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    
    # get context with trigger being tagged and its argument mask list
    prefix = tokenizer.tokenize(' '.join(context[:trigger_start]), add_prefix_space=True)
    tgt = tokenizer.tokenize(' '.join(context[trigger_start: trigger_end]), add_prefix_space=True)
    suffix = tokenizer.tokenize(' '.join(context[trigger_end:]), add_prefix_space=True)
    context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
    return input_template, output_template, context_tag_trigger

def create_instance_seq(ex, ontology_dict, index=0, id2entity=None, tokenizer=None, roles_to_label = None):
    context, offset = get_context(ex, index, 450)
    labels = [0]*len(context)

    for arg in ex['event_mentions'][index]['arguments']:
        entity = id2entity[arg['entity_id']]
        start = entity['start'] - offset
        end = entity['end'] - offset
        if start < 0 or end >= len(context):
            # print(f"skip:{start},{end}-{arg['role']}")
            continue
        # if ' '.join(context[start:end]) != entity['text']:
        #     print(' '.join(context[start:end]), entity['text'])
        # assert ' '.join(context[start:end]) == entity['text']
        for i in range(start,end):
            labels[i] = roles_to_label[arg['role']]


    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset

    event_type = ex['event_mentions'][index]['event_type']
    
    # get context with trigger being tagged and its argument mask list
    prefix, prefix_labels = tokenize_with_labels(context[:trigger_start], labels[:trigger_start], tokenizer, type='bert')
    event_type_token, event_type_token_labels = tokenize_with_labels([str(ontology_dict[event_type]['i-label'])], [0], tokenizer, type='bert')
    tgt, tgt_labels = tokenize_with_labels(context[trigger_start: trigger_end], labels[trigger_start: trigger_end], tokenizer, type='bert')
    suffix, suffix_labels = tokenize_with_labels(context[trigger_end:], labels[trigger_end:], tokenizer, type='bert')
    context_tag_trigger = prefix + ['<', ] + event_type_token + tgt + ['>', ] + suffix
    labels_tag_trigger = prefix_labels + [0,] + event_type_token_labels + tgt_labels + [0,] + suffix_labels

    assert len(context_tag_trigger) == len(labels_tag_trigger)

    return context_tag_trigger, labels_tag_trigger

def create_instance_two_event(ex, ontology_dict, index_i=0, index_j=0, id2entity=None, tokenizer=None):
    input_template_i, output_template_i = get_template(ex, index_i, ontology_dict, tokenizer)
    input_template_j, output_template_j = get_template(ex, index_j, ontology_dict, tokenizer)
    # context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH - WORDS_PER_EVENT * len(close_events))
    context, offset = get_context(ex, index_i, 325)

    trigger_i = ex['event_mentions'][index_i]['trigger']
    trigger_start_i = trigger_i['start'] - offset
    trigger_end_i = trigger_i['end'] - offset

    trigger_j = ex['event_mentions'][index_j]['trigger']
    trigger_start_j = trigger_j['start'] - offset
    trigger_end_j = trigger_j['end'] - offset
    
    # print(trigger_start_i, trigger_start_j)
    assert trigger_start_i <= trigger_start_j
    if trigger_end_j < len(context) and trigger_start_i < trigger_start_j:
        prefix = tokenizer.tokenize(' '.join(context[:trigger_start_i]), add_prefix_space=True)
        tgt_i = tokenizer.tokenize(' '.join(context[trigger_start_i: trigger_end_i]), add_prefix_space=True)
        mid = tokenizer.tokenize(' '.join(context[trigger_end_i:trigger_start_j]), add_prefix_space=True)
        tgt_j = tokenizer.tokenize(' '.join(context[trigger_start_j:trigger_end_j]), add_prefix_space=True)
        suffix = tokenizer.tokenize(' '.join(context[trigger_end_j:]), add_prefix_space=True)
        context_tag_trigger = prefix + [' <tgr>', ] + tgt_i + [' <tgr>', ] + mid + [' <tgr>', ] + tgt_j + [' <tgr>', ] + suffix
        input_template = input_template_i + [';', ] + input_template_j 
        output_template = output_template_i + [';', ] + output_template_j 
    else:
        prefix = tokenizer.tokenize(' '.join(context[:trigger_start_i]), add_prefix_space=True)
        tgt_i = tokenizer.tokenize(' '.join(context[trigger_start_i: trigger_end_i]), add_prefix_space=True)
        suffix = tokenizer.tokenize(' '.join(context[trigger_end_i:]), add_prefix_space=True)
        context_tag_trigger = prefix + [' <tgr>', ] + tgt_i + [' <tgr>', ] + suffix
        input_template = input_template_i 
        output_template = output_template_i 
    
    return input_template, output_template, context_tag_trigger


def create_instance_tag_other_event(ex, ontology_dict, index=0, close_events=None, id2entity=None, tokenizer=None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH - WORDS_PER_EVENT * len(close_events))
    # context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH)

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    if len(close_events):            

        # get the tokens need to be tagged (E2)
        # arg_1 = set(x['entity_id'] for x in ex['event_mentions'][index]['arguments'])
        add_tag = defaultdict(list)
        add_tag[(trigger_start, trigger_end)].append('trigger')
        for eid in close_events:
            arguments = ex['event_mentions'][eid]['arguments']
            event_type = simple_type(ex["event_mentions"][eid]['event_type'])
            trigger = ex["event_mentions"][eid]["trigger"]
            for arg in arguments:
                # if arg['entity_id'] in arg_1:
                if 'entity_id' in arg:
                    arg_start = id2entity[arg['entity_id']]['start'] - offset
                    arg_end = id2entity[arg['entity_id']]['end'] - offset
                else:
                    arg_start = arg['start'] - offset
                    arg_end = arg['end'] - offset
                # print(context[arg_start:arg_end])
                if arg_start < 0 or arg_end >= len(context):
                    continue
                add_tag[(arg_start,arg_end)].append((event_type, arg["role"]))
        
        # get the context with arguments for E2 being tagged.
        add_tag = list(add_tag.items())
        add_tag = sorted(add_tag, key=lambda x:x[0][0])
        
        context_tag_other_events = []
        curr = 0
        for arg in add_tag:
            pre_words = tokenizer.tokenize(' '.join(context[curr:arg[0][0]]), add_prefix_space=True)                
            arg_words = tokenizer.tokenize(' '.join(context[arg[0][0]:arg[0][1]]), add_prefix_space=True)
            if 'trigger' not in arg[1]:
                role_set = list(arg[1][0][1] for x in arg[1])
                # role_set = list(set(arg[1][0][1] for x in arg[1]))
                prefix = tokenizer.tokenize(' '.join(role_set), add_prefix_space=True)
                context_tag_other_events += pre_words + [" <tag>", ] + prefix + [' </tag>', ]  + arg_words
                # context_tag_other_events += pre_words + [" <tag>", ] + prefix + [':', ] + arg_words + [' </tag>', ]  
            else:
                context_tag_other_events += pre_words + [" <tgr>", ] + arg_words + [' <tgr>', ]
            curr = arg[0][1]

        suf_words = tokenizer.tokenize(' '.join(context[curr:]), add_prefix_space=True)
        context_tag_other_events += suf_words

        return input_template, output_template, context_tag_other_events

    # get context with trigger being tagged and its argument mask list
    prefix = tokenizer.tokenize(' '.join(context[:trigger_start]), add_prefix_space=True)
    tgt = tokenizer.tokenize(' '.join(context[trigger_start: trigger_end]), add_prefix_space=True)
    suffix = tokenizer.tokenize(' '.join(context[trigger_end:]), add_prefix_space=True)
    context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
        
    return input_template, output_template, context_tag_trigger

def create_instance_tag_comparing(ex, ontology_dict, index=0, close_events=None, id2entity=None, tokenizer=None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH - WORDS_PER_EVENT * len(close_events))
    # context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH)

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    original_mask = [0]*len(context)
    if len(close_events):            

        # highlight the arguments of the current event in the original context.
        for arg in ex['event_mentions'][index]['arguments']:
            arg_start = id2entity[arg['entity_id']]['start'] - offset
            arg_end = id2entity[arg['entity_id']]['end'] - offset
            if arg_start < 0 or arg_end >= len(context):
                continue
            for i in range(arg_start, arg_end):
                original_mask[i] = 1

        # get the tokens need to be tagged (arguments of other events)
        add_tag = defaultdict(list)
        add_tag[(trigger_start, trigger_end)].append('trigger')
        for eid in close_events:
            arguments = ex['event_mentions'][eid]['arguments']
            event_type = simple_type(ex["event_mentions"][eid]['event_type'])
            trigger = ex["event_mentions"][eid]["trigger"]
            for arg in arguments:
                # if arg['entity_id'] in arg_1:
                if 'entity_id' in arg:
                    arg_start = id2entity[arg['entity_id']]['start'] - offset
                    arg_end = id2entity[arg['entity_id']]['end'] - offset
                else:
                    arg_start = arg['start'] - offset
                    arg_end = arg['end'] - offset
                if arg_start < 0 or arg_end >= len(context):
                    continue
                add_tag[(arg_start,arg_end)].append((event_type, arg["role"]))
        
        # get the context with arguments for E2 being tagged.
        add_tag = list(add_tag.items())
        add_tag = sorted(add_tag, key=lambda x:x[0][0])
        
        context_tag_other_events = []
        context_tag_other_events_mask = []
        curr = 0
        for arg in add_tag:
            pre_words, pre_words_labels = tokenize_with_labels(context[curr:arg[0][0]], original_mask[curr:arg[0][0]], tokenizer)
            arg_words, arg_words_labels = tokenize_with_labels(context[arg[0][0]:arg[0][1]], original_mask[arg[0][0]:arg[0][1]], tokenizer)
            if 'trigger' not in arg[1]:
                # prefix = self.tokenizer.tokenize(' '.join(f"{x[0]} - {x[1]}" for x in arg[1])+' :', add_prefix_space=True)
                role_set = list(arg[1][0][1] for x in arg[1])
                prefix = tokenizer.tokenize(' '.join(role_set), add_prefix_space=True)
                context_tag_other_events += pre_words + [" <tag>", ] + prefix + [' </tag>', ]  + arg_words
                prefix_labels = [0]*len(prefix)
                context_tag_other_events_mask += pre_words_labels + [0] + prefix_labels + [0] + arg_words_labels
                # context_tag_other_events += pre_words + [" <tag>", ] + prefix + arg_words + [' </tag>', ]
                # context_tag_other_events_mask += pre_words_labels + [0] + prefix_labels + arg_words_labels + [0]
            else:
                context_tag_other_events += pre_words + [" <tgr>", ] + arg_words + [' <tgr>', ]
                context_tag_other_events_mask += pre_words_labels + [0] + arg_words_labels + [0]
            curr = arg[0][1]

        suf_words, suf_words_labels = tokenize_with_labels(context[curr:], original_mask[curr:], tokenizer)
        context_tag_other_events += suf_words
        context_tag_other_events_mask += suf_words_labels

    # get context with trigger being tagged and its argument mask list
    prefix, prefix_labels = tokenize_with_labels(context[:trigger_start], original_mask[:trigger_start], tokenizer)
    tgt, tgt_labels = tokenize_with_labels(context[trigger_start: trigger_end], original_mask[trigger_start:trigger_end], tokenizer)
    suffix, suffix_labels = tokenize_with_labels(context[trigger_end:], original_mask[trigger_end:], tokenizer)
    
    context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
    context_tag_trigger_mask = prefix_labels + [0] + tgt_labels + [0] + suffix_labels

    if len(close_events):
        return input_template, output_template, context_tag_trigger, context_tag_trigger_mask, context_tag_other_events, context_tag_other_events_mask
    return input_template, output_template, context_tag_trigger, context_tag_trigger_mask, context_tag_trigger, context_tag_trigger_mask
