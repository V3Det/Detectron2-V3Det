# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import itertools
import json
import os

import mmengine
import numpy as np
import torch
from nltk.corpus import wordnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann', default='datasets/V3Det/annotations/v3det_2023_v1_train.json')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true')
    parser.add_argument('--use_wn_name', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    if 'categories' not in data:
        categories = data
    else:
        categories = data['categories']
    cat_names = [x['name'] for x in sorted(categories, key=lambda x: x['id'])]
    if 'synonyms' in categories[0]:
        if args.use_wn_name:
            synonyms = [
                [xx.name() for xx in wordnet.synset(x['synset']).lemmas()] \
                    if x['synset'] != 'stop_sign.n.01' else ['stop_sign'] \
                    for x in sorted(categories, key=lambda x: x['id'])]
        else:
            synonyms = [x['synonyms'] if 'synonyms' in x else [] for x in \
                sorted(categories, key=lambda x: x['id'])]
    else:
        synonyms = []
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [
            x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names
        ]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
            for x in synonyms]
    elif args.prompt == 'scene':
        sentences = [
            'a photo of a {} in the scene'.format(x) for x in cat_names
        ]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
            for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
        sum(len(x) for x in sentences_synonyms))
    if args.model == 'clip':
        import clip
        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])
                ],
                                          dim=0)
            else:
                text_features = model.encode_text(text)
        print('text_features.shape', text_features.shape)
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.cpu().numpy()
    elif args.model in ['bert', 'roberta']:
        from transformers import AutoTokenizer, AutoModel
        if args.model == 'bert':
            model_name = 'bert-large-uncased'
        if args.model == 'roberta':
            model_name = 'roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        inputs = tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_outputs = model(**inputs)
            outputs = model_outputs.pooler_output
        text_features = outputs.detach().cpu()
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.numpy()
        print('text_features.shape', text_features.shape)
    else:
        assert 0, args.model
    suffix = args.ann.split('/')[-1].replace('.json', '_clip_a+cname.npy')
    out_dir = 'datasets/metadata'
    mmengine.mkdir_or_exist(out_dir)
    out_path = os.path.join(out_dir, suffix)
    print('Saving to', out_path)
    np.save(open(out_path, 'wb'), text_features)
