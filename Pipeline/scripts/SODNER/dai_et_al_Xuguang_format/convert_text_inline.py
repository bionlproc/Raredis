'''Update date: 2020-Jan-13'''
import argparse
import args
from collections import defaultdict
import os
from os.path import basename, splitext


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--input_ann", default="/data/dai031/Experiments/CADEC/tokens.ann", type=str)
    parser.add_argument("--input_tokens", default="/data/dai031/Experiments/CADEC/tokens", type=str)
    parser.add_argument("--output_filepath", default="/data/dai031/Experiments/CADEC/text-inline", type=str)
    parser.add_argument("--no_doc_info", action="store_true")

    args, _ = parser.parse_known_args()
    return args


def output_sentence(f, tokens, mentions, doc=None):
    def check_mention_text(tokens, mentions):
        for mention in mentions:
            tokenized_mention = []
            indices = [int(i) for i in mention[0].split(",")]
            for i in range(0, len(indices), 2):
                start, end = indices[i], indices[i + 1]
                tokenized_mention += tokens[start:end + 1]

            if "".join(tokenized_mention) != mention[2].replace(" ", ""):
                print("%s (original) vs %s (tokenized)" % (mention[2], " ".join(tokenized_mention)))

    if doc is not None:
        f.write("Document: %s\n" % doc)
    f.write("%s\n" % (" ".join(tokens)))
    check_mention_text(tokens, mentions)
    mentions = ["%s, %s %s" % (m[3], m[0], m[1]) for m in mentions]
    f.write("%s\n" % ("|".join(mentions)))

    with open(os.path.join(args.input_ann, doc + ".ann"), "r") as in_f:
        rels = []
        for line_ann in in_f:
            line_ann = line_ann.strip()
            if line_ann[0] != "R": continue
            spr = line_ann.split()
            ent1_id = spr[2][5:]
            ent2_id = spr[3][5:]
            relation = spr[1]
            rels.append([ent1_id, ent2_id, relation])

        relations = ["%s %s %s" % (r[0], r[1], r[2]) for r in rels]
        f.write("%s\n\n" % ("|".join(relations)))


def load_mentions(filepath):
    mentions = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            sp = line.strip().split("\t")

            # changed 5 to 6
            assert len(sp) == 6

            # Added ent_id
            doc, sent_idx, label, indices, mention, ent_id = sp
            mentions[(doc, int(sent_idx))].append((indices, label, mention, ent_id))
    return mentions


if __name__ == "__main__":
    # args = parse_parameters()
    mentions = load_mentions(args.output_tokens_ann)

    with open(args.inline_output_filepath, "w") as out_f:
        with open(args.output_filepath_token) as in_f:
            pre_doc, sent_idx = None, 0
            tokens = []
            for line in in_f:
                if len(line.strip()) == 0:
                    if len(tokens) > 0:
                        assert pre_doc is not None
                        output_sentence(out_f, tokens, mentions.get((pre_doc, sent_idx), []),
                                        None if args.no_doc_info else pre_doc)
                        sent_idx += 1
                        tokens = []
                    continue
                sp = line.strip().split()
                token, doc, _, _ = sp
                if pre_doc is None:
                    pre_doc = doc
                if pre_doc != doc:
                    pre_doc = doc
                    sent_idx = 0
                    assert len(tokens) == 0
                tokens.append(token)
            if len(tokens) > 0:
                assert pre_doc is not None
                output_sentence(out_f, tokens, mentions.get((pre_doc, sent_idx), []),
                                None if args.no_doc_info else pre_doc)
