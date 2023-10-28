'''Update date: 2020-Jan-13'''
import argparse, os
import args


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--input_filepath", default="/data/dai031/Experiments/CADEC/text-inline", type=str)
    parser.add_argument("--output_dir", default="/data/dai031/Experiments/CADEC/split", type=str)

    args, _ = parser.parse_known_args()
    return args


def output_sentence(sentences, filepath):
    with open(filepath, "w") as f:
        for sentence in sentences:
            temp_dict = dict()
            xyz = sentence[2].split("|")
            for x in xyz:
                x = x.split(" ")
                ent_id = x[0][:x[0].find(r',')]
                tokens = [int(k) for k in x[1].split(",")]
                label = x[2]
                tokens.append(label)
                temp_dict[ent_id] = tokens

            abc = sentence[3].split("|")
            rel_list = []
            for a in abc:
                rel_list.append(a.split(" "))

            f.write("%s\n" % sentence[0])
            f.write("%s\n" % sentence[1])
            f.write("%s\n" % sentence[2])
            f.write("%s\n" % sentence[3])
            f.write("\n")


if __name__ == "__main__":
    # train_set = [l.strip() for l in open("/Users/shashankgupta/Documents/Raredis/dai_et_al/split/train.id").readlines()]
    # dev_set = [l.strip() for l in open("/Users/shashankgupta/Documents/Raredis/dai_et_al/split/dev.id").readlines()]
    # test_set = [l.strip() for l in open("/Users/shashankgupta/Documents/Raredis/dai_et_al/split/test.id").readlines()]

    # args = parse_parameters()
    if not os.path.exists(args.txt_file_path):
        os.mkdir(args.txt_file_path)

    sentences = []
    with open(args.inline_output_filepath) as f:
        for line in f:
            doc = line.strip().replace("Document: ", "")
            tokens = next(f).strip()
            mentions = next(f).strip()
            relations = next(f).strip()
            sentences.append((doc, tokens, mentions, relations))
            assert next(f).strip() == ""

    output_sentence(sentences, os.path.join(args.txt_file_path, "dev.txt"))