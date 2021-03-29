import numpy as np
import tensorflow as tf
import os
import time
from build_model import TextModel
from shapley import compute_scores, turn_list
from preprocess import tokenization, extract, modeling
import pickle

np.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bert_dir = './models/uncased_L-12_H-768_A-12'

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "sst-2", "The name of the task to train.")
flags.DEFINE_string(
    "bert_config_file", bert_dir + "/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", bert_dir + "/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_integer("layer", "-1", "extract layer index")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("train_bs", 32, "Batch size for training.")
flags.DEFINE_integer("eval_bs", 8, "Batch size for evaluation.")
flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_epochs", 2,
                     "Total number of training epochs to perform.")
flags.DEFINE_string("method", "SampleShapley", "Shapley method")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    processors = {
        "sst-2": extract.Sst2Processor,
        "cola": extract.ColaProcessor,
    }
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    if FLAGS.task_name == "sst-2":
        FLAGS.data_dir = "data/sst-2"
        FLAGS.init_checkpoint = "models/sst-2/model.ckpt-6313"
    elif FLAGS.task_name == "cola":
        FLAGS.data_dir = "data/cola"
        FLAGS.init_checkpoint = "models/cola/model.ckpt-801"

    processor = processors[task_name]()

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    # ------------------- preprocess dataset -------------------
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    max_seq_length = FLAGS.max_seq_length

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))

    # ----------------------- build models ---------------------
    tf.reset_default_graph()
    model = TextModel(FLAGS.bert_config_file, FLAGS.init_checkpoint, max_seq_length, num_labels)
    model.start_session()

    method = FLAGS.method
    if method not in ['SampleShapley', 'Singleton']:
        print("Not Supported Shapley")
    else:
        print(method)
    print('Making explanations...')

    for (i, example) in enumerate(eval_examples):
        st = time.time()
        print('explaining the {}th sample...'.format(i))
        tokens_a = tokenizer.tokenize(example.text_a)
        a_len = len(tokens_a)
        print(tokens_a)
        print('tokens length is', a_len)
        feature = extract.convert_single_example(i, example, label_list, max_seq_length, tokenizer)

        pre_slist = list(range(a_len))
        output_tree = list(range(a_len))
        tree_values = []
        # construct a binary tree
        for h in range(a_len - 1):
            pre_slen = len(pre_slist)
            totcombs = []
            ratios = []
            stn = {}

            # compute B, phi{a,b,...} for each point
            tot_values = {}
            for k in range(pre_slen):
                scores = compute_scores(pre_slist, k, feature,
                                        a_len, model.predict, method)
                if len(scores) == 2:
                    b = 0
                    subtree = [b, scores[0:1]]
                else:
                    b = scores[0] - np.sum(scores[1:])
                    subtree = [b, scores[1:]]
                tot_values[k] = subtree

            locs = []
            for j in range(pre_slen - 1):
                coal = turn_list(pre_slist[j]) + turn_list(pre_slist[j + 1])
                now_slist = pre_slist[:j]  # elems before j
                now_slist.append(coal)
                if j + 2 < pre_slen:
                    now_slist = now_slist + pre_slist[j + 2:]  # elems after j+1

                totcombs.append(now_slist)
                # compute shapley values of now pair combination
                score = compute_scores(now_slist, j, feature,
                                       a_len, model.predict, method)
                nowb = score[0] - np.sum(score[1:])
                nowphis = score[1:]

                lt = tot_values[j][1]
                rt = tot_values[j + 1][1]
                avgphis = (nowphis + np.concatenate((lt, rt))) / 2
                len_lt = lt.shape[0]

                b_lt = tot_values[j][0]
                b_rt = tot_values[j + 1][0]
                b_local = nowb - b_lt - b_rt
                contri_lt = b_lt + np.sum(avgphis[:len_lt])
                contri_rt = b_rt + np.sum(avgphis[len_lt:])

                # additional two metrics
                extra_pre_slist_l = list(pre_slist)
                extra_pre_slist_l.pop(j+1)
                extra_score_l = compute_scores(extra_pre_slist_l, j, feature,
                                               a_len, model.predict, method)
                psi_intra_l = extra_score_l[0] - np.sum(extra_score_l[1:])
                psi_intra_l = psi_intra_l - b_lt

                extra_pre_slist_r = list(pre_slist)
                extra_pre_slist_r.pop(j)
                extra_score_r = compute_scores(extra_pre_slist_r, j, feature,
                                               a_len, model.predict, method)
                psi_intra_r = extra_score_r[0] - np.sum(extra_score_r[1:])
                psi_intra_r = psi_intra_r - b_rt
                psi_intra = (psi_intra_l + psi_intra_r)
                psi_inter = b_local - psi_intra
                t = abs(psi_inter) / (abs(psi_intra) + abs(psi_inter))
                # end additional metrics

                locs.append([b_local, contri_lt, contri_rt, b_lt, b_rt, t, nowb])

            for j in range(pre_slen - 1):
                loss = 0.0
                if j - 1 >= 0:
                    loss = loss + abs(locs[j - 1][0])

                if j + 2 < pre_slen:
                    loss = loss + abs(locs[j + 1][0])

                all_info = loss + abs(locs[j][0]) + abs(locs[j][1]) + abs(locs[j][2])
                metric = abs(locs[j][0]) / all_info
                sub_metric = loss / all_info

                ratios.append(metric)

                stn[j] = {'r': metric,
                          's': sub_metric,
                          'Bbetween': locs[j][0],
                          'Bl': locs[j][3],
                          'Br': locs[j][4],
                          't': locs[j][5],
                          'B([S])': locs[j][6],
                        }
            stn['base_B'] = tot_values
            coalition = np.argmax(np.array(ratios))
            pre_slist = totcombs[coalition]
            stn['maxIdx'] = coalition
            stn['after_slist'] = pre_slist
            print('coalition:', coalition)
            print('after_slist:', pre_slist)

            tree_values.append(stn)

	        # generate a new nested list by adding elements into a empty list
            tmp_list = []
            for z in range(len(output_tree)):
                if z == coalition:
                    tmp_list.append(list((output_tree[z], output_tree[z + 1], stn[z])))
                elif z == coalition + 1:
                    continue
                else:
                    tmp_list.append(output_tree[z])
            output_tree = tmp_list.copy()

        save_path = 'binary_trees/' + FLAGS.task_name.upper()
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_pkl = save_path + '/stn_' + str(i) + '.pkl'
        with open(save_pkl, "wb") as f:
            contents = {"sentence": tokens_a,
                        "tree": output_tree,
                        "tree_values": tree_values,
                        }
            pickle.dump(contents, f)

        print('Time spent is {}'.format(time.time() - st))


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    # parse flags and run main()
    tf.app.run()
