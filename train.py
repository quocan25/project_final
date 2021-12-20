from tqdm import tqdm
from lib import remove_tone_line, _save_pickle, _load_pickle, create_look_ahead_mask, create_masks
from transformer_class import Transformer
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np


configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


# with open('data/train_tieng_viet.txt', 'r', encoding='utf-8') as f:
#     train_output = f.readlines()
#
# print('Number of sequences: ', len(train_output))
# print('First sequence: ', train_output[0])
#
# train_idx_500k = []
# train_opt_500k = []
# train_ipt_500k = []
# val_idx_50k = []
# val_opt_50k = []
# val_ipt_50k = []
# test_idx_50k = []
# test_opt_50k = []
# test_ipt_50k = []
#
# for i in tqdm(range(600000)):
#     [idx, origin_seq] = train_output[i].split('\t')
#     try:
#         non_acc_seq = remove_tone_line(origin_seq)
#     except:
#         print('error remove tone line at sequence {}', str(i))
#         continue
#     if i < 500000:
#         train_idx_500k.append(idx)
#         train_opt_500k.append(origin_seq)
#         train_ipt_500k.append(non_acc_seq)
#     elif i < 550000:
#         val_idx_50k.append(idx)
#         val_opt_50k.append(origin_seq)
#         val_ipt_50k.append(non_acc_seq)
#     else:
#         test_idx_50k.append(idx)
#         test_opt_50k.append(origin_seq)
#         test_ipt_50k.append(non_acc_seq)


# _save_pickle('train_tv_idx_500k.pkl', train_idx_500k)
# _save_pickle('val_tv_idx_50k.pkl', val_idx_50k)
# _save_pickle('test_tv_idx_50k.pkl', test_idx_50k)
# print(val_idx_50k)
# print(val_opt_50k)
# print(val_ipt_50k)

train_ipt_500k = _load_pickle("data/train_tv_ipt_500k.pkl")
train_opt_500k = _load_pickle("data/train_tv_opt_500k.pkl")

val_ipt_50k = _load_pickle("data/val_tv_ipt_50k.pkl")
val_opt_50k = _load_pickle("data/val_tv_opt_50k.pkl")
t1 = time.time()
train_examples = tf.data.Dataset.from_tensor_slices((train_ipt_500k, train_opt_500k))
val_examples = tf.data.Dataset.from_tensor_slices((val_ipt_50k, val_opt_50k))

# tokenizer_ipt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (ipt.numpy() for (ipt, opt) in train_examples), target_vocab_size=2**13)
#
# tokenizer_opt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
#     (opt.numpy() for (ipt, opt) in train_examples), target_vocab_size=2**13)


# _save_pickle('tokenizer/tokenizer_ipt.pkl', tokenizer_ipt)
# _save_pickle('tokenizer/tokenizer_opt.pkl', tokenizer_opt)

tokenizer_ipt = _load_pickle('tokenizer/tokenizer_ipt.pkl')
tokenizer_opt = _load_pickle('tokenizer/tokenizer_opt.pkl')

print("Create token", time.time() - t1)

BUFFER_SIZE = 20000
BATCH_SIZE = 128


def encode(ipt, opt):
    ipt = [tokenizer_ipt.vocab_size] + tokenizer_ipt.encode(
        ipt.numpy()) + [tokenizer_ipt.vocab_size + 1]

    opt = [tokenizer_opt.vocab_size] + tokenizer_opt.encode(
        opt.numpy()) + [tokenizer_opt.vocab_size + 1]

    return ipt, opt


def tf_encode(ipt, opt):
    result_ipt, result_opt = tf.py_function(encode, [ipt, opt], [tf.int64, tf.int64])
    result_ipt.set_shape([None])
    result_opt.set_shape([None])
    return result_ipt, result_opt


MAX_LENGTH = 64


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


t2 = time.time()
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)
print("Map data ", time.time() - t2)

#  POSITION ENCODING

# # MASKING
#
#
# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
#
#
# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)


# SET HYPERPARAMETERS


num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_ipt.vocab_size + 2
target_vocab_size = tokenizer_opt.vocab_size + 2
dropout_rate = 0.1

#  OPTIMIZER


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


# LOSS AND METRICS
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


# TRAINING AND CHECKPOINTTING
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# def create_masks(inp, tar):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inp)
#
#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inp)
#
#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#     return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "checkpoints/train_500k"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')

EPOCHS = 50
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> non_diacritic, tar -> diacritic

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = [tokenizer_ipt.vocab_size]
    end_token = [tokenizer_ipt.vocab_size + 1]

    # inp sentence is non_diacritic, hence adding the start and end token
    inp_sentence = start_token + tokenizer_ipt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is exist diacritic, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_opt.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_opt.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_ipt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_ipt.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_opt.decode([i]) for i in result
                            if i < tokenizer_opt.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_opt.decode([i for i in result
                                               if i < tokenizer_opt.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


translate("tieng Viet la ngon ngu trong sang nhat the gioi")
