#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]

import fire
import json
import os
import numpy as np
import tensorflow as tf
import tflex
import random
import model, sample, encoder

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def interact_model(
    model_name='117M',
    asker=None,
    responder=None,
    restore_from=None,
    seed=None,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    penalize=0,
    prompt=None
):
    """
    Interactively chat with the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :penalize=0.0 : Float value controlling "used" penalty. Implements repetition
     reduction (similar to CTRL) if set to a value > 0. A decent setting might be 0.85
     with temperature 0.3 and top_k 40.
    """
    if asker is None:
        raise Exception("Add a name present in the training dataset that you will be chatting as")
    if responder is None:
        raise Exception("Add a name present in the training dataset that gpt will be chatting as")

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tflex.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p, penalize=penalize
        )

        saver = tflex.Saver()
        if restore_from is None:
          restore_from = os.path.join('models', model_name)
        ckpt = tflex.latest_checkpoint(restore_from)
        saver.restore(sess, ckpt)

        input_ = ''
        time = 1924862493344
        while True:
            time = increase_time(time)
            input_ =  input_ + f'({time}) {asker}: ' + input(f"{asker}: ")
            time = increase_time(time)
            input_ = input_ + f'\n ({time}) {responder}: '
            if len(input_) > 1 and input_.endswith('\n'):
                input_ = input_[:-1]
            context_tokens = enc.encode(input_)
            out = sess.run(output, feed_dict={
                context: [context_tokens]
            })[:, len(context_tokens):]
            enc.decode(out[0])
            text = enc.decode(out[0]).split(f') {asker}', 1)[0]
            print(f'\n ({time}) {responder}: ' + text.rsplit('(', 1)[0])
            input_ = input_ + text
            sys.stdout.flush()


def increase_time(time):
    # increase timestamp for each message seen by gpt
    return time + random.randint(100, 1000)


if __name__ == '__main__':
    fire.Fire(interact_model)
