# Get word timestamps
```Python
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

model = EncDecCTCModel.restore_from(restore_path=config.NEMO_MODEL)
vocabulary = asr_model.decoder.vocabulary + ['']    # add blank token
scorer = Scorer(LM_MODEL_ALPHA, LM_MODEL_BETA, model_path=LM_MODEL_PATH, vocabulary=vocabulary)
decoder = BeamDecoder(vocab, LM_MODEL_BW, ext_scorer=scorer)
audio_files = ['test.wav', 'test1.wav']
probs_out = model.transcribe(audio_files, logprobs=True, batch_size=1)

for i, probs in enumerate(probs_out):
    probs = softmax(probs)
    decoder.reset()
    predictions = decoder.decode(probs)
    
    words = predictions[0][1].split()
    timesteps = decoder.get_word_timestamps_python()
    words = [
        [words[i], (ts[0] * NEMO_MODEL_TIMESTEP_MS, ts[1] * NEMO_MODEL_TIMESTEP_MS)]
        for i, ts in enumerate(timesteps)
        if words[i] != 'ee' and words[i] != 'eee' and words[i] != 'e'
    ]
```


[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/documentation-github.io-blue.svg)](https://nvidia.github.io/OpenSeq2Seq/html/index.html)
<div align="center">
  <img src="./docs/logo-shadow.png" alt="OpenSeq2Seq" width="250px">
  <br>
</div>

# OpenSeq2Seq: toolkit for distributed and mixed precision training of sequence-to-sequence models

OpenSeq2Seq main goal is to allow researchers to most effectively explore various
sequence-to-sequence models. The efficiency is achieved by fully supporting
distributed and mixed-precision training.
OpenSeq2Seq is built using TensorFlow and provides all the necessary
building blocks for training encoder-decoder models for neural machine translation, automatic speech recognition, speech synthesis, and language modeling.

## Documentation and installation instructions 
https://nvidia.github.io/OpenSeq2Seq/

## Features
1. Models for:
   1. Neural Machine Translation
   2. Automatic Speech Recognition
   3. Speech Synthesis
   4. Language Modeling
   5. NLP tasks (sentiment analysis)
2. Data-parallel distributed training
   1. Multi-GPU
   2. Multi-node
3. Mixed precision training for NVIDIA Volta/Turing GPUs

## Software Requirements
1. Python >= 3.5
2. TensorFlow >= 1.10
3. CUDA >= 9.0, cuDNN >= 7.0 
4. Horovod >= 0.13 (using Horovod is not required, but is highly recommended for multi-GPU setup)

## Acknowledgments
Speech-to-text workflow uses some parts of [Mozilla DeepSpeech](https://github.com/Mozilla/DeepSpeech) project.

Beam search decoder with language model re-scoring implementation (in `decoders`) is based on [Baidu DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech).

Text-to-text workflow uses some functions from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).

## Disclaimer
This is a research project, not an official NVIDIA product.

## Related resources
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)
* [OpenNMT](http://opennmt.net/)
* [Neural Monkey](https://github.com/ufal/neuralmonkey)
* [Sockeye](https://github.com/awslabs/sockeye)
* [TF-seq2seq](https://github.com/google/seq2seq)
* [Moses](http://www.statmt.org/moses/)

## Paper
If you use OpenSeq2Seq, please cite [this paper](https://arxiv.org/abs/1805.10387)
```
@misc{openseq2seq,
    title={Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq},
    author={Oleksii Kuchaiev and Boris Ginsburg and Igor Gitman and Vitaly Lavrukhin and Jason Li and Huyen Nguyen and Carl Case and Paulius Micikevicius},
    year={2018},
    eprint={1805.10387},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
