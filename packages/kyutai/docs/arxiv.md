---
title: "Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling"
source: "https://arxiv.org/html/2509.08753v1"
author: Neil Zeghidour, Eugene Kharitonov, Manu Orsini, Václav Volhejn, Gabriel de Marmiesse, Edouard Grave, Patrick Pérez, Laurent Mazaré, Alexandre Défossez
published: 2025-09-10
created: 2025-09-13
description: Kyutai paper on Delayed Streams Modeling (DSM) framework for streaming sequence-to-sequence learning
tags:
  - "clippings"
  - "kyutai"
  - "streaming"
  - "sequence-to-sequence"
  - "ASR"
  - "TTS"
---

# Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling

**arXiv:2509.08753v1 [cs.CL] 10 Sep 2025**

Neil Zeghidour*, Eugene Kharitonov, Manu Orsini, Václav Volhejn, Gabriel de Marmiesse, Edouard Grave, Patrick Pérez, Laurent Mazaré, Alexandre Défossez*

*Equal contribution  
Kyutai  
{neil, eugene, alex}@kyutai.org

## Abstract

We introduce Delayed Streams Modeling (DSM), a flexible formulation for streaming, multimodal sequence-to-sequence learning. Sequence-to-sequence generation is often cast in an offline manner, where the model consumes the complete input sequence before generating the first output timestep. Alternatively, streaming sequence-to-sequence rely on learning a policy for choosing when to advance on the input stream, or write to the output stream. DSM instead models already time-aligned streams with a decoder-only language model. By moving the alignment to a pre-processing step, and introducing appropriate delays between streams, DSM provides streaming inference of arbitrary output sequences, from any input combination, making it applicable to many sequence-to-sequence problems. In particular, given text and audio streams, automatic speech recognition (ASR) corresponds to the text stream being delayed, while the opposite gives a text-to-speech (TTS) model. We perform extensive experiments for these two major sequence-to-sequence tasks, showing that DSM provides state-of-the-art performance and latency while supporting arbitrary long sequences, being even competitive with offline baselines. Code, samples and demos are available at [github.com/kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/).

## 1 Introduction

We are interested in streaming sequence-to-sequence (seq2seq) learning, i.e. predicting an output sequence as we process an input sequence synchronously, as opposed to offline seq2seq where inputs are recorded entirely before producing the output sequence. The latter class of offline models was introduced for a diverse set of tasks such as handwriting recognition, automatic speech recognition (ASR) or machine translation, by designing modality-dependent input encoders, typically coupled with a text decoder. Although this asymmetry between input processing and output generation facilitated the adoption of this framework in many tasks, it also led to a divergence of model architectures across modalities. As an example, a Tacotron text-to-speech (TTS) model would differ from an ASR model such as LAS. The advent of decoder-only Transformers for text language modeling reduced the gap between input and output processing by allowing a single model to process a simple concatenation of tokens. In parallel, neural compression algorithms that can transform images and audio into discrete tokens analogous to text allowed integrating these modalities along text sequences. Thus, a decoder-only model can be used for seq2seq tasks such as ASR, TTS, spoken dialogue, visual understanding or image generation. Furthermore, inputs and outputs are interchangeable in this framework, meaning a single model can be trained for generation in both directions: AudioPALM performs TTS and ASR, while CM3Leon provides both image captioning and generation. Yet, a major limitation of these decoder-only approaches is their incompatibility with streaming. First, their prefix-based formulation requires access to the full input sequence before generation, which prevents real-time inference and inherently limits the maximum input length. Second, modalities operate at differing framerates: audio or video tokens are typically sampled regularly, while text tokens represent linguistic units pronounced with varying durations. This prevents applications such as meeting transcription or continuous translation.

Another popular approach is to learn an alignment policy between modalities, using architectures such as Transducers, or specific attention formulations. At inference, the policy decision will change which modules to execute at each step, which is detrimental to batching. Besides, learning the policy requires train-time exploration, which can be costly. As noted by previous work, a simple *wait-k* policy can be used, especially for same modality sequence-to-sequence modeling.

In this work, we present Delayed Streams Modeling (DSM), a framework for streaming sequence-to-sequence learning across modalities. We make a simplifying assumption compared with previous *wait-k* based methods, namely that both modalities are aligned to a shared framerate as a pre-processing step. DSM uses a decoder-only model to process as many parallel token streams as there are I/O sequences. This multistream architecture allows for a synchronous autoregressive modeling of aligned sequences which—when coupled with a finite context—provides real-time, streaming generation over infinite input sequences. Moreover, by operating at a constant framerate, DSM allows for batching, a feature rarely provided by streaming models. The second key component of DSM, inspired by the *wait-k* policy, is the introduction of a delay between streams to control the quality/latency trade-off: shifting a sequence B such that it is delayed w.r.t. sequence A allows for a better prediction of the former based on the latter. With appropriate delays, a DSM model can be trained to continuously predict any combination of output sequences from any combination of input sequences. To illustrate the abilities of the DSM framework, we train speech-text models for ASR and TTS. We show how DSM provides a state-of-the-art tradeoff between latency—as low as a few hundred milliseconds—and quality, while providing long-form synthesis and transcription, along with precise word timestamps that locate where they are pronounced.

## 2 Related Work

**Streaming Sequence-to-Sequence Learning.** Most streaming seq2seq literature has focused on speech-to-text tasks, in particular ASR and translation. Monotonic and local attention respectively allow for causal attention of outputs with respect to inputs along with handling arbitrarily long sequences. A common limitation of streaming models is their incompatibility with batching when using an inference policy, or the lack of symmetry meaning that specific models must be used for speech-to-text and text-to-speech. Previous approaches using Transformer decoder-only models typically require non-standard attention, and separate calls to the backbone per modality. In contrast, DSM allows for batching and accelerated inference, using only standard attention, with all modalities fused to limit the number of steps in the backbone decoder. In the context of this paper, this allows DSM to be trained for state-of-the-art ASR or TTS (see Figure 1), as shown in Section 4, with its performance being even competitive with offline approaches.

**Multimodal language models.** Transformer-based autoregressive models are the current main approach to sequence-to-sequence problems. They were introduced for machine translation, and were soon extended to multimodal tasks, such as ASR or visual understanding, by designing modality-specific encoders. More recently, neural codecs have provided compact, discrete representations of images and audio that remove the need for modality-specific encoders inside the generative model, while providing a symmetrical processing of inputs and outputs which allows performing bidirectional tasks (e.g. speech-to-text and text-to-speech) with a single architecture. Previous work introduced a multistream decoder architecture for spoken dialogue, which predicts text and audio tokens in a streaming fashion, later applied to real-time speech translation. In this work we extend this approach, in order to reach state-of-the-art performance on the two most competitive speech-text tasks, namely ASR and TTS. Moreover, while previous work operates with a delay specified before training, we propose delay conditioning for inference-time latency control without retraining. Our TTS covers both monologue and controllable dialog generation, a topic that was studied by CoVoMix, although at a lower sample rate (8 kHz) and not streaming.

**Figure 1:** Delayed streams modeling for speech-text tasks. Depending on which stream is delayed with respect to the other, we solve either an ASR or a TTS task. For TTS, we further need an action stream for the model to let us know when it is ready to receive a new word.

## 3 Method

**Notation.** We wish to solve a sequence-to-sequence task between two domains X and Y. Each domain consists of sequences of vectors of all possible lengths.

In the case where either X_t or Y_t is discrete-valued, we can use a one-hot representation for it. We assume that we are given a joint probability distribution over the outer product domain X×Y, and that we have the random variables X∈X and Y∈Y, along with the joint distribution P[X,Y] = p(X,Y).

We also introduce T∈ℕ (resp. T') the random variable indicating the length of X (resp. Y), along with the marginals p(X) and p(Y). For any sequence Z, and index t, we denote Z_<t = (Z_1,...,Z_{t-1}), potentially empty if t ≤ 0. We similarly define Z_≤t, Z_≥t, and Z_>t.

**Sequence-to-sequence as joint modeling.** Let's assume for this paragraph that X is the set of all possible monophonic waveforms sampled at 24 kHz, and Y is made of sequences of one-hot encoded vectors over a set of words. Intuitively, we assume there exists a coupling p(X,Y) such that p(X,Y) is high if Y represents the transcription of X, or conversely, if X represents a speech utterance of the text given by Y. Formally, the task of ASR corresponds to sampling from the distribution P[Y|X], while the task of TTS corresponds to sampling from the distribution P[X|Y]. Thus, each task can be solved by accurately estimating both probability distributions:

q(X,Y) ≈ P[Y|X], q'(Y,X) ≈ P[X|Y]

For simplicity, we now only focus on estimating P[Y|X], the inverse task being obtained by exchanging the definition of X and Y. We thus call X the input domain, and Y the output domain.

**Auto-regressive modeling of Y.** A good candidate for estimating P[Y|X] is auto-regressive modeling, with a Transformer model, under the extra assumption that the output domain Y can be discretized. Thus, one would estimate:

q(y|X,Y_<t) ≈ P[Y_t = y|X,Y_<t]

One can then sample Y auto-regressively, knowing X. Due to the lack of explicit structure between the time grid t of X and t' of Y, one would usually condition on the entirety of X, e.g. when using Transformer based models, either by prefixing the entire sequence X before the generation Y, or by providing X through cross-attention layers, which is mathematically equivalent. This forbids the use of the model in a streaming fashion, as the entire input signal X must be known ahead of time, and cannot be extended once the generation of Y has started. Such methods often require explicit and manual chunking and stitching operations, which also reduces their ability to be efficiently batched. Conversely, aligning X and Y to the same frame rate allows for batched streaming inference.

**Aligning sequences for streaming prediction.** We assume that both domains X and Y can share the same time grid, e.g. (X_t)∈ℝ^{T×d} and (Y_t)∈ℝ^{T×d'}. We call two such aligned sequences *streams*. Then one can simply model:

q_aligned(y|X_≤t,Y_<t) ≈ P[Y_t = y|X_≤t,Y_<t]

Given X~p(X), we sample auto-regressively from this equation, *with a streaming context* X:

Ỹ_1 ~ q_aligned(Ỹ_1|X_1), Ỹ_t ~ q_aligned(Ỹ_t|X_≤t,Ỹ_<t)

We would want that given X~p(X), then (X,Ỹ)~(X,Y), so that in particular P[Ỹ|X] ≈ P[Y|X]. However this needs not be the case unless certain conditions are met.

**The importance of causality.** In particular, for (X,Ỹ)~(X,Y) to be true, Y_>t must be independent of X_>t, knowing X_≤t. To realize that, one can look at a simple counter-example taking X_t~B(0.5) independent Bernoulli variables, and Y_t = X_t ⊕ X_{t+1} the XOR of X_t and X_{t+1}. Clearly P[Y_t|X_≤t,Y_<t]~B(0.5) for all t, yet, given X=(0,1), one would have:

Y_1 = 1 a.s., Ỹ_1 ~ B(0.5)

Thus Y|X and Ỹ|X have different distributions. Intuitively, given that we do not sample X but teacher-force real-world data, we must ensure that when sampling Ỹ_t, no future value of X_>t might end up in "contradiction" with the value we sampled.

**Delaying the output stream.** In practice, this is achieved by delaying the output stream Y_t by a number of steps τ > 0. Thus, we replace the previous equation by:

q_τ(y|X_≤{t+τ},Y_<t) ≈ P[Y_t = y|X_≤{t+τ},Y_<t]

and define Ỹ^τ, similarly to the procedure described earlier. Perfect independence is hard to achieve: in the case of ASR, a named entity might be ambiguous without context, and only future development in a discussion would resolve this ambiguity. Taking τ = T recovers the prefixing or cross-attention approaches presented earlier. In practice, there is a trade-off between the level of independence of Y_t with X_{>t+τ}, and the latency of the method.

**Figure 2:** DSM Architecture. Transformer is fed with the streaming input X_t. After a delay τ, a sampler is fed with the output of the backbone samples Ỹ_t. At the next step, the backbone receives both the sampled value and next streaming input, whose embeddings are summed.

**Architecture.** DSM contains three components: (i) an auto-regressive backbone, (ii) an input embedder for X and Y into the backbone, and (iii) a sampler for Ỹ^τ conditioned on the output of the backbone. The backbone can be a Transformer architecture, optionally equipped with cross-attention layers to provide further non-streaming contextual information. The embedder for X and Y can be learnt embedding tables in the case where both domains are discrete. The embeddings are summed before going into the backbone. On the output side, we mask the loss on the tokens of X and only compute cross-entropy on Y. Finally, the conditional sampler can be a linear layer applied to the output of the backbone to derive logits if Y is discrete. It could also be a flow or diffusion model conditioned on the output of the backbone for the continuous case.

### 3.1 Representations of the speech and text domains

We demonstrate the DSM framework on ASR and TTS, where the two domains are text and audio.

**Audio.** Given a waveform w∈ℝ^{d_s·f_s} with the duration in seconds d_s and the sample rate f_s = 24 kHz, we turn it into a more compact latent space using the Mimi codec, giving us a sequence of tensors Z^audio ∈ ℝ^{d_s·f_r×d_audio}, with a frame rate of f_r = 12.5 Hz. This latent space is discretized with Residual Vector Quantization (RVQ), giving us a set of Q ∈ [1,32] coarse-to-fine discrete values per time step with cardinality N_a = 2048, each coming from one codebook in the RVQ, giving a quantized representation Z^{q-audio} ∈ {1,...,N_a}^{d_s·f_r×Q}.

**Text.** We tokenize text using a vocabulary of N_t, specifically trained on speech data transcriptions. Two tokens have a special meaning: PAD (indicating the absence of words at this time) and WORD (indicating the start of a new word). Given a transcript, with word-level timestamps, of a waveform of duration d_s, its aligned text representation is Z^text ∈ {1,...N_t}^{d_s·f_r}. For each word in the transcript represented by tokens (x_1,...,x_n) ∈ {1,...,N_t}^n and starting at s ∈ ℝ^+ seconds, we define its start index i = floor(s·f_r), and store it as Z^text_i ← WORD, Z^text_{i+1} ← x_1, Z^text_{i+2} ← x_2, etc. Any step in Z^text not assigned by a word token is given the special value PAD.

### 3.2 DSM for automatic speech recognition: DSM-ASR

For ASR, we consider X = Z^{q-audio} and Y = Z^text. By predicting the word tokens of Y, we learn to transcribe audio, while computing the loss on PAD and WORD tokens trains the model to predict the precise boundaries of each word. At inference time, we teacher-force the audio tokens of X and sample the full sequence Z^text to obtain a transcription along with timestamps with a precision of 80ms (frame size). This is allowed by the fact that we apply a constant delay to all words in the sequence, meaning we only need to shift the output timestamps back by the same value to recover the true timestamps.

**Deriving aligned speech-text data.** We are looking for fine-grained alignment between speech and text, however speech datasets are typically aligned at the level of the sentence. Conveniently, whisper-timestamped provides automatic transcriptions with word-level timestamps. We rely on these pseudo-labels for the pretraining phase of DSM-ASR. We then finetune on a mixture of public datasets with ground-truth transcripts (see details in Section 4.2), which pose two challenges. First, the automatic transcriptions extracted by Whisper in pretraining are formatted with capitalization and punctuation, but the level of formatting varies a lot between datasets. To address this, we train a 300M prefix-LM for automatic formatting, on a dataset of formatted Whisper transcripts. A second challenge is that these ground-truth transcripts do not have word-level alignment. We derive those by producing pseudo-transcripts with Whisper, and reconciling them with the formatted transcript using a Dynamic Time Warping algorithm.

**Delay conditioning for inference-time control.** As shown in Section 4.3.1, transcription quality is heavily dependent on the delay between audio and text. Thus, training DSM-ASR with a fixed delay requires choosing a latency/quality trade-off beforehand, and retraining a new model for each delay, despite the training task remaining fundamentally the same. To instead control this trade-off at inference, we train DSM-ASR over random delays, sampled for each sequence. The model is additionally conditioned on a cosine embedding of the delay (expressed in milliseconds), added to the inputs. Experiments in Section 4.3.1 compare this model to the models trained with a fixed delay and show that the effective delay precisely respects the conditioning value.

### 3.3 DSM for text-to-speech

We further apply DSM to TTS, taking X = Z^text, Y = Z^{q-audio}. We use a stream delay of 1.28s (or 16 steps) on the output audio. For sampling along the Q dimension in Z^{q-audio}, we use a RQ-Transformer as a sampler, i.e. a smaller Transformer conditioned on the output of the backbone at each timestep and performing autoregressive modeling along the Q dimension. All the backbone inputs (generated audio tokens and next word token input) are fed through learnt embeddings and summed. We are confronted with the problem that the input domain is no longer plain text, but text properly padded for time alignment. While at train time we can teacher-force the ground-truth padded text, this is not the case for a novel text to synthesize at inference time.

**Action output stream.** We add an extra stream to the TTS outputs, whose goal is to predict whether the next input text token will be a WORD token or not. This special input token indicates that a new word is starting, and that its tokens are going to follow as inputs. This extra stream controls an inference-time *action*: when predicted by the model, we will feed as input the text tokens for the next word over the next time steps. While these are being fed, the model is not allowed to output another WORD action. The action output stream is not fed back into the model as it is redundant with the text stream input.

**Lookahead second text stream.** The action stream allows the model to predict the next word position, although the model has no knowledge of its content for making that decision. The delay between text and audio only provides context for the audio generation, however, the decision on where to insert pauses and words has no such context. Given a sequence of words m_1, m_2, ..., the lookahead text stream feeds the tokens of the words m_{i+l} to the backbone while the primary text feed contains the tokens of words m_i.

**Speaker conditioning.** We provide speaker embeddings for up to 5 speakers. Each speaker is represented by a 10s audio extract of the same speaker outside of the training segment. Speakers are identified using the diarization tool Pyannote in the training data. If more than 5 speakers are present in the segment, only 5 at random are kept for the speaker embeddings. If less than 5 speakers are present, the remaining speaker slots are filled with learnt padding values. Each speaker audio extract is encoded with a *speaker encoder* and results in a speaker embedding with a fixed dimension. We concatenate the speaker embedding from the different speakers, sum them with an absolute positional embedding, and feed them through cross-attention layers to the backbone. The speaker encoder has the same architecture as the encoder of the Mimi codec, and is initialized with its weights. We keep the weights of the convolutional layers frozen for stability, but let its Transformer layers be fine-tuned in an end-to-end fashion with the language model conditioned on it.

**Change of turn tokens.** We indicate change of turns between the first speaker in the speaker embedding, called the *main speaker*, and any other speaker. When the main speaker starts talking, their first word is prefixed with a special MAIN token in the text stream. When another speaker starts speaking after the main speaker, a special OTHER token is inserted. At inference time, we can thus make controllable dialogs by feeding the model with speaker embeddings for the two speakers, and controlling the change of turn by inserting the MAIN and OTHER special tokens.

**Classifier free guidance.** We use classifier free guidance (CFG), both with respect to the speaker conditioning, and also with respect to the text, that is, we replace at inference time the logits for a given timestep t and codebook index q, given α ≥ 1, with:

l_{t,q} = l^∅_{t,q} + α(l^{text,speaker}_{t,q} - l^∅_{t,q})

where l^∅_{t,q} are the logit estimates obtained by feeding no text, action or lookahead inputs to the model, and no speaker embedding, and l^{text,speaker}_{t,q} are the conditioned logits estimates. No CFG is applied on the action stream logits. The model is trained with an independent dropout of 20% on the speaker embedding and on the input text. Unless stated otherwise, we use α = 2.

## 4 Experiments

The paper continues with extensive experimental results showing DSM-ASR achieving 6.4% WER (competitive with top offline models) while being the only streaming model among top ASR systems, and DSM-TTS providing state-of-the-art performance with 100x real-time throughput on a single H100 GPU.

## 5 Conclusion

We introduce Delayed Streams Modeling, a flexible framework for streaming sequence-to-sequence learning. DSM provides a remarkable trade-off between quality and latency, and an unprecedented throughput among streaming models. Focusing on speech-text tasks, DSM-ASR is the first streaming ASR model to provide timestamped, formatted transcripts that competes with the top offline models, while DSM-TTS is competitive with non-streaming baselines while being the only model providing long form synthesis. In future work, we will extend DSM to more sequential multimodal tasks. In particular, one limitation of our approach is the need for aligned domains, which reduces the amount of gold-standard ground-truth data that can be used for training.

**Societal impact.** We acknowledge that streaming naturalistic speech with voice conditioning opens up both opportunities in inclusive human-machine interactions and risks of fraudulent impersonation. Addressing the latter requires that public access to such technologies is accompanied by proper user terms, voice verification mechanisms, and resilient watermarking of generated content. Given the limitations of such existing approaches, in particular for open source models, we have not open sourced the voice conditioning module for our best TTS model, only providing pre-computed speaker embeddings.