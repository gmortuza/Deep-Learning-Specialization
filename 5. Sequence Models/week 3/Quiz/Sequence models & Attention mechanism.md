1. Consider using this encoder-decoder model for machine translation.
![Image 1](img/1.png)
This model is a “conditional language model” in the sense that the encoder portion (shown in green) is modeling the probability of the input sentence *x*.

    - [ ] True
    - [x] False

2. In beam search, if you increase the beam width BB, which of the following would you expect to be true? Check all that apply.
    - [x] Beam search will run more slowly
    - [x] Beam search will use up more memory.
    - [x] Beam serach will generally find better solutions(i.e. do a better job maximizing *P(y|x)*)
    - [ ] Beam search will converge after fewer steps.

3. In machine translation, if we carry out beam search without using sentence normalization, the algorithm will tend to output overly short translations.

    - [x] True
    - [ ] False

4. Suppose you are building a speech recognition system, which uses an RNN model to map from audio clip xx to a text transcript yy. Your algorithm uses beam search to try to find the value of yy that maximizes P(y∣x). 
On a dev set example, given an input audio clip, your algorithm outputs the transcript 
y = “I’m building an A Eye system in Silly con Valley.”, whereas a human gives a much superior transcript 
y\* = “I’m building an AI system in Silicon Valley.”
According to your model,
P(y|x) = 1.09 * 10<sup>-7</sup>
P(y|x) = 7.21 * 10<sup>-8</sup>
Would you expect increasing the beam width B to help correct this example?



    - [x] No, because P(y\*|x) ≤ P(y|x) indicates the error should be attributed to the RNN rather than to the search algorithm.
    - [ ] No, because P(y\*|x) ≤ P(y|x) indicates the error should be attributed to the search algorithm rather than to the RNN.
    - [ ] Yes, because P(y\*|x) ≤ P(y|x) indicates the error should be attributed to the RNN rather than to the search algorithm.
    - [ ] Yes, because  P(y\*|x) ≤ P(y|x) indicates the error should be attributed to the search algorithm rather than to the RNN.

5. Continuing the example from Q4, suppose you work on your algorithm for a few more weeks, and now find that for the vast majority of examples on which your algorithm makes a mistake, P(y\*|x) > P(y|x) This suggest you should focus your attention on improving the search algorithm.

    - [x] True
    - [ ] False

6. Consider the attention model for machine translation.
![Model](img/6.png)
Further, here is the formula for α<sup><t,t’></sup>.
![equation](img/6_1.png)
Which of the following statements about α<sup><t,t’></sup> are true? Check all that apply.

    - [x] We expect  α<sup><t,t’></sup> to be generally larger for values of a<t> that are highly relevant to the value the network should output for y<t’>. (Note the indices in the superscripts.)
    - [ ] We expect  α<sup><t,t’></sup> to be generally larger for values of a<sup><t></sup> that are highly relevant to the value the network should output for y<sup><t’></sup>. (Note the indices in the superscripts.)
    - [ ] ∑<sub>t</sub> α<sup> <t,t’></sup> =1 (Note the summation is over t.)
    - [x] ∑<sub>t'</sub> α<sup> <t,t’></sup> =1 (Note the summation is over t'.)
    
7. The network learns where to “pay attention” by learning the values e<sup><t,t’></sup>, which are computed using a small neural network: We can't replace s<sup><t−1></sup> with s<sup>t</sup> as an input to this neural network. This is because s<sup><t></sup> depends on α<sup><t,t’></sup> which in turn depends on e<sup><t,t’></sup> ; so at the time we need to evalute this network, we haven’t computed s<sup><t></sup> yet.
    
    - [x] True
    - [ ] False

8. Compared to the encoder-decoder model shown in Question 1 of this quiz (which does not use an attention mechanism), we expect the attention model to have the greatest advantage when:

    - [x] The input sequence length T<sub>x</sub> is large.
    - [ ] The input sequence length T<sub>x</sub> is small.

9. Under the CTC model, identical repeated characters not separated by the “blank” character (_) are collapsed. Under the CTC model, what does the following string collapse to?
__c_oo_o_kk___b_ooooo__oo__kkk

    - [ ] cokbok
    - [x] cookbook
    - [ ] cook book
    - [ ] coookkboooooookkk

10. In trigger word detection, x<sup><t></sup> is:

    - [x] Features of the audio (such as spectrogram features) at time t.
    - [ ] The tt-th input word, represented as either a one-hot vector or a word embedding.
    - [ ] Whether the trigger word is being said at time t.
    - [ ] Whether someone has just finished saying the trigger word at time t.

