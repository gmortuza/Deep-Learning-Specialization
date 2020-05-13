# Recurrent Neural Networks

1. Suppose your training examples are sentences (sequences of words). Which of the following refers to the j<sup>th</sup> word in the i<sup>th</sup> training example?

	- [x] x<sup>(i)\<j> </sup>
	- [ ] x<sup>\<i>(j) </sup>
	- [ ] x<sup>(j)\<i> </sup>
	- [ ] x<sup>\<j>(i) </sup>

2. Consider this RNN:
![Image 2](img/2.png)
This specific type of architecture is appropriate when:  
	- [x] T<sub>x</sub> = T<sub>y</sub>  
	- [ ] T<sub>x</sub> < T<sub>y</sub>  
	- [ ] T<sub>x</sub> > T<sub>y</sub>  
	- [ ] T<sub>x</sub> = 1  

3. To which of these tasks would you apply a many-to-one RNN architecture? (Check all that apply).
![Image 3](img/3.png)  
	- [ ] Speech recognition (input an audio clip and output a transcript)  
	- [x] Sentiment classification (input a piece of text and output a 0/1 to denote positive or negative sentiment)  
	- [ ] Image classification (input an image and output a label)  
	- [x] Gender recognition from speech (input an audio clip and output a label indicating the speaker’s gender)  

4. You are training this RNN language model.
![Image 4](img/4.png)
At the t<sup>th</sup> time step, what is the RNN doing? Choose the best answer.  
	- [ ] Estimating P(y<sup>\<1></sup>, y<sup>\<2></sup>, ...., y<sup>\<t-1></sup>)
	- [ ] [] Estimating P(y<sup>\<1></sup>)
	- [x] Estimating P(y<sup>\<t></sup> | y<sup>\<1></sup>, y<sup>\<2></sup>, ...., y<sup>\<t-1></sup>)
	- [x] Estimating P(y<sup>\<t></sup> | y<sup>\<1></sup>, y<sup>\<2></sup>, ...., y<sup>\<t></sup>)
