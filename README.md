# WIP WIP WIP WIP WIP

# dingocar-research
A stripped back version of the donkeycar ml models for tinkering with and exploring some crazy ideas I have.

# Idea 1

I've been working on ways to make a model trained using simulated data transfer to the real world. The _naive_ approach would be to train on a mixture of simulated data and real-world data. The hope being that the model learns the similar features shared by the simulated and real-world environments resulting in a model that transfers well between the two.

The reality is, beacuse deep learning models are annoyingly good at function approximation the tend to just learn two different ways to do the same task. Intuitavelly you can think of it was the simulated images taking one path through the network while the real-world data takes another path. In ML speek this is know as; _the model exhibiting bi-modal behaviour_.

To attack this problem I have read about a few differing approaches.

1. Trick the model into _thinking_ it is looking at real-world data when infact it is looking at a simulated image that has been pre-processed to look like a real-world image. This is the approach taken by [1](https://arxiv.org/pdf/1709.07857.pdf), [2](https://web.stanford.edu/~jaustinb/posters/CS236.pdf) and [myself](https://www.youtube.com/watch?v=uNJ_bljizAM). All these attempts utialize a Cycle Consiistant Generative Adversiral Network archetecture, aka [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf) to transform simulated images to something that looks like a real-world image. Visually these transforms appear pretty bloody good.

**TODO: Embed video**

This approach seems intuatively like it should work, and in practice it shows some promice but unfortunately its not the silver buliet everyone was hoping form :-( 

2. Randomize the crap out of the simulated world so the model is forced to learn a extreemly robust solution. I personally love this idea, though I have not tried it myself yet. This technique was used by [Open AI](https://openai.com/blog/learning-dexterity/) to train their robotic hand to manipulate a cube in simulation and then deploy the model in the real-world. Also [3](https://arxiv.org/pdf/1703.06907.pdf) [4](https://arxiv.org/pdf/1611.04201.pdf) and [5](https://arxiv.org/pdf/1710.06537.pdf) have adoped a similar approach. What is cool about these approaches is not only are the visual features randomized but also the physics. What is even coolerer is when a recurent model is used, (Open AI and [5] did this) the model does a spot of meta-learning. As the agent interacts with its environment over a few time steps it is able to _infer_ how the physics of that world behaves and adjust it's policy to suit. Another way I like to think of the domain randomization approach is it is almost the polar oposite of the Cycle GAN approach. Rather than trying to make the simulated data _look like_ the real-world. This approach removes anything the model could focus on that is not essential to the task at hand. This technique does tend to take longer to converge than if you just had real-world data, but I think that's a small price to pay for the ability to train in simulation.

3. **TO DO: Mention This** [6](https://arxiv.org/pdf/1909.11825.pdf)

## My approach

This approach is sort-of similar to point 3 in that I want to prevent them model learning a bi-modal solution to the problem. It is also similar to the Cycle GAN approach in that I want to learn features from simulated data that are indistinguishable from real-world features. 

The keys is I'm focusing on the _feature space_ directly NOT _pixel space_ (ie: relying on image reconstruction). To do this, in addition to the 2 heads for steering and throttle control. I also have 2 additional heads for sim/real clasification and what I call _smoosh_ (because I don't know the term for forcing one distribution to look like another). 

Our goal is, when presented with an input image from either the sim or real-world we dont want any domain specific features to be present in the feature vecture. Put another way, given the feature vector, we don't want to be able to reliabily predict wether the input was a simulated or real-world image.

To achieve this, as far as I know optimizing for 'confusion' while simultainously optimizing for accuracy is not really a thing...? (hit me up if you have a better way, there is probably one out there). Neively I tried the following first time round:

`loss = mse(steering) + mse(throttle) - mse(is_real_classification)`

The logic being that `loss = mse(steering) + mse(throttle) !!!PLUS!!! mse(sim_real_classification)` would optimise for steering and throttle while making the sim\_real\_classification *good* as telling a sim from real. So `loss = mse(steering) + mse(throttle) !!!MINUS!!! mse(is_real_classification)` would optimise for steering and throttle while making the sim\_real\_classification *bad*.

 At train when presented with the feature vector we want the sim/real classification layer to get good at telling the difference between simulated and real images, and we want 
