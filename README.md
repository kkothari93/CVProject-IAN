# CVProject-IAN

This is an implementation of the introspective adverserial networks as proposed by [Brock et al](https://arxiv.org/abs/1609.07093) for a CS543:Computer Vision class project. This project was our first dive into generative modeling into neural nets and hence, we took this opportunity to learn more about the literature in generative modeling and try our hands in implementing various variants that people have presented. All our results were on face images from celebA dataset.

## GANs attempted

We coded all GANs from scratch in order to test our understanding of the papers. We managed to get decent results in all variants with the same hyperparameters. However, the results were not as great as those shown in the papers. Some of our results are shown later.

-- [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets): We implement a standard fully convolutional GAN to generate face images using the JS divergence metric (i.e. maximizing log-likelihood of samples)

-- [wGAN](https://arxiv.org/abs/1701.07875): We maintain the same convolution structure but instead of an f-divergence we use a distance metric between distribution (Wasserstein distance). We also tried a supposedly improved variant of this GAN (one with gradient penalty that is based on unit norm gradient property of an optimal critic) however, we did not see much improvement in our result.

-- [swGAN](https://arxiv.org/abs/1803.11188): Again, same architecture, but a different way of estimating the Wasserstein distance metric.

## Front end

-- We borrowed most of our front end from the original authors of the paper and wrote just an API to interface our models with their specifications.

-- A demo of the interface in action can be found [here](https://www.youtube.com/watch?v=91MqRQ8sTig).

## Things we could not complete

-- We trained all modules in the network together and this is why the generator sample quality was not that great. We wanted to train the inference network separately after training the GAN modules. We would expect to see significant improvements.

-- The interface could be vastly improved (made non-hacky).


