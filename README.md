# Deep Value Function Networks

This repository is the official code of Deep Value Function Networks (DVFN)

DVFN is neural-networks based stagewise decomposition algorithm for large-scale multistage stochastic programming problems

![figure_0](https://user-images.githubusercontent.com/105804347/169448524-932f1486-e376-4a8c-965a-4860e0c83ea0.jpg)

Manuscript : [DVFN_Manuscript.pdf](https://github.com/NeurIPS-2022/DVFN/files/9129678/DVFN_Manuscript.pdf)

Appendix : [DVFN_Appendix.pdf](https://github.com/NeurIPS-2022/DVFN/files/9129679/DVFN_Appendix.pdf)

## Version of Packages

numpy == 1.19.5

sympy == 1.8

tensorflow == 2.5.0

## Numerical Example: Production Planning

1 Variables and Formulation

![image](https://user-images.githubusercontent.com/105804347/170276127-e16a8398-153c-4aaa-b5bb-5df7afe9a310.png)

![image](https://user-images.githubusercontent.com/105804347/170278021-904ff57d-22f0-4f28-8ffa-f0b337b1d49b.png)

2 Comparison of first stage solution

![image](https://user-images.githubusercontent.com/105804347/170279720-659e8ddb-4170-49fa-840f-a5da75c97b4a.png)

![image](https://user-images.githubusercontent.com/105804347/170281566-85738445-5a89-4d66-b33b-af139ee674f3.png)

3 Perturbation analysis on maximum resource

![image](https://user-images.githubusercontent.com/105804347/170280339-6dbf30fc-4341-49dc-a7d9-65926f3fd5c0.png)

## Numerical Example: Energy Planning

1 Variables and Formulation

![image](https://user-images.githubusercontent.com/105804347/170278369-315517b5-a104-425c-903d-018445f93baf.png)

![image](https://user-images.githubusercontent.com/105804347/170278804-f3d7c929-948f-4fdc-82b6-a04d116186b4.png)

2 Comparison of first stage solution

![image](https://user-images.githubusercontent.com/105804347/170280804-ddb8b883-e8db-4775-9eb6-bcc47ab93e7a.png)

![image](https://user-images.githubusercontent.com/105804347/170280900-6b28987b-5fd4-41fc-8ddb-a09c43e425b9.png)

3 Perturbation analysis on hydro generation cost and thermal generation cost

![image](https://user-images.githubusercontent.com/105804347/170281194-20e42d3d-6429-48b8-9bd0-521794235c3f.png)

## Hyperparameter Tuning (* indicates the selected hyperparameter setting)

![image](https://user-images.githubusercontent.com/105804347/170280117-63bd0a30-f37d-4807-be4f-1ce34ebe5a8c.png)

## Effect of Activation Functions on Input Convex Neural Networks

We adopted Input Convex Neural Networks (ICNNs) to approximate the value function since it is generally convex on mild condition (See Appendix A). To preserve convex structure, ICNNs require convex activation function. The examples of convex activation functions are relu, leaky-relu, elu, selu and softplus. In this paper, we used elu activation function which is continuous, differentiable on everywhere and partially linear. Together with the previous results, we investigate the effect of activation function on ICNNs. As Appendix D, we solved the production and energy planning problems with DVFN for 20 times with different activation functions, and the box plots of logarithms of gradient loss of stage 0 ICNNs are represented. As expected, the elu function shows good performance, and it was newly verified the the softplus function is also a good choice.

![boxplot_ICNN](https://user-images.githubusercontent.com/105804347/179458542-2f0a4661-f213-4201-88ff-f4eca2ffe826.png)
![boxplot_ICNN](https://user-images.githubusercontent.com/105804347/179458639-683c3642-df31-41a6-9a09-700f189aa83a.png)
