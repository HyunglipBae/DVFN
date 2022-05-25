# Deep Value Function Networks

This repository is the official code of Deep Value Function Networks (DVFN)

DVFN is neural-networks based stagewise decomposition algorithm for large-scale multistage stochastic programming problems

![figure_0](https://user-images.githubusercontent.com/105804347/169448524-932f1486-e376-4a8c-965a-4860e0c83ea0.jpg)

## Version of Packages

numpy == 1.19.5

sympy == 1.8

tensorflow == 2.5.0

## Numerical Examples

1. Production Planning

1.1 Variables

![image](https://user-images.githubusercontent.com/105804347/170276127-e16a8398-153c-4aaa-b5bb-5df7afe9a310.png)

1.2 Formulation

![image](https://user-images.githubusercontent.com/105804347/170278021-904ff57d-22f0-4f28-8ffa-f0b337b1d49b.png)

2. Energy Planning

2.1 Variables

![image](https://user-images.githubusercontent.com/105804347/170278369-315517b5-a104-425c-903d-018445f93baf.png)

2.2 Formulation

![image](https://user-images.githubusercontent.com/105804347/170278804-f3d7c929-948f-4fdc-82b6-a04d116186b4.png)

## Results

1. Hyperparameter tuning (* indicates the selected hyperparameter setting)

![image](https://user-images.githubusercontent.com/105804347/170280117-63bd0a30-f37d-4807-be4f-1ce34ebe5a8c.png)

2. Production Planning

2.1 Comparison of first stage solution

![image](https://user-images.githubusercontent.com/105804347/170279720-659e8ddb-4170-49fa-840f-a5da75c97b4a.png)

![image](https://user-images.githubusercontent.com/105804347/170279820-d8f24476-c414-4e42-8ace-ecde7754877c.png)

2.2 Perturbation analysis on maximum resource

![image](https://user-images.githubusercontent.com/105804347/170280339-6dbf30fc-4341-49dc-a7d9-65926f3fd5c0.png)

3. Energy Planning

3.1 Comparison of first stage solution

![image](https://user-images.githubusercontent.com/105804347/170280804-ddb8b883-e8db-4775-9eb6-bcc47ab93e7a.png)

![image](https://user-images.githubusercontent.com/105804347/170280900-6b28987b-5fd4-41fc-8ddb-a09c43e425b9.png)

3.2 Perturbation analysis on hydro generation cost and thermal generation cost

![image](https://user-images.githubusercontent.com/105804347/170281194-20e42d3d-6429-48b8-9bd0-521794235c3f.png)
