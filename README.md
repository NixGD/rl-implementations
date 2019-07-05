# vpg
A from-scratch implementation of a simple Vanilla Policy Gradient.

The algorithm supports multiple different methods to estimate the gradient.  These all take the form of the right hand side
![equation](https://spinningup.openai.com/en/latest/_images/math/1485ca5baaa09ed99fbcc54ba600e36852afd36c.svg)
where &#934;<sub>t</sub> can be one of several different options (controlled by the `method` parameter):

 - `trajectory`: The total reward from the trajectory.
 - `togo`: The reward to go, accumulated from all later action (_a<sub>i</sub>_ for all _i &geq; t_)
 - `value baseline` (default): The difference between the reward to go and the estimated value of the state.

 While all these have the right gradient in expecation, they have different variances.  For quickest training, use `value baseline`
