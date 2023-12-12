# EconCBF
Optimal Control Theory and Control Barrier Functions applied to Economics and Finance

_Abstract_ - Control barrier functions (CBFs) and safety-critical control have seen a rapid increase in popularity in recent years, predominantly applied to systems in aerospace, robotics and neural network controllers. Control barrier functions can provide a computationally efficient method to monitor arbitrary primary controllers and enforce state constraints to ensure overall system safety. One area that has yet to take advantage of the benefits offered by CBFs is the field of finance and economics. This manuscript re-introduces three applications of traditional control to economics, and develops and implements CBFs for such problems. We consider the problem of optimal advertising for the deterministic and stochastic case and Merton's portfolio optimization problem. Numerical simulations are used to demonstrate the effectiveness of using traditional control solutions in tandem with CBFs and stochastic CBFs to solve such problems in the presence of state constraints.

**To Run the Code:**
1) Install the necessary packages using pip install -r requirements.txt
2) Run any of the examples by calling python _example_.py where _example_ is the name of the script

For more details on each problem, see [Econ_CBFs_vanWijk.pdf](/Econ_CBFs_vanWijk.pdf)

_Optimal Advertising (deterministic)_

The optimal advertising problem introduced in [Weber 2011](https://mitpress.mit.edu/9780262015738/optimal-control-theory-with-applications-in-economics/) considers the problem of maximizing the discounted profits of a firm using advertising.

_Optimal Advertising (stochastic)_

The stochastic control example used is another example of optimal advertising, but with slight model modifications, as well as stochasticity injected into the system. The problem is a variation of the [Vidale-Wolfe](https://pubsonline.informs.org/doi/abs/10.1287/opre.5.3.370) advertising model. In this example, the goal is still to maximize discounted profits, but since stochasticity is introduced, the expected value of the discounted profits must be maximized.

_Merton's Portfiolio Optimization_

For the final application of CBFs to economics and finance, a classic portfolio optimization example is used. First introduced by [Merton 1969](https://www.jstor.org/stable/1926560), the Merton problem is a problem in continuous-time finance wherein an investor seeks to maximize the expected discounted wealth by trading in a risky asset and risk-free bank account. 
