from IPython.display import display as disp, Markdown as Md, Math as Mt
from sympy import latex
TeXCode = lambda x: print(latex(x))
TeXit = lambda x:disp(Md(latex(x)))