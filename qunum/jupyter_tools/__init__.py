from IPython.display import display as disp, Markdown as Md, Math as Mt
from sympy import latex
def TeXCode(x)->None:
    print(latex(x))
    return 
def TeXit(x)->None:
    disp(Md(latex(x)))
    return

