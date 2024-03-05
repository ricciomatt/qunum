from IPython.display import display as disp, Markdown as Md, Math as Mt
from sympy import latex
from .plotting import setup_plotly
def TeXCode(x)->str:
    print(latex(x))
    return latex(x)
def TeXit(x)->None:
    disp(Md(f"${latex(x)}$"))
    return
def TeXdisp(x:str, n_dollar=1)->None:
    if(n_dollar != 1):
        disp(Md('$$'+x+'$$'))
    else:
        disp(Md('$'+x+'$'))
    return