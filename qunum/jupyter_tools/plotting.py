from plotly.offline import iplot, iplot_mpl, init_notebook_mode
from plotly import express as px, graph_objects as go, subplots as plty_sub, io as pio
def setup_plotly(default_template:str = 'presentation', nb:bool = True)->None:
    pio.templates.default = default_template
    init_notebook_mode(nb)
    return
def example_template():
    pass
