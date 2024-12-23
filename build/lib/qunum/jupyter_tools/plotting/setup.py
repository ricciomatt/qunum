from plotly.offline import iplot, iplot_mpl, init_notebook_mode
from plotly import express as px, graph_objects as go, subplots as plty_sub, io as pio

def setup_plotly(default_template:str = 'presentation', online:bool = True)->None:
    custom_template = {
    "layout": {
        "font": {
            "family": "Arial, sans-serif",
            "size": 14,
            "color": "#333333"
        },
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "title": {
            "font": {
                "size": 20,
                "color": "#000000"
            }
        },
        "xaxis": {
            "title": {
                "font": {
                    "size": 16,
                    "color": "#000000"
                }
            },
            "tickfont": {
                "size": 14,
                "color": "#000000"
            }
        },
        "yaxis": {
            "title": {
                "font": {
                    "size": 16,
                    "color": "#000000"
                }
            },
            "tickfont": {
                "size": 14,
                "color": "#000000"
            }
        },
        'height':750,
        'width':1500
    }
    }
    # Register the custom template
    pio.templates['custom_template'] = custom_template

    # Set the default template
    pio.templates.default = 'custom_template'

    pio.templates.default = default_template
    init_notebook_mode(online)
    return


def example_template():
    pass


