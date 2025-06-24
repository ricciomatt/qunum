from typing import Any, Iterable, Generator
from plotly import express as px
from json import load
from plotly import graph_objects as go
from plotly.offline import iplot, init_notebook_mode
from plotly import io as pio
import os
from copy import copy
import numpy as np

class PlotIt:
    def __init__(self, data:list[dict]|None = None, frames:list[list[dict]]|None = None, offline:bool = False, connected:bool = False)->None:
        self.offline = offline
        self.connected = connected
        if(offline):
            init_notebook_mode(connected=connected)
        a = plotly_configs()
        self.configs:dict = a['configs']
        self.layout:dict = a['layout']
        self.def_plot:dict = a["defaults"]['plot']
        self.polar_plot:dict = a['defaults']['polar_plot']
        self.curve_names = set({})
        if(data is None):
            self.data:list[dict] = []
        else:
            self.data:list[dict] = data
        if(frames is None):
            self.frames:list[list[dict]] = [[]]
        else:
            self.frames:list[list[dict]] = frames
        self.animate_menu = a['defaults']['animate_menu']
        return
    
    def update_configs(self, **kwargs:dict)->None:
        self.configs = unpack_assign(kwargs, copy(self.configs))
        return
    
    def update_layout(self, **kwargs:dict)->None:
        self.layout = unpack_assign(kwargs, copy(self.layout))
        return 
    
    def update_data(self, curve_name_or_idx:int|str, **kwargs)->None:
        if(isinstance(curve_name_or_idx, str)):
            curve_name_or_idx = self.curve_name_to_idx(curve_name_or_idx)
        self.data[curve_name_or_idx].update(kwargs)
        return

    def add_subplot(self, domain:dict[str:tuple[float,float], str:tuple[float,float]] = {'x':(.5, 1), 'y':(0,1)}, kind:str = 'cartesian', fix_overlap:bool = True, overlap:str='y')->None:
        match overlap:
            case 'x':
                no = 'y'
            case 'y':
                no = 'x'
            case _:
                raise ValueError('Overlap Must be "x" or "y"')
        axis = (list(filter(lambda x: f'{no}axis' in x, list(self.layout.keys()))))
        def add_cartesian():
            ax_def = plotly_configs()['defaults']['axis']
            n = len(axis)
            self.layout[f'yaxis{n+1}'] = copy(ax_def)
            self.layout[f'xaxis{n+1}'] = copy(ax_def)
            if(fix_overlap):
                dx = 1 - (domain[no][1]-domain[no][0])
                for i in range(1,n+1):
                    match i:
                        case 1:
                            t = f'{no}axis'    
                        case _:
                            t = '{no}axis{i}'.format(i = i, no = no)
                    self.layout[t]['domain'] = (self.layout[t]['domain'][0]*dx, self.layout[t]['domain'][1]*dx) 
            self.layout[f'yaxis{n+1}']['domain'] = domain['y']
            self.layout[f'xaxis{n+1}']['domain'] = domain['x']
            self.layout[f'yaxis{n+1}']['anchor'] = f'x{n+1}'
            self.layout[f"xaxis{n+1}"]['anchor'] = f"y{n+1}"
            return 
        def add_polar():
            ax_def = plotly_configs()['defaults']['polar']
            n = len(axis)
            self.layout[f'yaxis{n+1}'] = copy(ax_def)
            self.layout[f'xaxis{n+1}'] = copy(ax_def)
            if(fix_overlap):
                dx = 1 - (domain[no][1]-domain[no][0])
                for i in range(1,n+1):
                    match i:
                        case 1:
                            t = f'{no}axis'    
                        case _:
                            t = '{no}axis{i}'.format(i = i, no = no)
                    self.layout[t]['domain'] = (self.layout[t]['domain'][0]*dx, self.layout[t]['domain'][1]*dx) 
            self.layout[f'polar{n+1}']['domain'] = domain
            return
        
        match kind.lower():
            case 'cartesian':
                add_cartesian()
            case 'polar':
                add_polar()
            case _:
                raise ValueError('Can Only add Cartesian or Polar subplots, {kind} is not supported'.format(kind = kind))
        return

    def show(self)->None:
        Fig = self.getFig()
        if(self.offline):
            return iplot(Fig, config=self.configs)
        else:
            return Fig.show(config=self.configs)
    
    def to_image(self, file:str = f'{os.getcwd()}/defualt.png', format='png', **kwargs)->None:
        pio.write_image(self.getFig(), file=file, format = format,  **kwargs)
        return

    def getFig(self)->go.Figure:
        match (self.data, self.frames):
            case data, frames if data == [] and frames == [[]]:
                data = None
                frames = None
                layout = self.layout
            case data, frames if data != [] and frames == [[]]:
                data = self.data
                frames = None
                layout = self.layout
            case data,frames if  data != [] and frames != [[]]:
                frames = self.frames
                data = self.data
                layout = copy(self.layout)
                if("updatemenus" not in layout):
                    layout['updatemenus'] = self.animate_menu
                else:
                    layout['updatemenus'].extend(self.animate_menu)
            case data, frames if data == [] and frames != [[]]:
                frames = self.frames
                data = None
                layout = copy(self.layout)
                if("updatemenus" not in layout):
                    layout['updatemenus'] = self.animate_menu
                else:
                    layout['updatemenus'].extend(self.animate_menu)
              
        return go.Figure(data=data, frames=frames, layout = layout)

    def remove_data(self, curve_name:str)->None:
        self.data = list(filter(lambda x: x['name']!=curve_name, self.data))
        return

    def ins_curve_name(self,curve_name:str|None, l:int = 0)->str:
        if(curve_name is None):
            return self.ins_curve_name(f'Curve{l}', l)
        if(curve_name not in self.curve_names):
            self.curve_names.add(curve_name)
            return curve_name
        else:
            return self.ins_curve_name(None, l+1)
    
    def extend_data(self, *data:tuple[dict]|Generator[dict, None, None])->None:
        for d in data:
            self.add_data(**d)
        return 

    def add_data(self, **data_in:dict)->None:
        if('name' in data_in):
            nm = data_in['name']
        else:
            nm = None
        nm = self.ins_curve_name(nm, 0)
        if('type' in data_in):
            tp  = data_in['type']
        else:
            tp = None
        match tp:
            case 'scatterpolar':
               plot = copy(self.polar_plot)
            case 'scatter':
                plot = copy(self.def_plot)
            case None:
                plot = copy(self.def_plot)
            case _:
                plot = copy(self.def_plot)
        plot['name'] = nm
        plot.update(data_in)
        self.data.append(plot)
        return

    def add_to_frame(self, frame_idx:int, **data:dict)->None:
        self.frames[frame_idx].append(data)
        return
    
    def add_frame(self, data:list[dict])->None:
        self.frames.append([ self.chk_data(data_in=d) for d in data])
        return

    def update_animation(self, *data:list[list[dict]])->None:
        for d in data:
            self.add_frame(data = d)
        return 

    def animation_speed(self, frame_duration:int = 20, transition_duration:int = 20)->None:
        self.animate_menu[0]['buttons'][0]['args'][1]["frame"]["duration"] = frame_duration
        self.animate_menu[0]['buttons'][0]['args'][1]["transition"]["duration"]= transition_duration
        return 

    def chk_data(self, **data_in:dict)->dict:
        if('name' in data_in):
            nm = data_in['name']
        else:
            nm = None
        nm = self.ins_curve_name(nm, 0)
        if('type' in data_in):
            tp  = data_in['type']
        else:
            tp = None
        match tp:
            case 'scatterpolar':
               plot = copy(self.polar_plot)
            case 'scatter':
                plot = copy(self.def_plot)
            case None:
                plot = copy(self.def_plot)
            case _:
                raise ValueError('Must Provide valid For plot type: {tp} is not valid'.format(tp = tp))
        plot['name'] = nm
        plot.update(data_in)
        return plot        

    def curve_name_to_idx(self, curve_name:str, data:list[dict])->int:
        idx = np.where(np.fromiter(map(lambda x: x['name']==curve_name, data), dtype=np.bool_))[0]
        try:
            return idx[0]
        except:
            raise ValueError['Curve, {curve_name} not found in {data}'.format(curve_name=curve_name)]
        
    def remove_frame(self, frame_idx:int)->None:
        del self.frames[frame_idx]
        return 
    
    def remove_data_from_frame(self, frame_idx:int, curve_name:str)->None:
        self.frames[frame_idx] = filter(lambda x: x['name']!=curve_name, self.frames[frame_idx])
        return 

    def __repr__(self)->str:
        self.show()
        return ''
    
    def reset(self)->None:
        self.__init__(offline = self.offline, connected = self.connected)
        return

def plotly_configs()->dict[str:dict, str:dict, str:dict]:
    with open('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/plotting_configs.json', 'r') as k:
        configs = load(k)
    return configs

def unpack_assign(kwargs:dict, tgt:dict)->dict:
    for key in kwargs:
        if(isinstance(kwargs[key], dict)):
            tgt[key] = unpack_assign(kwargs[key], tgt[key])
        else: 
            tgt[key] = kwargs[key]
    return tgt