

class SmartMeta:
    def __init__(self,*args, **kwargs)->None:
        if(kwargs is None):
            kwargs = {}
        if(args is None):
            args = []
        self.kwargs = kwargs
        self.args = args
        return