

class Goal:
    
    def __init__(self,function,index):
        self.index = index
        self.function = function
        
    def __call__(self, state):
        return self.function(state)
    
    def __str__(self):
        return "Goal {}".format(self.index)
    
    