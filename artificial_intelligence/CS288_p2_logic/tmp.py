import logicPlan

from logic import PropSymbolExpr as PSE

model = {PSE("North",100):True, PSE("P",3,4,1):True,PSE("P",3,3,1):False, PSE("West",0):True,PSE("GhostScary"):True, PSE("West",2):False,PSE("South",1):True, PSE("East",0):False}

actions = ['North', 'South', 'East', 'West']

plan = logicPlan.extractActionSequence(model, actions)
print(plan)
