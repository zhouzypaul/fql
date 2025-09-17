from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.cfgrl import CFGRLAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    cfgrl=CFGRLAgent,
)
