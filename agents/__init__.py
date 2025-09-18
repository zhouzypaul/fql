from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iql_diffusion import IQLDiffusionAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.cfgrl import CFGRLAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iql_diffusion=IQLDiffusionAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    cfgrl=CFGRLAgent,
)
