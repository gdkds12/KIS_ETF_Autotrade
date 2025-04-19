from enum import Enum, auto

class DiscordRequestType(Enum):
    ORDER_CONFIRMATION = auto()
    GENERAL_NOTIFICATION = auto()
    ALERT = auto()
    CYCLE_STATUS = auto() 