# Client Link: https://NPFLBAAEVOXXUYZK.anvil.app/QED54JFBPJMZBQPITDWWVL75

from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):

  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    self.card_2.visible = False
    #anvil.Notification("Please make sure server is running!", timeout=1).show()
    # Any code you write here will run when the form opens.

  def button_1_click(self, **event_args):
    if not ((self.brand1Box.text and not self.brand1Box.text.isspace()) and (self.brand2Box.text and not self.brand2Box.text.isspace()) and (self.limitBox.text and int(self.limitBox.text)>=2000)):
      anvil.Notification("Check Input Values", timeout=1).show()
      return
    self.card_2.visible = False
    winner,objlist,num=anvil.server.call('comparator2',self.brand1Box.text,
                                  self.brand2Box.text,self.limitBox.text,
                                  int(self.radio_button_1.get_group_value()))
    self.card_2.visible = True
    self.image_1.source,self.image_2.source,self.image_3.source,self.image_4.source = objlist
    self.winnerLabel.text = str.title(winner)+' is more positively perceived by Twitter'
    self.tweetnum.text='(on the basis of positivity:negativity ratio of '+str(num)+' recent tweets)'
    anvil.Notification("Execution Complete.", timeout=2).show()
    return
