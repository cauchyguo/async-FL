from time import sleep

from client.NormalClient import NormalClient
from utils.GlobalVarGetter import GlobalVarGetter

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver,DelaySimulator
from core.handlers.Handler import Handler

class FLEClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = 0

    def customize_upload(self):
        self.upload_item("group_id", self.group_id)

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(GroupSetter(), UpdateReceiver)
        self.handler_chain.add_handler_before(DelaySetter(), DelaySimulator)


class GroupSetter(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.group_id = client.message_queue.get_from_downlink(client.client_id, "group_id")
        return request
    
class DelaySetter(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.delay = client.message_queue.get_from_downlink(client.client_id, "delay")
        return request
    # def upload(self, data_sum, weights):
    #     update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
    #                    "time_stamp": self.time_stamp, "group_id": self.global_var['client_group_mapping'][self.client_id]}
    #     self.message_queue.put_into_uplink(update_dict)
