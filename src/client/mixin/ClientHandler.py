import copy
import time

from core.handlers.Handler import Handler


class UpdateSender(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.upload()
        return request


class UpdateReceiver(Handler):
    def _handle(self, request):
        client = request.get('client')
        weights_buffer = client.message_queue.get_from_downlink(client.client_id, 'weights')
        state_dict = client.model.state_dict()
        for k in weights_buffer:
            if client.training_params[k]:
                state_dict[k] = copy.deepcopy(weights_buffer[k])
        del weights_buffer
        client.model.load_state_dict(state_dict)
        client.time_stamp = client.message_queue.get_from_downlink(client.client_id, 'time_stamp')
        client.schedule_t = client.message_queue.get_from_downlink(client.client_id, 'schedule_time_stamp')
        client.receive_notify()
        return request


class DelaySimulator(Handler):
    def _handle(self, request):
        client = request.get('client')
        train_time = request.get('train_time')
        # print('train_time',train_time)
        sleep_time = max(client.delay - train_time,0)
        if hasattr(client, 'delay_simulate'):
            client.delay_simulate()
        else:
            time.sleep(sleep_time)
        return request

